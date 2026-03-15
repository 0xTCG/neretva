
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from common_cyp import *

BASES = ['A', 'C', 'G', 'M', 'N', 'P', 'T']
TOP_CONFI = 1 - (1e-300)

def classify_mutation_type(allele, position, variant):
    if variant.is_novel:
        return 'type6-novel'
    
    # Determine position type
    position_has_functional = any(v.is_major for v in position.variants.values() if v.is_major)
    position_has_minor = any(v.is_minor for v in position.variants.values() if v.is_minor)
    
    # Check what variant this allele actually has
    if variant.is_wildtype:
        if position_has_functional:
            return 'type1-func-ref'  # Functional position, allele has reference
        elif position_has_minor:
            return 'type3-minor-ref'  # Minor position, allele has reference
        else:
            return 'type7-wildtype'  # Default to minor-ref for other wildtype
    else:
        # Allele has a mutation
        if variant.is_major:
            return 'type2-func-mut'  # Functional position, allele has the mutation
        elif variant.is_minor:
            return 'type4-minor-mut'  # Minor position, allele has the mutation
        else:
            return 'type6-novel'  # Unknown mutation type

def get_confidence_for_type(g,a,p, mutation_type):
    confidences = {
            'type1-func-ref': TOP_CONFI,
            'type2-func-mut': 0.8,
            'type3-minor-ref': 0.8,
            'type4-minor-mut': TOP_CONFI,
            # 'type5-cross-gene': TOP_CONFI,
            'type6-novel': TOP_CONFI,
            'type7-wildtype': TOP_CONFI
        }

    return confidences.get(mutation_type)

def logit(p, eps=1e-7):
    if not isinstance(p, torch.Tensor):
        p = torch.tensor(p, dtype=torch.float32)
    
    p = torch.clamp(p, eps, 1 - eps)  # Avoid log(0)
    return torch.log(p / (1 - p))


def create_sparse_priors(sample, db):
    PRIOR_MU_DEFAULT = -1000.0
    PRIOR_LOGVAR_DEFAULT = -7.0
    sparse_prior_mus = []
    sparse_prior_logvars = []
    valid_indices_set = set(sample.valid_indices) 
    for allele in sample.valid_alleles:
        for pos_obj in allele.generatable_positions:
            allele_variants = []
            for variant in pos_obj.variants.values():
                if allele.extended_allele_vector[variant.index] == 1:
                    allele_variants.append(variant)
            
            assert len(allele_variants) == 1
            allele_variant = allele_variants[0]
            
            mut_type = classify_mutation_type(allele, pos_obj, allele_variant)
            confidence = get_confidence_for_type(allele.gene.name, allele.name, pos_obj.position, mut_type)
            
            if mut_type in ['type1-func-ref', 'type2-func-mut']:
                log_var = -8.0
            elif mut_type in ['type3-minor-ref', 'type4-minor-mut']:
                log_var = -8.0
            else:
                log_var = PRIOR_LOGVAR_DEFAULT
            
            num_other_variants = len(pos_obj.variants.values()) - 1
            non_variant_confidence = (1.0 - confidence) / num_other_variants if num_other_variants > 0 else 0.0
            
            position_mus = []
            position_logvars = []
            
            for base in BASES:
                if allele_variant.variant == base:
                    position_mus.append(logit(confidence))
                elif any(variant.variant == base for variant in pos_obj.variants.values()):
                    position_mus.append(logit(non_variant_confidence))
                else:
                    position_mus.append(PRIOR_MU_DEFAULT)
                position_logvars.append(log_var)
            
            sparse_prior_mus.extend(position_mus)
            sparse_prior_logvars.extend(position_logvars)

    return torch.tensor(sparse_prior_mus), torch.tensor(sparse_prior_logvars)
    
def decode_allele_beta(allele_idx, allele, sample, db, learned_beta):
    if allele_idx == 0:
        print(f"\n=== DECODE DEBUG ===")
        print(f"learned_beta type: {type(learned_beta)}")
        print(f"learned_beta shape: {learned_beta.shape}")
        print(f"learned_beta[:2, :5]: {learned_beta[:2, :5]}")
        print(f"learned_beta sum: {learned_beta.sum().item():.6f}")
        print(f"learned_beta device: {learned_beta.device if hasattr(learned_beta, 'device') else 'N/A'}")
        
    allele_name = f"{allele.gene.name}*{allele.name}"
    allele_row = learned_beta[allele_idx]  # Get this allele's row
    db_allele_row = torch.tensor(sample.beta[allele_idx])  # Always use sample.beta as DB reference
    
    position_data = {}
    
    for pos_obj in allele.generatable_positions:
        gene_name = pos_obj.gene.name
        pos_key = (gene_name, pos_obj.position)  # Use (gene, position) tuple as key
        
        if pos_key not in position_data:
            position_data[pos_key] = {
                'gene': gene_name,
                'position': pos_obj.position,
                'gene_position': f"{gene_name}:{pos_obj.position}",
                'ref_base': pos_obj.ref_base,
                'database_base': None,
                'mutation_type': None,
                'learned_probs': {base: 0.0 for base in BASES},
                'db_probs': {base: 0.0 for base in BASES}
            }
        
        # Now find the corresponding mutation indices for this position
        for variant in pos_obj.variants.values():
            if variant.index in sample.valid_indices:
                # Find the mutation index in the valid_indices
                mut_idx = sample.valid_indices.index(variant.index)
                
                # Fill in learned probabilities
                learned_prob = allele_row[mut_idx].item()
                position_data[pos_key]['learned_probs'][variant.variant] = learned_prob
                
                # Fill in database probabilities (from sample.beta)
                db_prob = db_allele_row[mut_idx].item()
                position_data[pos_key]['db_probs'][variant.variant] = db_prob
    
    # Determine database base and mutation type for each position
    for pos_key, pos_info in position_data.items():
        gene_name, position_num = pos_key
        pos_obj = db.genes[gene_name].positions[position_num]
        
        # Find database expected base (what sample.beta says this allele should have)
        db_probs = pos_info['db_probs']
        database_base = max(db_probs.items(), key=lambda x: x[1])[0]  # Base with highest prob in sample.beta
        pos_info['database_base'] = database_base
        
        # Determine mutation type using the database expected variant
        pos_info['mutation_type'] = get_mutation_type_from_variant(allele, pos_obj, database_base)
        
        # Find learned base (highest probability)
        learned_probs = pos_info['learned_probs']
        max_base = max(learned_probs.items(), key=lambda x: x[1])
        pos_info['learned_base'] = max_base[0]
        pos_info['learned_base_prob'] = max_base[1]
    
    # Convert to sorted list of dictionaries
    result = []
    # Sort by gene name first, then by position
    for pos_key in sorted(position_data.keys(), key=lambda x: (x[0], x[1])):
        pos_info = position_data[pos_key]
        
        row = {
            'allele': allele_name,
            'gene_position': pos_info['gene_position'],
            'gene': pos_info['gene'],
            'position': pos_info['position'],
            'ref_base': pos_info['ref_base'],
            'database_base': pos_info['database_base'],
            'mutation_type': pos_info['mutation_type'],
            'learned_base': pos_info['learned_base'],
            'learned_prob': f"{pos_info['learned_base_prob']:.12f}",
        }
        
        # Add individual base probabilities
        for base in BASES:
            row[f'{base}_prob'] = f"{pos_info['learned_probs'][base]:.12f}"
        
        # Add database comparison
        row['db_base'] = pos_info['database_base']
        row['db_base_prob'] = f"{pos_info['db_probs'][pos_info['database_base']]:.3f}"
        
        # KEY FIX: differs_from_db compares learned vs original database expectation
        row['differs_from_db'] = pos_info['learned_base'] != pos_info['database_base']
        
        result.append(row)
    
    return result

def get_mutation_type_from_variant(allele, position, expected_base):

    # Check if this is a cross-gene position (infection)
    if position.gene.name != allele.gene.name:
        return "cross_gene"
    
    # Find the variant that matches the expected base
    expected_variant = None
    for variant in position.variants.values():
        if variant.variant == expected_base:
            expected_variant = variant
            break
    
    if expected_variant is None:
        return "unknown"
    
    # Use variant properties directly
    if expected_variant.is_wildtype:
        # Check if this position has any functional or minor mutations
        has_functional = any(v.is_major for v in position.variants.values() if v.is_major is not None)
        has_minor = any(v.is_minor for v in position.variants.values() if v.is_minor is not None)
        
        if has_functional:
            return "functional_ref"
        elif has_minor:
            return "minor_ref"
        else:
            return "wildtype"
    else:
        # Non-wildtype variant
        if expected_variant.is_major:
            return "functional_mut"
        elif expected_variant.is_minor:
            return "minor_mut"
        elif expected_variant.is_novel:
            return "novel"
        else:
            return "unknown"
def decode_all_alleles_beta(sample, db, learned_beta, save_to_file=None):
    all_results = {}
    
    print("\n" + "="*130)
    print("DECODED BETA MATRICES FOR ALL ALLELES")
    print("="*130)
    
    for allele_idx, allele in enumerate(sample.valid_alleles):
        allele_name = f"{allele.gene.name}*{allele.name}"
        
        allele_data = decode_allele_beta(allele_idx, allele, sample, db, learned_beta)
        
        all_results[allele_name] = allele_data
        
        cross_gene_count = sum(1 for row in allele_data if row['mutation_type'] == 'cross_gene')
        same_gene_count = len(allele_data) - cross_gene_count
        differs_count = sum(1 for row in allele_data if row['differs_from_db'])
        
        print(f"\n {allele_name} ({len(allele_data)} positions: {same_gene_count} same-gene, {cross_gene_count} cross-gene, {differs_count} differences)")
        print("-" * 130)
        
        # FIX: Added M to header
        print(f"{'Gene:Pos':<15} {'Ref':<3} {'DB':<3} {'Learned':<8} {'Diff':<4} {'Type':<15} {'Prob':<8} {'A':<8} {'C':<8} {'G':<8} {'M':<8} {'N':<8} {'P':<8} {'T':<8}")
        print("-" * 130)
        
        for row in allele_data:
            differs = "🔴" if row['differs_from_db'] else "✅"
            
            gene_pos = row['gene_position']
            if row['mutation_type'] == 'cross_gene':
                gene_pos = f"🔄{gene_pos}"
            
            # FIX: Added M_prob and increased precision to .6f
            print(f"{gene_pos:<15} "
                  f"{row['ref_base']:<3} "
                  f"{row['database_base']:<3} "
                  f"{row['learned_base']:<8} "
                  f"{differs:<4} "
                  f"{row['mutation_type']:<15} "
                  f"{row['learned_prob']:<8} "
                  f"{row['A_prob']:<8} "
                  f"{row['C_prob']:<8} "
                  f"{row['G_prob']:<8} "
                  f"{row['M_prob']:<8} "
                  f"{row['N_prob']:<8} "
                  f"{row['P_prob']:<8} "
                  f"{row['T_prob']:<8}")

    return all_results

def create_functional_observation_counts(sample, db):
    num_mutations = len(sample.valid_indices)
    
    observed_counts = torch.zeros(num_mutations)
    
    for mut_idx, variant_idx in enumerate(sample.valid_indices):
        variant = db.variants()[variant_idx]
        observed_counts[mut_idx] = variant.coverage.count
    
    return observed_counts

def create_functional_indices(sample, db):
    functional_indices = torch.zeros(len(sample.valid_indices), dtype=torch.bool)
    functional_positions = {gene.name: {pos for (pos, _) in gene.functional.keys() if gene.positions[pos].is_valid} for gene in db.genes.values()}
    
    for _, allele in enumerate(sample.valid_alleles):
        for func_pos in functional_positions[allele.gene.name]:
            pos_obj = allele.gene.positions.get(func_pos)
            for variant in pos_obj.variants.values():
                functional_indices[sample.valid_indices.index(variant.index)] = True
    
    return functional_indices

# def set_expected_position_strength(sample, db, lb_factor=0.8, ub_factor=1.2, min_lb=1.0):
#     for allele in sample.valid_alleles:
#         if not allele.enabled:
#             continue
#         allele.expected_position_strength = {}
#         for pos_obj in allele.generatable_positions:
        
#             for variant in pos_obj.variants.values():
#                 lb = lb_factor * sample.expected_coverage
#                 ub = ub_factor * sample.expected_coverage

#                 allele.set_position_strength(pos_obj, variant.variant, lb, ub)
def set_expected_position_strength(sample, db, lb_factor=0.8, ub_factor=1.2, min_lb=1.0):
    

    position_to_count = {}
    for allele in sample.valid_alleles:
        if not allele.enabled:
            continue
        for pos_obj in allele.generatable_positions:
            position = pos_obj.position
            if position not in position_to_count:
                pos_count = sum(v.coverage.count for v in pos_obj.variants.values())
                position_to_count[position] = pos_count
    
    if len(position_to_count) > 0:
        mean_position_count = sum(position_to_count.values()) / len(position_to_count)
    else:
        mean_position_count = sample.expected_coverage
    
    position_expected_coverage = {}
    for pos, count in position_to_count.items():
        if mean_position_count > 0:
            position_expected_coverage[pos] = (count / mean_position_count) * sample.expected_coverage
        else:
            position_expected_coverage[pos] = sample.expected_coverage
    
    for allele in sample.valid_alleles:
        if not allele.enabled:
            continue
        allele.expected_position_strength = {}
        
        for pos_obj in allele.generatable_positions:
            position = pos_obj.position
            pos_exp_cov = position_expected_coverage.get(position, sample.expected_coverage)
            total_non_zero = sum(1 for v in pos_obj.variants.values() if v.coverage.count > 0)
            for variant in pos_obj.variants.values():
                is_same_gene = (variant.gene.name == allele.gene.name)
                
                if is_same_gene:
                    lb =  min(lb_factor * pos_exp_cov, variant.coverage.count)
                    ub =  min(ub_factor * pos_exp_cov, variant.coverage.count)
                    

                    variant.lb = lb
                    variant.ub = ub
                else:
                    # Cross-gene: bounded by actual infection reads
                    if allele.extended_allele_vector[variant.index] != 1:
                        continue
                    total_infection_reads = sum(
                        len(read_ids)
                        for read_ids in allele.infection_sources[variant].values()
                    )
                    ub = min(pos_exp_cov, total_infection_reads)
                    lb = ub
                
                allele.set_position_strength(pos_obj, variant.variant, lb, ub)
