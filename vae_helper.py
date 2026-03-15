
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from common_0620B import *

# BASES = ['A', 'T', 'C', 'G', 'P', 'N']
BASES = ['A','C','G','N','P','T']
# TOP_CONFI = 0.9999999999999999999
TOP_CONFI = 1 - (1e-300)

def classify_mutation_type(allele, position, variant):
  
    if position.position == 288:
        check = 1
    # Check if this is a cross-gene variant
    if variant.position.gene.name != allele.gene.name:
        return 'type5-cross-gene'
    
    # if variant.is_N:
    #     pass
    # # Check if this is a novel mutation (not in database)
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
    
    # if g in ['KIR2DS2']:
    if g in ['KIR2DP1']:
        if p in [288]:
                # print('[capture] 288 from KIR2DP1')
                confidences = {
            
                    'type1-func-ref': 0.20,
                    'type2-func-mut': TOP_CONFI,
                    'type3-minor-ref': 0.80,
                    # 'type4-minor-mut': 0.80,
                    # 'type4-minor-mut': TOP_CONFI,
                    'type4-minor-mut': 0.90,
                    
                    'type5-cross-gene': TOP_CONFI,
                    'type6-novel': TOP_CONFI,
                    'type7-wildtype': TOP_CONFI
                }
        elif p in [9181]:
            confidences = {
            
                    'type1-func-ref': TOP_CONFI,
                    'type2-func-mut': 0.20,
                    'type3-minor-ref': 0.80,
                    # 'type4-minor-mut': 0.80,
                    # 'type4-minor-mut': TOP_CONFI,
                    'type4-minor-mut': 0.90,
                    
                    'type5-cross-gene': TOP_CONFI,
                    'type6-novel': TOP_CONFI,
                    'type7-wildtype': TOP_CONFI
                }
        else:
            
            confidences = {
            
                    'type1-func-ref': 0.80,
                    'type2-func-mut': TOP_CONFI,
                    'type3-minor-ref': 0.80,
                    # 'type4-minor-mut': 0.80,
                    # 'type4-minor-mut': TOP_CONFI,
                    'type4-minor-mut': 0.90,
                    
                    'type5-cross-gene': TOP_CONFI,
                    'type6-novel': TOP_CONFI,
                    'type7-wildtype': TOP_CONFI
                }
     
    elif g in ['KIR2DL5']:
     
        confidences = {
        
                'type1-func-ref': TOP_CONFI,
                'type2-func-mut': TOP_CONFI,
                'type3-minor-ref': 0.80,
                # 'type4-minor-mut': 0.80,
                # 'type4-minor-mut': TOP_CONFI,
                'type4-minor-mut': 0.90,
                
                'type5-cross-gene': TOP_CONFI,
                'type6-novel': TOP_CONFI,
                'type7-wildtype': TOP_CONFI
            }
    # elif g in ['KIR2DL1']:
    #     confidences = {

    #                 'type1-func-ref': TOP_CONFI,
    #                 'type2-func-mut': TOP_CONFI,
    #                 'type3-minor-ref': 0.90,
    #                 'type4-minor-mut': 0.90,
    #                 'type5-cross-gene': TOP_CONFI,
    #                 'type6-novel': TOP_CONFI,
    #                 'type7-wildtype': TOP_CONFI
    #             }
    elif g in ['KIR2DL1']:
        if p in [13433]:
            confidences = {

                        'type1-func-ref': TOP_CONFI,
                        'type2-func-mut': 0.30,
                        'type3-minor-ref': 0.90,
                        'type4-minor-mut': 0.90,
                        'type5-cross-gene': TOP_CONFI,
                        'type6-novel': TOP_CONFI,
                        'type7-wildtype': TOP_CONFI
                    }
        elif p in [5757, 13416]:
             confidences = {

                        'type1-func-ref': 0.10,
                        'type2-func-mut': TOP_CONFI,
                        'type3-minor-ref': 0.90,
                        'type4-minor-mut': 0.90,
                        'type5-cross-gene': TOP_CONFI,
                        'type6-novel': TOP_CONFI,
                        'type7-wildtype': TOP_CONFI
                    }
        elif p in [141, 642, 1983, 2467, 2468, 7049, 11611, 12629, 13643]:
            confidences = {

                            'type1-func-ref': TOP_CONFI,
                            'type2-func-mut': TOP_CONFI,
                            'type3-minor-ref': 0.30,
                            'type4-minor-mut': 0.90,
                            'type5-cross-gene': TOP_CONFI,
                            'type6-novel': TOP_CONFI,
                            'type7-wildtype': TOP_CONFI
                    }
        else:
            confidences = {

                        'type1-func-ref': TOP_CONFI,
                        'type2-func-mut': TOP_CONFI,
                        'type3-minor-ref': 0.90,
                        'type4-minor-mut': 0.90,
                        'type5-cross-gene': TOP_CONFI,
                        'type6-novel': TOP_CONFI,
                        'type7-wildtype': TOP_CONFI
                    }
    elif g in ['KIR2DL5']:
            confidences = {

            'type1-func-ref': TOP_CONFI,
            'type2-func-mut': TOP_CONFI,
            'type3-minor-ref': 0.60,
            'type4-minor-mut': 0.60,
            'type5-cross-gene': TOP_CONFI,
            'type6-novel': TOP_CONFI,
            'type7-wildtype': TOP_CONFI
        }
    # elif g in ['KIR2DL4']:
    #         confidences = {

    #         'type1-func-ref': TOP_CONFI,
    #         'type2-func-mut': TOP_CONFI,
    #         'type3-minor-ref': 0.99,
    #         'type4-minor-mut': 0.99,
    #         'type5-cross-gene': TOP_CONFI,
    #         'type6-novel': TOP_CONFI,
    #         'type7-wildtype': TOP_CONFI
    #     }
    elif g in ['KIR3DL3']:    

             confidences = {
                'type1-func-ref': TOP_CONFI,
                'type2-func-mut': TOP_CONFI,
                'type3-minor-ref': 0.30,
                'type4-minor-mut': 0.60,
                'type5-cross-gene': TOP_CONFI,
                'type6-novel': TOP_CONFI,
                'type7-wildtype': TOP_CONFI
            }

    elif g in ['KIR3DL2','KIR3DP1']:    
        confidences = {

            'type1-func-ref': TOP_CONFI,
            'type2-func-mut': TOP_CONFI,
            'type3-minor-ref': 0.30,
            'type4-minor-mut': 0.30,
            'type5-cross-gene': TOP_CONFI,
            'type6-novel': TOP_CONFI,
            'type7-wildtype': TOP_CONFI
        }
 

    elif g in ['KIR3DL1']:
        if p in [18, 1884, 6515, 6798, 7710, 9522, 10519]:
              confidences = {

            'type1-func-ref': TOP_CONFI,
            'type2-func-mut': TOP_CONFI,
            # 'type3-minor-ref': 0.90,
            # 'type3-minor-ref': TOP_CONFI,
            'type3-minor-ref': 0.30,
            
            'type4-minor-mut': 0.99,
            'type5-cross-gene': TOP_CONFI,
            'type6-novel': TOP_CONFI,
            'type7-wildtype': TOP_CONFI
        }
        else:
            confidences = {

                'type1-func-ref': TOP_CONFI,
                'type2-func-mut': TOP_CONFI,
                # 'type3-minor-ref': 0.90,
                # 'type3-minor-ref': TOP_CONFI,
                'type3-minor-ref': 0.95,
                
                'type4-minor-mut': 0.95,
                'type5-cross-gene': TOP_CONFI,
                'type6-novel': TOP_CONFI,
                'type7-wildtype': TOP_CONFI
            }
        
    elif g in ['KIR2DL4']:    
        confidences = {

            'type1-func-ref': TOP_CONFI,
            'type2-func-mut': TOP_CONFI,
            'type3-minor-ref': 0.30,
            'type4-minor-mut': 0.90,
            'type5-cross-gene': TOP_CONFI,
            'type6-novel': TOP_CONFI,
            'type7-wildtype': TOP_CONFI
        }
    else:
        confidences = {
            'type1-func-ref': TOP_CONFI,
            'type2-func-mut': TOP_CONFI,
            'type3-minor-ref': TOP_CONFI,
            'type4-minor-mut': TOP_CONFI,
            'type5-cross-gene': TOP_CONFI,
            'type6-novel': TOP_CONFI,
            'type7-wildtype': TOP_CONFI
        }

    return confidences.get(mutation_type, TOP_CONFI)

def create_priors(sample, db):
    # valid_positions = [pos for gene in db.genes.values() for pos in gene.positions.values() if pos.is_valid]
    # In create_priors, order positions to match sample.valid_indices
    position_order = []
    for idx in sample.valid_indices:
        variant = db.variants()[idx]
        pos_key = (variant.gene.name, variant.pos)
        if pos_key not in position_order:
            position_order.append(pos_key)

    valid_positions = [db.genes[gene].positions[pos] for gene, pos in position_order]
    num_alleles = len(sample.valid_alleles)
    num_positions = len(valid_positions)
    num_bases = len(BASES)
    PRIOR_MU_DEFAULT = -10.0
    PRIOR_LOGVAR_DEFAULT = -7.0
    PRIOR_LOG_VAR_HIGH_1 = -15
    PRIOR_LOG_VAR_HIGH_3 = -80
    PRIOR_LOG_VAR_HIGH_2 = -12
    PRIOR_LOG_VAR_MINOR = 2.0
    prior_mus = torch.full((num_alleles, num_positions, num_bases), PRIOR_MU_DEFAULT)
    prior_logvars = torch.full((num_alleles, num_positions, num_bases), PRIOR_LOGVAR_DEFAULT)
    encountered = set()
    mask = torch.zeros(num_alleles, num_positions, num_bases)

    for allele_idx, allele in enumerate(sample.valid_alleles):
        if allele.gene.name == 'KIR3DS1':
            check = 1
        for pos_idx, pos_obj in enumerate(valid_positions):
            # if allele.gene.name == 'KIR2DL1' and allele.name == '0030202' and pos_obj.gene.name == 'KIR2DP1' and pos_obj.position == 7996:
            if allele.gene.name == 'KIR2DL5A' and allele.name == '0010101' and pos_obj.gene.name == 'KIR2DL5B' and pos_obj.position == 8014:
            
            # if allele.gene.name == 'KIR2DL1' and allele.name == '0030202' and pos_obj.gene.name == 'KIR2DP1' and pos_obj.position in [v.pos for v in db.genes['KIR2DL1'].alleles['0030202'].infection_sources]:
                check = 1
                encountered.add(pos_obj.position)

            has_variant_at_position = any(
                allele.extended_allele_vector[variant.index] == 1 
                for variant in pos_obj.variants.values()
            )
            if has_variant_at_position:
                mask[allele_idx, pos_idx, :] = 1.0  # All 6 bases get 1
            allele_variant = None
            for variant in pos_obj.variants.values():
                if allele.extended_allele_vector[variant.index] == 1:
                    allele_variant = variant
                    break
            
            if allele_variant:
                if allele.gene.name == 'KIR3DL3' and allele.name == '0020602' and pos_obj.position == 1876:
                    r = 1
                mut_type = classify_mutation_type(allele, pos_obj, allele_variant)
                # if allele.gene.name in ['KIR3DL2','KIR3DL3', 'KIR3DP1']:
                confidence = get_confidence_for_type(allele.gene.name, mut_type)
           
                
                if mut_type in ['type1-func-ref', 'type2-func-mut']:
                    log_var = PRIOR_LOG_VAR_HIGH_1
                elif mut_type in ['type3-minor-ref', 'type4-minor-mut']:
                    # log_var = PRIOR_LOG_VAR_MINOR
                    # log_var = PRIOR_LOGVAR_DEFAULT
                    log_var = PRIOR_LOG_VAR_HIGH_2

                else:
                    log_var = PRIOR_LOGVAR_DEFAULT

                num_other_variants = len(pos_obj.variants.values()) - 1
                non_variant_confidence = (1.0 - confidence) / num_other_variants if num_other_variants > 0 else 0.0
                
                for base_idx, base in enumerate(BASES):
                        if allele_variant.variant == base:
                            prior_mus[allele_idx, pos_idx, base_idx] = logit(confidence)
                        elif any(variant.variant == base for variant in pos_obj.variants.values()):
                            prior_mus[allele_idx, pos_idx, base_idx] = logit(non_variant_confidence)
                        else:
                            prior_mus[allele_idx, pos_idx, base_idx] = PRIOR_MU_DEFAULT
                        
                        prior_logvars[allele_idx, pos_idx, base_idx] = log_var
    
    return prior_mus, prior_logvars, mask


def logit(p, eps=1e-7):
    if not isinstance(p, torch.Tensor):
        p = torch.tensor(p, dtype=torch.float32)
    
    p = torch.clamp(p, eps, 1 - eps)  # Avoid log(0)
    return torch.log(p / (1 - p))



def debug_create_priors(sample, db):
   
    # Collect ALL valid positions from ALL genes
    valid_positions = [pos for gene in db.genes.values() for pos in gene.positions.values() if pos.is_valid]
    
    print(f"[DEBUG] Total valid positions across all genes: {len(valid_positions)}")
    print(f"[DEBUG] Valid alleles: {len(sample.valid_alleles)}")
    
    # Print some position info
    for i, pos in enumerate(valid_positions[:5]):  # First 5 positions
        print(f"[DEBUG] Position {i}: Gene {pos.gene.name}, pos {pos.position}, ref_base {pos.ref_base}")
        print(f"[DEBUG]   Variants: {list(pos.variants.keys())}")
    
    num_alleles = len(sample.valid_alleles)
    num_positions = len(valid_positions)
    num_bases = len(BASES)  # Should be 6
    
    print(f"[DEBUG] Creating prior tensors: ({num_alleles}, {num_positions}, {num_bases})")
    
    PRIOR_MU_DEFAULT = -2.0
    PRIOR_LOGVAR_DEFAULT = -7.0
    prior_mus = torch.full((num_alleles, num_positions, num_bases), PRIOR_MU_DEFAULT)
    prior_logvars = torch.full((num_alleles, num_positions, num_bases), PRIOR_LOGVAR_DEFAULT)
    
    # Debug: Check a specific allele
    test_allele_name = "KIR3DL3*0020602"
    test_allele_idx = None
    
    for allele_idx, allele in enumerate(sample.valid_alleles):
        allele_name = f"{allele.gene.name}*{allele.name}"
        if allele_name == test_allele_name:
            test_allele_idx = allele_idx
            print(f"[DEBUG] Found test allele at index {allele_idx}")
            break
    
    # Process each allele
    for allele_idx, allele in enumerate(sample.valid_alleles):
        allele_name = f"{allele.gene.name}*{allele.name}"
        
        if allele_idx == test_allele_idx:
            print(f"[DEBUG] Processing test allele: {allele_name}")
        
        for pos_idx, pos_obj in enumerate(valid_positions):
            # Find what variant this allele has at this position
            allele_variant = None
            for variant in pos_obj.variants.values():
                if allele.extended_allele_vector[variant.index] == 1:
                    allele_variant = variant
                    break
            
            if allele_variant and allele_idx == test_allele_idx and pos_idx < 5:
                print(f"[DEBUG]   Position {pos_idx} ({pos_obj.gene.name}:{pos_obj.position}): variant {allele_variant.variant}")
            
            if allele_variant:
                mut_type = classify_mutation_type(allele, pos_obj, allele_variant)
                confidence = get_confidence_for_type(mut_type)
                
                if allele_idx == test_allele_idx and pos_idx < 5:
                    print(f"[DEBUG]     Mutation type: {mut_type}, confidence: {confidence}")
                
                # Set log variance based on mutation type
                if mut_type in ['type1-func-ref', 'type2-func-mut']:
                    log_var = -8.0
                elif mut_type in ['type3-minor-ref', 'type4-minor-mut']:
                    log_var = -7.0
                else:
                    log_var = PRIOR_LOGVAR_DEFAULT

                # Calculate probabilities for all bases at this position
                num_other_variants = len(pos_obj.variants.values()) - 1
                non_variant_confidence = (1.0 - confidence) / num_other_variants if num_other_variants > 0 else 0.0
                
                for base_idx, base in enumerate(BASES):
                    if allele_variant.variant == base:
                        prior_mus[allele_idx, pos_idx, base_idx] = logit(confidence)
                    elif any(variant.variant == base for variant in pos_obj.variants.values()):
                        prior_mus[allele_idx, pos_idx, base_idx] = logit(non_variant_confidence)
                    else:
                        prior_mus[allele_idx, pos_idx, base_idx] = PRIOR_MU_DEFAULT
                    
                    prior_logvars[allele_idx, pos_idx, base_idx] = log_var
                    
                    if allele_idx == test_allele_idx and pos_idx < 2 and base_idx < 3:
                        print(f"[DEBUG]     Base {base}: mu={prior_mus[allele_idx, pos_idx, base_idx]:.3f}")
    
    print(f"[DEBUG] Final prior tensor shapes: {prior_mus.shape}, {prior_logvars.shape}")
    
    # Check if the test allele has reasonable priors
    if test_allele_idx is not None:
        print(f"[DEBUG] Test allele priors sample:")
        for pos_idx in range(min(3, num_positions)):
            pos_obj = valid_positions[pos_idx]
            print(f"[DEBUG]   Pos {pos_idx} ({pos_obj.gene.name}:{pos_obj.position}):")
            for base_idx, base in enumerate(BASES):
                mu_val = prior_mus[test_allele_idx, pos_idx, base_idx].item()
                if mu_val > -1.5:  # Only print high-confidence bases
                    print(f"[DEBUG]     {base}: {mu_val:.3f}")
    
    return prior_mus, prior_logvars

def create_sparse_priors(sample, db):
    # BASES = ['A', 'C', 'G', 'N', 'P', 'T']
    PRIOR_MU_DEFAULT = -1000.0
    PRIOR_LOGVAR_DEFAULT = -7.0
    
    sparse_prior_mus = []
    sparse_prior_logvars = []
    
    for allele in sample.valid_alleles:
        for pos_obj in allele.generatable_positions:
            # Find what variant this allele has at this position
            allele_variant = None
            if allele.gene.name == 'KIR2DL5A' and allele.name == '0010101' and pos_obj.gene.name == 'KIR2DL5B' and pos_obj.position == 8014:
                check = 1

            for variant in pos_obj.variants.values():
                if allele.extended_allele_vector[variant.index] == 1:
                    allele_variant = variant
                    break
            
            if allele_variant:
                mut_type = classify_mutation_type(allele, pos_obj, allele_variant)
                confidence = get_confidence_for_type(allele.gene.name, mut_type)
             
                if mut_type in ['type1-func-ref', 'type2-func-mut']:
                    log_var = -8.0
                elif mut_type in ['type3-minor-ref', 'type4-minor-mut']:
                    log_var = -7.0
                else:
                    log_var = PRIOR_LOGVAR_DEFAULT
                
                num_other_variants = len(pos_obj.variants.values()) - 1
                non_variant_confidence = (1.0 - confidence) / num_other_variants if num_other_variants > 0 else 0.0
                
                # Create 6-dimensional prior for this (allele, position) pair
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
                
                sparse_prior_mus.extend(position_mus)  # Flatten to 1D
                sparse_prior_logvars.extend(position_logvars)
            else:
                assert False
                
    return torch.tensor(sparse_prior_mus), torch.tensor(sparse_prior_logvars)

def create_sparse_priors_V2(sample, db):
    PRIOR_MU_DEFAULT = -1000.0
    PRIOR_LOGVAR_DEFAULT = -7.0
    sparse_prior_mus = []
    sparse_prior_logvars = []
    
    for allele in sample.valid_alleles:
        for pos_obj in allele.generatable_positions:
            # Find ALL variants this allele has at this position
            allele_variants = []
            for variant in pos_obj.variants.values():
                if allele.extended_allele_vector[variant.index] == 1:
                    allele_variants.append(variant)
            
            if len(allele_variants) == 1:
                allele_variant = allele_variants[0]
                
                mut_type = classify_mutation_type(allele, pos_obj, allele_variant)
                confidence = get_confidence_for_type(allele.gene.name, allele.name, pos_obj.position, mut_type)
                
                if mut_type in ['type1-func-ref', 'type2-func-mut']:
                    log_var = -8.0
                elif mut_type in ['type3-minor-ref', 'type4-minor-mut']:
                    log_var = -7.0
                else:
                    log_var = PRIOR_LOGVAR_DEFAULT
                
                num_other_variants = len(pos_obj.variants.values()) - 1
                non_variant_confidence = (1.0 - confidence) / num_other_variants if num_other_variants > 0 else 0.0
                
                # Create 6-dimensional prior for this (allele, position) pair
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
                
            elif len(allele_variants) > 1:
                # Multiple variants - use evidence-based priors
                # has to be cross gene
                total_evidence = 0
                variant_evidence = {}
                
                # Calculate evidence for each variant from infection sources
                for variant in allele_variants:
                    evidence_count = 0
                    # if hasattr(allele, 'infection_sources') and variant in allele.infection_sources:
                    for pos_dict in allele.infection_sources[variant].values():
                        evidence_count += len(pos_dict)
                    variant_evidence[variant] = evidence_count
                    total_evidence += evidence_count
                
                # Debug print for the specific case
                if allele.gene.name == 'KIR2DS3' and allele.name == '0010301' and pos_obj.gene.name == 'KIR2DS2' and pos_obj.position == 10190:
                    print(f"Multiple variants for {allele.gene.name}*{allele.name} at {pos_obj.gene.name}:{pos_obj.position}")
                    for variant in allele_variants:
                        print(f"  {variant.variant}: {variant_evidence[variant]} evidence")
                    print(f"  Total evidence: {total_evidence}")
                
                # Create priors based on evidence ratios
                position_mus = []
                position_logvars = []
                
                for base in BASES:
                    if total_evidence > 0:
                        # Find evidence for this base
                        evidence_for_base = 0
                        for variant in allele_variants:
                            if variant.variant == base:
                                evidence_for_base = variant_evidence[variant]
                                break
                        
                        if evidence_for_base > 0:
                            prob = evidence_for_base / total_evidence
                            position_mus.append(logit(prob))
                        else:
                            position_mus.append(PRIOR_MU_DEFAULT)
                    else:
                        # No evidence - fall back to equal distribution among variants
                        if any(variant.variant == base for variant in allele_variants):
                            prob = 1.0 / len(allele_variants)
                            position_mus.append(logit(prob))
                        else:
                            position_mus.append(PRIOR_MU_DEFAULT)
                    
                    # Use default log variance for multi-variant cases
                    position_logvars.append(PRIOR_LOGVAR_DEFAULT)
                
                sparse_prior_mus.extend(position_mus)
                sparse_prior_logvars.extend(position_logvars)
                
            else:
                assert False, f"No variants found for generatable position {pos_obj.gene.name}:{pos_obj.position}"
    
    return torch.tensor(sparse_prior_mus), torch.tensor(sparse_prior_logvars)
    
def decode_allele_beta(allele_idx, allele, sample, db, learned_beta):
 
    allele_name = f"{allele.gene.name}*{allele.name}"
    allele_row = learned_beta[allele_idx]  # Get this allele's row
    db_allele_row = torch.tensor(sample.beta[allele_idx])  # Always use sample.beta as DB reference
    
    # Use allele.generatable_positions to get the actual positions for this allele
    position_data = {}
    
    # Enumerate through the allele's generatable positions
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
            'learned_prob': f"{pos_info['learned_base_prob']:.3f}",
        }
        
        # Add individual base probabilities
        for base in BASES:
            row[f'{base}_prob'] = f"{pos_info['learned_probs'][base]:.3f}"
        
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
    
    print("\n" + "="*120)
    print("DECODED BETA MATRICES FOR ALL ALLELES")
    print("="*120)
    
    for allele_idx, allele in enumerate(sample.valid_alleles):
        allele_name = f"{allele.gene.name}*{allele.name}"
        
        # Decode this allele (no db_beta parameter needed)
        allele_data = decode_allele_beta(allele_idx, allele, sample, db, learned_beta)
        
        all_results[allele_name] = allele_data
        
        # Count differences and cross-gene positions
        cross_gene_count = sum(1 for row in allele_data if row['mutation_type'] == 'cross_gene')
        same_gene_count = len(allele_data) - cross_gene_count
        differs_count = sum(1 for row in allele_data if row['differs_from_db'])
        
        # Print summary for this allele
        print(f"\n {allele_name} ({len(allele_data)} positions: {same_gene_count} same-gene, {cross_gene_count} cross-gene, {differs_count} differences)")
        print("-" * 120)
        
        # Print header: Ref, DB, Learned, Diff, Type, Prob, A-T
        print(f"{'Gene:Pos':<15} {'Ref':<3} {'DB':<3} {'Learned':<8} {'Diff':<4} {'Type':<15} {'Prob':<6} {'A':<6} {'C':<6} {'G':<6} {'N':<6} {'P':<6} {'T':<6}")
        print("-" * 120)
        
        # Print each position with reordered columns
        for row in allele_data:
            differs = "🔴" if row['differs_from_db'] else "✅"
            
            # Highlight cross-gene positions
            gene_pos = row['gene_position']
            if row['mutation_type'] == 'cross_gene':
                gene_pos = f"🔄{gene_pos}"  # Add cross-gene indicator
            
            print(f"{gene_pos:<15} "
                  f"{row['ref_base']:<3} "
                  f"{row['database_base']:<3} "
                  f"{row['learned_base']:<8} "
                  f"{differs:<4} "
                  f"{row['mutation_type']:<15} "
                  f"{row['learned_prob']:<6} "
                  f"{row['A_prob']:<6} "
                  f"{row['C_prob']:<6} "
                  f"{row['G_prob']:<6} "
                  f"{row['N_prob']:<6} "
                  f"{row['P_prob']:<6} "
                  f"{row['T_prob']:<6}")

    return all_results

def create_functional_observation_counts(sample, db):
    num_mutations = len(sample.valid_indices)
    
    observed_counts = torch.zeros(num_mutations)
    
    # For each variant, get its observed count
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


# def create_gene_masks(sample, db, functional_indices):
#     gene_names = []
#     gene_masks_list = []
    
#     for gene in db.genes.values():
#         gene_mask = torch.zeros_like(functional_indices, dtype=torch.bool)
        
#         for mut_idx in range(len(functional_indices)):
#             if functional_indices[mut_idx]:
#                 variant = db.variants()[sample.valid_indices[mut_idx]]
#                 if variant.gene.name == gene.name:
#                     gene_mask[mut_idx] = True
        
#         if gene_mask.sum() > 0:
#             gene_names.append(gene.name)
#             gene_masks_list.append(gene_mask)
    
#     gene_masks_stacked = torch.stack(gene_masks_list) if gene_masks_list else torch.empty(0, len(functional_indices))
    
#     return gene_masks_stacked, gene_names
def create_gene_masks(sample, db, functional_indices):
    import numpy as np
    
    functional_indices_np = functional_indices.cpu().numpy() if torch.is_tensor(functional_indices) else functional_indices
    num_mutations = len(functional_indices_np)
    
    gene_assignments = [''] * num_mutations
    for mut_idx in np.where(functional_indices_np)[0]:
        variant = db.variants()[sample.valid_indices[mut_idx]]
        gene_assignments[mut_idx] = variant.gene.name
    
    gene_names = []
    gene_masks_list = []
    
    for gene in db.genes.values():
        mask = np.array([g == gene.name for g in gene_assignments], dtype=bool)
        if mask.sum() > 0:
            gene_names.append(gene.name)
            gene_masks_list.append(torch.from_numpy(mask))
    
    gene_masks_stacked = torch.stack(gene_masks_list) if gene_masks_list else torch.empty(0, num_mutations, dtype=torch.bool)
    
    return gene_masks_stacked, gene_names

# def create_gene_masks(sample, db, functional_indices):
#     print('np new version')
#     import numpy as np
    
#     functional_indices_np = functional_indices.cpu().numpy() if torch.is_tensor(functional_indices) else functional_indices
#     num_mutations = len(functional_indices_np)
    
#     gene_assignments = [''] * num_mutations
#     for mut_idx in np.where(functional_indices_np)[0]:
#         variant = db.variants()[sample.valid_indices[mut_idx]]
#         gene_assignments[mut_idx] = variant.gene.name
    
#     gene_names = []
#     gene_masks_list = []
    
#     for gene in db.genes.values():
#         mask = np.array([g == gene.name for g in gene_assignments], dtype=bool)
#         if mask.sum() > 0:
#             gene_names.append(gene.name)
#             gene_masks_list.append(torch.from_numpy(mask))
    
#     gene_masks_stacked = torch.stack(gene_masks_list) if gene_masks_list else torch.empty(0, num_mutations, dtype=torch.bool)
    
#     return gene_masks_stacked, gene_names