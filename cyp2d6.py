# %% Imports & Initialization
import pickle
import os
import sys
sys.path.append('/data/qinghuiz/inumanag-kir/aldy-kir-fast-geny')
import copy
import re
import numpy as np
import multiprocessing as mp
import subprocess as sp
import collections
from pprint import pprint
import pysam
import math
import mappy
import torch
from tqdm import tqdm

import os
import tensorflow as tf
import numpy as np
import argparse
from shutil import which
from pprint import pprint
import pickle

from helper import *

class SimpleSample:
    def __init__(self, path):
        self.path = path


#%%
def get_variant_info(aldy_sample, db, gene_name):
    from aldy.gene import Mutation
    
    gene = db.genes[gene_name]
    cov = aldy_sample.coverage
    
    indel_eqs = aldy_sample._indel_sites_eqs 
    indel_sites = aldy_sample._indel_sites  
    
    for pos, position in gene.positions.items():
        ref_base = position.ref_base
        
        for variant_key, variant in position.variants.items():
            if variant_key == ref_base:
                variant.coverage.count = cov[Mutation(pos, "_")]
            elif variant_key in 'ACGT':
                variant.coverage.count = cov[Mutation(pos, f"{ref_base}>{variant_key}")]
            elif variant_key == 'N':
                count = 0
                for (indel_pos, indel_op), (off, on) in indel_sites.items():
                    if indel_pos == pos and indel_op.startswith('del'):
                        count += on
                if count == 0:
                    for op, cov_list in cov._coverage.get(pos, {}).items():
                        if op.startswith('del'):
                            count += len(cov_list)
                variant.coverage.count = count
            elif variant_key == 'P':
                count = 0
                for (indel_pos, indel_op), (off, on) in indel_sites.items():
                    if indel_pos == pos and indel_op.startswith('ins'):
                        count += on
                if count == 0:
                    for op, cov_list in cov._coverage.get(pos, {}).items():
                        if op.startswith('ins'):
                            count += len(cov_list)
                variant.coverage.count = count
            elif variant_key == 'M':
                # Multi-SNP - get coverage from aldy directly
                count = 0
                if hasattr(position, 'multi_snp_ops') and position.multi_snp_ops:
                    for op in position.multi_snp_ops:
                        cov_list = cov._coverage.get(pos, {}).get(op, [])
                        count += len(cov_list)
                variant.coverage.count = count


#%% Allele Filtering - First Pass (functional mutations only)
def filter_allele_functional(a, sample, db):
    """
    First pass filter: Only check functional mutation coverage.
    Does NOT check region coverage (that comes after CN estimation).
    """
    # Disable fusion alleles
    if '#' in a.name:
        a.enabled = False
        return
    
    for (pos, op) in a.gene.functional:
        position = a.gene.positions[pos]
        
        if (pos, op) in a.func:
            # Check multi-SNP FIRST (before checking '>')
            if '.' in op and '>' in op:
                required_base = 'M'
            elif '>' in op:
                required_base = convert_op_to_variant(op)
            elif op.startswith('del'):
                required_base = 'N'
            elif op.startswith('ins'):
                required_base = 'P'
            else:
                required_base = position.ref_base
        else:
            # Allele doesn't have this functional mutation
            # Check if allele has a DIFFERENT mutation at this position
            allele_has_other_mutation_at_pos = any(
                p == pos for (p, o) in (a.func | a.minor)
            )
            
            if allele_has_other_mutation_at_pos:
                continue  # Skip - allele has different mutation here
            else:
                required_base = position.ref_base 
        
        coverage = position.variants[required_base].coverage.count
        
        if coverage < sample.min_coverage:
            print(f'{a.name} was filtered out as {pos} {op} needs {required_base} but not covered')
            a.enabled = False
            return
    
    a.enabled = True


#%% Allele Filtering - Second Pass (CN-based for CYP2D6)
def filter_allele_by_cn_config(sample, db, aldy_gene, cn_config_cn, cn_threshold=0.5):
    """
    Second pass filter for CYP2D6: Filter out alleles whose cn_config has CN < threshold.
    Call this AFTER CN estimation.
    """
    filtered_count = 0
    
    for allele in sample.valid_alleles:
        if not allele.enabled:
            continue
        
        major_name = db.minor_to_major[allele.name]
        major_allele = aldy_gene.alleles.get(major_name)
        
        if major_allele:
            cn_config_name = major_allele.cn_config
            cn_val = cn_config_cn.get(cn_config_name, 1.0)
            
            if cn_val < cn_threshold:
                print(f'{allele.name} was filtered out: cn_config={cn_config_name} has CN={cn_val:.2f} < {cn_threshold}')
                allele.enabled = False
                filtered_count += 1
    
    print(f"[CN filter] Filtered {filtered_count} alleles based on cn_config CN")
    
    # Update valid_alleles list
    sample.valid_alleles = [a for a in sample.valid_alleles if a.enabled]
    
    return filtered_count


#%% Position and variant utilities
def populate_valid_positions(sample, db, GENE):
    for gene in db.genes.values():
        for position in gene.positions.values():
            position.is_valid = False
    
    for allele in db.alleles():
        if not allele.enabled:
            continue
        gene = allele.gene

        for mutation in gene.functional:
            pos, op = mutation
            gene.positions[pos].is_valid = True
        
        for mutation in allele.func:
            pos, op = mutation
            gene.positions[pos].is_valid = True
        
        for mutation in allele.minor:
            pos, op = mutation
            gene.positions[pos].is_valid = True
    
    valid_variants = {v for gene in db.genes.values() for pos in gene.positions.values() 
                 if pos.is_valid for v in pos.variants.values()}
    
    valid_alleles = [a for a in db.alleles() if a.enabled]
    sample.valid_alleles = valid_alleles
    sample.valid_indices = sorted([v.index for v in valid_variants])

    position_order = []
    for idx in sample.valid_indices:
        variant = db.variants()[idx]
        pos_key = variant.pos  
        if pos_key not in position_order:
            position_order.append(pos_key)

    valid_positions = [db.genes[GENE].positions[pos] for pos in position_order]
    for allele in sample.valid_alleles:
        allele.generatable_positions = []
        for pos_obj in valid_positions:
            allele.generatable_positions.append(pos_obj)
    
    total_variants = sum(len(pos.variants) for gene in db.genes.values() for pos in gene.positions.values())
    sample.beta = np.zeros((len(valid_alleles), total_variants), dtype=np.float64)
    
    for i, allele in enumerate(valid_alleles):
        sample.beta[i] = allele.extended_allele_vector
    
    # Filter and normalize
    valid_indices_set = set(sample.valid_indices)
    for col in range(sample.beta.shape[1]):
        if col not in valid_indices_set:
            sample.beta[:, col] = 0

    for row in range(sample.beta.shape[0]):
        row_sum = np.sum(sample.beta[row])
        if row_sum > 0:
            sample.beta[row] = sample.beta[row] / row_sum

    sample.beta = sample.beta[:, sample.valid_indices]
    
    unique_indices, seen = [], set()
    [unique_indices.append(i) or seen.add(tuple(sample.beta[i])) for i in range(len(sample.beta)) if tuple(sample.beta[i]) not in seen]
    sample.beta, sample.valid_alleles = sample.beta[unique_indices], [sample.valid_alleles[i] for i in unique_indices]
    [setattr(allele, 'index', i) for i, allele in enumerate(sample.valid_alleles)]


def zero_out_low_coverage_variants(sample, db, coverage_threshold=3):
    for gene in db.genes.values():
        for position in gene.positions.values():
            for variant in position.variants.values():
                if variant.coverage.count <= coverage_threshold:
                    variant.coverage.count = 0


#%% Region and CN utilities
def get_allele_region_mask(allele, gene, region_names):
    cn_config = gene.cn_configs.get(allele.cn_config, gene.cn_configs['1'])
    return [cn_config.cn[0].get(r, 1) for r in region_names]


def create_region_mask(sample, gene, db):
    region_names = [r for r in gene.regions[0].keys() 
                    if r.startswith('e') or r.startswith('i')]
    
    mask = torch.tensor([
        get_allele_region_mask(gene.alleles[db.minor_to_major[a.name]], gene, region_names) 
        for a in sample.valid_alleles
    ], dtype=torch.float32)
    
    return region_names, mask


def get_region_coverage(aldy_sample, gene):
    cov = aldy_sample.coverage
    
    region_cov = {}
    for region_name in gene.regions[0].keys():
        if region_name.startswith('e') or region_name.startswith('i'):
            region_cov[region_name] = cov.region_coverage(0, region_name)
    
    return region_cov


def get_unique_cn_configs(aldy_gene):
    """Get unique cn_configs and their region masks."""
    cn_config_info = {}
    
    region_names = [r for r in aldy_gene.regions[0].keys() 
                    if r.startswith('e') or r.startswith('i')]
    
    for cn_name, cn_config in aldy_gene.cn_configs.items():
        mask = [cn_config.cn[0].get(r, 1) for r in region_names]
        cn_config_info[cn_name] = {
            'mask': mask,
            'cn_config': cn_config
        }
    
    return region_names, cn_config_info


def create_cn_config_region_mask(aldy_gene):
    """
    Create region mask at cn_config level for cn_configs that have at least one region.
    Skip cn_configs with all-zero masks (full deletions like *5).
    Returns: (region_names, cn_config_names, mask tensor, skipped_cn_configs)
    """
    region_names, cn_config_info = get_unique_cn_configs(aldy_gene)
    
    cn_config_names = []
    masks = []
    skipped_cn_configs = []
    
    for cn_name in sorted(cn_config_info.keys()):
        mask = cn_config_info[cn_name]['mask']
        
        # Skip cn_configs with all-zero masks (can't estimate CN from regions)
        if sum(mask) == 0:
            print(f"  Skipping cn_config '{cn_name}': all regions = 0 (full deletion)")
            skipped_cn_configs.append(cn_name)
            continue
        
        cn_config_names.append(cn_name)
        masks.append(mask)
    
    mask_tensor = torch.tensor(masks, dtype=torch.float32) if masks else torch.zeros(0, len(region_names))
    
    print(f"\nCN configs for estimation: {cn_config_names}")
    print(f"Skipped CN configs (all-zero): {skipped_cn_configs}")
    print(f"Region mask shape: {mask_tensor.shape}")
    
    for i, cn_name in enumerate(cn_config_names):
        print(f"  {cn_name}: {mask_tensor[i].tolist()}")
    
    return region_names, cn_config_names, mask_tensor, skipped_cn_configs


def estimate_cyp2d6_cn(aldy_sample, aldy_gene, db):
    """
    Estimate CN for each structural pattern (cn_config) in CYP2D6.
    Skip cn_configs with all-zero region masks.
    """
    from cn_estimator import run_cn_estimator
    
    # Get region coverage
    region_names, cn_config_names, cn_config_mask, skipped_cn_configs = create_cn_config_region_mask(aldy_gene)
    region_cov = get_region_coverage(aldy_sample, aldy_gene)
    region_normalized = torch.tensor([region_cov[r] for r in region_names], dtype=torch.float32)
    
    print(f"\nRegion coverage:")
    for r, c in zip(region_names, region_normalized.tolist()):
        print(f"  {r}: {c:.1f}")
    
    # Run CN estimator at cn_config level (only for non-zero cn_configs)
    if len(cn_config_names) > 0:
        cn_config_cn = run_cn_estimator(
            region_cov=region_normalized,
            region_mask=cn_config_mask,
            gene_names=cn_config_names,
            max_cn=6.0,
            num_iterations=500,
            delta=0.5
        )
    else:
        cn_config_cn = {}
    
    # Add skipped cn_configs with CN = 0
    for skipped in skipped_cn_configs:
        cn_config_cn[skipped] = 0.0
    
    print("\n[CN Config CN Results]")
    for cn_name in sorted(cn_config_cn.keys()):
        cn_val = cn_config_cn[cn_name]
        is_skipped = "(skipped - full deletion)" if cn_name in skipped_cn_configs else ""
        print(f"  {cn_name}: {cn_val:.2f} {is_skipped}")
    
    # Store CN per cn_config on the gene
    db.genes['CYP2D6'].cn_config_cn = cn_config_cn
    db.genes['CYP2D6'].skipped_cn_configs = skipped_cn_configs
    
    # Total gene CN is sum of all cn_configs (excluding skipped ones which are 0 anyway)
    total_cn = sum(cn_config_cn.values())
    db.genes['CYP2D6'].estimated_cn = total_cn
    print(f"\n  Total CYP2D6 CN: {total_cn:.2f}")
    
    return cn_config_cn, cn_config_names, skipped_cn_configs

def print_cn_config_info(aldy_gene):
    """Debug: Print cn_config region expectations."""
    region_names = [r for r in aldy_gene.regions[0].keys() 
                    if r.startswith('e') or r.startswith('i')]
    
    print(f"\nRegions: {region_names}")
    print("\nCN Config region masks:")
    
    for cn_name, cn_config in aldy_gene.cn_configs.items():
        mask = [cn_config.cn[0].get(r, 1) for r in region_names]
        print(f"  {cn_name}: {mask}")
    
    print("\nAllele -> CN Config examples:")
    seen_configs = set()
    for name, allele in aldy_gene.alleles.items():
        if allele.cn_config not in seen_configs:
            seen_configs.add(allele.cn_config)
            print(f"  *{name}: cn_config='{allele.cn_config}'")


#%% Main execution
if __name__ == "__main__":
    from aldy.gene import Gene
    from aldy.profile import Profile
    from aldy.sam import Sample
    from aldy.common import script_path
    from common_cyp import Database, convert_op_to_variant
    
    GENE = 'CYP2D6'
    # GENE = sys.argv[1]

    gene = Gene(script_path(f"aldy.resources.genes/{GENE.lower()}.yml"), genome="hg19")
    
    profile = Profile.load(gene, "illumina")
    path = '/project/shared/aldy-data/wgs/NA12145.wgs.cram'
    # path = sys.argv[2]
    
    aldy_sample = Sample(
        gene=gene,
        profile=profile,
        path=path,
        reference="/project/shared/aldy-data/Homo_sapiens_assembly19_1000genomes_decoy.fasta"
    )
    
    db = Database(f"{GENE.lower()}.pkl")
    sample = SimpleSample(path)
    sample.min_coverage = 3
    sample.expected_coverage = aldy_sample.coverage.diploid_avg_coverage() / 2
    
    # Get variant coverage info
    get_variant_info(aldy_sample, db, GENE)
    
    # Debug: Print CN config info for CYP2D6
    if GENE == 'CYP2D6':
        print_cn_config_info(db.genes[GENE].aldy_gene)


#%% First pass: Filter alleles by functional mutation coverage
if __name__ == "__main__":
    from common_cyp import convert_op_to_variant
    
    g = db.genes[GENE]
    aldy_gene = g.aldy_gene
    
    print("\n=== First Pass: Functional Mutation Filtering ===")
    ax = {}
    for a in g.alleles.values():
        filter_allele_functional(a, sample=sample, db=db)

        if a.enabled:
            major_allele = db.minor_to_major[a.name]
            group_key = major_allele
            ax.setdefault(group_key, []).append(a.name)
    
    for major, minors in ax.items():
        alleles_str = '; '.join(minors)
        print(f"[filter] {g.name} {major} => {alleles_str}")
    
    valid_count = sum(1 for a in g.alleles.values() if a.enabled)
    print(f"[filter] valid_alleles after first pass = {valid_count}")


#%% CN Estimation (before second pass filtering)
if __name__ == "__main__":
    from cn_estimator import run_cn_estimator
    
    print("\n=== CN Estimation ===")
    
    if GENE == 'CYP2D6':
        # CYP2D6: Estimate CN at structural pattern (cn_config) level
        cn_config_cn, cn_config_names = estimate_cyp2d6_cn(
            aldy_sample, 
            db.genes[GENE].aldy_gene, 
            db
        )
    else:
        # CYP2C8, 2C9, 2C19: Simple gene-level CN
        region_names, region_mask = create_region_mask(sample, db.genes[GENE].aldy_gene, db)
        region_cov = get_region_coverage(aldy_sample, db.genes[GENE].aldy_gene)
        region_normalized = torch.tensor([region_cov[r] for r in region_names], dtype=torch.float32)
        
        gene_region_mask = torch.ones(1, len(region_names), dtype=torch.float32)
        
        gene_cn = run_cn_estimator(
            region_cov=region_normalized,
            region_mask=gene_region_mask,
            gene_names=[GENE],
            max_cn=4.0,
            num_iterations=500,
            delta=0.5
        )
        
        print("\n[Gene CN Results]")
        cn_raw = gene_cn[GENE]
        print(f"  {GENE}: {cn_raw:.2f}")
        
        # For CYP2C8/9/19, assume diploid
        db.genes[GENE].estimated_cn = 2.0


#%% Second pass: Filter alleles by CN config (CYP2D6 only)
if __name__ == "__main__":
    if GENE == 'CYP2D6':
        print("\n=== Second Pass: CN Config Filtering ===")
        
        # First populate valid_alleles for filtering
        sample.valid_alleles = [a for a in db.alleles() if a.enabled]
        
        # Filter out alleles whose cn_config has CN < threshold
        filter_allele_by_cn_config(
            sample, 
            db, 
            db.genes[GENE].aldy_gene, 
            cn_config_cn, 
            cn_threshold=0.5
        )
        
        valid_count = len(sample.valid_alleles)
        print(f"[filter] valid_alleles after second pass = {valid_count}")


#%% Prep for solver
if __name__ == "__main__":
    print("\n=== Preparing for VAE ===")
    zero_out_low_coverage_variants(sample, db)
    populate_valid_positions(sample, db, GENE)
    print_variant_counts(sample, db)


#%% Run VAE
if __name__ == "__main__":
    from vae_cyp import *
    from vae_cyp_helper import *
    
    print("\n=== Running VAE ===")
    
    total_mut_counts = sum(variant.coverage.count 
                     for gene in db.genes.values() 
                     for pos in gene.positions.values() 
                     if pos.is_valid 
                     for variant in pos.variants.values())
    
    mut_counts_tensor = torch.tensor([db.variants()[i].coverage.count for i in sample.valid_indices], dtype=torch.float32)
    valid_allele_names = [f'{a.gene.name}*{a.name}' for a in sample.valid_alleles]
    
    bam_id = path.split('/')[-1]
    
    densities, densities_un, learnt_beta = run_vae(
        total_mut_counts,   
        valid_allele_names,   
        mut_counts_tensor,       
        sample=sample,
        db=db,       
        num_iterations=3000
    )

#%% Process results
if __name__ == "__main__":
    print("\n=== Processing Results ===")
    
    major_allele_densities = {}
    major_allele_densities_unnorm = {}
    threshold = 0.1
    
    for i, density in enumerate(densities):
        allele_name = valid_allele_names[i]
        gene_name, allele_id = allele_name.split('*')
        major_key = allele_id.split('.')[0]
        
        if major_key not in major_allele_densities:
            major_allele_densities[major_key] = 0
            major_allele_densities_unnorm[major_key] = 0
        major_allele_densities[major_key] += density
        major_allele_densities_unnorm[major_key] += densities_un[i]

    # Filter by threshold and group by gene
    filtered_by_gene = {}
    for major_key in major_allele_densities.keys():
        norm_val = major_allele_densities[major_key]
        if norm_val > threshold:
            if GENE not in filtered_by_gene:
                filtered_by_gene[GENE] = []
            filtered_by_gene[GENE].append((major_key, norm_val, major_allele_densities_unnorm[major_key]))

    # Apply gene-specific selection 
    final_results = []
    for gene_name, alleles in filtered_by_gene.items():
        alleles.sort(key=lambda x: x[1], reverse=True)
        if gene_name in ['CYP2C8', 'CYP2C9', 'CYP2C19']:
            final_results.extend(alleles[:2])
        else:
            final_results.extend(alleles)
    
    print('\n[Raw]:')
    for major_key, norm_val, unnorm_val in final_results:
        print(f'{GENE}*{major_key}: Normalized={norm_val:.4f}, Unnormalized={unnorm_val:.2f}')


#%% Final output
if __name__ == "__main__":
    print('\n[Alleles]')
    
    if GENE == 'CYP2D6':
        # CYP2D6: Use cn_config-aware CN assignment
        aldy_gene = db.genes[GENE].aldy_gene
        gene_cn = db.genes[GENE].estimated_cn
        cn_config_cn = db.genes[GENE].cn_config_cn
        
        for major_key, norm_val, unnorm_val in final_results:
            # Get cn_config for this major allele
            major_allele = aldy_gene.alleles.get(major_key)
            
            if major_allele:
                cn_config = major_allele.cn_config
                cn_config_cn_val = cn_config_cn.get(cn_config, 1.0)
                
                # Calculate allele CN based on proportion
                allele_cn = max(1, round(float(norm_val) * gene_cn))
                
                print(f'# {GENE}*{major_key} (cn_config={cn_config}, cn_config_cn={cn_config_cn_val:.2f})')
                for _ in range(allele_cn):
                    print(f'{GENE}*{major_key}')
            else:
                # Fallback
                allele_cn = max(1, round(float(norm_val) * gene_cn))
                for _ in range(allele_cn):
                    print(f'{GENE}*{major_key}')
    else:
        # CYP2C8/9/19: Simple CN assignment
        gene_cn = db.genes[GENE].estimated_cn

        for major_key, norm_val, unnorm_val in final_results:
            allele_cn = max(1, round(float(norm_val) * gene_cn))
            for _ in range(allele_cn):
                print(f'{GENE}*{major_key}')

# %%