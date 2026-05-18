from common_0620B import *

from collections import defaultdict
import numpy as np


def estimate_CN(sample, db, min_coverage_threshold=3, debug=False):
    gene_copy_numbers = {}
    gene_stats = defaultdict(lambda: {'total_coverage': 0, 'position_count': 0, 'positions': []})
    
    for gene_name, gene in sorted(db.genes.items()):
        for pos_num in sorted(gene.positions.keys()):
            position = gene.positions[pos_num]
      
            position_coverage = 0
            is_multimap_position = False
            
            for _, variant in position.variants.items():
                if len(variant.infected_alleles) > 0:
                    is_multimap_position = True
                         
                position_coverage += variant.coverage.count
            
            if not is_multimap_position and position_coverage >= min_coverage_threshold:
                gene_stats[gene_name]['total_coverage'] += position_coverage
                gene_stats[gene_name]['position_count'] += 1
                gene_stats[gene_name]['positions'].append((pos_num, position_coverage))
                
    
        stats = gene_stats[gene_name]
        if stats['position_count'] > 0 and sample.expected_coverage > 0:
            estimated_cn = stats['total_coverage'] / (stats['position_count'] * sample.expected_coverage)
            gene_copy_numbers[gene_name] = estimated_cn

        else:
            gene_copy_numbers[gene_name] = 0.0
            if debug and stats['position_count'] == 0:
                print(f"\n[{gene_name}] No usable positions found (all multi-map or low coverage)")
    
    return gene_copy_numbers

def bin_copy_numbers(gene_copy_numbers):
    gene_ranges = {
        'KIR2DL4': {1: (0.88888888, 1.78888888), 2: (1.78888888, 8.88888888)},
        'KIR2DP1': {1: (0.88888888, 1.58888888), 2: (1.58888888, 8.88888888)},


        'KIR3DL2': {1: (0.88888888, 1.78888888), 2: (1.78888888, 8.88888888)},
        'KIR3DL3': { 1: (0.88888888, 1.78888888), 2: (1.78888888, 8.88888888)},
        'KIR3DP1': { 1: (0.88888888, 1.38888888), 2: (1.38888888, 2.588888888), 3:(2.588888888, 8.8888888)},
    }
    
    binned_cn = {}
    for gene_name, estimated_cn in gene_copy_numbers.items():
        if gene_name in gene_ranges:
            ranges = gene_ranges.get(gene_name)
        else:
            binned_cn[gene_name] = -1
            continue

        
        for cn_value, (lower, upper) in ranges.items():
            if lower <= estimated_cn < upper:
                binned_cn[gene_name] = cn_value
                break
      
    return binned_cn


def set_expected_position_strength(sample, db, lb_factor=0.8, ub_factor=1.2, min_lb=1.0):
    for allele in sample.valid_alleles:
        if not allele.enabled:
            continue
        allele.expected_position_strength = {}
        for pos_obj in allele.generatable_positions:
            position = pos_obj.position
            
            total_non_zero = sum(1 for v in pos_obj.variants.values() if v.coverage.count > 0)
            
            for variant in pos_obj.variants.values():
                # if allele.extended_allele_vector[variant.index] != 1:
                #     continue
                is_same_gene = (variant.gene.name == allele.gene.name)
                if is_same_gene:
                    # adjust only if:
                    # 1. variant is observede
                    to_estimate = ['KIR2DL4','KIR2DP1','KIR3DL2','KIR3DL3','KIR3DP1']
                    to_estimate = []
                    if allele.gene.name in to_estimate  and allele.gene.estimated_cn == 2 and variant.coverage.count > 0 and not any(a.gene.name != allele.gene.name for a in variant.infected_alleles):
                        if total_non_zero == 1:
                            lb = ub = variant.coverage.count / 2
                            print(f'[homo]setting {allele.gene.name} {allele.name} {position} {variant.variant} coverage {lb}')
                        else:
                            lb = ub = variant.coverage.count
                            print(f'[hete]setting {allele.gene.name} {allele.name} {position} {variant.variant} coverage {lb}')

                    else:
                        # Same-gene: tightly bounded around expected coverage
                        lb = lb_factor * sample.expected_coverage
                        ub = ub_factor * sample.expected_coverage
                else:
                    # Cross-gene: bounded by actual infection reads
                    # Count actual infection reads for this allele
                    if allele.extended_allele_vector[variant.index] != 1:
                        continue
                    total_infection_reads = sum(
                        len(read_ids) 
                        for read_ids in allele.infection_sources[variant].values()
                    )
                    # Upper bound is the actual infection read count
                    ub = min(sample.expected_coverage, total_infection_reads)
                    # Lower bound: either half of upper bound, or minimum
                    # lb = max(min_lb, ub * 0.5)
                    lb = ub
                # allele.expected_position_strength[(position, variant.variant)] = (lb, ub)
                allele.set_position_strength(pos_obj, variant.variant, lb, ub)