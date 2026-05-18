# %% Imports & Initialization
# find /data/qinghuiz/inumanag-kir/data/sim -type f -name "*.fa" ! -name "*.extract.fa" | xargs -I{} sh -c 'python kir.py {} > LDA_res/$(basename {}).log'
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
#%%

import os

import tensorflow as tf
import numpy as np
import argparse
from shutil import which
from pprint import pprint
import pickle

#%%
from core.helper import * # needs attention of common class import override

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

#%%
if __name__ == "__main__":
    from aldy.gene import Gene
    from aldy.profile import Profile
    from aldy.sam import Sample
    from aldy.common import script_path
    from common_cyp import Database
    import argparse
    parser = argparse.ArgumentParser(description='CYP gene typing tool')
    # GENE = 'CYP4F2'
    # GENE = sys.argv[1]
    parser.add_argument('--input', help='Path to input FASTA/BAM file')
    parser.add_argument('--gene', help='CYP gene to genotype')
    parser.add_argument("--reference", "-r", type=str, required=True, help="Path to the human reference genome FASTA file")
    args = parser.parse_args()
    GENE = args.gene


    # GENE = 'CYP2C19'
    gene = Gene(script_path(f"aldy.resources.genes/{GENE.lower()}.yml"), genome="hg19")

    profile = Profile.load(gene, "illumina")
    # /project/shared/aldy-data/wgs/NA07055.wgs.cram'
    # path = '/project/shared/aldy-data/wgs/HG00276.wgs.cram'

    path = args.input
    aldy_sample = Sample(
        gene=gene,
        profile=profile,
        path=path,
        # reference="/project/shared/aldy-data/Homo_sapiens_assembly19_1000genomes_decoy.fasta"
        reference=args.reference

    )
    # aldy_sample.coverage._normalize_coverage()
    db = Database(f"data/{GENE.lower()}.pkl")
    sample = SimpleSample(path)
    sample.min_coverage = 3
    sample.expected_coverage = aldy_sample.coverage.diploid_avg_coverage() / 2

    get_variant_info(aldy_sample,db,GENE)

def get_insertion_coverage(aldy_sample, pos, ins_seq):
    cov = aldy_sample.coverage

    for (indel_pos, indel_op), (off, on) in aldy_sample._indel_sites.items():
        if indel_pos == pos and indel_op == f'ins{ins_seq}':
            return on

    for op, cov_list in cov._coverage.get(pos, {}).items():
        if op == f'ins{ins_seq}':
            return len(cov_list)

    return 0
#%%
def filter_allele(a):
    if '#' in a.name:
        a.enabled = False
        return

    for (pos, op) in a.gene.functional:
        position = a.gene.positions[pos]

        if (pos, op) in a.func:
            if '.' in op and '>' in op:
                required_base = 'M'
            elif '>' in op:
                required_base = convert_op_to_variant(op)
            elif op.startswith('del'):
                required_base = 'N'
            elif op.startswith('ins'):
                if pos == 42524928:
                    ins_seq = op[3:]
                    coverage = get_insertion_coverage(aldy_sample, pos, ins_seq)
                    if coverage < sample.min_coverage:
                        print(f'{a.name} was filtered out as {pos} {op} not covered (cov={coverage})')
                        a.enabled = False
                        return
                    continue
                else:
                    required_base = 'P'
            else:
                required_base = position.ref_base
        else:
            allele_has_other_mutation_at_pos = any(
                p == pos for (p, o) in (a.func | a.minor)
            )

            if allele_has_other_mutation_at_pos:
                continue
            else:
                required_base = position.ref_base

        coverage = position.variants[required_base].coverage.count

        if coverage < sample.min_coverage:
            print(f'{a.name} was filtered out as {pos} {op} needs {required_base} but not covered')
            a.enabled = False
            return

    a.enabled = True



if __name__ == "__main__":
    from common_cyp import *
    g = db.genes[GENE]
    # g = sys.argv[1]
    ax = {}
    for a in g.alleles.values():
        filter_allele(a)

        if a.enabled:
            major_allele = db.minor_to_major[a.name]
            group_key = major_allele

            ax.setdefault(group_key, []).append(a.name)

    for major, minors in ax.items():
        alleles_str = '; '.join(minors)
        print(f"[filter] {g.name} {major} => {alleles_str}")

    valid_count = sum(1 for a in g.alleles.values() if a.enabled)
    print(f"[filter] valid_alleles={valid_count}")

# enabled = ['1.001','1.002','1.004','1.016']
# for a in db.genes[GENE].alleles:
#     if not a in enabled:
#         db.genes[GENE].alleles[a].enabled = False
#%%
def populate_valid_positions(sample):
    for gene in db.genes.values():
        for position in gene.positions.values():
            position.is_valid = False

    for allele in db.alleles():
        if not allele.enabled:
            continue
        gene = allele.gene

        # # if allele == gene.wildtype:

        # for mutation in gene.functional:
        #     pos, op = mutation
        #     gene.positions[pos].is_valid = True
        # else:
        for mutation in allele.func:
            pos, op = mutation

            gene.positions[pos].is_valid = True

        # Process minor mutations
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

#%%
def zero_out_low_coverage_variants(sample, db, coverage_threshold=8):
    for gene in db.genes.values():
        for position in gene.positions.values():
            for variant in position.variants.values():
                if variant.coverage.count <= coverage_threshold:
                    variant.coverage.count = 0
                    # variant.coverage.covered_reads.clear()
                    # variant.coverage.covered_hits = []

#%%
def get_allele_region_mask(allele, gene, region_names):
    cn_config = gene.cn_configs.get(allele.cn_config, gene.cn_configs['1'])
    return [cn_config.cn[0].get(r, 1) for r in region_names]


def create_region_mask(sample, gene):
    # region_names = [r for r in gene.regions[0].keys()
    #                 if r.startswith('e') or r.startswith('i')]
    region_names = [r for r in gene.regions[0].keys() ]
    mask = torch.tensor([
        get_allele_region_mask(gene.alleles[db.minor_to_major[a.name]], gene, region_names)
        for a in sample.valid_alleles
    ], dtype=torch.float32)

    return region_names, mask

def get_region_coverage(aldy_sample, gene):
    cov = aldy_sample.coverage

    region_cov = {}
    for region_name in gene.regions[0].keys():
        # if region_name.startswith('e') or region_name.startswith('i'):
            region_cov[region_name] = cov.region_coverage(0, region_name)

    return region_cov

CYP2D6_CN_CONFIGS = ['1', '68.001','141.1001']

def get_unique_cn_configs(aldy_gene):
    # region_names = [r for r in aldy_gene.regions[0].keys()
    #                 if r.startswith('e') or r.startswith('i')]
    region_names = [r for r in aldy_gene.regions[0].keys()]
    cn_config_info = {}
    for cn_name, cn_config in aldy_gene.cn_configs.items():
        mask = [cn_config.cn[0].get(r, 1) for r in region_names]
        cn_config_info[cn_name] = {
            'mask': mask,
            'cn_config': cn_config
        }

    return region_names, cn_config_info

def filter_alleles_by_cn_config(sample, db, aldy_gene, selected_cn_configs):

    print(f"\n[Filtering alleles by cn_config]")

    # First pass: check which cn_configs have valid alleles
    cn_config_valid_alleles = {cn: [] for cn in selected_cn_configs}
    cn_config_all_alleles = {cn: [] for cn in selected_cn_configs}

    for allele in db.genes['CYP2D6'].alleles.values():
        major_name = db.minor_to_major[allele.name]
        major_allele = aldy_gene.alleles.get(major_name)

        if not major_allele:
            continue

        cn_config_name = major_allele.cn_config

        if cn_config_name in selected_cn_configs:
            cn_config_all_alleles[cn_config_name].append(allele)
            if allele.enabled:
                cn_config_valid_alleles[cn_config_name].append(allele)

    # Check for cn_configs with no valid alleles
    for cn_config in selected_cn_configs:
        if len(cn_config_valid_alleles[cn_config]) == 0:
            print(f"\n  WARNING: CN config '{cn_config}' has no valid alleles!")
            print(f"  Re-enabling alleles and filtering by their own functional mutations...")

            # Re-enable alleles under this cn_config and filter by their own func mutations
            rescued_count = 0
            for allele in cn_config_all_alleles[cn_config]:
                if rescue_allele_by_own_func(allele, sample, db):
                    rescued_count += 1
                    cn_config_valid_alleles[cn_config].append(allele)

            print(f"  Rescued {rescued_count} alleles for cn_config '{cn_config}'")

    # Second pass: disable alleles not in selected cn_configs
    filtered_count = 0
    kept_count = 0
    cn_config_alleles = {}

    for allele in db.genes['CYP2D6'].alleles.values():
        if not allele.enabled:
            continue

        major_name = db.minor_to_major[allele.name]
        major_allele = aldy_gene.alleles.get(major_name)

        if major_allele:
            cn_config_name = major_allele.cn_config

            if cn_config_name not in selected_cn_configs:
                allele.enabled = False
                filtered_count += 1
            else:
                kept_count += 1
                cn_config_alleles.setdefault(cn_config_name, []).append(major_name)

    print(f"\nKept alleles by cn_config:")
    for cn_config, majors in sorted(cn_config_alleles.items()):
        unique_majors = sorted(set(majors))
        print(f"  {cn_config}: {len(unique_majors)} major alleles")
        print(f"    {', '.join(f'*{m}' for m in unique_majors[:10])}" +
              (f" ... +{len(unique_majors)-10} more" if len(unique_majors) > 10 else ""))

    print(f"\n  Filtered: {filtered_count} alleles")
    print(f"  Kept: {kept_count} alleles")

def rescue_allele_by_own_func(allele, sample, db):
    """
    Re-enable an allele if its OWN functional mutations are covered.
    Only checks allele.func, not gene.functional.
    """
    if '#' in allele.name:
        return False

    # Check only the allele's own functional mutations
    for (pos, op) in allele.func:
        if pos not in allele.gene.positions:
            continue

        position = allele.gene.positions[pos]

        # Determine required base
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

        if required_base not in position.variants:
            return False

        coverage = position.variants[required_base].coverage.count

        if coverage < sample.min_coverage:
            return False

    allele.enabled = True
    return True

#%%
def linear_regression_cn(region_observed, cn_config_mask, cn_config_names, max_cn=6.0, num_iterations=500):
    from cn.cn_estimator import run_cn_estimator

    cn_estimates = run_cn_estimator(
        region_cov=region_observed,
        region_mask=cn_config_mask,
        gene_names=cn_config_names,
        max_cn=max_cn,
        num_iterations=num_iterations,
        delta=0.5,
        print_every=1000  # Suppress output
    )

    return cn_estimates
#%%
def create_cn_config_region_mask(aldy_gene, cn_configs_to_use=None):
    region_names, cn_config_info = get_unique_cn_configs(aldy_gene)

    if cn_configs_to_use is None:
        cn_configs_to_use = CYP2D6_CN_CONFIGS

    cn_config_names = []
    masks = []

    for cn_name in cn_configs_to_use:
        if cn_name not in cn_config_info:
            print(f"  Warning: cn_config '{cn_name}' not found, skipping")
            continue

        mask = cn_config_info[cn_name]['mask']
        cn_config_names.append(cn_name)
        masks.append(mask)

    mask_tensor = torch.tensor(masks, dtype=torch.float32) if masks else torch.zeros(0, len(region_names))

    print(f"\nUsing CN configs: {cn_config_names}")

    return region_names, cn_config_names, mask_tensor

def update_deletion_variant_counts(sample, db, aldy_gene, selected_cn_configs):
    print("\n[Updating N variant counts for deleted regions]")

    updated_count = 0

    for cn_config_name, cn_val in selected_cn_configs.items():
        if cn_config_name == '1':
            continue

        cn_config = aldy_gene.cn_configs.get(cn_config_name)
        if not cn_config:
            continue

        deleted_regions = [r for r, v in cn_config.cn[0].items() if v == 0]

        if not deleted_regions:
            continue

        # Use integer count
        expected_n_count = int(round(sample.expected_coverage * cn_val))
        print(f"  CN config '{cn_config_name}' (CN={cn_val:.0f}): deleted regions = {deleted_regions}")
        print(f"    Expected N count per position: {expected_n_count}")

        for pos, position in db.genes['CYP2D6'].positions.items():
            region = aldy_gene.region_at(pos)
            if not region:
                continue

            gene_idx, region_name = region
            if region_name in deleted_regions:
                n_variant = position.variants['N']
                old_count = n_variant.coverage.count or 0
                n_variant.coverage.count = int(old_count) + expected_n_count
                updated_count += 1

    print(f"  Total updated: {updated_count} positions")



def estimate_fusion(aldy_sample, aldy_gene, sample, db, gene_cn, max_cn=3, cn_threshold=1, respect_gene_cn=True, huber_delta=0.5):
    import itertools
    print("\n=== CYP2D6 Fusion Template CN Estimation ===")

    region_names, cn_config_names, cn_config_mask = create_cn_config_region_mask(aldy_gene, CYP2D6_CN_CONFIGS)
    region_cov = get_region_coverage(aldy_sample, aldy_gene)
    region_observed = torch.tensor([region_cov[r] for r in region_names], dtype=torch.float32)

    if gene_cn < 1.66666:
        total_cn_required = 1
    elif gene_cn < 2.66666:
        total_cn_required = 2
    elif gene_cn < 3.5:
        total_cn_required = 3
    elif gene_cn < 4.5:
        total_cn_required = 4
    else:
        total_cn_required = int(round(gene_cn))


    print(f"\nRegion coverage:")
    for r, c in zip(region_names, region_observed.tolist()):
        print(f"  {r}: {c:.1f}")

    print(f"\nCN configs: {cn_config_names}")
    print(f"Gene CN: {gene_cn:.2f} → required total: {total_cn_required}")
    print(f"Respect gene CN constraint: {respect_gene_cn}")

    def weighted_huber_loss(predicted, observed, delta, weights):
        diff = observed - predicted
        abs_diff = torch.abs(diff)
        quadratic = torch.clamp(abs_diff, max=delta)
        linear = abs_diff - quadratic
        per_region_loss = 0.5 * quadratic ** 2 + delta * linear
        return torch.sum(weights * per_region_loss)

    region_weights = torch.ones(len(region_names), dtype=torch.float32)
    for i, r in enumerate(region_names):
        if r.startswith('e'):
            region_weights[i] = 1.5
        if r in ['e1', 'e9', 'i1', 'i8']:
            region_weights[i] = 2.0

    num_configs = len(cn_config_names)
    best_loss = float('inf')
    best_cn = None
    num_evaluated = 0
    all_results = []

    if respect_gene_cn:
        print(f"Enumerating configurations that sum to {total_cn_required}...")
        for cn_tuple in itertools.product(range(max_cn + 1), repeat=num_configs):
            if sum(cn_tuple) != total_cn_required:
                continue

            num_evaluated += 1
            cn_vector = torch.tensor(cn_tuple, dtype=torch.float32)
            expected = cn_vector @ cn_config_mask
            loss = weighted_huber_loss(expected, region_observed, huber_delta * sample.expected_coverage, region_weights).item()
            all_results.append((cn_tuple, loss))

            if loss < best_loss:
                best_loss = loss
                best_cn = cn_tuple

        print(f"Evaluated {num_evaluated} configurations")

        if best_cn is None:
            print("WARNING: No valid configuration found! Falling back to unconstrained.")
            respect_gene_cn = False

    if not respect_gene_cn:
        print(f"Enumerating all configurations (CN 0-{max_cn})...")
        for cn_tuple in itertools.product(range(max_cn + 1), repeat=num_configs):
            num_evaluated += 1
            cn_vector = torch.tensor(cn_tuple, dtype=torch.float32)
            expected = cn_vector @ cn_config_mask
            loss = weighted_huber_loss(expected, region_observed, huber_delta * sample.expected_coverage, region_weights).item()
            all_results.append((cn_tuple, loss))

            if loss < best_loss:
                best_loss = loss
                best_cn = cn_tuple

        print(f"Evaluated {num_evaluated} configurations")

    # Show top 5
    all_results.sort(key=lambda x: x[1])
    print(f"\nTop 5 configurations:")
    for i, (cn_tuple, loss) in enumerate(all_results[:5]):
        config_str = ", ".join(f"{cn_config_names[j]}:{cn_tuple[j]}" for j in range(len(cn_tuple)) if cn_tuple[j] > 0)
        print(f"  {i+1}. loss={loss:.2f}: {config_str}")

    print(f"\nBest CN assignment (loss={best_loss:.2f}):")
    cn_config_cn = {}
    for i, cn_name in enumerate(cn_config_names):
        cn_config_cn[cn_name] = best_cn[i]
        if best_cn[i] > 0:
            print(f"  {cn_name}: {best_cn[i]}")

    print(f"  Total: {sum(best_cn)}")

    # Show expected vs observed
    best_cn_tensor = torch.tensor(best_cn, dtype=torch.float32)
    expected = best_cn_tensor @ cn_config_mask
    print(f"\nFit comparison:")
    for i, r in enumerate(region_names):
        obs = region_observed[i].item()
        exp = expected[i].item()
        diff = obs - exp
        print(f"  {r}: obs={obs:.1f}, exp={exp:.1f}, diff={diff:+.1f}")

    selected_cn_configs = {cn: float(cn_val) for cn, cn_val in cn_config_cn.items()
                          if cn_val >= cn_threshold}

    print(f"\n[Selected CN Configs (CN >= {cn_threshold})]")
    for cn_name, cn_val in sorted(selected_cn_configs.items(), key=lambda x: -x[1]):
        print(f"  {cn_name}: {int(cn_val)}")

    total_cn = sum(selected_cn_configs.values())
    print(f"\n  Total CYP2D6 CN: {int(total_cn)}")

    filter_alleles_by_cn_config(sample, db, aldy_gene, selected_cn_configs)

    return cn_config_cn, selected_cn_configs


def estimate_fusion_A(aldy_sample, aldy_gene, sample, db, gene_cn, max_cn=3, cn_threshold=1, respect_gene_cn=True, huber_delta=0.5):
    import itertools, math
    print("\n=== CYP2D6 Fusion Template CN Estimation ===")

    region_names, cn_config_names, cn_config_mask = create_cn_config_region_mask(aldy_gene, CYP2D6_CN_CONFIGS)
    region_cov = get_region_coverage(aldy_sample, aldy_gene)
    region_observed = torch.tensor([region_cov[r] for r in region_names], dtype=torch.float32)

    # Add CYP2D7 PCE region
    cov = aldy_sample.coverage
    for i in ['e1', 'e2', 'e3', 'e5', 'e6', 'e9', 'pce']:
        pce_cov = cov.region_coverage(1, i)  # regions[1] is CYP2D7
        pce_adjusted = max(0, pce_cov - 2.0)  # Subtract baseline 2 copies
        region_names.append(f'{i}_2d7')
        region_observed = torch.cat([region_observed, torch.tensor([pce_adjusted])])

        # Append PCE contribution to each CN config mask
        # '1' contributes 0, fusion configs contribute 1

        pce_contributions = [
            1 - cn_config_mask[ci][region_names.index(i)]
            for ci, cn_name in enumerate(cn_config_names)
        ]
        if i == 'pce': pce_contributions[0] = torch.tensor(0.)
        # print(i, pce_contributions, file=sys.stderr)
        pce_col = torch.tensor(pce_contributions, dtype=torch.float32).unsqueeze(1)
        cn_config_mask = torch.cat([cn_config_mask, pce_col], dim=1)

    allow_extra_68 = 0
    thresholds = [1+2/3, 2+2/3, 3.5, 4.5]
    for ti, t in enumerate(thresholds):
        if gene_cn < t:
            total_cn_required = ti + 1
            if cov.region_coverage(0, 'i1') - cov.region_coverage(0, 'e2') >= 1.0:
                # cn_estimate does not work sometimes with *68 (because tail is too long),
                # so we increase search space here until cn_estimate is fixed
                # TODO: fix, hacky!
                allow_extra_68 = math.floor(cov.region_coverage(0, 'i1') - cov.region_coverage(0, 'e2'))
            break
    else:
        total_cn_required = int(round(gene_cn))

    print(f"\nRegion coverage:")
    for r, c in zip(region_names, region_observed.tolist()):
        print(f"  {r}: {c:.1f}")

    print(f"\nCN configs: {cn_config_names}")
    print(f"PCE contributions: {dict(zip(cn_config_names, pce_contributions))}")
    print(f"Gene CN: {gene_cn:.2f} → required total: {total_cn_required}")
    print(f"Respect gene CN constraint: {respect_gene_cn}")

    def weighted_huber_loss(predicted, observed, delta, weights):
        diff = observed - predicted
        abs_diff = torch.abs(diff)
        quadratic = torch.clamp(abs_diff, max=delta)
        linear = abs_diff - quadratic
        per_region_loss = 0.5 * quadratic ** 2 + delta * linear
        return torch.sum(weights * per_region_loss)

    # Weight regions
    region_weights = torch.ones(len(region_names), dtype=torch.float32)
    for i, r in enumerate(region_names):
        if r.startswith('e'):
            region_weights[i] = 1.5
        if r in ['e1', 'e9', 'i1']:
            region_weights[i] = 3.0
        if r == 'pce_2d7':
            region_weights[i] = 3.0  # Weight PCE for fusion detection

    num_configs = len(cn_config_names)
    best_loss = float('inf')
    best_cn = None
    num_evaluated = 0
    all_results = []

    if respect_gene_cn:
        print(f"Enumerating configurations that sum to {total_cn_required}...")
        for cn_tuple in itertools.product(range(max_cn + 1), repeat=num_configs):
            if total_cn_required != sum(cn_tuple):
                if not (allow_extra_68 == cn_tuple[1] and total_cn_required == sum(cn_tuple) - allow_extra_68):
                    continue

            num_evaluated += 1
            cn_vector = torch.tensor(cn_tuple, dtype=torch.float32)
            expected = cn_vector @ cn_config_mask
            loss = weighted_huber_loss(expected, region_observed, huber_delta * sample.expected_coverage, region_weights).item()
            all_results.append((cn_tuple, loss))

            if loss < best_loss:
                best_loss = loss
                best_cn = cn_tuple

        print(f"Evaluated {num_evaluated} configurations")

        if best_cn is None:
            print("WARNING: No valid configuration found! Falling back to unconstrained.")
            respect_gene_cn = False

    if not respect_gene_cn:
        print(f"Enumerating all configurations (CN 0-{max_cn})...")
        for cn_tuple in itertools.product(range(max_cn + 1), repeat=num_configs):
            num_evaluated += 1
            cn_vector = torch.tensor(cn_tuple, dtype=torch.float32)
            expected = cn_vector @ cn_config_mask
            loss = weighted_huber_loss(expected, region_observed, huber_delta * sample.expected_coverage, region_weights).item()
            all_results.append((cn_tuple, loss))

            if loss < best_loss:
                best_loss = loss
                best_cn = cn_tuple

        print(f"Evaluated {num_evaluated} configurations")

    # Show top 5
    all_results.sort(key=lambda x: x[1])
    print(f"\nTop 5 configurations:")
    for i, (cn_tuple, loss) in enumerate(all_results[:5]):
        config_str = ", ".join(f"{cn_config_names[j]}:{cn_tuple[j]}" for j in range(len(cn_tuple)) if cn_tuple[j] > 0)
        print(f"  {i+1}. loss={loss:.2f}: {config_str}")

    print(f"\nBest CN assignment (loss={best_loss:.2f}):")
    cn_config_cn = {}
    for i, cn_name in enumerate(cn_config_names):
        cn_config_cn[cn_name] = best_cn[i]
        if best_cn[i] > 0:
            print(f"  {cn_name}: {best_cn[i]}")

    print(f"  Total: {sum(best_cn)}")

    # Show expected vs observed
    best_cn_tensor = torch.tensor(best_cn, dtype=torch.float32)
    expected = best_cn_tensor @ cn_config_mask
    print(f"\nFit comparison:")
    for i, r in enumerate(region_names):
        obs = region_observed[i].item()
        exp = expected[i].item()
        diff = obs - exp
        print(f"  {r}: obs={obs:.1f}, exp={exp:.1f}, diff={diff:+.1f}")

    selected_cn_configs = {cn: float(cn_val) for cn, cn_val in cn_config_cn.items()
                          if cn_val >= cn_threshold}

    config_str = ", ".join(f"{cn_name}:{int(cn_val)}" for cn_name, cn_val in sorted(selected_cn_configs.items(), key=lambda x: -x[1]))
    print(f"\n[Selected CN Configs] {config_str}")

    total_cn = sum(selected_cn_configs.values())
    print(f"  Total CYP2D6 CN: {int(total_cn)}")

    filter_alleles_by_cn_config(sample, db, aldy_gene, selected_cn_configs)

    return cn_config_cn, selected_cn_configs

#%%
if __name__ == "__main__":
    # prep for solver
    zero_out_low_coverage_variants(sample,db)
    populate_valid_positions(sample)
    # print_variant_counts(sample,db)

def rebuild_allele_vectors_for_fusion(db, aldy_gene):

    print("\n[Rebuilding allele vectors for fusion cn_configs]")

    gene = db.genes['CYP2D6']

    for allele in gene.alleles.values():
        major_name = db.minor_to_major.get(allele.name)
        if not major_name or major_name not in aldy_gene.alleles:
            continue

        cn_config_name = aldy_gene.alleles[major_name].cn_config
        if cn_config_name == '1':
            continue  # Normal allele, no changes needed

        cn_config = aldy_gene.cn_configs.get(cn_config_name)
        if not cn_config:
            continue

        deleted_regions = [r for r, v in cn_config.cn[0].items() if v == 0]
        if not deleted_regions:
            continue

        updated_count = 0

        for position in gene.positions.values():
            pos_num = position.position

            region = aldy_gene.region_at(pos_num)
            if not region:
                continue

            gene_idx, region_name = region
            if region_name in deleted_regions:
                # Zero out all variants except N
                for var_key, variant in position.variants.items():
                    if var_key == 'N':
                        allele.extended_allele_vector[variant.index] = 1
                    else:
                        allele.extended_allele_vector[variant.index] = 0
                updated_count += 1

        # if updated_count > 0:
        #     print(f"  {allele.name} (cn_config={cn_config_name}): updated {updated_count} positions to N")
#%%
if __name__ == "__main__":
    from cn.cn_estimator import run_cn_estimator

    region_names, region_mask = create_region_mask(sample, db.genes[GENE].aldy_gene)
    region_cov = get_region_coverage(aldy_sample, db.genes[GENE].aldy_gene)
    region_normalized = torch.tensor([region_cov[r] for r in region_names], dtype=torch.float32)

    # CYP 2C8,9,19 only, for CYP2D6, need fusion template level
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
    print(f"  {GENE}: {cn_raw:.2f} → {cn_raw}")
    db.genes[GENE].estimated_cn = cn_raw

    # CYP 2C8, 9, 19, round the estimated CN to 2
    # TODO: CYP 2D6, need to get fusion template CN
    if GENE in ['CYP2C8', 'CYP2C9', 'CYP2C19', 'CYP2B6', 'CYP3A5', 'CYP4F2', 'SLCO1B1', 'TPMT']:
        db.genes[GENE].estimated_cn = 2.0
    #%%
    # if GENE == 'CYP2D6': need to estimate the per - fusion template CN here.
    if GENE == 'CYP2D6':
        # CYP2D6: Enumerate all CN combinations
        cn_config_cn, selected_cn_configs = estimate_fusion_A(
            aldy_sample,
            db.genes[GENE].aldy_gene,
            sample,
            db,
            gene_cn=db.genes[GENE].estimated_cn,
            max_cn=3,
            cn_threshold=1
        )
        db.genes[GENE].estimated_cn = sum(selected_cn_configs.values())
        db.genes[GENE].cn_config_cn = cn_config_cn
        db.genes[GENE].selected_cn_configs = selected_cn_configs
        update_deletion_variant_counts(sample, db, db.genes[GENE].aldy_gene, selected_cn_configs)

        rebuild_allele_vectors_for_fusion(db, db.genes[GENE].aldy_gene)
        populate_valid_positions(sample)

#%%
#%%
def allocate_copies(proportions, total_cn):

    total_cn = int(round(total_cn))

    raw_copies = {a: p * total_cn for a, p in proportions.items()}

    floor_copies = {a: int(c) for a, c in raw_copies.items()}

    remainder = total_cn - sum(floor_copies.values())

    fractional = {a: raw_copies[a] - floor_copies[a] for a in raw_copies}
    sorted_by_frac = sorted(fractional.keys(), key=lambda a: -fractional[a])

    for i in range(remainder):
        floor_copies[sorted_by_frac[i]] += 1

    return {a: c for a, c in floor_copies.items() if c > 0}

def allocate_copies_simple(proportions, total_cn):
    total_cn = int(round(total_cn))
    raw_copies = {a: p * total_cn for a, p in proportions.items()}
    floor_copies = {a: int(c) for a, c in raw_copies.items()}

    for a in floor_copies:
        if floor_copies[a] == 0:
            floor_copies[a] = 1

    # If we now exceed total_cn, reduce from lowest proportion alleles
    while sum(floor_copies.values()) > total_cn and total_cn > 0:
        candidates = [a for a in floor_copies if floor_copies[a] > 1]
        if not candidates:
            break
        min_allele = min(candidates, key=lambda a: proportions[a])
        floor_copies[min_allele] -= 1

    remainder = total_cn - sum(floor_copies.values())
    if remainder > 0:
        fractional = {a: raw_copies[a] - floor_copies[a] for a in raw_copies}
        sorted_by_frac = sorted(fractional.keys(), key=lambda a: -fractional[a])
        for i in range(remainder):
            floor_copies[sorted_by_frac[i]] += 1

    return {a: c for a, c in floor_copies.items() if c > 0}

#%%
def allocate_copies_by_cn_config(proportions, selected_cn_configs, db, aldy_gene, sample):
    """
    Allocate copies respecting cn_config constraints.
    Uses valid_alleles to determine which cn_config each major allele belongs to.
    """
    # Build mapping from major allele -> cn_config based on VALID alleles
    major_to_cn_config = {}

    for allele in sample.valid_alleles:
        major_key = allele.name.split('.')[0]
        major_name = db.minor_to_major.get(allele.name)

        if major_name and major_name in aldy_gene.alleles:
            cn_config_name = aldy_gene.alleles[major_name].cn_config
        else:
            cn_config_name = '1'

        # If this major allele is already mapped to a cn_config,
        # prefer the one that's in selected_cn_configs
        if major_key in major_to_cn_config:
            existing_config = major_to_cn_config[major_key]
            if existing_config not in selected_cn_configs and cn_config_name in selected_cn_configs:
                major_to_cn_config[major_key] = cn_config_name
        else:
            major_to_cn_config[major_key] = cn_config_name

    print(f"\nMajor allele -> cn_config mapping:")
    for major, config in sorted(major_to_cn_config.items()):
        print(f"  *{major} -> {config}")

    # Group alleles by cn_config
    cn_config_alleles = {}
    for major_key, prop in proportions.items():
        cn_config_name = major_to_cn_config.get(major_key, '1')
        cn_config_alleles.setdefault(cn_config_name, {})[major_key] = prop

    # Allocate within each cn_config
    all_copies = {}
    fusion_allocated = False
    for cn_config_name, cn_val in selected_cn_configs.items():
        cn_for_config = int(round(cn_val))

        if cn_for_config == 0:
            continue

        allele_props = cn_config_alleles.get(cn_config_name, {})

        if not allele_props:
            print(f"  CN config '{cn_config_name}' (CN={cn_for_config}): NO ALLELES FOUND")
            continue

        # Normalize proportions within this cn_config
        total_prop = sum(allele_props.values())
        if total_prop > 0:
            normalized_props = {a: p / total_prop for a, p in allele_props.items()}
        else:
            normalized_props = {a: 1.0 / len(allele_props) for a in allele_props}

        # Allocate copies for this cn_config
        copies = allocate_copies(normalized_props, cn_for_config)
        all_copies.update(copies)
        if cn_config_name != '1':
            fusion_allocated = True
        print(f"  CN config '{cn_config_name}' (CN={cn_for_config}): {copies}")

    return all_copies, fusion_allocated

#%%
def check_rescue_68(aldy_sample, fusion_allocated, pce_threshold=0.9):
    cov = aldy_sample.coverage
    pce_val = cov.region_coverage(1, 'pce') - 2.0

    if pce_val <= pce_threshold:
        # print(f"  No *68 rescue: pce_2d7={pce_val:.1f} <= {pce_threshold}")
        return False

    if fusion_allocated:
        # print(f"  No *68 rescue: fusion allele already allocated")
        return False

    # print(f"  Rescue *68: pce_2d7={pce_val:.1f} > {pce_threshold}")
    return True

# check_rescue_68(aldy_sample, selected_cn_configs)
#%% with functional penalty
# run ae
#%% with functional penalty
# run ae
if __name__ == "__main__":
    from core.vae_cyp_B4 import *
    from core.vae_cyp_helper import *

    #%%
    total_mut_counts = sum(variant.coverage.count
                     for gene in db.genes.values()
                     for pos in gene.positions.values()
                     if pos.is_valid
                     for variant in pos.variants.values())
    total_mut_counts = int(round(total_mut_counts))
    mut_counts_tensor = torch.tensor([db.variants()[i].coverage.count for i in sample.valid_indices], dtype=torch.float32)
    valid_allele_names = [f'{a.gene.name}*{a.name}' for a in sample.valid_alleles]

    bam_id = path.split('/')[-1]
    if GENE == 'CYP4F2' and len(valid_allele_names) == 1:
        densities = np.array([1.0])
        densities_un = np.array([1.0])
        learnt_beta = None
    else:
        # densities = run_vae(sample.total_mutations, valid_allele_names, mut_counts_tensor,sample.beta,num_iterations=6000)
        densities, densities_un, learnt_beta = run_vae(
            total_mut_counts,
            valid_allele_names,
            mut_counts_tensor,
            sample = sample,
            db = db,
            num_iterations=3000
        )
    # debug_beta_matrix_detailed(beta, sample, db)

    #%%
    # ma = prepare_results_with_dummy(densities, valid_allele_names, 0.25)
    major_allele_densities = {}
    major_allele_densities_unnorm = {}
    threshold = 0.12

    # print('[Alleles]:')
    for i, density in enumerate(densities):
        allele_name = valid_allele_names[i]
        gene, allele_id = allele_name.split('*')
        major_key = allele_id.split('.')[0]
        # if density > threshold:
        # print(allele_id,density)
        if major_key not in major_allele_densities:
            major_allele_densities[major_key] = 0
            major_allele_densities_unnorm[major_key] = 0
        major_allele_densities[major_key] += density
        major_allele_densities_unnorm[major_key] += densities_un[i]
    
    # Report major alleles above threshold
    final_results = []
    # Filter by threshold and group by gene
    filtered_by_gene = {}
    for major_key in major_allele_densities.keys():
        norm_val = major_allele_densities[major_key]
        if norm_val > threshold:
            gene_name = GENE
            if gene_name not in filtered_by_gene:
                filtered_by_gene[gene_name] = []
            filtered_by_gene[gene_name].append((major_key, norm_val, major_allele_densities_unnorm[major_key]))

    # Apply gene-specific selection
    final_results = []
    for gene_name, alleles in filtered_by_gene.items():
        alleles.sort(key=lambda x: x[1], reverse=True)  # Sort by norm density
        if gene_name in ['CYP2C8', 'CYP2C9', 'CYP2C19']:
            final_results.extend(alleles[:2])
        else:
            final_results.extend(alleles)
    print('\n[Raw]:')
    for major_key, norm_val, unnorm_val in final_results:
        print(f'{GENE}*{major_key}: Normalized={norm_val:.4f}, Unnormalized={unnorm_val:.2f}')

    gene_cn = db.genes[GENE].estimated_cn

    # Build proportions dict from final_results
    proportions = {major_key: norm_val for major_key, norm_val, unnorm_val in final_results}

    if GENE == 'CYP2D6':
        # CYP2D6: Allocate by cn_config
        selected_cn_configs = db.genes[GENE].selected_cn_configs
        has_fusion = any(cn_config != '1' for cn_config in selected_cn_configs.keys())
        print("\nAllocating by cn_config:")
        if has_fusion:
            copies, fusion_allocated = allocate_copies_by_cn_config(
                proportions,
                selected_cn_configs,
                db,
                db.genes[GENE].aldy_gene,
                sample
            )
        else:
            copies = allocate_copies_simple(proportions, gene_cn)
            fusion_allocated = False
    elif GENE in ['CYP2C8', 'CYP2C9', 'CYP2C19']:
        # CYP2C8/9/19: Simple allocation
        gene_cn = db.genes[GENE].estimated_cn
        copies = allocate_copies_simple(proportions, gene_cn)
    else:
        sorted_alleles = sorted(proportions.keys(), key=lambda a: -proportions[a])
        if len(sorted_alleles) >= 2:
            copies = {sorted_alleles[0]: 1, sorted_alleles[1]: 1}
        elif len(sorted_alleles) == 1:
            copies = {sorted_alleles[0]: 2}
        else:
            copies = {}

    #%%
    print("\n[Alleles]")
    for major_key, cn in copies.items():
        for _ in range(cn):
            print(f'{GENE}*{major_key}')
    #%%
    if GENE == 'CYP2D6' and check_rescue_68(aldy_sample, fusion_allocated):
            print(f'{GENE}*68')
    #%%
    # decode_all_alleles_beta(sample,db,learnt_beta)
# %%
