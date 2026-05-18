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
from helper import * # needs attention of common class import override
from common_0620B import  *


#%%
# from V5_3 import run_vae
#%%
if __name__ == "__main__":
    # Use sys.argv when called from command line
    try:
        get_ipython().__class__.__name__
        NOTEBOOK = True
        print("[!!!] running in notebook")
    except:
        NOTEBOOK = False
        print("[!!!] running as script")

    THREADS = 16
    # if not NOTEBOOK and len(sys.argv) > 3:
    #     THREADS = int(sys.argv[3])
    print(f"[!!!] using {THREADS} threads")

#%%
if __name__ == "__main__":
    with timing("initializaiton"):
     
        db = Database("kir.pickle")

    
#%% Read BAM reads
class Hit:
    pass

class Sample:
    def __init__(self, path):
        self.path = path
        self.reads = []

    class Read:
        def __init__(self, name, pair, seq, id = -1, comment=None, qual=None):
            self.name = name
            self.id = id
            self.pair = pair
            self.seq = seq
            self.comment = comment
            self.alignments = {}

@timeit
def get_bam_reads(path: str) -> Sample:
    f = Sample(path)
    print(f'[sample] bam: {path}')
    regions = [
        "chr2:142731000-142734000",
        "chr12:98943000-98946000",
        "chr19:12733000-12736000",
        "chr19:41300000-41400000",
        "chr19:46000000-47000000",
        "chr19:52000000-56000000",
    ]
    with pysam.AlignmentFile(path) as bam:
        for contig in bam.header.references:
            if contig.startswith("chr19_"):
                regions.append(contig)
    for r in regions:
        with pysam.AlignmentFile(path) as bam:
            for read in bam.fetch(region=r):
                f.reads.append(Sample.Read(
                    read.query_name,
                    read.is_read2,
                    read.query_sequence,
                    read.query_qualities
                ))
    f.read_len = len(f.reads[0].seq)
    f.reads.sort(key=lambda r: (r.name, r.pair))
    for id, read in enumerate(f.reads):
        read.id = id
    return f

#%%
@timeit
def get_fq_reads(path: str) -> Sample:
    f = Sample(path)
    print(f'[sample] fasta: {path}')
    with pysam.FastxFile(path) as fq:
        for read in fq:
            name, pair = read.name.split("/")
            f.reads.append(Sample.Read(
                name,
                pair == "2",
                read.sequence,
                read.comment,
                '', # read.quality
            ))
    f.read_len = len(f.reads[0].seq)
    f.reads.sort(key=lambda r: (r.name, r.pair))
    for id, read in enumerate(f.reads):
        read.id = id
    return f

@timeit
def calculate_coverage(path: str, cn_region=("chr22", 19941772, 19969975)):
    # cn_region=("chr19", 46120067, 46124688)
    cn = collections.defaultdict(int)
    with pysam.AlignmentFile(path) as bam:
        for read in bam.fetch(region=f"{cn_region[0]}:{cn_region[1]}-{cn_region[2]}"):
            if read.reference_end is None: continue
            a = (read.reference_start, read.reference_end)
            b = (cn_region[1], cn_region[2])
            if a[0] <= b[0] <= a[1] or b[0] <= a[0] <= b[1]:
                start = read.reference_start
                if read.cigartuples is None: continue
                if read.is_supplementary: continue
                for op, size in read.cigartuples:
                    if op in [0, 7, 8, 2]:
                        for i in range(size):
                            cn[start + i] += 1
                        start += size
    cov = sum(cn.values()) / (cn_region[2] - cn_region[1]) / 2
    # if 'HG03516' in path: cov *= 2  # hack
    return round(cov)
#%%
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='KIR gene typing tool')
    parser.add_argument('--test', action='store_true', help='Run in test mode with simulated data', default=False)
    parser.add_argument('--input', help='Path to input FASTA/BAM file')
    parser.add_argument('--threads', type=int, default=16, help='Number of threads to use')
    parser.add_argument('--seed', type=int, default=42)
    
    if not NOTEBOOK:
        args = parser.parse_args()
        path = args.input
    else:
        args = parser.parse_args([])
        args.test = False
 

        files = []
        with open('input.fq', 'w') as outfile:
            f_number = 0
            for file_path in files:
                f_number += 1
                with open(file_path, 'r') as infile:
                    content = infile.read()
                    lines = content.strip().split('\n')
                    
                    for i in range(0, len(lines), 2):  # Process FASTA format (header + sequence)
                        if i + 1 < len(lines):
                            header = lines[i]
                            sequence = lines[i + 1]
                            
                            if header.startswith('>'):
                                # Extract read info and add file prefix
                                read_info = header[1:]  # Remove '>'
                                new_header = f">f{f_number}_{read_info.replace(' ', '_')}"
                                outfile.write(new_header + '\n')
                                outfile.write(sequence + '\n')
        # path = 'input.fq'
        path = '/project/shared/inumanag-kir/data/hprc/HG01258.final.cram'


    if path.endswith('.fa') or path.endswith('.fq'):
        sample = get_fq_reads(path)
        cov = 20
    else:
        sample = get_bam_reads(path)
        cov = calculate_coverage(path)
    
    if args.test:
        print(f"[!!!] running in test mode")
        sample.ground_truth = []
        # sample.preset_valid_alleles = ['KIR3DP1*0030101','KIR3DP1*007'] # only use those alleles as valid alleles
        
        sample.preset_valid_alleles = [] # only use those alleles as valid alleles
        # Extract ground truth from the input file name
        # Expected format: path/to/KIR2DL1-0010101-KIR2DL1-0020101.fq
        try:
            basename = os.path.basename(path)
            # Remove file extension
            basename = os.path.splitext(basename)[0]
            
            # Parse gene-allele pairs
            pairs = basename.split('-')
            for i in range(0, len(pairs), 2):
                if i+1 < len(pairs):
                    gene = pairs[i]
                    allele = pairs[i+1]
                    sample.ground_truth.append(f"{gene}*{allele}")
            
            print(f"[!!!] ground truth alleles: {sample.ground_truth}")
        except Exception as e:
            print(f"[!!!] Error parsing ground truth from filename: {e}")
            print(f"[!!!] Expected format: KIR2DL1-0010101-KIR2DL1-0020101.fq")
    sample.ground_truth = ['KIR3DL2*0010101']
    sample.expected_coverage = cov
    sample.min_coverage = 3 # math.floor(cov / 3)
    sample.total_error = 0
    print(f"[read] {os.path.basename(sample.path)}: {len(sample.reads):,} reads loaded ({sample.expected_coverage=:.1f}x)")
    # get_indices(sample,db)
    
    # create_mutation_indices_maps(sample,db)
    # get_allele_db(sample,db)
    
#%%
#-%% Map reads
def align_minimap(file, database, MAX_NM, threads):
    alignments = {g: {a: [] for a in db.genes[g].alleles} for g in db.genes}
    if not os.path.exists(database):
        with open(database, "w") as fo:
            for a in db.alleles():
                print(f">{a.gene.name}.{a.name}", file=fo)
                print(a.seq, file=fo)
    with timing("minimap2"):
        cmd = [
            "/cvmfs/soft.computecanada.ca/easybuild/software/2020/avx512/Core/minimap2/2.24/bin/minimap2",
            # "/Users/qinghui_zhou/Documents/Flow/minimap2/minimap2",
            # "/data/qinghuiz/inumanag-kir/minimap2/minimap2",

            "-x", "sr", "--secondary=yes",  # short-read preset. TODO: check if normal works or if -k needs decreasing
            "-c",  # calculate CIGAR
            "-P", "--dual=no",  # do all-to-all mapping
            "-t", str(threads),
            database, file,
        ]
        p = sp.Popen(cmd, stderr=sp.DEVNULL, stdout=sp.PIPE)
        total = 0
        for li, l in enumerate(iter(p.stdout.readline, "")):
            if li % 5_000_000 == 0: print(f"{li:,}", end='...')
            if not l: break

            rn, rl, qs, qe, st, ref, _, rs, re, _, _, _, nm, *_, cg = l.decode().split("\t")
            nm = int(nm[5:])
            rs, re, rl = int(rs), int(re), int(rl)
            if nm + rl - (re - rs) > MAX_NM: continue
            
            h = Hit()
            h.enabled = 0
            h.rid = int(rn)
            h.reversed = st != "+"
            h.cost = nm
            h.st, h.en = rs, re
            h.read_st, h.read_en = int(qs), int(qe)
            h.cigar = cg[5:].strip()
            # h.covered_muts = []
            # h.mutation_read_positions = {}
            h.covered_variants = []
            h.variant_read_positions = {}
            h.ops = None
            # g, a = ref.split(".")
            if ref.startswith("KIR2DL5."):
                parts = ref.split(".")
                g = parts[0]  # "KIR2DL5"
                a = ".".join(parts[1:])  # "A.0010101"
            else:
                g, a = ref.split(".", 1)
            alignments[g][a].append(h)
            sample.reads[h.rid].alignments.setdefault(g, {}).setdefault(a, []).append(h)
            total += 1
        p.wait()
        print()
        assert p.returncode == 0
        for a in db.alleles():
            alignments[a.gene.name][a.name].sort(key=lambda h: h.st)
    print(f"[minimap2] {total:,}/{li:,} mappings handled")
    return alignments

def filter_best_alignments_globally(sample):
    read_alignments = {} 
    
    for read in sample.reads:
        read_alignments[read.id] = []
        for gene_name in read.alignments:
            for allele_name, hits in read.alignments[gene_name].items():
                for hit in hits:
                    read_alignments[read.id].append((hit.cost, gene_name, allele_name, hit))
    
    for read_id, alignments in read_alignments.items():
        if not alignments:
            continue
            
        min_cost = min(cost for cost, _, _, _ in alignments)
        for cost, gene_name, allele_name, hit in alignments:
            if cost > min_cost:
                hit.enabled = False

#%%
if __name__ == "__main__":
    fa = f"{sample.path}.extract.fa"
    with timing("prepare FASTA for mapping"):
        with open(fa, "w") as fo:
            for ri, r in enumerate(sample.reads):
                print(f">{ri}", file=fo)
                print(f"{r.seq}", file=fo)
    MAX_NM = 3
    with open("kirdb_new.fa", 'w') as fo:
        for a in db.alleles():
            print(f">{a.gene.name}.{a.name}", file=fo)
            print(a.seq, file=fo)

    sample.alignments = align_minimap(fa, "kirdb_new.fa", MAX_NM, threads=THREADS)
    #%%
    filter_best_alignments_globally(sample)

#%%
def validate_allele_hit(r, seq, h, a, db):
    def get_cigar(a):
        l = re.split(r"([A-Z=])", a)[:-1]
        return [(int(l[i]), l[i + 1]) for i in range(0, len(l), 2)]
    
    if h.enabled is False:
        return (False, 0, [], [], {})
    
    r = mappy.revcomp(r) if h.reversed else r
    start, s_start = h.st, h.read_st
    
    covered_variants = set()
    covered_muts = set() # since variants are only ACTGNP, single N variant is ambiguous
    variant_read_positions = {}
    
    for size, op in get_cigar(h.cigar):
        if op == "D":
            for i in range(size):
                if a.mutmap.get(start + i, (2, 0))[0] < 2:
                    return (False, 0, [], [], {})
            start += size
        elif op == "I":
            if a.mutmap.get(start, (2, 0))[0] < 2:
                return (False, 0, [],[], {})
            s_start += size
        elif op == "S":
            s_start += size
        elif op == "M":
            for i in range(size):
                
                pos = start + i
                read_pos = s_start + i
                
                if (mm_info := a.mutmap.get(pos, (3, 0)))[0] < 3:
                    
                    if seq[pos] != r[read_pos]: #TODO: fishy since we need novel mutation, need to check if this is a novel mutation or a cross gene artifact
                        
                        return (False, 0, [],[], {})
                    else:
                        mmi, mutation = mm_info
                        gene_name = a.gene.name

                        position = db.genes[gene_name].positions[mutation[0]]
                        read_base = r[read_pos]
                        found_variant = None
                        mmi, mutation = mm_info
                        
                        if mmi == 0:
                            mutation_op = mutation[1]
                            found_mut = mutation
                            if mutation_op.startswith('ins'):
                                found_variant = position.variants['P']
                            elif mutation_op.startswith('del'):
                                found_variant = position.variants['N'] 
                            elif '>' in mutation_op:
                                found_variant = position.variants[mutation_op[2]]

                        elif mmi > 0:  # check wildtype / other bases? 
                            ref_base = position.ref_base
                            if read_base == seq[pos] == ref_base:
                                found_variant = position.variants[position.ref_base]
                            found_mut = (pos, '_')
                        if found_variant:
                            covered_variants.add(found_variant)
                            covered_muts.add(found_mut)
                            variant_read_positions[found_variant] = read_pos
                            
                            if found_variant.variant == 'N' and hasattr(found_variant, 'can_delete') and found_mut[1] in found_variant.can_delete:
                                for deletable_pos in found_variant.can_delete[found_mut[1]]:
                                        n_variant = db.genes[gene_name].positions[deletable_pos].variants['N']
                                 
                                        covered_variants.add(n_variant)
                                        variant_read_positions[n_variant] = read_pos
                                        covered_muts.add((deletable_pos, 'N'))

                                             
                    assert seq[pos] == r[read_pos]
            start += size
            s_start += size

    return (True, 0, covered_variants, covered_muts, variant_read_positions)

def filter_allele(a, thresh=0.58, verbose=False):

    a.enabled = True
    a.max_span = 0
    if a.gene.name == 'KIR2DL1' and a.name == '0020101':
        r = 1
    
    if verbose:
        print(f"\n=== Filtering {a.gene.name}*{a.name} ===")
    
    # # Calculate max span from enabled hits
    # for h in sample.alignments[a.gene.name][a.name]:
    #     if not h.enabled: 
    #         continue
    #     a.max_span = max(h.en - h.st, a.max_span)
    
    if verbose:
        print(f"Max span: {a.max_span}")
    
    # Create coverage array based on variant counts
    span = [0 for _ in range(len(a.seq) + 2)]
    
    # Fill span array with variant coverage counts
    for i in range(len(a.seq)):
        if i in a.gene.positions:
            position = a.gene.positions[i]
            # Sum all variant coverage at this position
            total_coverage = sum(variant.coverage.count for variant in position.variants.values())
            span[i] = total_coverage
    
    if verbose:
        print(f"Coverage array length: {len(span)}")
        print(f"Positions with coverage > 0: {sum(1 for x in span if x > 0)}")
    
    # Collect functional positions sorted by position
    functional_positions = []
    for i, (mmi, mm) in a.mutmap.items():
        if mmi >= 2: 
            continue
        if (mmi == 0 and mm in a.func) or mmi == 1:
            functional_positions.append(i)
    
    functional_positions.sort()
    
    if verbose:
        print(f"Functional positions: {len(functional_positions)} total")
        if len(functional_positions) <= 20:
            print(f"Functional positions: {functional_positions}")
    
    if not functional_positions:
        a.enabled = True
        if verbose:
            print("No functional positions found - allele enabled")
        return
    
    # Step 1: Find deletion ranges from BOTH ends
    if verbose:
        print("\n--- Step 1: Finding deletion ranges ---")
    

    # if a.gene.name in ['KIR3DL3']:
    #     front_func_del_alwd = 1 # front, mmi = 0 allowed
    #     middle_func_del_allowed =0 # middle, other func, widltype should observe
    #     middle_otr_func_del_allowed =2 # middle, other func, widltype should observe
    #     middle_exempt_pos = [ ]

    if a.gene.name in ['KIR3DL3']:
        front_func_del_alwd = 1 # front, mmi = 0 allowed
        middle_func_del_allowed =0 # middle, other func, widltype should observe
        middle_otr_func_del_allowed =0 # middle, other func, widltype should observe
        middle_exempt_pos = [ ]
    
    # elif a.gene.name in ['KIR2DL4']:
    #     front_func_del_alwd = 0 # front, mmi = 0 allowed
    #     middle_func_del_allowed =0 # middle, other func, widltype should observe
    #     middle_otr_func_del_allowed =0 # middle, other func, widltype should observe
    #     middle_exempt_pos = [9837,9877]

    elif a.gene.name in ['KIR3DL2']:
        front_func_del_alwd = 1 # front, mmi = 0 allowed
        middle_func_del_allowed =0 # middle, other func, widltype should observe
        middle_otr_func_del_allowed =1 # middle, other func, widltype should observe
        middle_exempt_pos = [3612]

    elif a.gene.name in ['KIR2DS5']:
        front_func_del_alwd = 0
        middle_exempt_pos = [6146, 6147, 9571]
        middle_func_del_allowed = 0
        middle_otr_func_del_allowed =0
    
    elif a.gene.name in ['KIR2DP1']:
        front_func_del_alwd = 2
        middle_exempt_pos = [3949, 3978, 4023, 5790, 9181]
        middle_func_del_allowed = 0
        middle_otr_func_del_allowed =1
    
    elif a.gene.name in ['KIR2DL1']:
        front_func_del_alwd = 0
        middle_exempt_pos = [5757, 13416,13433]
        middle_func_del_allowed = 0
        middle_otr_func_del_allowed =0
        
    else:
        front_func_del_alwd = 0
        middle_func_del_allowed =0 # middle, other func, widltype should observe
        middle_otr_func_del_allowed =1 # middle, other func, widltype should observe
        middle_exempt_pos = [ ]
    

    front_func_del_tot = 0
  # Find deletion from start (continuous uncovered positions from beginning)
    deletion_start_end = 0  # End of start deletion
    for pos in functional_positions:
        # Calculate actual variant coverage at this position
        mmi, mm = a.mutmap.get(pos, (None, None))
        w_pos = mm[0]
        position = a.gene.positions[w_pos]
        

        total_coverage = sum(variant.coverage.count for variant in position.variants.values())

        if mmi == 0:
                assert mm in a.func
                specific_coverage = 0
                for variant in position.variants.values():
                    # if variant.mutation == mm:
                    if mm in variant.mutations:
                        spec_coverage = variant.coverage.count
                        break
                
                if spec_coverage < sample.min_coverage  and mm[0] not in middle_exempt_pos:
                    # a.enabled = False
                    front_func_del_tot += 1
                    if front_func_del_tot > front_func_del_alwd:
                        a.enabled = False
                        if verbose:
                            print(f"FILTERED IMMEDIATELY in start deletion detection: Functional mutation {mm} at allele pos {pos} (wildtype pos {w_pos}) has insufficient coverage ({specific_coverage})")
                        else:
                            pass
                            # print(f'[info] {a.gene.name}*{a.name} filtered: functional mutation {mm} at pos {pos} has insufficient coverage ({specific_coverage})')
                        return

        if total_coverage < sample.min_coverage:
            deletion_start_end = max(deletion_start_end, w_pos + 1)
            if verbose:
                print(f"Start deletion: pos {w_pos} coverage {total_coverage} < {sample.min_coverage}")
        else:
            if verbose:
                print(f"Start deletion stopped at pos {w_pos} with coverage {total_coverage}")
            break  # Stop at first covered position
    
    # Find deletion from end (continuous uncovered positions from end)
    deletion_end_start = len(a.seq)  # Start of end deletion
    for i in range(len(functional_positions) - 1, -1, -1):
        pos = functional_positions[i]
        mmi, mm = a.mutmap.get(pos, (None, None))
        w_pos = mm[0]
        # Calculate actual variant coverage at this position
        position = a.gene.positions[w_pos]
        total_coverage = sum(variant.coverage.count for variant in position.variants.values())
        
        if mmi == 0:
                assert mm in a.func
                specific_coverage = 0
                for variant in position.variants.values():
                    # if variant.mutation == mm:
                    if mm in variant.mutations:
                        specific_coverage = variant.coverage.count
                        break
                
                if specific_coverage < sample.min_coverage:
                    a.enabled = False
                    if verbose:
                        print(f"FILTERED IMMEDIATELY in start deletion detection: Functional mutation {mm} at allele pos {pos} (wildtype pos {w_pos}) has insufficient coverage ({specific_coverage})")
                    else:
                        pass
                        # print(f'[info] {a.gene.name}*{a.name} filtered: functional mutation {mm} at pos {pos} has insufficient coverage ({specific_coverage})')
                    return
                
        if total_coverage < sample.min_coverage:
            deletion_end_start = min(deletion_end_start, w_pos)
            if verbose:
                print(f"End deletion: pos {pos} coverage {total_coverage} < {sample.min_coverage}")
        else:
            if verbose:
                print(f"End deletion stopped at pos {pos} with coverage {total_coverage}")
            break  # Stop at first covered position from end
    
    # Calculate total deletion length
    start_deletion_length = deletion_start_end
    end_deletion_length = max(0, len(a.seq) - deletion_end_start)
    total_deletion_length = start_deletion_length + end_deletion_length
    
    if verbose:
        print(f"Start deletion length: {start_deletion_length}")
        print(f"End deletion length: {end_deletion_length}")
        print(f"Total deletion length: {total_deletion_length}")
        print(f"Allele length: {len(a.seq)}")
    
    # Ensure deletions don't overlap (if they do, the whole allele is deleted)
    if deletion_start_end >= deletion_end_start:
        a.enabled = False
        if verbose:
            print(f"FILTERED: Complete deletion (start {deletion_start_end} >= end {deletion_end_start})")
        return
    
    deletion_ratio = total_deletion_length / len(a.seq)
    
    if verbose:
        print(f"Deletion ratio: {deletion_ratio:.1%} (threshold: {thresh:.1%})")
    
    # Require total deletion to be less than threshold% of allele length
    if deletion_ratio >= thresh:
        a.enabled = False
        if verbose:
            print(f"FILTERED: Total deletion too large ({deletion_ratio:.1%} >= {thresh:.1%})")
        return
    
    # Step 2: Check middle (non-deleted) range with nuanced rules
    if verbose:
        print(f"\n--- Step 2: Checking middle range [{deletion_start_end}, {deletion_end_start}) ---")
    
    middle_functional_positions = [pos for pos in functional_positions 
                                 if deletion_start_end <= pos < deletion_end_start]
    
    if verbose:
        print(f"Middle functional positions: {len(middle_functional_positions)}")
        print(f"Middle positions: {middle_functional_positions}")
    
    # Count mmi=0 and mmi=1 positions in middle range
    middle_mmi0_positions = []
    middle_mmi1_positions = []
    
    for pos in middle_functional_positions:
        mmi, mm = a.mutmap.get(pos, (None, None))
        if mmi == 0:
            middle_mmi0_positions.append(pos)
        elif mmi == 1:
            middle_mmi1_positions.append(pos)
    
    if verbose:
        print(f"Middle mmi=0 positions: {len(middle_mmi0_positions)}")
        print(f"Middle mmi=1 positions: {len(middle_mmi1_positions)}")
    
    # Apply different rules based on number of mmi=0 positions
    if len(middle_mmi0_positions) >= 0:
        if verbose:
            print("Using lenient rules (>=3 mmi=0 positions)")
        
        # Many functional mutations - allow some misses
        
        # For mmi=0 positions, check specific variant coverage
        mmi0_uncovered = 0
        for pos in middle_mmi0_positions:
            # if pos in a.gene.positions:
                mmi, mm = a.mutmap.get(pos, (None, None))
                position = a.gene.positions[mm[0]]
                
                # Find the specific variant for this mutation
                specific_coverage = 0
                for variant in position.variants.values():
                    # if variant.mutation == mm:
                    if mm in variant.mutations:
                        specific_coverage = variant.coverage.count
                        break
                
                if specific_coverage < sample.min_coverage and mm[0] not in middle_exempt_pos:
                    mmi0_uncovered += 1
                    if verbose:
                        print(f"  mmi=0 pos {pos} mutation {mm}: coverage {specific_coverage} < {sample.min_coverage}")
                elif verbose:
                    print(f"  mmi=0 pos {pos} mutation {mm}: coverage {specific_coverage} >= {sample.min_coverage} ✓")
        
        # For mmi=1 positions, check total position coverage
        mmi1_uncovered = 0
        for pos in middle_mmi1_positions:
                mmi, mm = a.mutmap.get(pos, (None, None))
                position = a.gene.positions[mm[0]]
                wildtype_coverage = 0
                wildtype_coverage = position.variants[position.ref_base].coverage.count

                if wildtype_coverage < sample.min_coverage and mm[0] not in middle_exempt_pos:
                    mmi1_uncovered += 1
                    if verbose:
                        print(f"  mmi=1 pos {pos}: total coverage {wildtype_coverage} < {sample.min_coverage}")
                elif verbose:
                    print(f"  mmi=1 pos {pos}: total coverage {wildtype_coverage} >= {sample.min_coverage} ✓")
        
        if verbose:
            print(f"mmi=0 uncovered: {mmi0_uncovered}/1 allowed")
            print(f"mmi=1 uncovered: {mmi1_uncovered}/1 allowed")
        
       
        if mmi0_uncovered > middle_func_del_allowed:
            mmi, mm = a.mutmap.get(middle_mmi0_positions[0], (None, None))
            a.enabled = False
            if verbose:
                print(f"FILTERED: {mmi0_uncovered} mmi=0 uncovered in middle")
            # else:
                # pass
                # print(f'[info] {a.gene.name}*{a.name} filtered: {mmi0_uncovered} mmi=0 uncovered in middle')
            return
        
        if mmi1_uncovered > middle_otr_func_del_allowed:
            mmi, mm = a.mutmap.get(middle_mmi1_positions[0], (None, None))
            a.enabled = False
            if verbose:
                print(f"FILTERED: {mmi1_uncovered} mmi=1 uncovered in middle")
            # else:
                # print(f'[info] {a.gene.name}*{a.name} filtered: {mmi1_uncovered} mmi=1 uncovered in middle')
            return
    
    # Check minor mutations using variant coverage
    if verbose:
        print(f"\n--- Step 3: Checking minor mutations ---")
        print(f"Minor mutations: {len(a.minor)}")
    
    minor_covered = {m: 0 for m in a.minor}
    
    for i, (mmi, mm) in a.mutmap.items():
        if mmi >= 2: 
            continue
        if (mmi == 0) and mm in a.minor:
            position = a.gene.positions[mm[0]]
            
            for variant in position.variants.values():
                # if variant.mutation == mm:
                if mm in variant.mutations:
                    if verbose:
                        print(f"  Minor mutation {mm} at pos {i}: coverage {variant.coverage.count}")
                    if variant.coverage.count >= sample.min_coverage:
                        minor_covered[mm] = 1
                    break
    
    a.uncovered = sum(1 for i in span if i < sample.min_coverage)
    a.minor_uncovered = sum(1 for i in minor_covered.values() if not i) if minor_covered else 0
    
    if verbose:
        print(f"Minor covered: {sum(minor_covered.values())}/{len(minor_covered)}")
        print(f"Minor uncovered: {a.minor_uncovered}")
    
    minor_uncovered_allowed = 0.95
    if minor_covered:
        if a.minor_uncovered / len(minor_covered) > minor_uncovered_allowed:
            a.enabled = False
            if verbose:
                print("FILTERED: All minor mutations lack sufficient coverage")
            else:
                print(f'[info] {a.gene.name}*{a.name} was filtered out because all of its minor mutations were not covered.\n')
        else:
            a.minor_miss = a.minor_uncovered / len(minor_covered)
            if verbose:
                print(f"Minor miss ratio: {a.minor_miss:.2%}")
    else:
        a.minor_miss = 0
        if verbose:
            print("No minor mutations to check")
    
    if verbose:
        if a.enabled:
            print(f"✓ PASSED: {a.gene.name}*{a.name} enabled")
        else:
            print(f"✗ FILTERED: {a.gene.name}*{a.name} disabled")
        print("=" * 50)

def filter_single_allele_variants(sample, db, min_alleles=2):
  
    
    # Step 1: Count how many alleles support each variant for each read
    read_variant_support = {}  # {read_id: {variant_key: set(allele_names)}}
    
    for read in sample.reads:
        read_variant_support[read.id] = {}
        
        for gene_name, allele_dict in read.alignments.items():
            for allele_name, hits in allele_dict.items():
                for hit in hits:
                    if hit.enabled and hasattr(hit, 'covered_variants'):
                        for variant in hit.covered_variants:
                            variant_key = (variant.gene.name, variant.pos, variant.variant)
                            
                            if variant_key not in read_variant_support[read.id]:
                                read_variant_support[read.id][variant_key] = set()
                            
                            read_variant_support[read.id][variant_key].add(f"{gene_name}*{allele_name}")
    
    # Step 2: For each read, check if it has ANY variants with sufficient support
    removed_count = 0
    
    for read in sample.reads:
        if read.id not in read_variant_support:
            continue
        
        # Check if this read has ANY variants supported by 2+ alleles
        has_well_supported_variants = any(
            len(supporting_alleles) >= min_alleles 
            for supporting_alleles in read_variant_support[read.id].values()
        )
        
        if has_well_supported_variants:
            # Read has some well-supported variants, filter the poorly supported ones
            variants_to_remove = []
            for variant_key, supporting_alleles in read_variant_support[read.id].items():
                if len(supporting_alleles) < min_alleles:
                    variants_to_remove.append((variant_key, supporting_alleles))
            
            # Remove poorly supported variants
            if variants_to_remove:
                for gene_name, allele_dict in read.alignments.items():
                    for allele_name, hits in allele_dict.items():
                        for hit in hits:
                            if hit.enabled and hasattr(hit, 'covered_variants'):
                                new_covered_variants = []
                                for variant in hit.covered_variants:
                                    variant_key = (variant.gene.name, variant.pos, variant.variant)
                                    
                                    keep_variant = True
                                    for remove_key, supporting_alleles in variants_to_remove:
                                        if variant_key == remove_key:
                                            keep_variant = False
                                            removed_count += 1
                                            # print(f"Removed variant {variant_key} from read {read.id} "
                                            #       f"(only supported by: {supporting_alleles})")
                                            break
                                    
                                    if keep_variant:
                                        new_covered_variants.append(variant)
                                
                                hit.covered_variants = new_covered_variants
        # else:
        #     # Read has NO well-supported variants - keep ALL variants
        #     total_variants = len(read_variant_support[read.id])
        #     if total_variants > 0:
        #         print(f"Read {read.id}: keeping all {total_variants} variants "
        #               f"(no variants have 2+ allele support)")
    
    print(f"Filtered {removed_count} variants (kept reads with no well-supported variants)")

def parse_allele_hits(a, sample, MAX_IS=2_000):

    for h in sample.alignments[a.gene.name][a.name]:
        result = validate_allele_hit(sample.reads[h.rid].seq, a.idx_seq, h, a, db)
        h.enabled, h.cover, h.covered_variants, h.covered_muts, h.variant_read_positions = result
      
        if a.gene.name == 'KIR3DL3' and a.name == '0020201':
            if h.rid == 1944:
                print(h.enabled, h.st,h.en,[str(a) for a in h.covered_variants])
    for h in sample.alignments[a.gene.name][a.name]:
        if not h.enabled: 
            continue
            
        if h.rid and sample.reads[h.rid - 1].name == sample.reads[h.rid].name:
            rp = sample.reads[h.rid - 1]
        elif h.rid + 1 < len(sample.reads) and sample.reads[h.rid + 1].name == sample.reads[h.rid].name:
            rp = sample.reads[h.rid + 1]
        else:
            continue
            
        for hp in rp.alignments.get(a.gene.name, {}).get(a.name, []):
            if not hp.enabled: 
                continue
            if abs(h.st - hp.st) < MAX_IS: 
                break
        else:
            # did not find a pair. maybe this is at the edge of the allele? if so, keep it!
            if not (h.st < MAX_IS or h.en + MAX_IS > len(a.idx_seq)):
                h.enabled = False
        # if a.gene.name == 'KIR3DL3' and a.name == '0020201':
        #     if h.rid == 1944:
        #         print(h.enabled, h.st,h.en,[str(a) for a in h.covered_variants])
        # if h.enabled:
        #     for found_variant in h.covered_variants:
        #         found_variant.coverage.add_coverage(h)
            
def parse_gene_alignments(g, args):
    
    for a in g.alleles.values():
        parse_allele_hits(a, sample)
    # filter_single_allele_variants(sample,db, min_alleles=2)
    for a in g.alleles.values():
        for h in sample.alignments[a.gene.name][a.name]:
            if h.enabled:
                for found_variant in h.covered_variants:
                    found_variant.coverage.add_coverage(h)
    for a in g.alleles.values():
        a.enabled = False
        if not args.test:
            filter_allele(a,verbose=False)
        else:
            if sample.preset_valid_alleles:
                for gt in sample.preset_valid_alleles:
                    gg, aa = gt.split('*')
                    if a.gene.name == gg and a.name == aa:
                        a.enabled = True
            else:
                # do not use allele filter in test mode
                g_enabled = False
                for gt in sample.ground_truth:
                    parts = gt.split('*')
                    if len(parts) == 2:
                        gg = parts[0]  # Gene name
                        aa = parts[1]  # Allele name
                        if gg == g.name:
                            g_enabled = True
                            break
                
                # Enable or disable alleles based on whether this gene is in ground truth
                for allele in g.alleles.values():
                    allele.enabled = g_enabled
    
    return g.name, g.alleles, {
        an: [(h.enabled, h.cover, h.covered_variants, h.covered_muts, h.variant_read_positions) 
             for h in hits] 
        for an, hits in sample.alignments[g.name].items()
    }

def populate_deleted_positions(sample, db):
    
    
    for gene in db.genes.values():
        for position in gene.positions.values():
            total_coverage = sum(variant.coverage.count for variant in position.variants.values())
            
            if total_coverage < sample.min_coverage:
                position.has_no_coverage = True
                print(f'{gene.name} {position.position} has no coverage')
            else:
                position.has_no_coverage = False


#%%
if __name__ == "__main__":
    with timing("filtering"):
        res = []
        for g in tqdm(db.genes.values(), desc="Processing genes"):
            res.append(parse_gene_alignments(g, args))
        
        for gn, ga, gal in res:
            db.genes[gn].alleles = ga
            for an, hits in sample.alignments[gn].items():
                for hi, h in enumerate(hits):
                    h.enabled, h.cover, h.covered_variants, h.covered_muts, h.variant_read_positions = gal[an][hi]
        
    for g in db.genes.values():
        if not args.test:
            ax = {}
            for a in g.alleles.values():
                if a.enabled:
                    major_allele = db.minor_to_major[g.name, a.name]
                    
                    # Handle KIR2DL5 grouping specially
                    if g.name == "KIR2DL5":
                        group_key = major_allele[:5]
                    else:
                        # For other genes: use first 3 characters as before
                        group_key = major_allele[:3]
                    
                    ax.setdefault(group_key, []).append((a.name, a.uncovered/len(a.idx_seq)))
            
            for a, aa in ax.items():
                al = '; '.join(f"{x[0]}: {x[1]:.1%}" for x in sorted(aa, key=lambda x: x[1]))
                print(f"[filter] {g.name} {a} => {al}")

    print(f"[filter] valid_alleles={sum(1 for a in db.alleles() if a.enabled):,}")
    populate_deleted_positions(sample, db)

#%%
def populate_valid_positions(sample):
    
    # First, reset all positions to invalid
    for gene in db.genes.values():
        for position in gene.positions.values():
            position.is_valid = False
    
    for allele in db.alleles():
        if not allele.enabled:
            continue
        gene = allele.gene
        # Process functional mutations
        # if allele == gene.wildtype and sum(a.enabled for a in gene.alleles.values()) == 1:
        if allele == gene.wildtype:

            for mutation in gene.functional:
                pos, op = mutation
                if not gene.positions[pos].has_no_coverage:
                    gene.positions[pos].is_valid = True
        else:
            for mutation in allele.func:
                pos, op = mutation
                if not gene.positions[pos].has_no_coverage:
                    # if pos in gene.positions:
                    gene.positions[pos].is_valid = True
            
            # Process minor mutations  
            for mutation in allele.minor:
                pos, op = mutation
                if not gene.positions[pos].has_no_coverage:
                    # if pos in gene.positions:
                    gene.positions[pos].is_valid = True


def identify_reads_with_variant(sample, db):
    reads_hosted_with_variants = {}
    
    for r in sample.reads:
        r.covered_variants = {}
        for g in r.alignments:
            for a in r.alignments[g]:
                for h in r.alignments[g][a]:
                    if h.enabled and len(h.covered_variants) > 0:
                        valid_variants = []
                        
                        # Check each variant for validity
                        for variant in h.covered_variants:
                            # Check if position is valid and apply major_only filter
                            if variant.position.is_valid :
                                valid_variants.append(variant)

                        if valid_variants and h.rid not in reads_hosted_with_variants:
                            reads_hosted_with_variants[h.rid] = {}
                        
                        # Store each valid variant with its position in the read
                        for variant in valid_variants:
                            if variant in h.variant_read_positions:
                                read_pos = h.variant_read_positions[variant]
                                reads_hosted_with_variants[h.rid][variant] = read_pos
                                r.covered_variants[variant] = read_pos
    return reads_hosted_with_variants

def analyze_homology(sample, db, reads_hosted_with_variants):    
    for rid in reads_hosted_with_variants:
        
        # Check the read's mapping on all alleles
        for gene, allele_dict in sample.reads[rid].alignments.items():
            for allele, hits in allele_dict.items():
                target_allele_obj = db.genes[gene].alleles[allele]
                if not target_allele_obj.enabled:
                    continue
                    
                for h in hits:
                    if not h.enabled:
                        continue
                        
                    for variant_obj, read_pos in reads_hosted_with_variants[rid].items():
                        source_gene = variant_obj.gene.name
                        target_gene = gene
                        
                        if source_gene == target_gene:
                            continue
                        
                        # Calculate infected position on target allele
                        infected_pos = h.st + read_pos
                        
                        # Validate infection with sequence matching
                        r = sample.reads[rid].seq
                        r = mappy.revcomp(r) if h.reversed else r
                        
                        if (infected_pos < len(target_allele_obj.seq) and
                            target_allele_obj.seq[infected_pos] == r[read_pos]):
                            # print('[info] Add infection.')
                            # Add infection using the allele method
                            target_allele_obj.add_infected_variant(
                                variant_obj,
                                infected_pos,
                                rid
                            )

def create_estimated_strength_tensor(sample, db):
    valid_alleles = sample.valid_alleles
    total_variants = sum(len(pos.variants) for gene in db.genes.values() for pos in gene.positions.values())
    
    estimated_strength = np.zeros((len(valid_alleles), total_variants), dtype=np.float64)
    
    for allele_idx, allele in enumerate(valid_alleles):
        for variant_idx in range(len(sample.valid_indices)):
            # Get the variant object from the valid_indices
            global_variant_idx = sample.valid_indices[variant_idx]
            variant_obj = db.variants()[global_variant_idx]
            
            variant_position = variant_obj.pos
            
            expected_strength = allele.get_position_strength(variant_position, default=0.0)
            
            estimated_strength[allele_idx, global_variant_idx] = expected_strength
    
    estimated_strength = estimated_strength[:, sample.valid_indices]
    
    sample.estimated_strength = estimated_strength
    
def populate_extended_allele_vector(sample, db):    
    # Step 1: Find reads with variants (new approach)
    reads_hosted_with_variants = identify_reads_with_variant(sample, db)
    
    # Step 2: Analyze homology and process infections
    analyze_homology(sample, db, reads_hosted_with_variants)
    
    # Step 3: Get valid variants
    valid_variants = {v for gene in db.genes.values() for pos in gene.positions.values() 
                     if pos.is_valid for v in pos.variants.values()}
    
    # Step 4: Create beta matrix using extended allele vectors
    valid_alleles = [a for a in db.alleles() if a.enabled]
    sample.valid_alleles = valid_alleles
    sample.valid_indices = sorted([v.index for v in valid_variants])
    
    position_order = []
    for idx in sample.valid_indices:
        variant = db.variants()[idx]
        pos_key = (variant.gene.name, variant.pos)
        if pos_key not in position_order:
            position_order.append(pos_key)

    valid_positions = [db.genes[gene].positions[pos] for gene, pos in position_order]
    

    for allele in sample.valid_alleles:
        allele.generatable_positions = []
        
        for pos_obj in valid_positions:  # SAME ORDER as create_priors
            # Check if this allele has any variant at this position
            has_variant = any(
                allele.extended_allele_vector[variant.index] == 1 
                for variant in pos_obj.variants.values()
            )
            
            if has_variant:
                allele.generatable_positions.append(pos_obj)
                
    total_variants = sum(len(pos.variants) for gene in db.genes.values() for pos in gene.positions.values())
    sample.beta = np.zeros((len(valid_alleles), total_variants), dtype=np.float64)
    
    # Fill beta matrix with extended allele vectors
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
    
    # check_allele_vector(sample)
    print(f"[dedupe] Kept {len(sample.valid_alleles)} unique alleles out of {len(valid_alleles)} total")
    print([(i, j, f"{sample.valid_alleles[i].gene.name}*{sample.valid_alleles[i].name}", f"{sample.valid_alleles[j].gene.name}*{sample.valid_alleles[j].name}") for i in range(len(sample.beta)) for j in range(i+1, len(sample.beta)) if np.array_equal(sample.beta[i], sample.beta[j])])
    

#%%
def zero_out_low_coverage_variants(sample, db, coverage_threshold=3):
    for gene in db.genes.values():
        for position in gene.positions.values():
            for variant in position.variants.values():
                if variant.coverage.count <= coverage_threshold:
                    # Clear covered_reads to make coverage.count = 0
                    variant.coverage.covered_reads.clear()
                    variant.coverage.covered_hits = []
 #%%
def create_cross_gene_strength_mask(sample, db):
 
    import torch
    num_alleles = len(sample.valid_alleles)
    num_mutations = len(sample.valid_indices)
    
    # Initialize with 1.0 (default strength multiplier)
    strength_mask = torch.ones(num_alleles, num_mutations)
    
    print("\nCross-gene strength adjustments:")
    print("=" * 80)
    
    for allele_idx, allele in enumerate(sample.valid_alleles):

        
        allele_name = f"{allele.gene.name}*{allele.name}"
        major_type = allele.name[:3]  # Get major allele type (001, 021, etc.)
        
        print(f"\n{allele_name} (major: {major_type}):")
        
        for pos_obj in allele.generatable_positions:
            if pos_obj.gene.name == allele.gene.name:
                continue  # Skip same-gene positions
            
            # Check all variants at this cross-gene position
            for variant in pos_obj.variants.values():
                if allele.extended_allele_vector[variant.index] == 1:
                    
                    # Count infection reads
                    if variant in allele.infection_sources:
                        total_reads = sum(len(read_ids) for read_ids in allele.infection_sources[variant].values())
                     
                        # Calculate strength as ratio to expected coverage
                        strength = total_reads / sample.expected_coverage
                        if strength > 1: strength = 1
                        # Find mutation index
                        if variant.index in sample.valid_indices:
                            mut_idx = sample.valid_indices.index(variant.index)
                            strength_mask[allele_idx, mut_idx] = strength
                            
                            print(f"  {pos_obj.gene.name}:{pos_obj.position}→{variant.variant}: "
                                  f"infection_reads={total_reads}, strength={strength:.2f}")
    
    return strength_mask

#%%
#%%
if __name__ == "__main__":
    # prep for solver
    zero_out_low_coverage_variants(sample,db)
    populate_valid_positions(sample)
    # db.genes['KIR3DL1'].positions[271].is_valid = False
    populate_extended_allele_vector(sample,db)
    # set_expected_position_strength(sample, db)
    # create_estimated_strength_tensor(sample,db)
    #%%
    print_variant_counts(sample,db)
    # filter_unique_patterns(sample)
    # from helper import *
    # save_mutations_new_implementation(sample,db,'new_avec.pkl')
    from coverage import *
    bcn = bin_copy_numbers(estimate_CN(sample,db))
    for g in bcn:
        if bcn[g]>=1:
            print(g,bcn[g])
            db.genes[g].estimated_cn = bcn[g]
    
#%% with functional penalty
def populate_gene_coverage(sample, db):
    def get_cigar(a):
        l = re.split(r"([A-Z=])", a)[:-1]
        return [(int(l[i]), l[i + 1]) for i in range(0, len(l), 2)]
    enabled_genes = [g for g in db.genes if any(a.enabled for a in db.genes[g].alleles.values())]

    enabled_genes_set = set(enabled_genes)
    
    for gene_name in enabled_genes:
        gene = db.genes[gene_name]
        gene.coverage = defaultdict(int)
        pos = 0
        for region in gene.wildtype.regions.values():
            region.start, region.end, region.reads = pos, pos + len(region.seq), set()
            region.read_spans = {}  # {read_id: (start, end)} in region coords
            pos += len(region.seq)
    
    processed = {g: set() for g in enabled_genes}
    
    for read in sample.reads:
        for gene_name, allele_alns in read.alignments.items():
            if gene_name not in enabled_genes_set:
                continue
            
            if read.id in processed[gene_name]:
                continue
            
            best_hit, best_allele = None, None
            for aname, hits in allele_alns.items():
                allele = db.genes[gene_name].alleles.get(aname)
                if allele is None or not allele.enabled:
                    continue
                for h in hits:
                    if h.enabled and (not best_hit or h.cost < best_hit.cost):
                        best_hit, best_allele = h, allele
            
            if not best_hit or not best_allele:
                continue
            
            processed[gene_name].add(read.id)
            gene = db.genes[gene_name]
            apos, rpos = best_hit.st, best_hit.read_st
            gpos_min, gpos_max = float('inf'), float('-inf')
            
            for size, op in get_cigar(best_hit.cigar):
                if op in "M=X":
                    for i in range(size):
                        gpos = best_allele.translate_allele_position_to_gene(apos + i)
                        gene.coverage[gpos] += 1
                        gpos_min, gpos_max = min(gpos_min, gpos), max(gpos_max, gpos)
                    apos += size
                    rpos += size
                elif op == "D":
                    apos += size
                elif op in "IS":
                    rpos += size
            
            for region in gene.wildtype.regions.values():
                if region.start <= gpos_max and gpos_min < region.end:
                    region.reads.add(read.id)
                    clipped_start = max(region.start, gpos_min)
                    clipped_end = min(region.end, gpos_max)
                    region.read_spans[read.id] = (clipped_start, clipped_end)

def populate_region_coverage(sample, db):
    enabled_genes = [g for g in db.genes if any(a.enabled for a in db.genes[g].alleles.values())]
    for gene_name in enabled_genes:
        gene = db.genes[gene_name]
        pos = 0
        for region in gene.wildtype.regions.values():
            rlen = len(region.seq)
            cov = sum(gene.coverage.get(p, 0) for p in range(pos, pos + rlen))
            region.coverage = (cov / rlen / sample.expected_coverage) if rlen > 0 else 0
            pos += rlen

    
#%%
def analyze_region_cross_mapping(sample, db):
    enabled_genes = [g for g in db.genes if any(a.enabled for a in db.genes[g].alleles.values())]

    read_regions = defaultdict(list)
    enabled_genes_set = set(enabled_genes)
    
    for gene_name in enabled_genes:
        gene = db.genes[gene_name]
        for region in gene.wildtype.regions.values():
            region.cross_map = defaultdict(list)  # {src_gene: [(start, end), ...]} in region coords
            region.cross_map_reads = defaultdict(set)
            for rid in region.reads:
                read_regions[rid].append((gene_name, region))
    
    print('[cn config]')
    for read in tqdm(sample.reads):
        for tgt_gene, region in read_regions.get(read.id, []):
            for src_gene, src_alns in read.alignments.items():
                if src_gene == tgt_gene:
                    continue
                
                if src_gene not in enabled_genes_set:
                    continue
                
                if read.id in region.cross_map_reads[src_gene]:
                    continue
                
                # Check if any enabled allele of src_gene has an enabled hit
                has_hit = False
                for aname, hits in src_alns.items():
                    allele = db.genes[src_gene].alleles.get(aname)
                    if allele is None or not allele.enabled:
                        continue
                    for h in hits:
                        if h.enabled:
                            has_hit = True
                            break
                    if has_hit:
                        break
                
                if has_hit:
                    region.cross_map_reads[src_gene].add(read.id)
                    if read.id in region.read_spans:
                        region.cross_map[src_gene].append(region.read_spans[read.id])
    
    def merge_len(intervals):
        if not intervals:
            return 0
        intervals.sort()
        merged = [intervals[0]]
        for st, en in intervals[1:]:
            if st <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], en))
            else:
                merged.append((st, en))
        return sum(en - st for st, en in merged)
    
    for gene_name in tqdm(enabled_genes):
        gene = db.genes[gene_name]
        for region in gene.wildtype.regions.values():
            rlen = len(region.seq)
            region.cross_map_rate = {
                src_gene: merge_len(spans) / rlen 
                for src_gene, spans in region.cross_map.items()
            } if rlen else {}
#%%
def populate_gene_cn_config(db):
    enabled_genes = [g.name for g in db.genes.values() 
            if any(a.enabled for a in g.alleles.values())]
    for gene_name in enabled_genes:
        gene = db.genes[gene_name]
        gene.cn_config = {}
        for other_gene_name in enabled_genes:
            other_gene = db.genes[other_gene_name]
            for rname, region in other_gene.wildtype.regions.items():
                if gene_name == other_gene_name:
                    gene.cn_config[(other_gene_name, rname)] = 1.0
                else:
                    gene.cn_config[(other_gene_name, rname)] = region.cross_map_rate.get(gene_name, 0.0) / 2


def get_gene_cn_tensors(db):
    enabled_genes = [g.name for g in db.genes.values() 
            if any(a.enabled for a in g.alleles.values())]
    region_keys = [(g_name, rname) for g_name in enabled_genes
                   for rname in db.genes[g_name].wildtype.regions.keys()]
    
    # region_mask: [num_genes, num_regions]
    region_mask = torch.zeros(len(enabled_genes), len(region_keys), dtype=torch.float32)
    for i, gene_name in enumerate(enabled_genes):
        gene = db.genes[gene_name]
        for j, key in enumerate(region_keys):
            region_mask[i, j] = gene.cn_config.get(key, 0.0)
    
    # region_cov: [num_regions]
    region_cov = torch.tensor([db.genes[g].wildtype.regions[r].coverage 
                               for g, r in region_keys], dtype=torch.float32)
    
    return region_mask, region_cov, enabled_genes, region_keys

#%%
def discretize_cn(gene, cn):
    gene_thresholds = {
        'KIR2DL1': [0.3, 1.3, 2.5, 3.5],
        'KIR2DL2': [0.3, 1.5, 2.5, 3.5],
        'KIR2DL3': [0.3, 1.3, 2.5, 3.5],
        'KIR2DL4': [0.3, 1.5, 2.8, 3.5],
        'KIR2DL5': [0.3, 1.5, 2.5, 3.5],
        'KIR2DP1': [0.3, 1.5, 2.5, 3.5],
        'KIR2DS1': [0.3, 1.5, 2.5, 3.5],
        'KIR2DS2': [0.3, 1.5, 2.5, 3.5],
        'KIR2DS3': [0.3, 1.5, 2.5, 3.5],
        'KIR2DS4': [0.3, 1.5, 2.5, 3.5],
        'KIR2DS5': [0.3, 1.5, 2.5, 3.5],
        'KIR3DL1': [0.3, 1.5, 2.5, 3.5],
        'KIR3DL2': [0.3, 1.3, 2.5, 3.5],
        'KIR3DL3': [0.3, 1.5, 2.5, 3.5],
        'KIR3DP1': [0.3, 1.2, 2.3, 3.5],
        'KIR3DS1': [0.3, 1.7, 3.2, 3.5],
    }
    
    thresholds = gene_thresholds.get(gene, [0.3, 1.3, 2.5, 3.5])
    
    if cn < thresholds[0]:
        return 0
    elif cn < thresholds[1]:
        return 1
    elif cn < thresholds[2]:
        return 2
    elif cn < thresholds[3]:
        return 3
    else:
        return 3

#%% CN Estimation
if __name__ == "__main__":
    import importlib
    import sys
    modules_to_reload = ['cn_estimator']
    # modules_to_reload = ['V9_2_3_1106', 'V9_helper_10172039', 'common_0620B']
    
    for module_name in modules_to_reload:
        if module_name in sys.modules:
            importlib.reload(sys.modules[module_name])
    from cn_estimator import run_cn_estimator
    
    populate_gene_coverage(sample, db)
    populate_region_coverage(sample, db)
    
    analyze_region_cross_mapping(sample, db)
    populate_gene_cn_config(db)
    
    region_mask, region_cov, gene_names, region_keys = get_gene_cn_tensors(db)
    gene_cn = run_cn_estimator(
        region_cov=region_cov,      
        region_mask=region_mask,
        gene_names=gene_names, 
        max_cn=5.0,
        num_iterations=1000,
        delta = 0.3
    )

    #%%
    print("\n[Gene CN Results]")
    for gene, cn in sorted(gene_cn.items(), key=lambda x: x[0]):
        print(f"  {gene}: {cn:.2f}")
        db.genes[gene].estimated_cn = discretize_cn(gene, cn)

#%% with functional penalty
# run ae
if __name__ == "__main__":
    # from V9_2 import *
    # from V9_2_3_10152034 import *
    # # from V9_2_3_1007_B import *
    # # from V9_2_3_1007 import *
    # from V9_helper_1007 import *
    
    from vae import *
    from vae_helper import *
    from common_0620B import *
    
    #%%
    total_mut_counts = sum(variant.coverage.count 
                     for gene in db.genes.values() 
                     for pos in gene.positions.values() 
                     if pos.is_valid 
                     for variant in pos.variants.values())
    mut_counts_tensor = torch.tensor([db.variants()[i].coverage.count for i in sample.valid_indices], dtype=torch.float32)
    valid_allele_names = [f'{a.gene.name}*{a.name}' for a in sample.valid_alleles]

    bam_id = path.split('/')[-1]
    # densities = run_vae(sample.total_mutations, valid_allele_names, mut_counts_tensor,sample.beta,num_iterations=6000)
    densities, densities_un, learnt_beta = run_vae(
        total_mut_counts,   
        valid_allele_names,   
        mut_counts_tensor,       
        sample = sample,
        db = db,       
        num_iterations=3000,
        bam_id = bam_id,
        seeds = [args.seed]
    )
    
    #%%
    # ma = prepare_results_with_dummy(densities, valid_allele_names, 0.25)
    major_allele_densities = {}
    major_allele_densities_unnorm = {}
    print('[Raw]:')
    # Group and sum densities by major allele type
    for i, density in enumerate(densities):
        allele_name = valid_allele_names[i]
        gene, allele_id = allele_name.split('*')
        if gene == 'KIR2DL5':
            ab, allele_id = allele_id.split('.')
            if ab == 'A':
                gene = 'KIR2DL5A'
            elif ab == 'B':
                gene = 'KIR2DL5B'
            else:
                assert False
            major_id = allele_id[:3]
        else:
            major_id = allele_id[:3] 
        # Create a key for this major allele
        major_key = f"{gene}*{major_id}"
        # Add this density to the major allele's total
        if major_key not in major_allele_densities:
            major_allele_densities[major_key] = 0
            major_allele_densities_unnorm[major_key] = 0
        major_allele_densities[major_key] += density
        major_allele_densities_unnorm[major_key] += densities_un[i]
    
    # Report major alleles above threshold
    threshold = 0.019

    # Filter by threshold and group by gene
    filtered_by_gene = {}
    for major_key in major_allele_densities.keys():
        norm_val = major_allele_densities[major_key]
        if norm_val > threshold:
            gene_name = major_key.split('*')[0]
            if gene_name not in filtered_by_gene:
                filtered_by_gene[gene_name] = []
            filtered_by_gene[gene_name].append((major_key, norm_val, major_allele_densities_unnorm[major_key]))
    
    # Apply gene-specific selection
    final_results = []
    for gene_name, alleles in filtered_by_gene.items():
        alleles.sort(key=lambda x: x[1], reverse=True)  # Sort by norm density
        if gene_name == 'KIR2DL5A':
            final_results.extend(alleles[:1])  # Top 1
        elif gene_name == 'KIR3DL3':
            final_results.extend(alleles[:2])  # Top 2
        else:
            final_results.extend(alleles)  # All
    
    # Print final results
    final_results.sort(key=lambda x: x[0])  # Sort by allele name
    for major_key, norm_val, unnorm_val in final_results:
        print(f"{major_key}: norm={norm_val:.6f}, unnorm={unnorm_val:.6f}")

    #### CN processing
    #%%
    # Group by gene
    gene_alleles = {}
    for major_key, norm_val, unnorm_val in final_results:
        gene_name = major_key.split('*')[0]
        if gene_name not in gene_alleles:
            gene_alleles[gene_name] = []
        gene_alleles[gene_name].append((major_key, norm_val, unnorm_val))
    
    # Special handling for KIR2DL5: keep only top 1 from each A and B
    if 'KIR2DL5A' in gene_alleles or 'KIR2DL5B' in gene_alleles:
        combined = []
        
        if 'KIR2DL5A' in gene_alleles:
            a_alleles = gene_alleles.pop('KIR2DL5A')
            a_alleles.sort(key=lambda x: x[1], reverse=True)
            combined.append(a_alleles[0])  # Top 1 from A
        
        if 'KIR2DL5B' in gene_alleles:
            b_alleles = gene_alleles.pop('KIR2DL5B')
            b_alleles.sort(key=lambda x: x[1], reverse=True)
            combined.append(b_alleles[0])  # Top 1 from B
        
        gene_alleles['KIR2DL5'] = combined

    # print("\n[Alleles]")
    final_calls = []

    for gene_name, alleles in gene_alleles.items():
        if gene_name == 'KIR2DL5':
            gene_cn = db.genes['KIR2DL5'].estimated_cn
        else:
            gene_cn = db.genes[gene_name].estimated_cn
        
        alleles.sort(key=lambda x: x[1], reverse=True)
        num_alleles = len(alleles)
        
        allele_cn = {major_key: 1 for major_key, _, _ in alleles}
        remaining = gene_cn - num_alleles
        if remaining > 0:
            most_abundant = alleles[0][0]
            allele_cn[most_abundant] += remaining
        
        for major_key, norm_val, _ in alleles:
            cn = allele_cn[major_key]
            final_calls.append((major_key, cn))

    print("\n[Alleles]")
    for major_key, cn in sorted(final_calls, key=lambda x: x[0]):
        for _ in range(cn):
            print(f"{major_key}")
# %%
