#%%
import pickle
import sys
from Bio import SeqRecord, Seq
from Bio.SeqFeature import SeqFeature, FeatureLocation

import aldy.gene
import aldy.common

from common_cyp import Gene, Allele


def create_cyp_pickle(aldy_yaml_path, output_pickle_path, gene_name=None, genome='hg19'):
    print(f"Loading Aldy gene from: {aldy_yaml_path}")
    
    aldy_gene = aldy.gene.Gene(aldy_yaml_path, genome=genome)
    
    print(f"  Gene: {aldy_gene.name}")
    print(f"  Major alleles: {len(aldy_gene.alleles)}")
    
    wildtype_name = _find_wildtype(aldy_gene)
    gene = Gene(gene=aldy_gene.name, wildtype=wildtype_name)
    gene.aldy_gene = aldy_gene
    gene.multi_snps = {}  # Track multi-SNP positions and their ops
    
    allele_count = 0
    
    for major_name, major_allele in aldy_gene.alleles.items():
        for minor_name, minor_allele in major_allele.minors.items():
            _create_allele(aldy_gene, major_allele, minor_allele, gene)
            allele_count += 1
        
        if len(major_allele.minors) == 0:
            _create_allele(aldy_gene, major_allele, None, gene)
            allele_count += 1
    
    print(f"  Created {allele_count} alleles")
  
    for allele_name, allele in gene.alleles.items():
        for pos, op in allele.ops:
            gene.mutations.setdefault((pos, op), set()).add(allele_name)
            # Track multi-SNPs
            if '.' in op and '>' in op:
                gene.multi_snps.setdefault(pos, set()).add(op)
    
    for (pos, op), mut_info in aldy_gene.mutations.items():
        annotation = mut_info[0] if mut_info else None
        if annotation:
            gene.functional[(pos, op)] = annotation
            # Track multi-SNPs in functional
            if '.' in op and '>' in op:
                gene.multi_snps.setdefault(pos, set()).add(op)
    
    # Store multi_snp_ops on positions for reference
    for pos, ops in gene.multi_snps.items():
        if pos in gene.positions:
            gene.positions[pos].multi_snp_ops = ops
    
    print(f"  Multi-SNP positions: {len(gene.multi_snps)}")
    
    minor_to_major = {}
    for major_name in gene.aldy_gene.alleles.keys():
        for minor_name in gene.aldy_gene.alleles[major_name].minors.keys():
            minor_to_major[minor_name] = major_name
    
    genes = {gene_name: gene}
    db_data = (genes, None, None, minor_to_major)
    
    print(f"Saving to: {output_pickle_path}")
    with open(output_pickle_path, 'wb') as f:
        pickle.dump(db_data, f)
    

def _find_wildtype(aldy_gene):
    for name, allele in aldy_gene.alleles.items():
        if len(allele.func_muts) == 0:
            return name
    return '1' if '1' in aldy_gene.alleles else list(aldy_gene.alleles.keys())[0]


def _create_allele(aldy_gene, major_allele, minor_allele, gene):
    if minor_allele is None:
        allele_name = major_allele.name
        all_muts = set(major_allele.func_muts)
    else:
        allele_name = minor_allele.name
        all_muts = set(major_allele.func_muts) | set(minor_allele.neutral_muts)
    
    allele_seq = _apply_mutations(aldy_gene, all_muts)
    record = SeqRecord.SeqRecord(
        Seq.Seq(allele_seq),
        id=allele_name,
        description=f"{aldy_gene.name}*{allele_name}"
    )
    
    for i, (start, end) in enumerate(aldy_gene.exons, 1):
        feature = SeqFeature(
            FeatureLocation(start - 1, end),
            type='exon',
            qualifiers={'number': [str(i)]}
        )
        record.features.append(feature)
    
    if aldy_gene.aminoacid and aldy_gene.exons:
        cds_feature = SeqFeature(
            FeatureLocation(aldy_gene.exons[0][0] - 1, aldy_gene.exons[-1][1]),
            type='CDS',
            qualifiers={'translation': [aldy_gene.aminoacid]}
        )
        record.features.append(cds_feature)
    
    allele = Allele(gene, allele_name, record)
    
    # Keep original ops (including multi-SNPs)
    allele.ops = [(mut.pos, mut.op) for mut in all_muts]
    
    for mut in major_allele.func_muts:
        allele.func.add((mut.pos, mut.op))
        allele.mutations.add((mut.pos, mut.op))
    
    if minor_allele is not None:
        for mut in minor_allele.neutral_muts:
            allele.minor.add((mut.pos, mut.op))
            allele.mutations.add((mut.pos, mut.op))
    
    gene.alleles[allele_name] = allele


def _apply_mutations(aldy_gene, mutations):
    st, ed = aldy_gene._lookup_range
    seq = list(aldy_gene[st:ed])
    
    for mut in mutations:
        pos, op = mut.pos, mut.op
        if '>' in op and '.' not in op:  # Single SNP only
            seq[pos - st] = op[2]
        elif '.' in op and '>' in op:  # Multi-SNP
            l, r = op.split('>')
            for i, (ref, alt) in enumerate(zip(l.split('.'), r.split('.'))):
                if ref != '.':
                    seq[pos - st + i] = alt
        elif op.startswith('ins'): 
            seq[pos - st] = op[3:] + seq[pos - st]
        elif op.startswith('del'):  
            for i in range(len(op) - 3):
                seq[pos - st + i] = ''
        
    return ''.join(seq)


#%%
if __name__ == '__main__':
    gene_name = 'CYP2D6'

    yaml_path = aldy.common.script_path(f"aldy.resources.genes/{gene_name.lower()}.yml")
    pickle_path = f'{gene_name.lower()}.pkl'
    
    create_cyp_pickle(yaml_path, pickle_path, gene_name)

# %%