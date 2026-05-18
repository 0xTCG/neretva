import os
import sys
import time
import logging
import collections
from datetime import datetime
from dataclasses import dataclass
import pickle
import numpy as np
from collections import defaultdict

@dataclass
class Region:
    gene: str
    name: str
    seq: str
    partial: bool = False
    pseudo: bool = False

    @property
    def is_exon(self):
        return self.name.startswith("e")

    @property
    def is_intron(self):
        return self.name.startswith("i")


@dataclass
class Allele:
    gene: object
    name: str
    regions: dict
    ops: list  # list of differences from the wildtype allele
    protein: str

    enabled: bool
    func: set
    minor: set
    mutations: set
    keystones: set

    def __init__(self, gene, name, record):
        """Initialize the allele from an IMGT (kir.dat) record"""

        self.gene = gene
        self.name = name
        self.protein = None
        self.description = record.description
        self.regions = {}
        for f in record.features:
            if f.type == "CDS":
                if (protein := f.qualifiers.get("translation", None)):
                    self.protein = protein[0] + "X"
            elif f.type != "source":
                name = f.type.lower()
                if name in ["exon", "intron"]:
                    name = name[0]
                name += str(
                    f.qualifiers.get(
                        "number", ["_3" if f.type.lower() in self.regions else ""]
                    )[0]
                )
                assert name not in self.regions, (name, self.regions.keys())
                self.regions[name] = Region(
                    gene,
                    name,
                    str(record.seq[int(f.location.start) : int(f.location.end)]),
                    "partial" in f.qualifiers,
                    "pseudo" in f.qualifiers,
                )

        self.enabled = True
        self.func = set()
        self.minor = set()
        self.mutations = set()
        self.keystones = set()
        self.original_allele_vector = [] # allele vector defined by database
        self.extended_allele_vector = [] # allele vector that carries mutation / variant on other genes, after infection
  
        # Track variants from multi-mapping using Variant objects
        self.original_variants = set()  # Variant objects from this allele's definition
        self.infected_variants = set()  # Variant objects from multi-mapping reads
        self.extended_variants = set()  # Union of original + infected variants
        
        # Track infection sources for debugging
        self.infection_sources = {}  # {Variant: {infected_position: set(read_ids)}}
        
        # Track which reads contributed to infections
        self.infection_reads = set()  # read IDs that caused infections to this allele
    
        self.expected_position_strength = {}  # {position: expected_strength/coverage}

    def __hash__(self):
        return hash((self.gene.name, self.name))

    def __eq__(self, other):
        if isinstance(other, Allele):
            return self.gene.name == other.gene.name and self.name == other.name
        return False

    def set_position_strength(self, position, variant, lb, ub):
        self.expected_position_strength[(position, variant)] = (lb, ub)
        
    def get_position_strength(self, position, variant):
        return self.expected_position_strength.get((position,variant),(0,0))
    
    def get_all_position_strengths(self):
        return self.expected_position_strength.copy()


    def add_infected_variant(self, variant_obj, infected_position, read_id, source_allele=None):
        """
        Add a variant that was infected from multi-mapping reads.
        Automatically updates the extended allele vector.
        """
        # Add to infected variants
        self.infected_variants.add(variant_obj)
        self.extended_variants.add(variant_obj)
        
        # Track infection source
        if variant_obj not in self.infection_sources:
            self.infection_sources[variant_obj] = {}
        if infected_position not in self.infection_sources[variant_obj]:
            self.infection_sources[variant_obj][infected_position] = set()
        self.infection_sources[variant_obj][infected_position].add(read_id)
        
        self.infection_reads.add(read_id)
        
        variant_obj.infected_alleles.add(self)
        
        self.extended_allele_vector[variant_obj.index] = 1
        
        # If this variant has a wildtype counterpart at the same position, set it to 0
        position_key = (variant_obj.gene, variant_obj.pos)
        if position_key in variant_obj.gene.positions:
            wildtype_variant = variant_obj.gene.positions[position_key].variants.get("_")
            if wildtype_variant:
                self.extended_allele_vector[wildtype_variant.index] = 0

    def clear_infections(self):
        """Clear all infection data."""
        # Remove this allele from infected variants
        for variant_obj in self.infected_variants:
            variant_obj.infected_alleles.discard(self)
            
        self.infected_variants.clear()
        self.extended_variants = self.original_variants.copy()
        self.infection_sources.clear()
        self.infection_reads.clear()
        self.extended_allele_vector = self.original_allele_vector.copy()


    def parse_mutations(self):
        """Find all core mutations that modify the protein."""
        from Bio.Seq import Seq

        for pos, op in self.ops:
            self.gene.mutations.setdefault((pos, op), set()).add(self.name)
            if (pos, op) in self.gene.functional:
                continue
            r, rs = self.gene.wildtype.region(pos)
            if r and r.is_exon and not r.pseudo:  # ignore pseudo-exons!
                if op.startswith("ins") or op.startswith("del"):
                    # Exonic indels are always functionsl
                    self.gene.functional[pos, op] = op[:3]
                else:
                    # For SNPs, check if the protein is modified
                    seq = list(r.seq)
                    seq[rs] = op[2]  # apply the change
                    prot = "".join(  # translate to protein
                        i.seq if i.name != r.name else "".join(seq)
                        for i in self.gene.wildtype.regions.values()
                        if i.is_exon and not i.pseudo
                    )
                    prot = str(Seq(prot).translate())
                    if prot != self.gene.wildtype.protein:
                        pfx = os.path.commonprefix([prot, self.gene.wildtype.protein])
                        p = f"{self.gene.wildtype.protein[len(pfx) - 1]}{len(pfx)}{prot[len(pfx)]}"
                        self.gene.functional[pos, op] = p

    def region(self, i):
        """Return the region name and offset of gene location."""

        st = 0
        for r in self.regions.values():
            if st <= i < st + len(r.seq):
                return r, i - st
            st += len(r.seq)
        # assert False, i
        return None, None
    

    def translate_allele_position_to_gene(self, allele_pos):
        """
        Translate position on allele sequence to position on gene/wildtype sequence
        using mutmap to find coordinate mapping, accounting for indels
        """
        closest_allele_pos = None
        closest_distance = float('inf')
        
        # First try to find closest allele position <= allele_pos
        for mutmap_allele_pos, (mmi, mutation) in self.mutmap.items():
            if mutmap_allele_pos <= allele_pos:
                distance = allele_pos - mutmap_allele_pos
                if distance < closest_distance:
                    closest_distance = distance
                    closest_allele_pos = mutmap_allele_pos
                    closest_gene_pos = mutation[0]
                    closest_op = mutation[1]
                    closest_mmi = mmi
        
        if closest_allele_pos is None:
            # No position <= allele_pos found, find closest position > allele_pos
            for mutmap_allele_pos, (mmi, mutation) in self.mutmap.items():
                if mutmap_allele_pos > allele_pos:
                    distance = mutmap_allele_pos - allele_pos
                    if distance < closest_distance:
                        closest_distance = distance
                        closest_allele_pos = mutmap_allele_pos
                        closest_gene_pos = mutation[0]
                        closest_op = mutation[1]
                        closest_mmi = mmi
            
            if closest_allele_pos is None:
                assert False, f"No mutmap positions found for allele {self.name}"
            
            # Calculate using the position ahead
            gene_pos = closest_gene_pos - (closest_allele_pos - allele_pos)
        else:
            # Calculate using the position behind or at
            offset = allele_pos - closest_allele_pos
            
            # Only adjust for indels if mmi == 0 (mutation is present in this allele)
            if closest_mmi == 0 and closest_op.startswith('ins'):
                ins_length = len(closest_op) - 3
                gene_pos = closest_gene_pos + offset - ins_length
            elif closest_mmi == 0 and closest_op.startswith('del'):
                del_length = len(closest_op) - 3
                gene_pos = closest_gene_pos + offset + del_length
            else:
                # SNP, no mutation, or mutation not in this allele
                gene_pos = closest_gene_pos + offset
        
        return gene_pos

    def translate_gene_position_to_allele(self, gene_pos):
        """
        Translate position on gene/wildtype sequence to position on allele sequence
        using mutmap to find coordinate mapping, accounting for indels
        """
        closest_allele_pos = None
        closest_distance = float('inf')
        
        # First try to find closest gene position <= gene_pos
        for allele_pos, (mmi, mutation) in self.mutmap.items():
            mutmap_gene_pos = mutation[0]
            
            if mutmap_gene_pos <= gene_pos:
                distance = gene_pos - mutmap_gene_pos
                if distance < closest_distance:
                    closest_distance = distance
                    closest_allele_pos = allele_pos
                    closest_gene_pos = mutmap_gene_pos
                    closest_op = mutation[1]
                    closest_mmi = mmi
        
        if closest_allele_pos is None:
            # No position <= gene_pos found, find closest position > gene_pos
            for allele_pos, (mmi, mutation) in self.mutmap.items():
                mutmap_gene_pos = mutation[0]
                
                if mutmap_gene_pos > gene_pos:
                    distance = mutmap_gene_pos - gene_pos
                    if distance < closest_distance:
                        closest_distance = distance
                        closest_allele_pos = allele_pos
                        closest_gene_pos = mutmap_gene_pos
                        closest_op = mutation[1]
                        closest_mmi = mmi
            
            if closest_allele_pos is None:
                assert False, f"No mutmap positions found for allele {self.name}"
            
            # Calculate using the position ahead
            allele_pos = closest_allele_pos - (closest_gene_pos - gene_pos)
        else:
            # Calculate using the position behind or at
            offset = gene_pos - closest_gene_pos
            
            # Only adjust for indels if mmi == 0 (mutation is present in this allele)
            if closest_mmi == 0 and closest_op.startswith('ins'):
                ins_length = len(closest_op) - 3
                allele_pos = closest_allele_pos + offset + ins_length
            elif closest_mmi == 0 and closest_op.startswith('del'):
                del_length = len(closest_op) - 3
                allele_pos = closest_allele_pos + offset - del_length
            else:
                # SNP, no mutation, or mutation not in this allele
                allele_pos = closest_allele_pos + offset
        
        return allele_pos
        
    def add_extra_positions_to_gene(self):
        """
        Add novel positions from self.extra_positions to gene.positions
        """
        if not hasattr(self, 'extra_positions'):
            assert False
        
        for allele_pos in self.extra_positions:
            gene_pos = self.translate_allele_position_to_gene(allele_pos)
            
            if gene_pos not in self.gene.positions:
                self.gene.positions[gene_pos] = Position(self.gene, gene_pos)
                print(f"Added novel position {gene_pos} to {self.gene.name} (from allele pos {allele_pos})")

    @property
    def seq(self):
        return "".join(s.seq for s in self.regions.values())

    @property
    def exons(self):
        return [r for r in self.regions.values() if r.is_exon]

    @property
    def introns(self):
        return [r for r in self.regions.values() if r.is_intron]

    @property
    def utrs(self):
        return [r for r in self.regions.values() if r.name.startswith("utr")]


class Gene:
    def __init__(self, gene, wildtype):
        self.name = gene
        self._wildtype = wildtype  # Wildtype allele
        self.alleles = {}
        self.functional = {}  # Functional mutations
        self.mutations = {}
        self.positions = {}
        # The start position in the final chr19kir chromosome (generated later)
        self.ref_start = 0
    def add(self, allele, record):
        self.alleles[allele] = Allele(self, allele, record)

    @property
    def wildtype(self):
        return self[self._wildtype]
    
    @property
    def variants(self):
        """Get all variants for this gene, sorted by position then variant"""
        all_variants = []
        for pos in sorted(self.positions.keys()):
            position = self.positions[pos]
            # Sort variants: wildtype first, then alphabetically
            for variant_key in sorted(position.variants.keys(), key=lambda x: (x != "_", x)):
                all_variants.append(position.variants[variant_key])
        return all_variants

    def __getitem__(self, i):
        return self.alleles[i]

    @property
    def seq(self):
        return self.wildtype.seq

    def yaml(self):
        """Generate Aldy database YAML"""

        maps, exons = {}, []
        start = 0
        en = 1
        for r in self.wildtype.regions.values():
            name = r.name  # handle skip exons
            if name.startswith('e'):
                name = f"e{en}"
            if name.startswith('i'):
                name = f"i{en}"; en += 1
            maps[name] = [
                self.ref_start + start + 1,
                self.ref_start + start + len(r.seq) + 1,
            ]
            if r.is_exon and not r.pseudo:
                exons.append([start + 1, start + len(r.seq) + 1])
            start += len(r.seq)
        return {
            "name": self.name,
            "version": "fresh",
            "generated": str(datetime.now()),
            "alleles": {
                f"{self.name}*{a.name}": {
                    "mutations": [
                        [pos + 1 - (1 if op.startswith("ins") else 0), op, "-"]
                        + ([f] if (f := self.functional.get((pos, op))) else [])
                        for (pos, op) in sorted(a.ops)
                    ]
                }
                for a in self.alleles.values()
            },
            "structure": {
                "genes": [self.name],
                "regions": {"hg38": maps},
                "cn_regions": list(maps.keys()),
            },
            "reference": {
                "name": self.name,
                "mappings": {
                    "hg38": [
                        "19kir",
                        self.ref_start + 1,
                        self.ref_start + (l := len(self.seq)) + 1,
                        "+",
                        f"M{l}",
                    ]
                },
                "exons": exons,
                "seq": self.seq,
            },
        }


class TimeInterval:
    def __init__(self, msg=""):
        self.start = time.time()
        self.msg = msg

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, *_):
        print(self.report(self.msg))

    def report(self, msg="") -> str:
        import psutil, resource
        process = psutil.Process()
        mi = process.memory_full_info()
        mp = mi.rss / (1024 ** 2)
        ru = resource.getrusage(resource.RUSAGE_SELF)
        mx = ru.ru_maxrss / 1024
        msg = 'Block' if not self.msg else self.msg
        msg = f"[time] {msg} took {self.elapsed():.2f}s ({mp:,} MB; {mx:,} MB)"
        return msg

    def elapsed(self) -> float:
        return time.time() - self.start

def timing(msg: str = "") -> TimeInterval:
    return TimeInterval(msg)

def timeit(fn):
    def f(*args, **kwargs):
        with timing(f"{fn.__name__}"):
            return fn(*args, **kwargs)
    return f



class dotdict(collections.defaultdict):
    """dot.notation access to dictionary attributes"""
    # https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, data=None):
        super().__init__(str)
        if data: self.update(data)


class Coverage:
    def __init__(self, variant):
        self.variant = variant  
        
        # self.count = 0
        self.covered_reads = set()  
        self.covered_hits = []     
        
        
        self.infection_sources = {}  
        
    @property
    def count(self):
        return len(self.covered_reads)
    
    def add_coverage(self,  hit=None):
        """Add coverage from a read"""
        # print(f"[info] ac {hit.rid}")
        # if read_id not in self.covered_reads:
        if hit.rid not in self.covered_reads:  # Check for duplicates
            self.covered_reads.add(hit.rid)
            self.covered_hits.append(hit)
        # print(self.count)

def convert_op_to_variant(op):
    if op.startswith('ins'): 
        # return "+" + op[3:]
        return "P"
    elif op.startswith('del'): 
        return "N"  
    elif len(op) == 3 and '>' in op:  
        return op[2] 
    else:
        assert False

class Variant:
    """Specific variants (A, C, T, G, N, P) at a position"""
    def __init__(self, position, variant):
        self.position = position
        self.gene = position.gene
        self.pos = position.position
        # self.op = None
        self.variant = variant
        # self.mutation_tuple = (self.gene, self.pos, self.variant)
        # Index in the mutation vector/beta matrix
        
        self.is_wildtype = (self.variant == position.ref_base)
        self.is_snp = self.variant in 'ACTG'
        self.is_N = self.variant == 'N'
        self.is_P = self.variant == 'P'
        
        self.is_major = None  
        self.is_minor = None
        self.is_novel = None
        
        # self.is_novel = None # not accountered base (major/minor) at this position/novel position
        self.coverage = Coverage(self) # accumulated coverage
        
        self.source_alleles = set() #original alleles that define this variant
        self.infected_alleles = set() #alleles from other genes that have this variant
        
        # self.mutations = None
        self.mutations = set() # a position, variant can have multiple indels, like KIR2DL1, 7049, insT, and 7049 insTT
        self.ops = set()
        self.has_deletion_op = None
        # if self.op:
        #     self.mutation = (self.pos, self.op)

        # self.has_deletion_op = self.op and self.op.startswith('del')

        # if self.has_deletion_op:
        #     assert self.is_N is True
        #     self.can_delete = [] # list of positions this variant can delete
        
    def __str__(self):
        return f"{self.gene.name}: {self.pos} {self.variant})"

class Position:
    def __init__(self, gene, position, ref_base=None):
        self.gene = gene
        self.position = position
        self.ref_base = ref_base # wildtype base/variant
        self.is_valid = False # whether we use this position to infer allele proportion
        
        # All variants at this position
        self.variants = {}  # variant_tuple -> Variant object
        
        self.coverage = Coverage(self)
        
        if not ref_base:
            # print(gene.name, position)
            self.ref_base = gene.wildtype.seq[position]
        
       

        # by default add the reference variant
        # ref_mutation = (position, "_")
        # self.variants[self.ref_base] = Variant(self, ref_mutation)

        @property
        def wildtype_variant(self):
            return self.variants[self.ref_base]
        
    def query_variant(self, variant):
        return self.variants[variant]
    
class Database:
    def __init__(self, path):
        with open(path, "rb") as fo:
            (self.genes, _, _, self.minor_to_major) = pickle.load(fo)
        # self.positions = []

        # we do not need a breakpoint, since we know if a variant is a major / not.
        
        self.populate_positions()
        # populate the variant.can_delete
        self.populate_nullable_positions()

        self.assign_index()
        self.populate_allele_vectors()
        self.max_index = 0

    def alleles(self):
        for g in self.genes.values():
            yield from g.alleles.values()
    
    def variants(self):
        """Render all variants ordered by index"""
        all_variants = []
        for gene in self.genes.values():
            for position in gene.positions.values():
                all_variants.extend(position.variants.values())
        
        return sorted([v for v in all_variants], key=lambda v: v.index)
    
    def populate_positions(self):
        # get all the positions and their indices 
        # replace get_indices    
        tot = 0
        for gene in self.genes.values():
            # all_pos = {pos for allele in gene.alleles.values() for pos, op in allele.ops}
            all_pos = {pos for pos, op in gene.mutations}
            tot += len(all_pos)
            # print(gene.name, sorted(all_pos))
            # TODO: include novel mutation positions
            # TODO: needs to be sorted
            
            positions = {pos: Position(gene, pos) for pos in all_pos}
            # num_vars = (sum (len(positions[position].variants.keys()) for position in positions))
            # print(f'[info] number of variants is {num_vars}')
            for pos in all_pos:
                position = positions[pos]
            
                for variant in ['A', 'C', 'G', 'T', 'N', 'P']:
                    position.variants[variant] = Variant(position, variant)

            # populate all the information for a position
            #TODO: handle deletions specially
            for mut in gene.mutations:
                pos, op = mut
                position = positions[pos]
                variant = convert_op_to_variant(op)
                position.variants[variant].ops.add(op)
                position.variants[variant].mutations.add(mut)

                if mut in gene.functional:
                    position.variants[variant].is_major = True
                    position.variants[variant].is_minor = False
                    position.variants[variant].is_wildtype = False

                else:
                    position.variants[variant].is_minor = True
                    position.variants[variant].is_major = False
                    position.variants[variant].is_wildtype = False


                if op.startswith('del'):
                    position.variants[variant].has_deletion_op = True
                    position.variants[variant].can_delete = {}
                    assert variant == 'N'
                    
            gene.positions = positions
    
    def _get_deletion_piece(self, deletion_variant, op):
        """Updated to work with specific deletion operation"""
        for allele in deletion_variant.gene.alleles.values():
            for piece in allele.pieces:
                wi, ai, seq, mut = piece
                if mut and mut == (deletion_variant.pos, op):
                    return piece
        return None
    
    def assign_index(self):
        current_index = 0
        
        for gene_name in sorted(self.genes.keys()):
            gene = self.genes[gene_name]
            
            for pos in sorted(gene.positions.keys()):
                position = gene.positions[pos]

                # Sort, wildtype first among the variants
                variant_keys = sorted(position.variants.keys())
                
                for variant_key in variant_keys:
                    variant = position.variants[variant_key]
                    variant.index = current_index
                    self.max_index = current_index
                    current_index += 1
        print(f'[info] total indices {current_index}')
    
    def populate_nullable_positions(self):
        tot = 0
        for gene_name in sorted(self.genes.keys()):
            gene = self.genes[gene_name]
            
            # Collect all deletion variants in this gene
            deletion_variants = []
            for position in gene.positions.values():
                n_variant = position.variants['N']
                if n_variant.ops and any(op.startswith('del') for op in n_variant.ops):
                    deletion_variants.append(n_variant)
            
            print(f"  {gene_name}: Processing {len(deletion_variants)} deletion variants")
            # tot += len(deletion_variants)
            
            for deletion_variant in deletion_variants:
                # Initialize can_delete as a dictionary: {deletion_op: [positions]}
                if not hasattr(deletion_variant, 'can_delete'):
                    deletion_variant.can_delete = {}
                tot += len(deletion_variant.ops)
                # Process each deletion operation in this N variant
                for op in deletion_variant.ops:
                    if not op.startswith('del'):
                        assert False
                    
                    deletion_piece = self._get_deletion_piece(deletion_variant, op)
                    if not deletion_piece:
                        assert False
                    
                    wi, ai, seq, mut = deletion_piece
                    if not (mut and mut[1].startswith('del')):
                        assert False
                    
                    deleted_sequence = mut[1][3:]
                    wildtype_start, wildtype_end = wi, wi + len(deleted_sequence)
                    nullable_positions = []
                    
                    for pos in sorted(gene.positions.keys()):
                        if wildtype_start <= pos < wildtype_end:
                            position_obj = gene.positions[pos]
                            
                            # Check if any other alleles have non-wildtype variants at this position
                            has_non_wildtype_variants = any(
                                variant.ops and not variant.is_wildtype
                                for variant in position_obj.variants.values()
                                if variant.variant not in ['N', 'P']  # Exclude N and P from this check
                            )
                            
                            if has_non_wildtype_variants:
                                nullable_positions.append(pos)
                    
                    # Store positions that can be deleted by this specific deletion operation
                    deletion_variant.can_delete[op] = nullable_positions
                    
                    if nullable_positions:
                        print(f"    Deletion {op} at {deletion_variant.pos}: can delete {len(nullable_positions)} positions")
            
        print(f'Total {tot} deletions processed')


    def populate_allele_vectors(self):
        total_variants = sum(len(pos.variants) for gene in self.genes.values() for pos in gene.positions.values())
        print(f'[info] total variants {total_variants}')
        
        for gene_name in sorted(self.genes.keys()):
            gene = self.genes[gene_name]
            for allele in gene.alleles.values():
                allele_vector = [0] * (self.max_index + 1)
                
                for position in gene.positions.values():
                    # Check if this allele has ANY insertion at this position
                    has_insertion = any((pos, op) in allele.ops 
                                    for pos, op in allele.ops 
                                    if pos == position.position and op.startswith('ins'))
                    
                    # Check if this allele has ANY deletion at this position  
                    has_deletion = any((pos, op) in allele.ops 
                                    for pos, op in allele.ops 
                                    if pos == position.position and op.startswith('del'))
                    
                    # Check if this allele has ANY SNP at this position
                    has_snp = any((pos, op) in allele.ops 
                                for pos, op in allele.ops 
                                if pos == position.position and '>' in op)
                    
                    if has_insertion:
                        # Mark P variant (regardless of specific insertion)
                        allele_vector[position.variants['P'].index] = 1
                    elif has_deletion:
                        # Mark N variant (regardless of specific deletion)
                        allele_vector[position.variants['N'].index] = 1
                    elif has_snp:
                        # Find the specific SNP and mark the target base
                        for pos, op in allele.ops:
                            if pos == position.position and '>' in op:
                                target_base = op[2]  # A>G -> G
                                allele_vector[position.variants[target_base].index] = 1
                                break
                    else:
                        # No mutations at this position - mark wildtype
                        allele_vector[position.variants[position.ref_base].index] = 1
                
                # Handle long deletions (N variants that can delete other positions)
                for position in gene.positions.values():
                    n_variant = position.variants['N']
                    if hasattr(n_variant, 'can_delete') and n_variant.can_delete:
                        # Check which specific deletions this allele has at this position
                        allele_deletions_at_position = [
                            op for pos, op in allele.ops 
                            if pos == position.position and op.startswith('del')
                        ]
                        
                        # For each deletion this allele has, mark the affected positions as N
                        for deletion_op in allele_deletions_at_position:
                            if deletion_op in n_variant.can_delete:
                                deletable_positions = n_variant.can_delete[deletion_op]
                                for deletable_pos in deletable_positions:
                                    if deletable_pos in gene.positions:
                                        pos_obj = gene.positions[deletable_pos]
                                        deletable_n_variant = pos_obj.variants['N']
                                        allele_vector[deletable_n_variant.index] = 1
                                        # Remove wildtype at deleted positions
                                        wildtype_variant = pos_obj.variants[pos_obj.ref_base]
                                        allele_vector[wildtype_variant.index] = 0
                
                allele.original_allele_vector = allele_vector
                allele.extended_allele_vector = allele_vector.copy()


