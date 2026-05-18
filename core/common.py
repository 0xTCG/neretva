import os
import sys
import time
import logging
import collections
from datetime import datetime
from dataclasses import dataclass
import pickle

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
        self.allele_vector = []

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
        self.positions = {} # mutation positions, including novel and database positions with known mutations 
        # The start position in the final chr19kir chromosome (generated later)
        self.ref_start = 0

    def add(self, allele, record):
        self.alleles[allele] = Allele(self, allele, record)

    @property
    def wildtype(self):
        return self[self._wildtype]

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
        logging.info(self.report(self.msg))

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
        
        self.count = 0
        self.covered_reads = set()  
        self.covered_hits = []     
        
        # self.quality_scores = []   
        # self.strand_bias = {'forward': 0, 'reverse': 0}
        
        self.infection_sources = {}  
        
    def add_read_coverage(self, read_id, hit=None, quality=None, is_reverse=False):
        """Add coverage from a read"""
        if read_id not in self.covered_reads:
            self.covered_reads.add(read_id)
            self.count += 1
            
        if hit:
            self.covered_hits.append(hit)
            
        if quality is not None:
            self.quality_scores.append(quality)
            
        strand = 'reverse' if is_reverse else 'forward'
        self.strand_bias[strand] += 1

class Variant:
    """Represents a specific variant (A, C, T, G, N, +seq) at a position"""
    # A position has multiple variants.
    def __init__(self, position, index, mutation, variant):
        self.position = position
        self.variant = variant
        self.gene = position.gene
        self.pos = position.position
        # Original mutation from allele definition (pos, op)
        self.mutation = mutation  # e.g., (123, "A>G") or (123, "delG")
        self.op = mutation[1] # "A>G", "delG", "insATG"
        self.mutation_tuple = (self.gene, self.pos, self.variant)
        # Index in the mutation vector/beta matrix
        self.index = index
        
        self.is_wildtype = (self.variant == position.ref_base)
        # self.is_deletion = (variant_str == 'N')
        self.is_deletion = self.variant == 'N'
        self.is_insertion = self.op.startswith('+')
        self.is_snp = variant in 'ACTG'
        
        self.is_major = None  
        self.is_minor = None
        
        self.coverage = Coverage(self)
        
        self.source_alleles = set() #original alleles that define this variant
        self.infected_alleles = set() #alleles from other genes that have this variant
    
    def __str__(self):
        return f"{self.gene}:{self.mutation})"


    def geny2one(pos, op):
        if op.startswith('ins'): return {(pos, "+" + op[3:])}
        # elif op.startswith('del'): return {(pos + i, "-") for i in range(len(op) - 3)}
        # approximate it with only the first base
        elif op.startswith('del'): return {(pos, "-" + op[3])}

        elif len(op) == 3: return {(pos, op[2])}
        assert(f'bad {op}')
        return {}

class Position:
    def __init__(self, gene, position, ref_base=None):
        self.gene = gene
        self.position = position
        self.ref_base = ref_base
        self.is_valid = True # whether we use this position to infer allele proportion
        
        # All variants at this position
        self.variants = {}  # variant_str -> Variant object
        
    def query_variant(self, variant_str, index=None, original_mutation=None):
        if variant_str not in self.variants:
            self.variants[variant_str] = Variant(self, variant_str, index, original_mutation)
        return self.variants[variant_str]
    
class Database:
    def __init__(self, path):
        with open(path, "rb") as fo:
            (self.genes, _, _, self.minor_to_major) = pickle.load(fo)
        # self.positions = []
        self.positions = []

    def alleles(self):
        for g in self.genes.values():
            yield from g.alleles.values()
    
    def populate_positions(self):
        # get all the positions and their indices 
        pass

    # def mutations(self):
    #     # to replace get_indices
    #     '''
    #     return all mutations itertating from all genes
    #     '''
    #     pass
