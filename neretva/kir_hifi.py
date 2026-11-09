import os
import sys
import argparse
import importlib.resources
from collections import Counter, defaultdict

import numpy as np
import pysam
import mappy
import torch

from core.helper import *
from common_0620B import *

STRUCTURAL_GENES = {'KIR3DP1'}
DEFAULT_REGION = 'chr19:52000000-56000000'


class Hit:
    _COMP = str.maketrans('ACGTacgt', 'TGCAtgca')

    def __init__(self, mappy_hit, gene_name, rid=None, seq=None):
        self.rid = rid
        self.seq = seq
        self.gene_name = gene_name
        self.r_st = mappy_hit.r_st
        self.r_en = mappy_hit.r_en
        self.q_st = mappy_hit.q_st
        self.q_en = mappy_hit.q_en
        self.mapq = mappy_hit.mapq
        self.NM = mappy_hit.NM
        self.cigar = mappy_hit.cigar
        self.strand = mappy_hit.strand
        self.is_primary = mappy_hit.is_primary
        self.enabled = True

        self.covered_positions = {}
        self.covered_variants = []
        self.variant_read_positions = {}

    @property
    def score(self):
        return (self.r_en - self.r_st) - self.NM

    def populate_covered_variants(self, gene):
        rp = self.r_st

        if self.strand == 1:
            walk_seq = self.seq
            qp = self.q_st
        else:
            walk_seq = self.seq[self.q_st:self.q_en].translate(self._COMP)[::-1]
            qp = 0

        for size, op in self.cigar:
            if op == 0:
                for i in range(size):
                    wt_pos = rp + i
                    if wt_pos in gene.positions:
                        obs_base = walk_seq[qp + i].upper()
                        pos_obj = gene.positions[wt_pos]
                        if obs_base not in pos_obj.variants:
                            continue
                        self.covered_positions[pos_obj] = obs_base
                        self.covered_variants.append(pos_obj.variants[obs_base])
                        self.variant_read_positions[pos_obj.variants[obs_base]] = qp + i
                rp += size
                qp += size

            elif op == 1:
                wt_pos = rp - 1
                if wt_pos in gene.positions:
                    pos_obj = gene.positions[wt_pos]
                    if 'P' in pos_obj.variants:
                        self.covered_positions[pos_obj] = 'P'
                        self.covered_variants.append(pos_obj.variants['P'])
                        self.variant_read_positions[pos_obj.variants['P']] = qp
                qp += size

            elif op == 2:
                for i in range(size):
                    wt_pos = rp + i
                    if wt_pos in gene.positions:
                        self.covered_positions[gene.positions[wt_pos]] = 'N'
                rp += size

            elif op == 4:
                qp += size


class Alignment:
    def __init__(self, rid, seq, hits):
        self.rid = rid
        self.seq = seq
        self.hits = hits

        self.best_by_gene = {}
        for h in hits:
            cur = self.best_by_gene.get(h.gene_name)
            if cur is None or h.score > cur.score:
                self.best_by_gene[h.gene_name] = h

        if hits:
            best_score = max(h.score for h in hits)
            self.best = [h for h in hits if h.score == best_score]
        else:
            self.best = []


class Sample:
    ALLELIC_PAIRS = (('KIR2DL2', 'KIR2DL3'), ('KIR3DL1', 'KIR3DS1'))
    PAIR_MARGIN = 0.0005

    REMAP_MIN_FRAC = 0.0
    REMAP_MAX_NM_RATE = 0.01
    REMAP_TIE = 10

    MIN_ALN_LEN = 5000
    GENE_ALN_FRAC = 0.7
    GENE_MARGIN_FRAC = 0.02
    GENE_MARGIN_FLOOR = 100

    class Read:
        def __init__(self, name, seq, qual=None, id=-1):
            self.name = name
            self.id = id
            self.seq = seq
            self.qual = qual

            self.alignment = None
            self.gene = None
            self.genes = set()
            self.alignments = {}

            self.covered_positions = {}

        def __len__(self):
            return len(self.seq)

        def __repr__(self):
            return f"Read({self.name}, {len(self.seq)}bp, gene={self.gene})"

    def __init__(self, path, db, region=DEFAULT_REGION, min_coverage=3):
        self.db = db
        self.path = path
        self.region = region
        self.min_coverage = min_coverage
        self.total_error = 0
        self.reads = []

        self._load(path)
        self._resolve_allelic_pairs()
        self._allele_remap_structural()
        self.populate_variant_coverage()

    def _load(self, bam_path):
        gene_aligners = {}
        for gname, gene in self.db.genes.items():
            idx = mappy.Aligner(seq=gene.seq, preset='map-hifi')
            assert idx, f"failed to build index for {gname}"
            gene_aligners[gname] = idx
        print(f"[index] per-gene indexes: {len(gene_aligners)} genes")

        seen = set()
        read_id = 0

        def _process_rec(rec):
            nonlocal read_id
            if rec.query_sequence is None:
                return
            if rec.is_secondary:
                return
            if 'H' in (rec.cigarstring or ''):
                return
            if rec.query_name in seen:
                return
            seen.add(rec.query_name)

            seq = rec.query_sequence
            qual = list(rec.query_qualities) if rec.query_qualities else None
            read = self.Read(rec.query_name, seq, qual, id=read_id)

            all_hits = []
            for gname, idx in gene_aligners.items():
                gene_len = len(self.db.genes[gname].seq)
                aln_thr = min(self.MIN_ALN_LEN, int(gene_len * self.GENE_ALN_FRAC))
                for mh in idx.map(seq):
                    h = Hit(mh, gname, rid=read_id, seq=seq)
                    if (h.r_en - h.r_st) >= aln_thr:
                        all_hits.append(h)

            if not all_hits:
                return

            read.alignment = Alignment(read.id, seq, all_hits)
            scores = {g: h.score for g, h in read.alignment.best_by_gene.items()}
            if scores:
                top = max(scores.values())
                margin = max(top * self.GENE_MARGIN_FRAC, self.GENE_MARGIN_FLOOR)
                read.genes = {g for g, s in scores.items() if s >= top - margin}
                read.gene = max(scores, key=scores.get)

            for gname, hit in read.alignment.best_by_gene.items():
                hit.populate_covered_variants(self.db.genes[gname])
            self.reads.append(read)
            read_id += 1

        with pysam.AlignmentFile(bam_path) as bam:
            contigs = {x['SN'] for x in bam.header['SQ']}
            chrom, _, span = self.region.partition(':')
            fetch_chrom = chrom if chrom in contigs else chrom.replace('chr', '')
            region = f"{fetch_chrom}:{span}" if span else fetch_chrom

            for rec in bam.fetch(region=region):
                _process_rec(rec)
            n1 = len(self.reads)
            print(f"[recruit] source 1: {region} -> {n1} KIR reads")

            alt_prefix = f"{fetch_chrom}_"
            for sq in bam.header['SQ']:
                name = sq['SN']
                if name.startswith(alt_prefix):
                    for rec in bam.fetch(contig=name):
                        _process_rec(rec)
            n2 = len(self.reads) - n1
            print(f"[recruit] source 2: alt contigs -> {n2} KIR reads")

        lens = sorted(len(r) for r in self.reads)
        self.read_len = lens[len(lens) // 2] if lens else 0
        print(f"[hifi] {len(self.reads):,} total KIR reads; "
              f"median length {self.read_len:,}")

        counts = Counter(r.gene for r in self.reads if r.gene)
        if counts:
            print("[hifi] reads per gene:")
            for gname, n in counts.most_common():
                print(f"       {gname:<10} {n:>5,}")

    @staticmethod
    def _op_base(op):
        if '>' in op:
            return op.split('>')[1][0]
        if op.startswith('del'):
            return 'N'
        if op.startswith('ins'):
            return 'P'
        return None

    def _allele_distance(self, hit, gene):
        cp = hit.covered_positions
        best = None
        for a in gene.alleles.values():
            d = hit.NM
            for pos, op in a.func | a.minor:
                po = gene.positions.get(pos)
                if po is None or po not in cp:
                    continue
                alt = self._op_base(op)
                if alt is None:
                    continue
                obs = cp[po]
                d += (obs == po.ref_base) - (obs == alt)
            if best is None or d < best:
                best = d
        return best

    def _resolve_allelic_pairs(self):
        moved = Counter()
        for a_name, b_name in self.ALLELIC_PAIRS:
            ga, gb = self.db.genes.get(a_name), self.db.genes.get(b_name)
            if ga is None or gb is None:
                continue
            for read in self.reads:
                if not read.alignment or a_name not in read.genes or b_name not in read.genes:
                    continue
                ha = read.alignment.best_by_gene.get(a_name)
                hb = read.alignment.best_by_gene.get(b_name)
                if not ha or not hb:
                    continue
                ra = self._allele_distance(ha, ga) / max(1, ha.r_en - ha.r_st)
                rb = self._allele_distance(hb, gb) / max(1, hb.r_en - hb.r_st)
                if abs(ra - rb) < self.PAIR_MARGIN:
                    continue
                win, lose = (a_name, b_name) if ra < rb else (b_name, a_name)
                read.genes.discard(lose)
                if read.gene == lose:
                    read.gene = win
                moved[f"{lose}->{win}"] += 1
        print(f"[pairs] reads assigned to one gene of an allelic pair: {dict(moved)}")

    def _allele_remap_structural(self):
        for gene_name in STRUCTURAL_GENES:
            if gene_name in self.db.genes:
                self._allele_remap_gene(gene_name)

    def _allele_remap_gene(self, gene_name):
        gene = self.db.genes[gene_name]

        allele_aligners = {}
        for aname, allele in gene.alleles.items():
            seq = getattr(allele, 'seq', None)
            if not seq or not getattr(allele, 'mutmap', None):
                continue
            idx = mappy.Aligner(seq=seq, preset='map-hifi')
            if idx:
                allele_aligners[aname] = (idx, seq)

        if not allele_aligners:
            print(f"[allele-remap] {gene_name}: no allele indexes built")
            return
        print(f"[allele-remap] {gene_name}: {len(allele_aligners)} allele indexes")

        remapped = 0
        n_rejected = 0
        winner_counts = Counter()

        for read in self.reads:
            if not read.alignment:
                continue
            if not hasattr(read, 'strict_genes'):
                read.strict_genes = set(read.genes)

            allele_hits = []
            for aname, (idx, aseq) in allele_aligners.items():
                for mh in idx.map(read.seq):
                    s = (mh.r_en - mh.r_st) - mh.NM
                    allele_hits.append((s, mh, aname))
            allele_hits.sort(key=lambda x: -x[0])

            if not allele_hits:
                continue

            assigned = False
            win_score = None
            support = set()
            for score, mh, aname in allele_hits:
                if assigned and win_score is not None and score < win_score - self.REMAP_TIE:
                    break
                allele = gene.alleles[aname]
                aseq = allele_aligners[aname][1]
                alen = mh.r_en - mh.r_st
                strict = (alen >= self.REMAP_MIN_FRAC * len(aseq)
                          and mh.NM <= self.REMAP_MAX_NM_RATE * alen)
                if assigned and not strict:
                    continue

                rp = mh.r_st
                if mh.strand == 1:
                    wseq = read.seq
                    qp = mh.q_st
                else:
                    wseq = read.seq[mh.q_st:mh.q_en].translate(Hit._COMP)[::-1]
                    qp = 0

                cov = {}
                cvr = []
                valid = True

                for size, cop in mh.cigar:
                    if not valid:
                        break
                    if cop == 0:
                        for i in range(size):
                            ap = rp + i
                            mm = allele.mutmap.get(ap)
                            if mm is None:
                                continue
                            mmi, mut = mm

                            read_base = wseq[qp + i].upper()
                            allele_base = aseq[ap].upper()
                            if read_base != allele_base:
                                valid = False
                                break

                            wp = mut[0]
                            if wp not in gene.positions:
                                continue
                            po = gene.positions[wp]

                            if mmi == 0:
                                mop = mut[1]
                                if mop.startswith('del'):
                                    obs = 'N'
                                elif mop.startswith('ins'):
                                    obs = 'P'
                                else:
                                    obs = read_base
                            else:
                                obs = read_base

                            if po not in cov and obs in po.variants:
                                cov[po] = obs
                                cvr.append(po.variants[obs])
                        rp += size
                        qp += size

                    elif cop == 1:
                        if allele.mutmap.get(rp, (2,))[0] < 2:
                            valid = False
                        qp += size

                    elif cop == 2:
                        for i in range(size):
                            if allele.mutmap.get(rp + i, (2,))[0] < 2:
                                valid = False
                                break
                        rp += size

                    elif cop == 4:
                        qp += size

                if not valid or not cov:
                    continue
                if strict:
                    if win_score is None:
                        win_score = score
                    if score >= win_score - self.REMAP_TIE:
                        support.add(aname)
                if assigned:
                    continue

                read.genes.add(gene_name)
                winner_counts[aname] += 1

                h = Hit.__new__(Hit)
                h.rid = read.id
                h.seq = read.seq
                h.gene_name = gene_name
                h.r_st = 0
                h.r_en = len(gene.seq)
                h.q_st = mh.q_st
                h.q_en = mh.q_en
                h.mapq = mh.mapq
                h.NM = mh.NM
                h.cigar = mh.cigar
                h.strand = mh.strand
                h.is_primary = True
                h.enabled = True
                h.covered_positions = cov
                h.covered_variants = cvr
                h.variant_read_positions = {}

                read.alignment.best_by_gene[gene_name] = h
                remapped += 1
                assigned = True

            if support:
                if not hasattr(read, 'remap_support'):
                    read.remap_support = {}
                read.remap_support[gene_name] = support
                read.strict_genes.add(gene_name)
            if not assigned and allele_hits:
                n_rejected += 1

        print(f"[allele-remap] {gene_name}: {remapped} remapped, {n_rejected} all-rejected")
        if winner_counts:
            print("[allele-remap] winners: "
                  + ", ".join(f"*{a}:{c}" for a, c in winner_counts.most_common(10)))

    def populate_variant_coverage(self):
        for read in self.reads:
            if not read.alignment:
                continue
            for gname in read.genes:
                hit = read.alignment.best_by_gene.get(gname)
                if not hit:
                    continue
                for pos_obj, obs in hit.covered_positions.items():
                    pos_obj.variants[obs].coverage.covered_reads.add(read.id)

    def populate_region_coverage(self):
        enabled_genes = [g for g in self.db.genes
                         if any(a.enabled for a in self.db.genes[g].alleles.values())]
        for gene_name in enabled_genes:
            gene = self.db.genes[gene_name]

            cov = defaultdict(int)
            seen = set()
            for read in self.reads:
                if gene_name not in read.genes:
                    continue
                hit = read.alignment.best_by_gene.get(gene_name)
                if not hit or read.id in seen:
                    continue
                seen.add(read.id)
                rp = hit.r_st
                for size, op in hit.cigar:
                    if op in (0, 7, 8):
                        for i in range(size):
                            cov[rp + i] += 1
                        rp += size
                    elif op == 2:
                        rp += size

            pos = 0
            for region in gene.wildtype.regions.values():
                rlen = len(region.seq)
                total = sum(cov.get(p, 0) for p in range(pos, pos + rlen))
                region.coverage = (total / rlen) if rlen > 0 else 0
                pos += rlen

    def _single_copy_coverage(self):
        gene = self.db.genes['KIR3DL3']
        total_bases = 0
        total_cov = 0
        for region in gene.wildtype.regions.values():
            rlen = len(region.seq)
            total_cov += region.coverage * rlen
            total_bases += rlen
        return round(total_cov / total_bases / 2) if total_bases else 30

    def calculate_coverage(self):
        self.populate_region_coverage()
        self.expected_coverage = self._single_copy_coverage()
        print(f"[coverage] expected single-copy coverage: {self.expected_coverage}")

    def _func_expected(self, a, allele_func_only=False):
        gene = a.gene
        func_expected = {}
        for pos, op in gene.functional:
            if pos not in gene.positions:
                continue
            position = gene.positions[pos]
            if (pos, op) in a.func:
                base = self._op_base(op)
                if base:
                    func_expected[pos] = base
            elif pos not in func_expected:
                if not allele_func_only and gene.name not in STRUCTURAL_GENES:
                    func_expected[pos] = position.ref_base
        return func_expected

    @staticmethod
    def _mismatches(hit, gene, expected):
        n_cov = 0
        mm = []
        for pos, exp in expected.items():
            pos_obj = gene.positions[pos]
            if pos_obj not in hit.covered_positions:
                continue
            obs = hit.covered_positions[pos_obj]
            if obs in ('N', 'P') and obs != exp:
                continue
            n_cov += 1
            if obs != exp:
                mm.append((pos, exp, obs))
        return n_cov, tuple(mm)

    def _minor_check(self, a, min_coverage):
        gene = a.gene
        minor_covered = {}
        for pos, op in a.minor:
            base = self._op_base(op)
            if base is None:
                continue
            minor_covered[(pos, op)] = gene.positions[pos].variants[base].coverage.count >= min_coverage

        if minor_covered and all(not v for v in minor_covered.values()):
            a.enabled = False

        a.minor_uncovered = sum(1 for v in minor_covered.values() if not v) if minor_covered else 0
        a.minor_miss = a.minor_uncovered / len(minor_covered) if minor_covered else 0

    def filter_allele(self, a, min_coverage=None, verbose=False,
                      allele_func_only=False, max_missing_func=0):
        if min_coverage is None:
            min_coverage = self.min_coverage

        gene = a.gene
        a.enabled = True

        func_expected = self._func_expected(a, allele_func_only=allele_func_only)

        if not func_expected:
            a.minor_uncovered = 0
            a.minor_miss = 0
            if verbose:
                print(f"  [filter] {gene.name}*{a.name}: no func_expected, pass")
            return

        n_fail = 0
        for pos, exp_base in func_expected.items():
            cov = gene.positions[pos].variants[exp_base].coverage.count
            if cov < min_coverage:
                n_fail += 1
                if verbose:
                    print(f"  [filter] {gene.name}*{a.name}: FAIL pos={pos} "
                          f"exp={exp_base} cov={cov} < {min_coverage}")
                if n_fail > max_missing_func:
                    a.enabled = False
                    return

        self._minor_check(a, min_coverage)

    def filter_db(self, verbose=False):
        self.valid_alleles = []
        self.populate_region_coverage()
        self.expected_coverage = self._single_copy_coverage()

        for gene_name in sorted(self.db.genes.keys()):
            gene = self.db.genes[gene_name]

            for a in gene.alleles.values():
                self.filter_allele(a, min_coverage=self.min_coverage, verbose=verbose)

            ax = {}
            for a in gene.alleles.values():
                if a.enabled:
                    major = self.db.minor_to_major[gene_name, a.name][:3]
                    ax.setdefault(major, []).append((a.name, a.minor_miss))
                    self.valid_alleles.append(a)

            for key, alleles in sorted(ax.items()):
                al = '; '.join(f"{n}: {m:.1%}" for n, m in sorted(alleles, key=lambda x: x[1]))
                print(f"[filter] {gene_name} *{key} => {al}")

        for gene_name, gene in self.db.genes.items():
            if any(a.enabled for a in gene.alleles.values()):
                continue
            if gene_name not in ['KIR2DP1']:
                continue

            candidates = []
            for aname, a in gene.alleles.items():
                if not a.func:
                    continue
                self.filter_allele(a, min_coverage=self.min_coverage, allele_func_only=True)
                if a.enabled:
                    candidates.append((len(a.func), aname, a))

            candidates.sort(key=lambda x: x[0])
            for i, (_, _, a) in enumerate(candidates):
                if i >= 2:
                    a.enabled = False

            if not candidates:
                print(f'second pass filter for gene {gene_name}')
                func_positions = [p for p, _ in gene.functional if p in gene.positions]
                covered = sum(1 for p in func_positions
                              if any(v.coverage.count >= self.min_coverage
                                     for v in gene.positions[p].variants.values()))
                if func_positions and covered / len(func_positions) > 0.9:
                    for aname, a in gene.alleles.items():
                        self.filter_allele(a, min_coverage=self.min_coverage,
                                           allele_func_only=True, max_missing_func=1)
                        if a.enabled:
                            candidates.append((len(a.func), aname, a))

            ax = {}
            for _, aname, a in candidates:
                if a.enabled:
                    major = self.db.minor_to_major[gene_name, aname][:3]
                    ax.setdefault(major, []).append((aname, a.minor_miss))
            for key, alleles in sorted(ax.items()):
                al = '; '.join(f"{n}: {m:.1%}" for n, m in sorted(alleles, key=lambda x: x[1]))
                print(f"[filter] {gene_name} *{key} => {al}")

        print(f"[filter] enabled alleles={sum(1 for a in self.db.alleles() if a.enabled):,}")

    def estimate_gene_cn_ratio(self):
        for gene_name, gene in self.db.genes.items():
            if gene_name == 'KIR3DL3':
                gene.estimated_cn = 2
                continue

            if not any(a.enabled for a in gene.alleles.values()):
                gene.estimated_cn = 0
                continue

            total_cov = 0
            total_len = 0
            for region in gene.wildtype.regions.values():
                rlen = len(region.seq)
                total_cov += region.coverage * rlen
                total_len += rlen

            if total_len == 0 or not self.expected_coverage:
                gene.estimated_cn = 0
                continue

            ratio = (total_cov / total_len) / self.expected_coverage
            max_cn = 3 if gene_name in ('KIR3DP1', 'KIR2DL4') else 2
            gene.estimated_cn = max(0, min(round(ratio), max_cn))
            print(f"[cn] {gene_name}: ratio={ratio:.2f} -> CN={gene.estimated_cn}")


def init_region_bounds(db):
    for gene in db.genes.values():
        gene.ref_start = 0
        pos = 0
        for region in gene.wildtype.regions.values():
            region.start = pos
            region.end = pos + len(region.seq)
            region.coverage = 0
            region.cross_map_rate = {}
            pos += len(region.seq)


def populate_coverage_flags(db):
    for gene in db.genes.values():
        for position in gene.positions.values():
            position.has_no_coverage = not any(
                v.coverage.count > 0 for v in position.variants.values())


def populate_valid_positions(sample, db):
    for gene in db.genes.values():
        for position in gene.positions.values():
            position.is_valid = False

    for allele in db.alleles():
        if not allele.enabled:
            continue
        gene = allele.gene
        if allele == gene.wildtype:
            for pos, op in gene.functional:
                if not gene.positions[pos].has_no_coverage:
                    gene.positions[pos].is_valid = True
        else:
            for pos, op in allele.func:
                if not gene.positions[pos].has_no_coverage:
                    gene.positions[pos].is_valid = True
            for pos, op in allele.minor:
                if not gene.positions[pos].has_no_coverage:
                    gene.positions[pos].is_valid = True


def zero_out_low_coverage_variants(sample, db, coverage_threshold=3):
    for gene in db.genes.values():
        for position in gene.positions.values():
            for variant in position.variants.values():
                if variant.coverage.count <= coverage_threshold:
                    variant.coverage.covered_reads.clear()
                    variant.coverage.covered_hits = []


def build_allele_matrix(sample, db):
    valid_variants = {v for gene in db.genes.values() for pos in gene.positions.values()
                      if pos.is_valid for v in pos.variants.values()}

    valid_alleles = [a for a in db.alleles() if a.enabled]
    sample.valid_alleles = valid_alleles
    sample.valid_indices = sorted(v.index for v in valid_variants)

    position_order = []
    for idx in sample.valid_indices:
        variant = db.variants()[idx]
        pos_key = (variant.gene.name, variant.pos)
        if pos_key not in position_order:
            position_order.append(pos_key)

    valid_positions = [db.genes[gene].positions[pos] for gene, pos in position_order]

    for allele in sample.valid_alleles:
        allele.generatable_positions = [
            pos_obj for pos_obj in valid_positions
            if any(allele.extended_allele_vector[variant.index] == 1
                   for variant in pos_obj.variants.values())
        ]

    total_variants = sum(len(pos.variants) for gene in db.genes.values()
                         for pos in gene.positions.values())
    sample.beta = np.zeros((len(valid_alleles), total_variants), dtype=np.float64)

    for i, allele in enumerate(valid_alleles):
        sample.beta[i] = allele.extended_allele_vector

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
    for i in range(len(sample.beta)):
        key = tuple(sample.beta[i])
        if key not in seen:
            seen.add(key)
            unique_indices.append(i)
    sample.beta = sample.beta[unique_indices]
    sample.valid_alleles = [sample.valid_alleles[i] for i in unique_indices]
    for i, allele in enumerate(sample.valid_alleles):
        allele.index = i

    print(f"[dedupe] Kept {len(sample.valid_alleles)} unique alleles out of {len(valid_alleles)} total")


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
                    gene.cn_config[(other_gene_name, rname)] = \
                        region.cross_map_rate.get(gene_name, 0.0) / 2


def get_gene_cn_tensors(db, expected_coverage):
    enabled_genes = [g.name for g in db.genes.values()
                     if any(a.enabled for a in g.alleles.values())]
    region_keys = [(g_name, rname) for g_name in enabled_genes
                   for rname in db.genes[g_name].wildtype.regions.keys()]

    region_mask = torch.zeros(len(enabled_genes), len(region_keys), dtype=torch.float32)
    for i, gene_name in enumerate(enabled_genes):
        gene = db.genes[gene_name]
        for j, key in enumerate(region_keys):
            region_mask[i, j] = gene.cn_config.get(key, 0.0)

    region_cov = torch.tensor(
        [db.genes[g].wildtype.regions[r].coverage / expected_coverage
         for g, r in region_keys], dtype=torch.float32)

    return region_mask, region_cov, enabled_genes, region_keys


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


def estimate_gene_cn_model(sample, db):
    from cn.cn_estimator import run_cn_estimator

    populate_gene_cn_config(db)
    region_mask, region_cov, gene_names, region_keys = get_gene_cn_tensors(
        db, sample.expected_coverage)
    gene_cn = run_cn_estimator(
        region_cov=region_cov,
        region_mask=region_mask,
        gene_names=gene_names,
        max_cn=5.0,
        num_iterations=1000,
        delta=0.3,
    )

    for gene in db.genes.values():
        gene.estimated_cn = 0

    print("\n[Gene CN Results]")
    for gene, cn in sorted(gene_cn.items(), key=lambda x: x[0]):
        print(f"  {gene}: {cn:.2f}")
        db.genes[gene].estimated_cn = discretize_cn(gene, cn)


def report_alleles(densities, densities_un, valid_allele_names, db, threshold=0.019):
    major_allele_densities = {}
    major_allele_densities_unnorm = {}

    print('[Raw]:')
    for i, density in enumerate(densities):
        allele_name = valid_allele_names[i]
        gene, allele_id = allele_name.split('*')
        if gene == 'KIR2DL5':
            ab, allele_id = allele_id.split('.')
            gene = f'KIR2DL5{ab}'
        major_key = f"{gene}*{allele_id[:3]}"
        if major_key not in major_allele_densities:
            major_allele_densities[major_key] = 0
            major_allele_densities_unnorm[major_key] = 0
        major_allele_densities[major_key] += density
        major_allele_densities_unnorm[major_key] += densities_un[i]

    filtered_by_gene = {}
    for major_key, norm_val in major_allele_densities.items():
        if norm_val > threshold:
            gene_name = major_key.split('*')[0]
            filtered_by_gene.setdefault(gene_name, []).append(
                (major_key, norm_val, major_allele_densities_unnorm[major_key]))

    final_results = []
    for gene_name, alleles in filtered_by_gene.items():
        alleles.sort(key=lambda x: x[1], reverse=True)
        if gene_name == 'KIR2DL5A':
            final_results.extend(alleles[:1])
        elif gene_name == 'KIR3DL3':
            final_results.extend(alleles[:2])
        else:
            final_results.extend(alleles)

    final_results.sort(key=lambda x: x[0])
    for major_key, norm_val, unnorm_val in final_results:
        print(f"{major_key}: norm={norm_val:.6f}, unnorm={unnorm_val:.6f}")

    gene_alleles = {}
    for major_key, norm_val, unnorm_val in final_results:
        gene_name = major_key.split('*')[0]
        gene_alleles.setdefault(gene_name, []).append((major_key, norm_val, unnorm_val))

    if 'KIR2DL5A' in gene_alleles or 'KIR2DL5B' in gene_alleles:
        combined = []
        for ab in ('KIR2DL5A', 'KIR2DL5B'):
            if ab in gene_alleles:
                al = gene_alleles.pop(ab)
                al.sort(key=lambda x: x[1], reverse=True)
                combined.append(al[0])
        gene_alleles['KIR2DL5'] = combined

    final_calls = []
    for gene_name, alleles in gene_alleles.items():
        gene_cn = db.genes[gene_name].estimated_cn
        alleles.sort(key=lambda x: x[1], reverse=True)

        allele_cn = {major_key: 1 for major_key, _, _ in alleles}
        remaining = gene_cn - len(alleles)
        if remaining > 0:
            allele_cn[alleles[0][0]] += remaining

        for major_key, norm_val, _ in alleles:
            final_calls.append((major_key, allele_cn[major_key]))

    print("\n[Alleles]")
    for major_key, cn in sorted(final_calls, key=lambda x: x[0]):
        for _ in range(cn):
            print(f"{major_key}")

    return final_calls


def main():
    parser = argparse.ArgumentParser(description='KIR gene typing tool for HiFi reads')
    parser.add_argument('--input', required=True, help='Path to aligned HiFi BAM/CRAM file')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--region', default=DEFAULT_REGION,
                        help='KIR locus to recruit reads from')
    parser.add_argument('--min-coverage', type=int, default=3,
                        help='Minimum variant coverage for allele filtering')
    parser.add_argument('--cn', choices=['model', 'ratio'], default='model',
                        help='Copy number from the Neretva estimator or from coverage ratio')
    parser.add_argument('--iterations', type=int, default=3000)
    args = parser.parse_args()

    path = args.input

    with timing("initializaiton"):
        db = Database(importlib.resources.files("neretva") / "data" / "kir.pickle")
        init_region_bounds(db)
        db.assign_index()
        db.populate_allele_vectors()

    with timing("preprocessing"):
        sample = Sample(path, db, region=args.region, min_coverage=args.min_coverage)
        sample.filter_db()
        sample.calculate_coverage()
        print(f"[read] {os.path.basename(path)}: {len(sample.reads):,} reads loaded "
              f"({sample.expected_coverage=:.1f}x)")

    with timing("prep for solver"):
        zero_out_low_coverage_variants(sample, db)
        populate_coverage_flags(db)
        populate_valid_positions(sample, db)
        build_allele_matrix(sample, db)
        print_variant_counts(sample, db)

    with timing("cn estimation"):
        if args.cn == 'model':
            estimate_gene_cn_model(sample, db)
        else:
            sample.estimate_gene_cn_ratio()

    from core.vae import run_vae

    total_mut_counts = sum(variant.coverage.count
                           for gene in db.genes.values()
                           for pos in gene.positions.values()
                           if pos.is_valid
                           for variant in pos.variants.values())
    mut_counts_tensor = torch.tensor(
        [db.variants()[i].coverage.count for i in sample.valid_indices],
        dtype=torch.float32)
    valid_allele_names = [f'{a.gene.name}*{a.name}' for a in sample.valid_alleles]

    densities, densities_un, learnt_beta = run_vae(
        total_mut_counts,
        valid_allele_names,
        mut_counts_tensor,
        sample=sample,
        db=db,
        num_iterations=args.iterations,
        bam_id=os.path.basename(path),
        seeds=[args.seed],
    )

    report_alleles(densities, densities_un, valid_allele_names, db)


if __name__ == "__main__":
    main()
