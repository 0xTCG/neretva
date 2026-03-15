#!/usr/bin/env python3
"""
CYP allele-level evaluation grouped by gene (CYP2C8, CYP2C9, CYP2C19, CYP2D6).
Filters: TECH=WGS, GENE starts with CYP.

Usage: python evaluate_cyp_allele.py results.csv
"""

import sys
import csv
from collections import defaultdict
from itertools import product


def parse_alleles(s):
    s = s.strip().strip('"')
    if not s or s == '-/-' or s == '.':
        return [[]]
    if ';' in s:
        depth = 0
        for ch in s:
            if ch == '(': depth += 1
            elif ch == ')': depth -= 1
            elif ch == ';' and depth == 0:
                parts = _split_outer(s, ';')
                solutions = []
                for part in parts:
                    for sol in _parse_single(part):
                        solutions.append(sol)
                return solutions if solutions else [[]]
    return _parse_single(s)


def _split_outer(s, sep):
    parts, depth, current = [], 0, ''
    for ch in s:
        if ch == '(': depth += 1
        elif ch == ')': depth -= 1
        if ch == sep and depth == 0:
            parts.append(current.strip()); current = ''
        else:
            current += ch
    if current.strip(): parts.append(current.strip())
    return parts


def _parse_single(s):
    s = s.strip().strip('"')
    if not s or s == '-/-' or s == '.':
        return [[]]
    has_parens = '(' in s
    slash_parts = _split_outer(s, '/')
    if not has_parens:
        alleles = []
        for sp in slash_parts:
            for pp in sp.split('+'):
                pp = pp.strip()
                if pp and pp != '-' and not pp.lower().startswith('unr'):
                    alleles.append(pp[:3])
        return [alleles]
    expanded = []
    for sp in slash_parts:
        sp = sp.strip()
        if not sp or sp == '-' or sp.lower().startswith('unr'): continue
        if sp.startswith('(') and sp.endswith(')'):
            expanded.append([x.strip()[:3] for x in sp[1:-1].split(',')])
        else:
            for pp in sp.split('+'):
                pp = pp.strip()
                if pp and pp != '-' and not pp.lower().startswith('unr'):
                    expanded.append([pp[:3]])
    if not expanded: return [[]]
    return [list(combo) for combo in product(*expanded)]


def get_correct(truth, pred):
    remaining_pred = list(pred)
    tp, matched = 0, [False]*len(truth)
    for p in list(remaining_pred):
        for i, t in enumerate(truth):
            if not matched[i] and p == t:
                matched[i] = True; tp += 1; remaining_pred.remove(p); break
    return tp, len(remaining_pred), sum(1 for m in matched if not m)


def best_solution_allele(truth, solutions):
    best = (0, 999, 999)
    for sol in solutions:
        tp, fp, fn = get_correct(truth, sol)
        if (fp+fn) < (best[1]+best[2]) or ((fp+fn) == (best[1]+best[2]) and tp > best[0]):
            best = (tp, fp, fn)
    return best


def main():
    if len(sys.argv) < 2:
        print("Usage: python evaluate_cyp_allele.py results.csv")
        sys.exit(1)

    path = sys.argv[1]

    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader)

    tools = []
    i = 4
    while i < len(header) - 1:
        if header[i] and header[i] != 'M' and header[i+1] == 'M':
            tools.append((header[i], i, i+1))
            i += 2
        else:
            i += 1

    rows = []
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if not row or not row[0]:
                continue
            if len(row) > 1 and row[1] != 'WGS':
                continue
            if len(row) > 2 and not row[2].startswith('CYP'):
                continue
            rows.append(row)

    genes_order = []
    gene_rows = defaultdict(list)
    for row in rows:
        gene = row[2]
        if gene not in gene_rows:
            genes_order.append(gene)
        gene_rows[gene].append(row)

    print(f"Loaded {len(rows)} CYP WGS rows, {len(tools)} tools, {len(genes_order)} genes\n")

    for gene in genes_order:
        print("=" * 95)
        print(f"Per Allele — {gene}")
        print("=" * 95)
        print(f"{'Tool':<20} {'n':>5} {'Correct':>8} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8}")
        print("-" * 95)

        for tool_name, call_col, m_col in tools:
            gtp = gfp = gfn = 0

            for row in gene_rows[gene]:
                if m_col >= len(row):
                    continue
                m = row[m_col].strip()
                if not m or m == '#':
                    continue
                truth = parse_alleles(row[3].strip().strip('"'))[0]
                pred_sols = parse_alleles(row[call_col].strip() if call_col < len(row) else '')
                tp, fp, fn = best_solution_allele(truth, pred_sols)
                gtp += tp; gfp += fp; gfn += fn

            n = gtp + gfn
            if n == 0 and gfp == 0:
                continue
            acc = gtp / n if n > 0 else 0
            prec = gtp / (gtp + gfp) if (gtp + gfp) > 0 else 0
            rec = gtp / n if n > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            print(f"{tool_name:<20} {n:>5} {gtp:>8} {acc:>9.3f} {prec:>10.3f} {rec:>8.3f} {f1:>8.3f}")

        print()


if __name__ == '__main__':
    main()