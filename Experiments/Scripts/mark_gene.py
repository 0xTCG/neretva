#!/usr/bin/env python3

import sys
import csv
from collections import defaultdict
from itertools import product

TOOL_ORDER = ['PING', 'T1K', 'Geny', 'kir-mapper','Locityper','Neretva']


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
                if pp and pp != '-':
                    alleles.append(pp[:3])
        return [alleles]
    expanded = []
    for sp in slash_parts:
        sp = sp.strip()
        if not sp or sp == '-': continue
        if sp.startswith('(') and sp.endswith(')'):
            expanded.append([x.strip()[:3] for x in sp[1:-1].split(',')])
        else:
            for pp in sp.split('+'):
                pp = pp.strip()
                if pp and pp != '-':
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


def fmt_pct(v): return f"{v*100:.1f}\\%"
def fmt_f1(v): return f"{v:.2f}"


def bold_max(values, fmt_func, higher_is_better=True):
    if not values or all(v is None for v in values):
        return ['---']*len(values)
    valid = [v for v in values if v is not None]
    if not valid: return ['---']*len(values)
    best_val = max(valid)
    return [f"\\textbf{{{fmt_func(v)}}}" if v is not None and v == best_val
            else (fmt_func(v) if v is not None else '---') for v in values]


def main():
    if len(sys.argv) < 2:
        print("Usage: python evaluate_per_gene_latex.py results_with_M.csv > tables.tex", file=sys.stderr)
        sys.exit(1)

    path = sys.argv[1]
    with open(path) as f:
        header = next(csv.reader(f))

    all_tools = []
    i = 4
    while i < len(header) - 1:
        if header[i] and header[i] != 'M' and header[i+1] == 'M':
            all_tools.append((header[i], i, i+1))
            i += 2
        else:
            i += 1

    tool_map = {n: (n, c, m) for n, c, m in all_tools}
    tools = [tool_map[tn] for tn in TOOL_ORDER if tn in tool_map]
    tool_names = [t[0] for t in tools]

    rows = []
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if not row or not row[0]: continue
            if len(row) > 1 and row[1] != 'HPRC': continue
            if len(row) > 2 and not row[2].startswith('KIR'): continue
            rows.append(row)

    genes_order = []
    gene_rows = defaultdict(list)
    for row in rows:
        gene = row[2]
        if gene not in gene_rows: genes_order.append(gene)
        gene_rows[gene].append(row)

    gene_n_sample = {}
    for gene in genes_order:
        gene_n_sample[gene] = sum(1 for row in gene_rows[gene]
                                   if row[3].strip().strip('"') not in ('', '-/-'))

    # ===== SAMPLE LEVEL =====
    sample_data = {}
    sample_totals = {tn: [0,0,0,0] for tn in tool_names}

    for gene in genes_order:
        for tn, cc, mc in tools:
            tp = fp = fn = n_tp = 0
            for row in gene_rows[gene]:
                if mc >= len(row): continue
                m = row[mc].strip()
                if not m or m == '#': continue
                ts = row[3].strip().strip('"')
                cs = row[cc].strip().strip('"') if cc < len(row) else ''
                ta = not ts or ts == '-/-'
                ca = not cs or cs == '-/-'
                if not ta: n_tp += 1
                if m == '1': tp += 1
                elif m == '0':
                    if not ca: fp += 1
                    if not ta and ca: fn += 1
            prec = tp/(tp+fp) if (tp+fp) > 0 else None
            rec = tp/(tp+fn) if (tp+fn) > 0 else None
            f1 = 2*prec*rec/(prec+rec) if prec and rec and (prec+rec) > 0 else None
            sample_data[(gene, tn)] = (n_tp, tp, prec, rec, f1)
            sample_totals[tn][0] += tp; sample_totals[tn][1] += fp
            sample_totals[tn][2] += fn; sample_totals[tn][3] += n_tp

    # ===== ALLELE LEVEL =====
    allele_data = {}
    allele_totals = {tn: [0,0,0] for tn in tool_names}

    for gene in genes_order:
        for tn, cc, mc in tools:
            gtp = gfp = gfn = 0
            for row in gene_rows[gene]:
                if mc >= len(row): continue
                m = row[mc].strip()
                if not m or m == '#': continue
                truth = parse_alleles(row[3].strip().strip('"'))[0]
                pred_sols = parse_alleles(row[cc].strip() if cc < len(row) else '')
                tp, fp, fn = best_solution_allele(truth, pred_sols)
                gtp += tp; gfp += fp; gfn += fn
            allele_totals[tn][0] += gtp; allele_totals[tn][1] += gfp; allele_totals[tn][2] += gfn
            n = gtp + gfn
            prec = gtp/(gtp+gfp) if (gtp+gfp) > 0 else None
            rec = gtp/n if n > 0 else None
            f1 = 2*prec*rec/(prec+rec) if prec and rec and (prec+rec) > 0 else None
            allele_data[(gene, tn)] = (n, gtp, prec, rec, f1)

    # ===== LATEX OUTPUT =====
    def print_tabular(data, totals, level):
        col_spec = 'l|r' + '|rrrr'*len(tools)
        print(f"\\resizebox{{\\textwidth}}{{!}}{{%")
        print(f"\\begin{{tabular}}{{{col_spec}}}")
        print(f"\\hline")

        h1 = ' & '
        for ti, tn in enumerate(tool_names):
            sep = '|' if ti < len(tool_names)-1 else ''
            h1 += f" & \\multicolumn{{4}}{{c{sep}}}{{\\textbf{{{tn}}}}}"
        print(h1 + ' \\\\')

        h2 = '\\textbf{Gene} & \\textbf{Total}'
        for _ in tools:
            h2 += ' & \\textbf{Correct} & \\textbf{Precision} & \\textbf{Recall} & $F_1$'
        print(h2 + ' \\\\')
        print('\\hline')

        for gene in genes_order:
            total = gene_n_sample[gene] if level == 'sample' else data[(gene, tool_names[0])][0]
            s = f"\\textit{{{gene}}} & {total}"
            precs = [data[(gene,tn)][2] for tn in tool_names]
            recs = [data[(gene,tn)][3] for tn in tool_names]
            f1s = [data[(gene,tn)][4] for tn in tool_names]
            corrects = [data[(gene,tn)][1] for tn in tool_names]
            bc = bold_max(corrects, lambda v: str(int(v)), higher_is_better=True)
            bp = bold_max(precs, fmt_pct)
            br = bold_max(recs, fmt_pct)
            bf = bold_max(f1s, fmt_f1)
            for ti, tn in enumerate(tool_names):
                s += f" & {bc[ti]} & {bp[ti]} & {br[ti]} & {bf[ti]}"
            print(s + ' \\\\')

        print('\\hline')
        if level == 'sample':
            grand = sum(gene_n_sample[g] for g in genes_order)
        else:
            grand = allele_totals[tool_names[0]][0] + allele_totals[tool_names[0]][2]
        s = f"All & {grand}"
        precs, recs, f1s, corrects = [], [], [], []
        for tn in tool_names:
            if level == 'sample':
                tp,fp,fn,n = totals[tn]
            else:
                tp,fp,fn = totals[tn]
                n = tp+fn
            prec = tp/(tp+fp) if (tp+fp) > 0 else 0
            rec = tp/(tp+fn) if (tp+fn) > 0 else (tp/n if n > 0 else 0)
            f1 = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0
            precs.append(prec); recs.append(rec); f1s.append(f1); corrects.append(tp)
        bc = bold_max(corrects, lambda v: str(int(v)), higher_is_better=True)
        bp = bold_max(precs, fmt_pct)
        br = bold_max(recs, fmt_pct)
        bf = bold_max(f1s, fmt_f1)
        for ti in range(len(tool_names)):
            s += f" & {bc[ti]} & {bp[ti]} & {br[ti]} & {bf[ti]}"
        print(s + ' \\\\')
        print('\\hline')
        print('\\end{tabular}')
        print('}%')

    print("\\begin{table*}[t]")
    print("\\caption{Per-gene performance comparison on KIR genes (HPRC dataset). The best results per gene are highlighted in bold.}")
    print("\\label{tab:per_gene}")
    print("\\centering")
    print()
    print("\\textbf{(a) Per Call (Sample-Level)}\\\\[2pt]")
    print_tabular(sample_data, sample_totals, 'sample')
    print()
    print("\\textbf{(b) Per Allele}\\\\[2pt]")
    print_tabular(allele_data, allele_totals, 'allele')
    print()
    print("\\end{table*}")


if __name__ == '__main__':
    main()