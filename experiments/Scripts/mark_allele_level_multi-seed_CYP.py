#!/usr/bin/env python3
"""
Evaluate CYP WGS results at allele level from CSV with pre-computed M columns.
With *5 deletion handling (CYP2D6), CYP2C19 equivalence, NxM expansion,
dot-stripping, multi-seed summary.
"""

import sys
import csv
import re
import numpy as np
from itertools import product

CYP2C19_EQUIV = {'38': '1'}


def _normalize(s, gene=''):
    s = re.sub(r'[A-Za-z*]+$', '', s)
    if s.startswith('rs'):
        return ''
    if '.' in s:
        s = s.split('.')[0]
    s = s[:3]
    if gene == 'CYP2C19' and s in CYP2C19_EQUIV:
        s = CYP2C19_EQUIV[s]
    return s


def _expand_copies(s, gene=''):
    m = re.match(r'^(\d+)x(\d+)$', s)
    if m:
        allele = _normalize(m.group(1), gene)
        count = int(m.group(2))
        return [allele] * count
    n = _normalize(s, gene)
    return [n] if n else []


def parse_alleles(s, gene=''):
    s = s.strip().strip('"')
    if not s or s == '-/-' or s == '.':
        return [[]]

    s = s.replace("?", "")

    if re.search(r'\dor\d', s):
        parts = re.split(r'(?<=\d)or(?=\d)', s)
        solutions = []
        for part in parts:
            for sol in parse_alleles(part.strip(), gene):
                solutions.append(sol)
        return solutions if solutions else [[]]

    if "," in s and "(" not in s:
        s = s.split(",")[0].strip()

    if ';' in s:
        depth = 0
        has_outer_semi = False
        for ch in s:
            if ch == '(': depth += 1
            elif ch == ')': depth -= 1
            elif ch == ';' and depth == 0:
                has_outer_semi = True
                break
        if has_outer_semi:
            parts = _split_outer(s, ';')
            solutions = []
            for part in parts:
                for sol in _parse_single(part, gene):
                    solutions.append(sol)
            return solutions if solutions else [[]]

    return _parse_single(s, gene)


def _split_outer(s, sep):
    parts = []
    depth = 0
    current = ''
    for ch in s:
        if ch == '(': depth += 1
        elif ch == ')': depth -= 1
        if ch == sep and depth == 0:
            parts.append(current.strip())
            current = ''
        else:
            current += ch
    if current.strip():
        parts.append(current.strip())
    return parts


def _parse_single(s, gene=''):
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
                    alleles.extend(_expand_copies(pp, gene))
        return [alleles]

    expanded = []
    for sp in slash_parts:
        sp = sp.strip()
        if not sp or sp == '-':
            continue
        if sp.startswith('(') and sp.endswith(')'):
            alts = []
            for x in sp[1:-1].split(','):
                n = _normalize(x.strip(), gene)
                if n:
                    alts.append(n)
            expanded.append(alts)
        else:
            sub_alleles = []
            for pp in sp.split('+'):
                pp = pp.strip()
                if pp and pp != '-':
                    sub_alleles.extend(_expand_copies(pp, gene))
            for a in sub_alleles:
                expanded.append([a])

    if not expanded:
        return [[]]

    solutions = []
    for combo in product(*expanded):
        solutions.append(list(combo))
    return solutions


def get_correct(truth, pred):
    remaining_pred = list(pred)
    tp = 0
    matched_truth = [False] * len(truth)

    for p in list(remaining_pred):
        for i, t in enumerate(truth):
            if not matched_truth[i] and p == t:
                matched_truth[i] = True
                tp += 1
                remaining_pred.remove(p)
                break

    fn = sum(1 for m in matched_truth if not m)
    fp = len(remaining_pred)
    return tp, fp, fn


def canonical_truth(truth_solutions):
    return max(truth_solutions, key=len)


def evaluate_d6(truth, pred_solutions):
    """CYP2D6: canonical truth + *5 deletion handling."""
    best = (0, 999, 999)
    best_pred = pred_solutions[0] if pred_solutions else []

    truth_del = truth.count('5')
    truth_clean = [a for a in truth if a != '5']

    for pred in pred_solutions:
        pred_del = pred.count('5')
        pred_clean = [a for a in pred if a != '5']

        explicit_match = min(truth_del, pred_del)
        remaining_del = truth_del - explicit_match
        del_fp = pred_del - explicit_match

        tp, fp, fn = get_correct(truth_clean, pred_clean)

        if fn == 0:
            implicit_del = max(0, len(truth) - len(pred))
            auto_del_tp = min(remaining_del, implicit_del)
        else:
            auto_del_tp = 0

        tp += explicit_match + auto_del_tp
        fn += remaining_del - auto_del_tp
        fp += del_fp

        errors = fp + fn
        best_errors = best[1] + best[2]
        if errors < best_errors or (errors == best_errors and tp > best[0]):
            best = (tp, fp, fn)
            best_pred = pred

    return best, truth, best_pred


def evaluate_standard(truth_solutions, pred_solutions):
    """Non-CYP2D6: original logic."""
    best = (0, 999, 999)
    best_truth = truth_solutions[0] if truth_solutions else []
    best_pred = pred_solutions[0] if pred_solutions else []
    for truth in truth_solutions:
        for pred in pred_solutions:
            tp, fp, fn = get_correct(truth, pred)
            if (fp + fn) < (best[1] + best[2]) or ((fp + fn) == (best[1] + best[2]) and tp > best[0]):
                best = (tp, fp, fn)
                best_truth = truth
                best_pred = pred
    return best, best_truth, best_pred


def evaluate_row(gene, truth_solutions, pred_solutions):
    if gene == 'CYP2D6':
        canon = canonical_truth(truth_solutions)
        return evaluate_d6(canon, pred_solutions)
    else:
        return evaluate_standard(truth_solutions, pred_solutions)


def main():
    if len(sys.argv) < 2:
        print("Usage: python evaluate_cyp_allele_multi.py results.csv [gene_prefix]")
        sys.exit(1)

    path = sys.argv[1]
    gene_prefix = sys.argv[2] if len(sys.argv) > 2 else ''

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
            if gene_prefix and len(row) > 2 and not row[2].startswith(gene_prefix):
                continue
            rows.append(row)

    print(f"Loaded {len(rows)} rows, {len(tools)} tools\n")

    allele_stats = {name: [0, 0, 0] for name, _, _ in tools}

    for row in rows:
        truth_str = row[3].strip().strip('"')
        gene = row[2].strip() if len(row) > 2 else ''
        truth_solutions = parse_alleles(truth_str, gene)

        for tool_name, call_col, m_col in tools:
            if m_col >= len(row):
                continue
            m = row[m_col].strip()
            if m not in ('0', '1'):
                continue

            call_str = row[call_col].strip() if call_col < len(row) else ''
            pred_solutions = parse_alleles(call_str, gene)
            (tp, fp, fn), _, _ = evaluate_row(gene, truth_solutions, pred_solutions)

            allele_stats[tool_name][0] += tp
            allele_stats[tool_name][1] += fp
            allele_stats[tool_name][2] += fn

    print("=" * 95)
    print("Per Allele")
    print("=" * 95)
    print(f"{'Tool':<20} {'n':>5} {'Correct':>8} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print("-" * 95)

    seed_metrics = []
    tool_metrics = []
    for tool_name, _, _ in tools:
        tp, fp, fn = allele_stats[tool_name]
        n = tp + fn
        if n == 0:
            continue
        acc = tp / n
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / n
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        print(f"{tool_name:<20} {n:>5} {tp:>8} {acc:>9.3f} {prec:>10.3f} {rec:>8.3f} {f1:>8.3f}")
        if tp > 0:
            tool_metrics.append((tool_name, n, tp, acc, prec, rec, f1))
        if re.match(r'^Seed\d+$', tool_name):
            seed_metrics.append((n, tp, acc, prec, rec, f1))

    # Seed summary
    if seed_metrics:
        arr = np.array(seed_metrics)
        means = np.mean(arr, axis=0)
        stds = np.std(arr, axis=0)
        labels = ['n', 'Correct', 'Accuracy', 'Precision', 'Recall', 'F1']
        print("\n" + "=" * 55)
        print(f"Seed Summary (n={len(seed_metrics)} seeds)")
        print("=" * 55)
        for label, m, s in zip(labels, means, stds):
            if label == 'n':
                print(f"  {label:<12} {m:.0f}")
            elif label == 'Correct':
                print(f"  {label:<12} {m:.1f} ± {s:.1f}")
            else:
                print(f"  {label:<12} {m:.3f} ± {s:.3f}")

    # LaTeX per-tool
    if tool_metrics:
        best_correct = max(m[2] for m in tool_metrics)
        best_acc = max(m[3] for m in tool_metrics)
        best_prec = max(m[4] for m in tool_metrics)
        best_rec = max(m[5] for m in tool_metrics)
        best_f1 = max(m[6] for m in tool_metrics)

        print("\nLaTeX (per-tool):")
        for name, n, tp, acc, prec, rec, f1 in tool_metrics:
            def bf(val, best, fmt):
                s = f"{val:{fmt}}"
                return f"\\bf {s}" if val == best else s
            line = (f"{name} & {n} & {bf(tp, best_correct, 'd')} & {bf(acc, best_acc, '.3f')} "
                    f"& {bf(prec, best_prec, '.3f')} & {bf(rec, best_rec, '.3f')} & {bf(f1, best_f1, '.3f')} \\\\")
            print(line)

    # LaTeX seed summary
    if seed_metrics:
        n_val = f"{means[0]:.0f}"
        print("\nLaTeX (seed summary):")
        print(f"Neretva & {n_val} & {means[1]:.1f} & {means[2]:.3f} & {means[3]:.3f} & {means[4]:.3f} & {means[5]:.3f} \\\\")
        print(f"        &     & {{\\tiny$\\pm${stds[1]:.1f}}} & {{\\tiny$\\pm${stds[2]:.3f}}} & {{\\tiny$\\pm${stds[3]:.3f}}} & {{\\tiny$\\pm${stds[4]:.3f}}} & {{\\tiny$\\pm${stds[5]:.3f}}} \\\\")

    # Wrong predictions
    print("\n" + "=" * 95)
    print("Wrong Predictions (Allele-Level)")
    print("=" * 95)
    for row in rows:
        name = row[0].strip()
        gene = row[2].strip() if len(row) > 2 else ''
        truth_str = row[3].strip().strip('"')
        truth_solutions = parse_alleles(truth_str, gene)
        is_d6 = gene == 'CYP2D6'

        for tool_name, call_col, m_col in tools:
            if m_col >= len(row):
                continue
            m = row[m_col].strip()
            if m not in ('0', '1'):
                continue
            call_str = row[call_col].strip() if call_col < len(row) else ''
            if not call_str:
                continue

            pred_solutions = parse_alleles(call_str, gene)
            (tp, fp, fn), best_truth, best_pred = evaluate_row(gene, truth_solutions, pred_solutions)

            if fp == 0 and fn == 0:
                continue

            if is_d6:
                truth_clean = [a for a in best_truth if a != '5']
                pred_clean = [a for a in best_pred if a != '5']
            else:
                truth_clean = list(best_truth)
                pred_clean = list(best_pred)

            missed = []
            matched_pred = [False] * len(pred_clean)
            for t in truth_clean:
                found = False
                for j, p in enumerate(pred_clean):
                    if not matched_pred[j] and t == p:
                        matched_pred[j] = True
                        found = True
                        break
                if not found:
                    missed.append(t)
            extra = [pred_clean[j] for j in range(len(pred_clean)) if not matched_pred[j]]

            detail = ""
            if missed:
                detail += f" missed=[{','.join(missed)}]"
            if extra:
                detail += f" extra=[{','.join(extra)}]"

            print(f"  {name:<12} {gene:<10} {tool_name:<15} truth={best_truth} pred={best_pred} tp={tp} fp={fp} fn={fn}{detail}")


if __name__ == '__main__':
    main()