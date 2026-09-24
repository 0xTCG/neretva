#!/usr/bin/env python3

import sys
import csv
import re
import numpy as np


def main():
    if len(sys.argv) < 2:
        print("Usage: python evaluate_cyp_multi.py results.csv [gene_prefix]")
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

    def compute_metrics(tool_name, call_col, m_col):
        tp = fp = fn = n = 0
        for row in rows:
            if m_col >= len(row):
                continue
            m = row[m_col].strip()
            if m not in ('0', '1'):
                continue
            n += 1
            truth_str = row[3].strip().strip('"')
            call_str = row[call_col].strip().strip('"') if call_col < len(row) else ''
            truth_absent = (not truth_str or truth_str == '-/-')
            call_absent = (not call_str or call_str == '-/-')
            if m == '1':
                tp += 1
            else:
                if not call_absent:
                    fp += 1
                if not truth_absent and call_absent:
                    fn += 1
        if n == 0:
            return None
        acc = tp / n
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        return n, tp, acc, prec, rec, f1

    print("=" * 95)
    print("Per Call (Sample-level)")
    print("=" * 95)
    print(f"{'Tool':<20} {'n':>5} {'Correct':>8} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print("-" * 95)

    seed_metrics = []
    tool_metrics = []
    for tool_name, call_col, m_col in tools:
        result = compute_metrics(tool_name, call_col, m_col)
        if result is None:
            continue
        n, tp, acc, prec, rec, f1 = result
        print(f"{tool_name:<20} {n:>5} {tp:>8} {acc:>9.3f} {prec:>10.3f} {rec:>8.3f} {f1:>8.3f}")
        if tp > 0:
            tool_metrics.append((tool_name, n, tp, acc, prec, rec, f1))
        if re.match(r'^Seed\d+$', tool_name):
            seed_metrics.append((n, tp, acc, prec, rec, f1))

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

    if seed_metrics:
        n_val = f"{means[0]:.0f}"
        correct_val = f"{means[1]:.1f}{{\\tiny$\\pm${stds[1]:.1f}}}"
        acc_val = f"{means[2]:.3f}{{\\tiny$\\pm${stds[2]:.3f}}}"
        prec_val = f"{means[3]:.3f}{{\\tiny$\\pm${stds[3]:.3f}}}"
        rec_val = f"{means[4]:.3f}{{\\tiny$\\pm${stds[4]:.3f}}}"
        f1_val = f"{means[5]:.3f}{{\\tiny$\\pm${stds[5]:.3f}}}"
        print(f"Neretva & {n_val} & {means[1]:.1f} & {means[2]:.3f} & {means[3]:.3f} & {means[4]:.3f} & {means[5]:.3f} \\\\")
        print(f"        &     & {{\\tiny$\\pm${stds[1]:.1f}}} & {{\\tiny$\\pm${stds[2]:.3f}}} & {{\\tiny$\\pm${stds[3]:.3f}}} & {{\\tiny$\\pm${stds[4]:.3f}}} & {{\\tiny$\\pm${stds[5]:.3f}}} \\\\")


if __name__ == '__main__':
    main()