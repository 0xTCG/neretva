#!/usr/bin/env python3
"""
Evaluate KIR HPRC results from CSV with pre-computed M (match) columns.
Computes mean ± std across Seed* and Neretva_JSD_* tools.
"""

import sys
import csv
import re
import numpy as np


def main():
    if len(sys.argv) < 2:
        print("Usage: python evaluate_csv.py results.csv")
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
            if len(row) > 1 and row[1] != 'HPRC':
                continue
            if len(row) > 2 and not row[2].startswith('KIR'):
                continue
            rows.append(row)

    print(f"Loaded {len(rows)} KIR HPRC rows, {len(tools)} tools\n")

    def compute_metrics(tool_name, call_col, m_col):
        tp = fp = fn = n_truth_present = 0
        for row in rows:
            if m_col >= len(row):
                continue
            m = row[m_col].strip()
            if not m:
                continue
            truth_str = row[3].strip().strip('"')
            call_str = row[call_col].strip().strip('"') if call_col < len(row) else ''
            truth_absent = (not truth_str or truth_str == '-/-')
            call_absent = (not call_str or call_str == '-/-')
            if m == '#':
                continue
            if not truth_absent:
                n_truth_present += 1
            if m == '1':
                tp += 1
            elif m == '0':
                if not call_absent:
                    fp += 1
                if not truth_absent and call_absent:
                    fn += 1
        if n_truth_present == 0:
            return None
        acc = tp / n_truth_present
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        return n_truth_present, tp, acc, prec, rec, f1

    print("=" * 95)
    print("Per Call (Sample-level)")
    print("=" * 95)
    print(f"{'Tool':<20} {'n':>5} {'Correct':>8} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print("-" * 95)

    groups = {
        'Neretva': (r'^Seed\d+$', []),
        'w/o JSD': (r'^Neretva_JSD_\d+$', []),
    }

    for tool_name, call_col, m_col in tools:
        result = compute_metrics(tool_name, call_col, m_col)
        if result is None:
            continue
        n, tp, acc, prec, rec, f1 = result
        print(f"{tool_name:<20} {n:>5} {tp:>8} {acc:>9.3f} {prec:>10.3f} {rec:>8.3f} {f1:>8.3f}")
        for group_label, (pattern, metrics_list) in groups.items():
            if re.match(pattern, tool_name):
                metrics_list.append((n, tp, acc, prec, rec, f1))

    for group_label, (pattern, group_metrics) in groups.items():
        if not group_metrics:
            continue
        arr = np.array(group_metrics)
        means = np.mean(arr, axis=0)
        stds = np.std(arr, axis=0)
        labels = ['n', 'Correct', 'Accuracy', 'Precision', 'Recall', 'F1']
        print("\n" + "=" * 55)
        print(f"{group_label} Summary (n={len(group_metrics)} seeds)")
        print("=" * 55)
        for label, m, s in zip(labels, means, stds):
            if label == 'n':
                print(f"  {label:<12} {m:.0f}")
            elif label == 'Correct':
                print(f"  {label:<12} {m:.1f} ± {s:.1f}")
            else:
                print(f"  {label:<12} {m:.3f} ± {s:.3f}")

        print("\nLaTeX:")
        print(f"{group_label} & {means[0]:.0f} & {means[1]:.1f} & {means[2]:.3f} & {means[3]:.3f} & {means[4]:.3f} & {means[5]:.3f} \\\\")
        print(f"        &     & {{\\tiny$\\pm${stds[1]:.1f}}} & {{\\tiny$\\pm${stds[2]:.3f}}} & {{\\tiny$\\pm${stds[3]:.3f}}} & {{\\tiny$\\pm${stds[4]:.3f}}} & {{\\tiny$\\pm${stds[5]:.3f}}} \\\\")


if __name__ == '__main__':
    main()