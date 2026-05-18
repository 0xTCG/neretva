import sys
import csv


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

    print("=" * 95)
    print("Per Call (Sample-level)")
    print("=" * 95)
    print(f"{'Tool':<20} {'n':>5} {'Correct':>8} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print("-" * 95)

    for tool_name, call_col, m_col in tools:
        tp = 0
        fp = 0
        fn = 0
        n_truth_present = 0

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
                    fp += 1      # tool called something wrong
                if not truth_absent and call_absent:
                    fn += 1      # tool missed a present gene

        if n_truth_present == 0:
            continue
        acc = tp / n_truth_present
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        print(f"{tool_name:<20} {n_truth_present:>5} {tp:>8} {acc:>9.3f} {prec:>10.3f} {rec:>8.3f} {f1:>8.3f}")


if __name__ == '__main__':
    main()