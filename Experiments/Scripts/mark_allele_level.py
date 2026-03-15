import sys
import csv
from itertools import product


def parse_alleles(s):
    s = s.strip().strip('"')
    if not s or s == '-/-' or s == '.':
        return [[]]

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
                for sol in _parse_single(part):
                    solutions.append(sol)
            return solutions if solutions else [[]]

    return _parse_single(s)


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
        if not sp or sp == '-':
            continue
        if sp.startswith('(') and sp.endswith(')'):
            alts = [x.strip()[:3] for x in sp[1:-1].split(',')]
            expanded.append(alts)
        else:
            sub_alleles = []
            for pp in sp.split('+'):
                pp = pp.strip()
                if pp and pp != '-':
                    sub_alleles.append(pp[:3])
            for a in sub_alleles:
                expanded.append([a])

    if not expanded:
        return [[]]

    solutions = []
    for combo in product(*expanded):
        solutions.append(list(combo))
    return solutions


def get_correct(truth, pred):
    remaining_truth = list(truth)
    remaining_pred = list(pred)

    tp = 0
    matched_truth = [False] * len(remaining_truth)

    for p in list(remaining_pred):
        for i, t in enumerate(remaining_truth):
            if not matched_truth[i] and p == t:
                matched_truth[i] = True
                tp += 1
                remaining_pred.remove(p)
                break

    fn = sum(1 for m in matched_truth if not m)
    fp = len(remaining_pred)
    return tp, fp, fn


def best_solution_allele(truth, solutions):
    best = (0, 999, 999)
    for sol in solutions:
        tp, fp, fn = get_correct(truth, sol)
        if (fp + fn) < (best[1] + best[2]) or ((fp + fn) == (best[1] + best[2]) and tp > best[0]):
            best = (tp, fp, fn)
    return best


def main():
    if len(sys.argv) < 2:
        print("Usage: python evaluate_allele_from_m.py results_with_M.csv")
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

    allele_stats = {name: [0, 0, 0] for name, _, _ in tools}

    for row in rows:
        truth_str = row[3].strip().strip('"')
        truth_solutions = parse_alleles(truth_str)
        truth = truth_solutions[0]

        for tool_name, call_col, m_col in tools:
            if m_col >= len(row):
                continue
            m = row[m_col].strip()
            call_str = row[call_col].strip() if call_col < len(row) else ''

            if not m:
                continue

            if m == '#':
                continue

            pred_solutions = parse_alleles(call_str)
            tp, fp, fn = best_solution_allele(truth, pred_solutions)

            allele_stats[tool_name][0] += tp
            allele_stats[tool_name][1] += fp
            allele_stats[tool_name][2] += fn

    print("=" * 95)
    print("Per Allele")
    print("=" * 95)
    print(f"{'Tool':<20} {'n':>5} {'Correct':>8} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print("-" * 95)

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


if __name__ == '__main__':
    main()