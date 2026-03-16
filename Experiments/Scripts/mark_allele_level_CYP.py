

import sys
import csv
import re
from collections import defaultdict
from itertools import product

CYP_GENES = {'CYP2C8', 'CYP2C9', 'CYP2C19', 'CYP2D6'}

CYP_SAMPLES = {
    "HG00276", "NA07348", "NA12006", "NA18524", "NA18868", "NA19095", "NA19226",
    "HG00436", "NA07357", "NA12145", "NA18526", "NA18942", "NA19109", "NA19239",
    "HG00589", "NA10831", "NA12156", "NA18540", "NA18952", "NA19122", "NA19789",
    "HG01190", "NA10847", "NA12717", "NA18544", "NA18959", "NA19143", "NA19819",
    "NA06991", "NA10851", "NA12813", "NA18552", "NA18966", "NA19147", "NA19908",
    "NA07000", "NA10854", "NA12873", "NA18564", "NA18973", "NA19174", "NA19917",
    "NA07019", "NA11832", "NA18484", "NA18565", "NA18980", "NA19176", "NA19920",
    "NA07029", "NA11839", "NA18509", "NA18617", "NA18992", "NA19178", "NA20296",
    "NA07055", "NA11993", "NA18518", "NA18855", "NA19003", "NA19207", "NA20509",
    "NA07056", "NA12003", "NA18519", "NA18861", "NA19007", "NA19213", "NA21781",
}

CYP_TOOLS = ['Neretva', 'ALDY', 'StellarPGX', 'Cyrius', 'stargazer', 'PYPGX', 'Astrolabe']


def normalize_allele(allele, gene=""):
    allele = allele.strip()
    if not allele:
        return allele
    if "." in allele:
        allele = allele.split(".")[0]
    if allele != "-":
        allele = re.sub(r'[A-Za-z]+$', '', allele)
    if allele.endswith("*"):
        allele = allele[:-1]
    if gene == "CYP2D6" and allele == "5":
        return "-"
    if gene == "CYP2C19" and allele == "38":
        return "1"
    return allele


def parse_alleles_cyp(s, gene=""):
    s = s.strip().strip('"')
    if not s or s == '-/-' or s == '.':
        return []
    if ';' in s:
        results = []
        for part in s.split(';'):
            results.extend(_parse_one_call(part.strip(), gene))
        return results
    return _parse_one_call(s, gene)


def _parse_one_call(s, gene):
    s = s.strip()
    if not s or s == '-/-':
        return []
    s = re.sub(r'(\d+)x(\d+)', lambda m: '+'.join([m.group(1)] * int(m.group(2))), s)
    if '(' in s:
        return _parse_with_parens(s, gene)
    return _parse_simple(s, gene)


def _parse_simple(s, gene):
    if '/' not in s:
        return []
    parts = s.split('/')
    if len(parts) != 2:
        return []
    alleles = []
    for part in parts:
        part = part.strip()
        if '+' in part:
            for a in part.split('+'):
                a = normalize_allele(a.strip(), gene)
                if a:
                    alleles.append(a)
        else:
            a = normalize_allele(part, gene)
            if a:
                alleles.append(a)
    return [tuple(sorted(alleles))]


def _parse_with_parens(s, gene):
    match = re.match(r'^([^/]+)/\(([^)]+)\)$', s)
    if match:
        left = match.group(1).strip()
        options = [o.strip() for o in match.group(2).split(',')]
        results = []
        for opt in options:
            results.extend(_parse_simple(f"{left}/{opt}", gene))
        return results
    match = re.match(r'^\(([^)]+)\)/([^/]+)$', s)
    if match:
        options = [o.strip() for o in match.group(1).split(',')]
        right = match.group(2).strip()
        results = []
        for opt in options:
            results.extend(_parse_simple(f"{opt}/{right}", gene))
        return results
    match = re.match(r'^\(([^)]+)\)/\(([^)]+)\)$', s)
    if match:
        left_opts = [o.strip() for o in match.group(1).split(',')]
        right_opts = [o.strip() for o in match.group(2).split(',')]
        results = []
        for l, r in product(left_opts, right_opts):
            results.extend(_parse_simple(f"{l}/{r}", gene))
        return results
    return _parse_simple(s, gene)


def get_correct(truth, pred, gene=""):
    remaining_pred = list(pred)
    tp = 0
    matched = [False] * len(truth)

    for p in list(remaining_pred):
        for i, t in enumerate(truth):
            if not matched[i] and p == t:
                matched[i] = True
                tp += 1
                remaining_pred.remove(p)
                break


    if gene == "CYP2D6" and len(remaining_pred) == 0:
        for i, t in enumerate(truth):
            if not matched[i] and t == '-':
                matched[i] = True
                tp += 1

    fn = sum(1 for m in matched if not m)
    fp = len(remaining_pred)
    return tp, fp, fn


def best_match(truth_solutions, pred_solutions, gene=""):
    best = (0, 999, 999)
    for truth in truth_solutions:
        for pred in pred_solutions:
            tp, fp, fn = get_correct(list(truth), list(pred), gene)
            if (fp + fn) < (best[1] + best[2]) or \
               ((fp + fn) == (best[1] + best[2]) and tp > best[0]):
                best = (tp, fp, fn)
    return best


def count_truth_alleles(truth_solutions):
    if not truth_solutions:
        return 0
    return len(truth_solutions[0])


def main():
    if len(sys.argv) < 2:
        print("Usage: python evaluate_cyp_allele.py results_with_M.csv", file=sys.stderr)
        sys.exit(1)

    path = sys.argv[1]

    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader)

    all_tools = []
    i = 4
    while i < len(header) - 1:
        if header[i] and header[i] != 'M' and header[i+1] == 'M':
            all_tools.append((header[i], i, i+1))
            i += 2
        else:
            i += 1

    tool_map = {n: (n, c, m) for n, c, m in all_tools}
    tools = [tool_map[tn] for tn in CYP_TOOLS if tn in tool_map]

    rows = []
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if not row or not row[0]:
                continue
            if row[0].strip() not in CYP_SAMPLES:
                continue
            if len(row) > 1 and row[1] != 'WGS':
                continue
            gene = row[2] if len(row) > 2 else ''
            if gene not in CYP_GENES:
                continue
            rows.append(row)

    genes_order = []
    gene_rows = defaultdict(list)
    for row in rows:
        gene = row[2]
        if gene not in gene_rows:
            genes_order.append(gene)
        gene_rows[gene].append(row)

    gene_sort = ['CYP2C8', 'CYP2C9', 'CYP2C19', 'CYP2D6']
    genes_order = [g for g in gene_sort if g in gene_rows]

    print(f"Loaded {len(rows)} CYP WGS rows, {len(tools)} tools, {len(genes_order)} genes\n")

    for gene in genes_order:
        n_total = 0
        for row in gene_rows[gene]:
            truth_str = row[3].strip().strip('"')
            truth_solutions = parse_alleles_cyp(truth_str, gene)
            n_total += count_truth_alleles(truth_solutions)

        print("=" * 95)
        print(f"Per Allele — {gene}")
        print("=" * 95)
        print(f"{'Tool':<20} {'n':>5} {'Correct':>8} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8}")
        print("-" * 95)

        for tool_name, call_col, m_col in tools:
            total_tp = total_fp = total_fn = 0

            for row in gene_rows[gene]:
                truth_str = row[3].strip().strip('"')
                truth_solutions = parse_alleles_cyp(truth_str, gene)

                if not truth_solutions:
                    continue

                if m_col >= len(row) or not row[m_col].strip():
                    continue
                m = row[m_col].strip()
                if m == '#':
                    continue

                call_str = row[call_col].strip().strip('"') if call_col < len(row) else ''
                pred_solutions = parse_alleles_cyp(call_str, gene)

                if not pred_solutions:
                    total_fn += count_truth_alleles(truth_solutions)
                    continue

                tp, fp, fn = best_match(truth_solutions, pred_solutions, gene)
                total_tp += tp
                total_fp += fp
                total_fn += fn

            acc = total_tp / n_total if n_total > 0 else 0
            prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
            rec = total_tp / n_total if n_total > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            print(f"{tool_name:<20} {n_total:>5} {total_tp:>8} {acc:>9.3f} {prec:>10.3f} {rec:>8.3f} {f1:>8.3f}")

        print()


if __name__ == '__main__':
    main()