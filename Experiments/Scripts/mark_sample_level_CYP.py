

import sys
import csv
from collections import defaultdict

CYP_GENES = {'CYP2C8', 'CYP2C9', 'CYP2C19', 'CYP2D6'}

CYP_TOOLS = ['Neretva', 'ALDY', 'StellarPGX', 'Cyrius', 'stargazer', 'PYPGX', 'Astrolabe']

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


def main():
    if len(sys.argv) < 2:
        print("Usage: python evaluate_cyp_csv.py results.csv")
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
        print("=" * 95)
        print(f"Per Call (Sample-level) — {gene}")
        print("=" * 95)
        print(f"{'Tool':<20} {'n':>5} {'Correct':>8} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8}")
        print("-" * 95)

        for tool_name, call_col, m_col in tools:
            tp = fp = fn = n_truth_present = 0

            for row in gene_rows[gene]:
                truth_str = row[3].strip().strip('"')
                truth_absent = (not truth_str or truth_str == '-/-')

                if not truth_absent:
                    n_truth_present += 1

                if m_col >= len(row):
                    continue
                m = row[m_col].strip()
                if not m or m == '#':
                    continue

                call_str = row[call_col].strip().strip('"') if call_col < len(row) else ''
                call_absent = (not call_str or call_str == '-/-')

                if m == '1':
                    tp += 1
                elif m == '0':
                    if not call_absent:
                        fp += 1
                    if not truth_absent and call_absent:
                        fn += 1

            if n_truth_present == 0 and fp == 0:
                continue
            acc = tp / n_truth_present if n_truth_present > 0 else 0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            print(f"{tool_name:<20} {n_truth_present:>5} {tp:>8} {acc:>9.3f} {prec:>10.3f} {rec:>8.3f} {f1:>8.3f}")

        print()


if __name__ == '__main__':
    main()