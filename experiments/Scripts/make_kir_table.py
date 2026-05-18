

import argparse
import csv
import os
import re
from collections import defaultdict

GENES = [
    "KIR2DL1", "KIR2DL2", "KIR2DL3", "KIR2DL4", "KIR2DL5A", "KIR2DL5B",
    "KIR2DP1", "KIR2DS1", "KIR2DS2", "KIR2DS3", "KIR2DS4", "KIR2DS5",
    "KIR3DL1", "KIR3DL2", "KIR3DL3", "KIR3DP1", "KIR3DS1"
]


def parse_truth_cell(cell):
    """Parse ground truth cell like `0030204;`0030205 into sorted allele list (major only)."""
    if not cell or not cell.strip():
        return []
    return sorted([a.strip().lstrip("`")[:3] for a in cell.split(";") if a.strip()])


def format_alleles(alleles):
    """Format allele list: [] -> -/-, [a] -> a/-, [a,b] -> a/b, [a,b,c] -> a+b/c"""
    if not alleles:
        return "-/-"
    if len(alleles) == 1:
        return f"{alleles[0]}/-"
    if len(alleles) == 2:
        return f"{alleles[0]}/{alleles[1]}"
    return "+".join(alleles[:-1]) + "/" + alleles[-1]


def parse_log(path):
    """Extract predictions from log file. Returns dict: gene -> sorted allele list."""
    preds = defaultdict(list)
    try:
        with open(path) as f:
            content = f.read()
    except:
        return preds
    
    match = re.search(r'\[Alleles\]\s*\n(.*?)(?:\n\s*Command being timed|\Z)', content, re.DOTALL)
    if not match:
        return preds
    
    for line in match.group(1).strip().split('\n'):
        m = re.match(r'(KIR\w+)\*(\S+)', line.strip())
        if m:
            preds[m.group(1)].append(m.group(2))
    
    return {g: sorted(a) for g, a in preds.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth", required=True)
    parser.add_argument("--results", required=True)
    args = parser.parse_args()

    # Load ground truth
    truth = {}
    with open(args.truth) as f:
        for row in csv.DictReader(f):
            sid = row['ID'].replace('.bam', '')
            truth[sid] = {g: parse_truth_cell(row.get(g, '')) for g in GENES}

    # Output header
    print("ID,Gene,Ground_Truth,Prediction")

    for sid in sorted(truth.keys()):
        # Find log file
        log_path = None
        for pat in [f"{sid}.final.cram.log", f"{sid}.cram.log", f"{sid}.bam.log", f"{sid}.log"]:
            p = os.path.join(args.results, pat)
            if os.path.exists(p):
                log_path = p
                break
        
        preds = parse_log(log_path) if log_path else {}

        for gene in GENES:
            gt = format_alleles(truth[sid][gene])
            pr = format_alleles(preds.get(gene, []))
            print(f"{sid},{gene},{gt},{pr}")


if __name__ == "__main__":
    main()