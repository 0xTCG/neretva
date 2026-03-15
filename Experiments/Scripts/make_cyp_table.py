#%%
import sys
import csv
import os
import re
from collections import defaultdict

def parse_log_file(log_path):
    alleles = []
    in_alleles_section = False
    
    try:
        with open(log_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line == '[Alleles]':
                    in_alleles_section = True
                    continue
                if in_alleles_section:
                    if not line:
                        break
                    match = re.search(r'\*(\d+)', line)
                    if match:
                        alleles.append(match.group(1))
    except Exception as e:
        print(f"Error reading {log_path}: {e}")
        return None
    
    if not alleles:
        return None
    
    alleles_sorted = sorted(alleles, key=lambda x: int(x))
    return '/'.join(alleles_sorted)

def main(ground_csv, results_dir, output_csv):
    genes = ['CYP2C8', 'CYP2C9', 'CYP2C19', 'CYP2D6']
    
    # Build a lookup: (sample_id, gene) -> prediction
    predictions = {}
    
    for gene in genes:
        gene_dir = os.path.join(results_dir, gene)
        if not os.path.isdir(gene_dir):
            print(f"Warning: {gene_dir} not found")
            continue
        
        for filename in os.listdir(gene_dir):
            if filename.endswith('.log'):
                sample_id = filename[:-4]  # Remove .log
                log_path = os.path.join(gene_dir, filename)
                pred = parse_log_file(log_path)
                if pred:
                    predictions[(sample_id, gene)] = pred
    
    # Read ground truth and add predictions
    rows = []
    with open(ground_csv, 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        header.append('Prediction')
        rows.append(header)
        
        for row in reader:
            if len(row) < 2:
                continue
            sample_id = row[0].strip()
            gene = row[1].strip()
            
            pred = predictions.get((sample_id, gene), '')
            row.append(pred)
            rows.append(row)
    
    # Write output
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    
    print(f"Output written to {output_csv}")
    print(f"Total predictions found: {len(predictions)}")

if __name__ == '__main__':
    ground_csv = 'Ground_truth/cyp_ground.csv'
    results_dir = '../results/results_CYP_WGS'
    output_csv = 'Sheets/cyp_results_original.csv'
    
    main(ground_csv, results_dir, output_csv)

#%%