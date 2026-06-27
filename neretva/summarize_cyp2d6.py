import glob
import re
import sys
import csv


def extract_results(gene, input_folder):
    results = {}
    files = sorted(glob.glob(f'{input_folder}/*.log'))
    
    for f in files:
        id_ = f.split('/')[-1].replace('.log', '')
        with open(f) as log:
            content = log.read()
        matches = list(re.finditer(r'\[Alleles\]\s*\n', content))
        if matches:
            last_match = matches[-1]
            remaining = content[last_match.end():]
            lines = remaining.split('\n')
            
            alleles = []
            for line in lines:
                stripped = line.strip()
                if stripped.startswith(f'{gene}*'):
                    alleles.append(stripped.replace(f'{gene}*', ''))
                elif stripped == '':
                    continue  # Skip empty lines
                else:
                    break  # Stop at first non-allele, non-empty line
            
            results[id_] = sorted(alleles)
            print(f"DEBUG {id_}: {alleles}")
        else:
            results[id_] = []
            print(f"DEBUG {id_}: no [Alleles] found")
    
    return results


def evaluate(gene, input_folder, ground_file, output_file='summarized.csv'):
    """Extract results and evaluate against ground truth"""
    
    predictions = extract_results(gene, input_folder)
    
    ground = {}
    with open(ground_file, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            id_ = row['ID'].strip()
            if id_:
                ground[id_] = sorted(row['GeT-RM'].strip().split(';')) if row['GeT-RM'].strip() else []
    
    results = []
    for id_, gt_alleles in ground.items():
        pred_alleles = predictions.get(id_, [])
        correct = 1 if gt_alleles == pred_alleles else 0
        results.append([id_, ';'.join(gt_alleles), ';'.join(pred_alleles), correct])
    
    total = len(results)
    correct = sum(r[3] for r in results)
    accuracy = correct / total if total > 0 else 0
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ID', 'GroundTruth', 'Prediction', 'Correct'])
        writer.writerows(results)
        writer.writerow([])
        writer.writerow(['Summary'])
        writer.writerow(['Total', total])
        writer.writerow(['Correct', correct])
        writer.writerow(['Errors', total - correct])
        writer.writerow(['Accuracy', f'{accuracy:.2%}'])
    
    print(f"\nAccuracy: {correct}/{total} = {accuracy:.2%}")
    print(f"Errors: {total - correct}")
    
    for r in results:
        if r[3] == 0:
            print(f"  {r[0]}: GT={r[1]}, Pred={r[2]}")
    
    print(f"Output: {output_file}")



if __name__ == '__main__':
    gene = 'CYP2D6'  
    input_folder = sys.argv[1]  
    ground_file = 'cyp2d6_wgs.csv'
    output_file = sys.argv[2] if len(sys.argv) > 4 else 'summarized.csv'
    evaluate(gene, input_folder, ground_file, output_file)
    