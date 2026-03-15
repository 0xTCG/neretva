# Neretva

## How to Run

### KIR
```bash
python kir.py --input [bam_path] > [out_path]
```

### CYP (Currently supports CYP2C8, CYP2C9, CYP2C19, CYP2D6)
```bash
python cyp.py CYP2C8 [bam_path] > [out_path]
# Or
python cyp.py CYP2C9 [bam_path] > [out_path]
python cyp.py CYP2C19 [bam_path] > [out_path]
python cyp.py CYP2D6 [bam_path] > [out_path]
# Or
./eval_CYP.sh
```
