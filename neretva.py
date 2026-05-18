import sys
import os

CYP_GENES = {"cyp2b6", "cyp2c8", "cyp2c9", "cyp2c19", "cyp2d6", "cyp3a5", "cyp4f2"}

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Neretva: genotyping of highly polymorphic genes")
    parser.add_argument("gene", help=f"Gene to genotype: kir, {', '.join(sorted(CYP_GENES))}")
    parser.add_argument("--input", "-i", required=True, help="Path to input BAM/CRAM file")
    parser.add_argument("--reference", "-r", help="Path to human reference genome FASTA")
    parser.add_argument("--threads", type=int, default=16, help="Number of threads)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--mapper", help="Path to minimap2 binary")
    args = parser.parse_args()

    gene = args.gene.lower()
    here = os.path.dirname(os.path.abspath(__file__))

    if gene == "kir":
        cmd = [sys.executable, os.path.join(here, "kir.py"),
               "--input", args.input,
               "--threads", str(args.threads),
               "--seed", str(args.seed)]
        if args.mapper:
            cmd += ["--mapper", args.mapper]
    elif gene in CYP_GENES:
        if not args.reference:
            parser.error("--reference is required for CYP genes")
        cmd = [sys.executable, os.path.join(here, "cyp.py"),
               "--gene", gene.upper(),
               "--input", args.input,
               "--reference", args.reference]
    else:
        parser.error(f"Unknown gene: {args.gene}. Supported: kir, {', '.join(sorted(CYP_GENES))}")

    os.execv(cmd[0], cmd)

if __name__ == "__main__":
    main()