#!/usr/bin/env python3
"""
Neretva unified entry point.

Usage:
    python neretva.py kir <bam_path>
    python neretva.py cyp2c8 <bam_path>
    python neretva.py cyp2c9 <bam_path>
    python neretva.py cyp2c19 <bam_path>
    python neretva.py cyp2d6 <bam_path>
"""

import sys
import os
import subprocess

CYP_GENES = {"cyp2c8", "cyp2c9", "cyp2c19", "cyp2d6"}

def usage():
    print(__doc__.strip())
    sys.exit(1)

def main():
    if len(sys.argv) < 3:
        usage()

    gene = sys.argv[1].lower()
    bam_path = sys.argv[2]

    here = os.path.dirname(os.path.abspath(__file__))

    if gene == "kir":
        cmd = [sys.executable, os.path.join(here, "kir.py"), "--input", bam_path]
    elif gene in CYP_GENES:
        cmd = [sys.executable, os.path.join(here, "cyp.py"), gene.upper(), bam_path]
    else:
        print(f"Unknown gene: {sys.argv[1]}")
        print(f"Supported: kir, {', '.join(sorted(CYP_GENES))}")
        sys.exit(1)

    os.execv(cmd[0], cmd) 

if __name__ == "__main__":
    main()