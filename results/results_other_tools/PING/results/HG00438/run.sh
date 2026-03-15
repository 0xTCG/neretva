#!/bin/bash
set -euo pipefail

module load StdEnv/2020
module load gcc/9.3.0
module load r/4.3.1
module load samtools/1.17
module load bowtie2/2.5.1
module load bcftools/1.16
module load htslib/1.16

export R_LIBS_USER="$HOME/opt/R/library"

cd "$HOME/results/ping/runs/HG00438"
Rscript run_ping_HG00438.R > run.log 2>&1
