#!/usr/bin/env bash
set -euo pipefail
sample="$1"
gene="$2"

RUN_ROOT=/home/pouria/opt/src/astrolabe/runs/wgs
OUT_DIR=$RUN_ROOT/out
LOG_DIR=$RUN_ROOT/logs
CONF_DIR=$RUN_ROOT/conf

ASTRO_DIR=/home/pouria/opt/src/astrolabe/astrolabe-0.8.7.2
RUNNER=$ASTRO_DIR/run-astrolabe.sh
CONF=$CONF_DIR/astrolabe.ini

REF_FASTA=/project/shared/aldy-data/Homo_sapiens_assembly19_1000genomes_decoy.fasta
CRAM_DIR=$RUN_ROOT/crams_local
INTERVALS=$CONF_DIR/pgx7.intervals

cram="${CRAM_DIR}/${sample}.wgs.cram"
vcf_sorted="${OUT_DIR}/${sample}.pgx7.sorted.vcf.gz"

out="${OUT_DIR}/${sample}_${gene}.astrolabe.txt"
log="${LOG_DIR}/${sample}_${gene}.astrolabe.log"

if [[ -s "$out" ]]; then
  echo "[SKIP] $sample $gene"
  exit 0
fi

test -s "$cram"
test -s "${cram}.crai"
test -s "$vcf_sorted"
test -s "${vcf_sorted}.tbi"
test -s "$CONF"
test -s "$INTERVALS"

bash "$RUNNER" \
  -conf "$CONF" \
  -ref GRCh37 \
  -inputVCF "$vcf_sorted" \
  -inputBam "$cram" \
  -skipVcfQC -skipBamQC \
  -outFile "$out" \
  -intervals "$INTERVALS" \
  -fasta "$REF_FASTA" \
  -targets "$gene" \
  > "$log" 2>&1

echo "[DONE] $sample $gene"
