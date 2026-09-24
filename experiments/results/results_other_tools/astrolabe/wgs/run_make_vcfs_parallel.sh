#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT=/home/pouria/opt/src/astrolabe/runs/wgs
CONF_DIR=$RUN_ROOT/conf
OUT_DIR=$RUN_ROOT/out
LOG_DIR=$RUN_ROOT/logs

SAMPLES_LIST=$RUN_ROOT/samples_ready_7genes.list
BED3=$CONF_DIR/pgx7.bed3

REF_FASTA=/project/shared/aldy-data/Homo_sapiens_assembly19_1000genomes_decoy.fasta
CRAM_DIR=/project/shared/aldy-data/wgs

PAR=/cvmfs/soft.computecanada.ca/gentoo/2020/usr/bin/parallel
JOBS=8

mkdir -p "$OUT_DIR" "$LOG_DIR"
test -s "$SAMPLES_LIST"
test -s "$BED3"
test -s "$REF_FASTA"
test -s "${REF_FASTA}.fai"

# one-time silence
"$PAR" --citation >/dev/null 2>&1 || true

nohup "$PAR" --bar -j"$JOBS" \
  --joblog "$LOG_DIR/vcf_parallel.joblog" \
  --results "$LOG_DIR/vcf_parallel.results" \
  "$RUN_ROOT/make_one_vcf.sh" {} \
  :::: "$SAMPLES_LIST" \
  > "$LOG_DIR/nohup_vcf_parallel.out" 2>&1 &

echo "VCF parallel launched. Monitor:"
echo "  tail -f $LOG_DIR/nohup_vcf_parallel.out"
echo "  tail -f $LOG_DIR/vcf_parallel.joblog"
