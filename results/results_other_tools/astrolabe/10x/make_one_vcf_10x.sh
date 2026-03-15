#!/usr/bin/env bash
set -euo pipefail
S="$1"

module purge >/dev/null 2>&1 || true
module load StdEnv/2020 gcc/9.3.0 >/dev/null
module load htslib/1.16 >/dev/null
module load bcftools/1.16 >/dev/null

RUN_ROOT=/home/pouria/opt/src/astrolabe/runs/10x
CONF_DIR=$RUN_ROOT/conf
OUT_DIR=$RUN_ROOT/out
LOG_DIR=$RUN_ROOT/logs
BAM_DIR=$RUN_ROOT/bams_local

BED3=$CONF_DIR/pgx7.bed3
REF=/project/shared/aldy-data/Homo_sapiens_assembly19_1000genomes_decoy.fasta
BAM=$BAM_DIR/${S}.bam

VCF_RAW=$OUT_DIR/${S}.pgx7.raw.bcf
VCF_UNSORT=$OUT_DIR/${S}.pgx7.vcf.gz
VCF_SORT=$OUT_DIR/${S}.pgx7.sorted.vcf.gz

# اگر قبلاً ساخته شده، رد شو
if [[ -s "$VCF_SORT" && -s "${VCF_SORT}.tbi" ]]; then
  echo "[VCF EXISTS] $S"
  exit 0
fi

# sanity
test -s "$BED3"
test -s "$REF"
test -s "${REF}.fai"
test -s "$BAM"
test -s "${BAM}.bai"

rm -f "$VCF_RAW" "$VCF_UNSORT" "${VCF_UNSORT}.tbi" "$VCF_SORT" "${VCF_SORT}.tbi"

bcftools mpileup -f "$REF" -Ou -R "$BED3" "$BAM" --threads 2 \
| bcftools call -mv -Ob --threads 2 -o "$VCF_RAW"

bcftools view -i "QUAL>=20" "$VCF_RAW" -Oz -o "$VCF_UNSORT"
bcftools sort -Oz -o "$VCF_SORT" "$VCF_UNSORT"
tabix -f -p vcf "$VCF_SORT"

echo "[VCF DONE] $S"
