#!/usr/bin/env bash
set -euo pipefail

### ---- paths ----
ASTRO_DIR=/home/pouria/opt/src/astrolabe/astrolabe-0.8.7.2
RUNNER=$ASTRO_DIR/run-astrolabe.sh

RUN_ROOT=/home/pouria/opt/src/astrolabe/runs/wgs
CONF_DIR=$RUN_ROOT/conf
OUT_DIR=$RUN_ROOT/out
LOG_DIR=$RUN_ROOT/logs

SAMPLES_LIST=$RUN_ROOT/samples_ready_7genes.list   # 70 samples (one per line)
GENES_LIST=$RUN_ROOT/genes7.list                   # 7 genes (one per line)

REF_FASTA=/project/shared/aldy-data/Homo_sapiens_assembly19_1000genomes_decoy.fasta
CRAM_DIR=/project/shared/aldy-data/wgs

THREADS_PER_JOB=1      # Astrolabe خودش threads داره، ولی بهتره job-level کنترل کنیم
JOBS=8                 # تعداد job همزمان (با منابع سرور تنظیم کن)

### ---- load modules (bcftools + tabix + parallel) ----
module purge || true
module load StdEnv/2020 gcc/9.3.0
module load bcftools/1.16
module load htslib/1.13
module load parallel

mkdir -p "$CONF_DIR" "$OUT_DIR" "$LOG_DIR"

### ---- sanity checks ----
test -x "$RUNNER"
test -f "$REF_FASTA"
test -f "${REF_FASTA}.fai"
test -f "$SAMPLES_LIST"
test -f "$GENES_LIST"

### ---- build astrolabe.ini (IMPORTANT: rois.file should be prefix, astrolabe adds ref) ----
CONF="$CONF_DIR/astrolabe.ini"
cat > "$CONF" <<EOF
astrolabe.threads = ${THREADS_PER_JOB}
astrolabe.sens = 0.99
astrolabe.spec = 0.99

astrolabe.rois.source = file
# IMPORTANT: astrolabe appends ref (e.g. GRCh37) => it will read ${ASTRO_DIR}/etc/GRCh37
astrolabe.rois.file = ${ASTRO_DIR}/etc/

astrolabe.warehouse.file =

astrolabe.bam_qual.threshold = 2
astrolabe.bam_depth.threshold = 10
astrolabe.vcf_coverage.threshold = 0.75

roi.CYP2D6.conf = ${ASTRO_DIR}/etc/GRCh37/CYP2D6/roi.CYP2D6.ini
roi.CYP2D6.mode = wgs
EOF

### ---- per-sample prep: build pgx7 VCF once per sample ----
make_sample_vcf() {
  local sample="$1"
  local cram="${CRAM_DIR}/${sample}.wgs.cram"

  local bed3="${CONF_DIR}/pgx7.bed3"
  local vcf_raw="${OUT_DIR}/${sample}.pgx7.raw.bcf"
  local vcf_unsorted="${OUT_DIR}/${sample}.pgx7.vcf.gz"
  local vcf_sorted="${OUT_DIR}/${sample}.pgx7.sorted.vcf.gz"

  # اگر قبلاً ساخته شده، دوباره نساز
  if [[ -s "$vcf_sorted" && -s "${vcf_sorted}.tbi" ]]; then
    echo "[VCF] exists: $sample"
    return 0
  fi

  if [[ ! -f "$bed3" ]]; then
    echo "ERROR: missing $bed3 . First create it (pgx7.bed3)." >&2
    return 1
  fi

  echo "[VCF] building: $sample"
  rm -f "$vcf_raw" "$vcf_unsorted" "${vcf_unsorted}.tbi" "$vcf_sorted" "${vcf_sorted}.tbi"

  bcftools mpileup -f "$REF_FASTA" -Ou -R "$bed3" "$cram" --threads 4 \
    | bcftools call -mv -Ob --threads 4 -o "$vcf_raw"

  bcftools view -i "QUAL>=20" "$vcf_raw" -Oz -o "$vcf_unsorted"
  bcftools sort -Oz -o "$vcf_sorted" "$vcf_unsorted"
  tabix -f -p vcf "$vcf_sorted"
}

### ---- per-gene astrolabe call ----
run_one() {
  local sample="$1"
  local gene="$2"

  local cram="${CRAM_DIR}/${sample}.wgs.cram"
  local vcf_sorted="${OUT_DIR}/${sample}.pgx7.sorted.vcf.gz"
  local intervals="${CONF_DIR}/pgx7.intervals"

  local out="${OUT_DIR}/${sample}_${gene}.astrolabe.txt"
  local log="${LOG_DIR}/${sample}_${gene}.astrolabe.log"

  if [[ -s "$out" ]]; then
    echo "[SKIP] $sample $gene (exists)"
    return 0
  fi

  if [[ ! -s "$vcf_sorted" || ! -s "${vcf_sorted}.tbi" ]]; then
    echo "[ERR] missing VCF for $sample ($vcf_sorted)" >&2
    return 2
  fi

  bash "$RUNNER" \
    -conf "$CONF" \
    -ref GRCh37 \
    -inputVCF "$vcf_sorted" \
    -inputBam "$cram" \
    -skipVcfQC -skipBamQC \
    -outFile "$out" \
    -intervals "$intervals" \
    -fasta "$REF_FASTA" \
    -targets "$gene" \
    > "$log" 2>&1

  echo "[DONE] $sample $gene"
}

export -f run_one
export RUNNER ASTRO_DIR CONF_DIR OUT_DIR LOG_DIR REF_FASTA CRAM_DIR CONF

### ---- ensure bed3 + intervals exist ----
# اگر pgx7.bed3/intervals رو قبلاً ساختی، این بخش کاری نداره.
if [[ ! -f "${CONF_DIR}/pgx7.bed3" ]]; then
  echo "ERROR: ${CONF_DIR}/pgx7.bed3 not found." >&2
  echo "Create it from your pgx7.bed first (see notes below)." >&2
  exit 1
fi

if [[ ! -f "${CONF_DIR}/pgx7.intervals" ]]; then
  echo "ERROR: ${CONF_DIR}/pgx7.intervals not found." >&2
  exit 1
fi

### ---- build VCFs for all samples first (sequential, easier to debug) ----
echo "== Step 1: build per-sample pgx7 VCFs =="
while read -r s; do
  [[ -z "$s" ]] && continue
  make_sample_vcf "$s"
done < "$SAMPLES_LIST"

### ---- create job table: sample<tab>gene ----
JOB_TSV="${CONF_DIR}/jobs_wgs_7x70.tsv"
: > "$JOB_TSV"
while read -r s; do
  [[ -z "$s" ]] && continue
  while read -r g; do
    [[ -z "$g" ]] && continue
    printf "%s\t%s\n" "$s" "$g" >> "$JOB_TSV"
  done < "$GENES_LIST"
done < "$SAMPLES_LIST"

echo "== Step 2: run astrolabe with GNU parallel =="
echo "Jobs: $(wc -l < "$JOB_TSV")  |  Parallel workers: $JOBS"
parallel --colsep '\t' -j "$JOBS" --joblog "${LOG_DIR}/parallel.joblog" --lb \
  'run_one {1} {2}' :::: "$JOB_TSV"

echo "== All done =="
