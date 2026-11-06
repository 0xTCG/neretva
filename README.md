<p align="center">
  <img src="assets/logo.png" width="100" alt="Neretva logo">
</p>

# Neretva

**Neural Variational Inference for Allele-level Genotyping of Highly Polymorphic Genes**

Neretva is a unified framework that models the genotyping problem as a probabilistic latent variable model and employs auto-encoding variational Bayes (AEVB) for inference. It supports both CYP pharmacogene and KIR gene families, achieving competitive or improved accuracy over existing state-of-the-art tools.

> Zhou Q, Ahmadi SP, Numanagić I. *Neretva: Neural Variational Inference for Allele-level Genotyping of Highly Polymorphic Genes.* Under review, 2026.

---

## Supported Genes

| Family | Genes |
|--------|-------|
| **KIR** | All 17 KIR genes |
| **CYP** | CYP2B6, CYP2C8, CYP2C9, CYP2C19, CYP2D6, CYP3A5|

---

## Installation

### Requirements
- Python ≥ 3.9
- CUDA-compatible GPU (recommended)
- [minimap2](https://github.com/lh3/minimap2) (for KIR genotyping)
- [samtools](https://www.htslib.org/) (to index inputs)

### Setup
```bash
pip install git+https://github.com/0xTCG/neretva.git
```

> 💡 You can also use `uv pip install` for faster setup if you have `uv`.

> ⚠️ You might need to install git, gcc and python-devel (or equivalent packages) to compile the dependencies.

> ‼️ **Warning:** some dependencies (such as Aldy) might not work well in Conda environments. Use uv or pip instead.

### minimap2
```bash
curl -L https://github.com/lh3/minimap2/releases/download/v2.31/minimap2-2.31_x64-linux.tar.bz2 | tar -jxf -
./minimap2-2.31_x64-linux/minimap2 --version
```
or `conda install -c bioconda minimap2`. Pass the binary path to `--mapper`.


---

## Input

- KIR: BAM/CRAM aligned to **GRCh38**, coordinate-sorted and indexed. FASTQ/FASTA also accepted
- CYP: BAM/CRAM aligned to **hg19**, coordinate-sorted and indexed, with `--reference`

### Reference FASTA

| Family | Assembly | Download |
|--------|----------|----------|
| KIR | GRCh38, 1000 Genomes analysis set with decoy and HLA contigs | [`GRCh38_full_analysis_set_plus_decoy_hla.fa`](https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/technical/reference/GRCh38_reference_genome/GRCh38_full_analysis_set_plus_decoy_hla.fa) |
| CYP | GRCh37, Broad `assembly19` with decoy contigs | [`Homo_sapiens_assembly19_1000genomes_decoy.fasta`](https://storage.googleapis.com/gcp-public-data--broad-references/Homo_sapiens_assembly19_1000genomes_decoy/Homo_sapiens_assembly19_1000genomes_decoy.fasta) |

```bash
samtools faidx Homo_sapiens_assembly19_1000genomes_decoy.fasta
```

### Indexing the input
```bash
samtools index sample.cram   # or sample.bam
```

CRAM is decoded through htslib, which fetches reference sequences by MD5 and caches them under `~/.cache/hts-ref`. This needs network access on first use; to work offline, pre-populate the cache with samtools' `misc/seq_cache_populate.pl` from the FASTA above.

---

## Usage

### KIR genotyping
```bash
neretva kir --input <bam_path/fastq_path> --mapper <minimap2 path>
```

### Pharmacogene genotyping
```bash
neretva <gene> --input <bam_path> --reference <ref_fasta>
```

### Full options
```
neretva <gene> --input <bam> [options]
```

| Argument | Description |
|----------|-------------|
| `gene` | Gene to genotype: `kir`, `cyp2b6`, `cyp2c8`, `cyp2c9`, `cyp2c19`, `cyp2d6`, `cyp3a5`|
| `--input`| Path to input BAM/CRAM/FASTQ file (required) |
| `--reference`| Path to human reference genome FASTA (required for CYP) |
| `--threads` | Number of threads, default: 16 (KIR only) |
| `--seed` | Random seed, default: 42 |
| `--mapper` | Path to minimap2 binary (KIR only) |
| `--config` | Path to region config JSON, default: packaged `kir_regions.json` (KIR only) |

### Different assembly or loci

No assembly coordinate is hard-coded in the model. The loci Neretva pulls reads from live in [`neretva/data/kir_regions.json`](neretva/data/kir_regions.json):

| Key | Meaning |
|-----|---------|
| `regions` | intervals fetched from the input BAM/CRAM |
| `alt_contig_prefix` | ALT and decoy contigs to add, matched by name prefix |
| `cn_region` | single-copy interval used to set the per-copy coverage baseline |
| `assembly` | label only, for provenance |

To genotype against a different assembly, a different contig naming scheme (`19` rather than `chr19`), or a different gene family, copy the file, edit the coordinates, and pass it with `--config`:

```bash
neretva kir --input sample.bam --mapper $(which minimap2) --config my_regions.json
```

The allele database is held separately in `neretva/data/`, so a new gene family also needs its own reference allele set.

---

## Quick Test

A small test FASTA is included to verify installation:

```bash
neretva kir --input neretva/sample/test.fa --mapper $(which minimap2)
```

From an installed package rather than a clone:

```bash
neretva kir --mapper $(which minimap2) \
  --input "$(python -c 'import importlib.resources as r; print(r.files("neretva")/"sample"/"test.fa")')"
```

Expected output:
```
[Alleles]
KIR2DL1*003
KIR2DS3*001
```
## Reproducing Paper Results

See the [`experiments/`](experiments/experiment.ipynb) notebook for full benchmark reproduction, including all tool outputs and evaluation scripts.

---

## Citation

If you use Neretva in your research, please cite:

```bibtex
@article {Zhou2026.02.03.703582,
      author = {Zhou, Qinghui and Ahmadi, Seyed Pouria and Numanagi{\'c}, Ibrahim},
      title = {Neretva: Neural Variational Inference for Allele-level Genotyping of Highly Polymorphic Genes},
      elocation-id = {2026.02.03.703582},
      year = {2026},
      doi = {10.64898/2026.02.03.703582},
      publisher = {Cold Spring Harbor Laboratory},
      URL = {https://www.biorxiv.org/content/early/2026/03/16/2026.02.03.703582},
      eprint = {https://www.biorxiv.org/content/early/2026/03/16/2026.02.03.703582.full.pdf},
      journal = {bioRxiv}
}
```
---

## License

[MIT License](LICENSE)

---
