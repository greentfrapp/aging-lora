# Barreiro / Randolph 2021 (REJECTED)

**Status**: Evaluated as a Case 1 training cohort; **rejected 2026-04-24** because (a) GEO has no donor-age metadata, and (b) the release is genotype-multiplexed without demultiplexing files in the archive. Terekhova 2023 was promoted from the deferred Case 3 pool to replace it.

## Paper and accession

**Paper.** Randolph, Fiege, Thielen, Mickelson, Shiratori et al., "Single-cell RNA-sequencing reveals pervasive but highly cell type-specific genetic ancestry effects on the response to viral infection," *Science* 374 (6571), 2021. PubMed 34822289. Senior author: Luis Barreiro (University of Chicago). (Note: early scaffold referred to this as "Nedelec 2022 Nature Immunology" — wrong attribution; corrected.)

**Accession**: GEO `GSE162632`

- Series title: "Single-cell RNA-sequencing reveals pervasive but highly cell type-specific genetic ancestry effects on the response to viral infection"
- Project: PRJNA682434
- Platform: GPL24676 (Illumina NovaSeq 6000)
- 292 total GSMs across scRNA-seq, bulk RNA-seq, time-course RNA-seq
- Supplementary `GSE162632_RAW.tar` (789 MB) contains 30 scRNA-seq GSMs as standard 10x MTX triplets (barcodes.tsv.gz + features.tsv.gz + matrix.mtx.gz each)

## What we downloaded

Before the rejection decision, we successfully downloaded and extracted `GSE162632_RAW.tar`:

- 30 GSMs × 3 files each = 90 files
- Naming pattern: `GSMxxxxxxx_B{1..15}-c{1|2}-10X_{matrix.mtx,barcodes.tsv,features.tsv}.gz`
- Total ~1.6 GB on disk at `data/cohorts/raw/GSE162632/`

## Why we rejected it (the two blockers)

### 1. No donor-age metadata in GEO

Direct GEO query via `GEOparse` returned:

```
GSE162632 characteristics_ch1 keys (across first 30 GSMs):
  ['cell type', 'treatment']
```

Every sample has only two characteristics:
- `cell type: peripheral blood mononuclear cells`
- `treatment: mock- and IAV-infected`

No `age`, `donor_id`, `sex`, or any demographic information. The Randolph 2021 paper focuses on genetic ancestry effects on infection response — donor demographics were likely withheld from the public deposit for identifiability, which is common for paired-genotype scRNA-seq studies. The sc-ImmuAging paper presumably obtained ages by direct request to the Barreiro lab for the cohort they cite as "42 healthy European donors" (different from the 15 donors visible in the public RAW release).

### 2. Genotype-multiplexed without demux files

The series `overall_design` states:

> *Multiplexed single-cell RNA expression profiles of control (mock-infected) and influenza A virus (IAV)-infected peripheral blood mononuclear cells (PBMCs) collected from African and European American donors.*

Each capture (each GSM) pools:
- Multiple donors (requires genotype-VCF-based demultiplexing via Demuxlet / Vireo / souporcell)
- Both conditions (mock + IAV — requires a second-stage demultiplexing)

The `-c1`/`-c2` suffix in the filename was initially interpreted as mock vs. IAV condition, but the GEO `treatment` field shows `mock- and IAV-infected` for both c1 and c2 captures — so c1/c2 are two technical captures of the same pooled donor/condition mix, not condition-separated samples. Without the paper's genotype VCFs (not in the archive), per-donor and per-condition assignment is impossible.

## Unblocking requirements

Revival would require:
1. Email the Barreiro lab for per-donor ages (`barreirolabchicago@gmail.com` or Luis Barreiro directly at U. Chicago)
2. Obtain genotype VCFs (not in the public archive)
3. Run Demuxlet / Vireo / souporcell for per-cell donor assignment
4. A second-stage demultiplex for mock vs. IAV condition

Estimated effort: **multiple weeks**. Cost/benefit flipped unfavourably because:
- Barreiro/Randolph's 15-donor visible cohort is below the 80-donor LOCO-primary threshold — at best it would have been an exploratory fold.
- Terekhova 2023 offers 166 donors with public ages and no demultiplexing required — moves a fold from exploratory to primary status.

So we dropped Barreiro and promoted Terekhova. See `FUTURE_WORK.md` "Barreiro / Randolph 2021 (GSE162632) revival" for the revisit triggers.

## Known issues / caveats beyond the blockers

- **Attribution error in early scaffold**: the first version of this project referenced the Barreiro 2022 cohort as "Nedelec 2022 Nature Immunology." That was wrong — the actual paper is Randolph 2021 *Science*. Nedelec 2022 is a different paper from the same lab. Corrected throughout `roadmap/`, `HUMAN_TASKS.md`, and the download scripts.
- **The sc-ImmuAging paper's "42 healthy European donors" figure** does not match the 15 B* donors visible in the public RAW archive. sc-ImmuAging likely used a lab-internal demultiplexed release, not the public GEO deposit. Cannot be reproduced from public data alone.

## Role in the study — none

Not included. Files remain on disk at `data/cohorts/raw/GSE162632/` (1.6 GB extracted + 789 MB tar) pending user deletion. Delete once the three-cohort Case 1 pipeline is fully validated (conservative — don't delete while fallback options are still being evaluated).

## Related references

- Randolph et al. 2021 *Science* 374, DOI 10.1126/science.abg0928 — the actual paper
- `FUTURE_WORK.md` — revival triggers documented
- `HUMAN_TASKS.md #1` — early (historical) download task; marked complete as "files on disk, cohort dropped"
