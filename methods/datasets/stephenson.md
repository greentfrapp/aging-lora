# Stephenson 2021 COVID-19 PBMC

## Paper and accessions

**Paper.** Stephenson et al., "Single-cell multi-omics analysis of the immune response in COVID-19," *Nature Medicine* 27, 904–916 (2021). DOI: [10.1038/s41591-021-01329-2](https://www.nature.com/articles/s41591-021-01329-2). Senior author: Muzlifah Haniffa (Newcastle).

**Accessions** (same dataset, multiple mirrors — this is precisely the issue the leakage audit had to navigate):

- **CellxGene** collection `ddfad306-714d-4cc0-9985-d9072820c530` → dataset `c17079d3-204f-487e-bc54-d63bb947a5a2` → h5ad `c17079d3-204f-487e-bc54-d63bb947a5a2.h5ad` **← the source we use**
- **ArrayExpress** `E-MTAB-10026` — canonical European-style accession; referenced in Methods and most citations
- **HCA** project `Covid19PBMC` (native HCA Project ID) — this is how **scFoundation** ingested the data in its pretraining corpus (caught 2026-04-24 in MOESM5 row 81; caused the leakage-audit overlap we'd otherwise have missed)

## Access

Direct HTTPS via CellxGene, no auth. ~6.6 GB download.

```
curl -L -o c17079d3-204f-487e-bc54-d63bb947a5a2.h5ad \
  https://datasets.cellxgene.cziscience.com/c17079d3-204f-487e-bc54-d63bb947a5a2.h5ad
```

Destination: `data/cohorts/raw/stephenson_covid_portal/c17079d3-204f-487e-bc54-d63bb947a5a2.h5ad`.

## Schema (as shipped by CellxGene)

- **Shape**: 647,366 cells × 24,245 genes
- **`.obs` columns** (38 total): patient QC (`n_genes`, `total_counts_mt`, `pct_counts_mt`), clinical (`Collection_Day`, `Swab_result`, `Status`, `Worst_Clinical_Status`, `Outcome`, `Days_from_onset`, `time_after_LPS`, `Smoker`), `Site`, and CellxGene-canonical (`donor_id`, `cell_type`, `assay`, `disease`, `sex`, `development_stage`, `self_reported_ethnicity`, `is_primary_data`, and all `*_ontology_term_id` variants)
- **`.var` columns**: `feature_is_filtered`, `feature_name`, `feature_reference`, `feature_biotype`, `feature_length`, `feature_type` (Ensembl IDs as `var_names`)
- **`.raw.X`**: raw integer counts (CellxGene convention)
- **`.disease` values**: `COVID-19` (majority), `respiratory system disorder`, `normal`
- **Assay**: `10x 3' transcription profiling` (single value)

## Healthy filter

We keep only cells where `disease == 'normal'`. This drops 542,443 cells (83% of the release is COVID+ patients), leaving **104,923 healthy-donor cells from 29 donors**.

## Age precision — mixed

**Critical finding**: the CellxGene release carries age in `obs['development_stage']` with two formats mixed across donors:

- **Exact year** for 11 donors (e.g. `"21-year-old stage"`, `"44-year-old stage"`, `"62-year-old stage"`) → parses via regex to integer years.
- **Decade bin** for 18 donors (e.g. `"third decade stage"`, `"fourth decade stage"`, `"sixth decade stage"`) → midpoint substitution: third → 25, fourth → 35, fifth → 45, sixth → 55, seventh → 65, eighth → 75, ninth → 85.

Per-donor distribution of healthy Stephenson:
```
fourth decade stage     7 donors  → 35.0 yr midpoint
third decade stage      6 donors  → 25.0
sixth decade stage      4 donors  → 55.0
fifth decade stage      1         → 45.0
exact (distinct years)  11 donors → 21, 38, 40, 44, 57, 58, 62, 63, 64, 70, 73
```

The harmonizer writes `obs['age_precision']` ∈ {`exact`, `decade`} per cell so downstream code can down-weight or exclude decade-coded donors from calibration analyses. Per `methods/pretrained_lasso_sanity_check.md`, decade-precision is ±5 yr — acceptable as noise relative to the 7–11 yr MAEs the clocks produce.

## Cell-type labels

CellxGene-curated cell types with their canonical map:

| CellxGene label | Canonical |
|---|---|
| CD4-positive, alpha-beta T cell (+ subtypes) | CD4+ T |
| CD8-positive, alpha-beta T cell (+ subtypes) | CD8+ T |
| CD14-positive monocyte | Monocyte |
| CD16-positive, CD56-dim NK / CD16-neg CD56-bright NK | NK |
| naive / memory / class-switched memory B | B |
| platelet, plasmablast, mature NK T, gamma-delta T, MAIT, pDC, T-helper 22 | Other → dropped |

## Harmonization applied

1. Disease filter: `disease == 'normal'` → 104,923 cells.
2. Age parse: exact or decade-bin midpoint; 0 donors dropped for unparseable age.
3. Adult-only: passes (all healthy ≥ 21).
4. Cell type in canonical 5: retains 78,850 cells (drops 26,073 of the Other-bucket types).

Per-cell-type breakdown from the Stephenson-only harmonization run:

| Cell type | n_cells | n_donors |
|---|---:|---:|
| CD4+ T | 22,429 | 24 |
| CD8+ T | 20,410 | 24 |
| Monocyte | 10,748 | 23 |
| NK | 15,884 | 29 |
| B | 9,379 | 29 |

Per-donor NK and B are present for all 29 donors; CD4+ T / CD8+ T / Monocyte drop 5–6 donors where no cells of those types passed QC.

## Known issues / caveats

- **Accession mirror divergence** — the three IDs (CellxGene ddfad306, ArrayExpress E-MTAB-10026, HCA-Covid19PBMC) all reference the same underlying cohort. A leakage audit keyed only on the ArrayExpress ID would miss scFoundation's overlap (scFoundation ingested via the HCA Project ID). This was the core methodological finding documented in `methods/leakage_audit_notes.md`.
- **18/29 donors have decade-only ages** — limits calibration precision to ±5 yr for those donors. Acceptable because Stephenson is already exploratory-only (below the 80-donor primary threshold).

## Role in the study

**Exploratory training cohort.** 29 donors is below the 80-donor threshold for primary LOCO fold status. The `loco_stephenson` fold exists in `data/loco_folds.json` but is flagged `primary: false`.

**Leakage status per foundation model** (from `data/leakage_audit.csv`):
- scGPT: **overlapping** (CellxGene Census 2023-05-15 ingestion)
- Geneformer: clean (V1 June-2021 predates; verified zero hits in Genecorpus-30M supp)
- scFoundation: **overlapping** (ingested via `HCA-Covid19PBMC` project ID — found in Hao 2024 Nat Methods MOESM5 row 81)
- UCE: **overlapping** (CellxGene Census mid-late 2023 ingestion)

So only Geneformer rows on `loco_stephenson` are clean for asterisk-free reporting.

## Used in

- Task 1c-v1: harmonizer sanity run (regression-tested end-to-end on Stephenson alone; 78,850 cells × 5 types; validated obs schema + `age_precision` flag)
- Task 1c-v2: three-cohort harmonization (secondary training cohort)
- Phase 3 pilot: secondary table for Geneformer; scGPT/UCE/scFoundation rows carry leakage asterisks
