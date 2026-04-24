# OneK1K (Yazar et al. 2022)

## Paper and accessions

**Paper.** Yazar et al., "Single-cell eQTL mapping identifies cell type-specific genetic control of autoimmune disease," *Science* 376 (6589), 2022. DOI: [10.1126/science.abf3041](https://www.science.org/doi/10.1126/science.abf3041). Project name: OneK1K (a thousand donors, a thousand cells each).

**Accessions** (same underlying dataset, multiple mirrors):
- **CellxGene** collection `dde06e0f-ab3b-46be-96a2-a8082383c4a1` → dataset `3faad104-2ab8-4434-816d-474d8d2641db` → h5ad `a3f5651f-cd1a-4d26-8165-74964b79b4f2.h5ad` **← the source we use**
- GEO `GSE196830` — raw RAW.tar with 75 pooled CSV captures + genotype demultiplexing files (26 GB; superseded, retained only as a fallback per FUTURE_WORK.md)
- `www.onek1k.org` — project portal

## Access

Direct HTTPS, no auth. ~4.2 GB download.

```
curl -L -o a3f5651f-cd1a-4d26-8165-74964b79b4f2.h5ad \
  https://datasets.cellxgene.cziscience.com/a3f5651f-cd1a-4d26-8165-74964b79b4f2.h5ad
```

Destination: `data/cohorts/raw/onek1k_cellxgene/a3f5651f-cd1a-4d26-8165-74964b79b4f2.h5ad` (4.2 GB, MD5 `bb9fa77aa59331dc12f96a31734c7121`).

## Schema (as shipped by CellxGene)

- **Shape**: 1,248,980 cells × 35,528 genes
- **`.obs` columns**: `orig.ident`, `nCount_RNA`, `nFeature_RNA`, `percent.mt`, `donor_id`, `pool_number`, `predicted.celltype.l2`, `predicted.celltype.l2.score`, **`age`** (int32), `tissue_ontology_term_id`, `assay_ontology_term_id`, `disease_ontology_term_id`, `cell_type_ontology_term_id`, `self_reported_ethnicity_ontology_term_id`, `development_stage_ontology_term_id`, `sex_ontology_term_id`, `is_primary_data`, `suspension_type`, `tissue_type`, **`cell_type`**, **`assay`** (`10x 3' v2`), **`disease`** (all `normal`), `sex`, `tissue`, `self_reported_ethnicity`, `development_stage`, `observation_joinid`
- **`.var` columns**: Ensembl IDs as `var_names`; `feature_name` = HGNC symbol; standard CellxGene columns (`feature_is_filtered`, `feature_reference`, `feature_biotype`, `feature_length`, `feature_type`)
- **`.raw.X`**: raw integer counts (CellxGene convention puts raw in `.raw.X`, normalized in `.X`)
- **Ages**: integer years, 19–97 range, no NaN
- **Donors**: 981 unique (one duplicate pair removed in curation vs. the paper's 982)

## Cell-type ontology IDs observed in the CellxGene release

| Ontology ID | Label | Cells | Canonical map |
|---|---|---:|---|
| CL:0000904 | central memory CD4+ alpha-beta T | 289,000 | CD4+ T |
| CL:0000895 | naive thymus-derived CD4+ alpha-beta T | 259,012 | CD4+ T |
| CL:0000905 | effector memory CD4+ alpha-beta T | 31,261 | CD4+ T |
| CL:0000624 | CD4+ alpha-beta T | 773 | CD4+ T |
| CL:0000815 | regulatory T | 26,531 | CD4+ T |
| CL:0000913 | effector memory CD8+ alpha-beta T | 161,051 | CD8+ T |
| CL:0000900 | naive thymus-derived CD8+ alpha-beta T | 52,538 | CD8+ T |
| CL:0000907 | central memory CD8+ alpha-beta T | 16,409 | CD8+ T |
| CL:0000625 | CD8+ alpha-beta T | 305 | CD8+ T |
| CL:0000623 | natural killer | 164,933 | NK |
| CL:0000938 | CD16-neg CD56-bright NK | 7,006 | NK |
| CL:0001054 | CD14+ monocyte | 36,130 | Monocyte |
| CL:0002396 | CD14-low CD16+ monocyte | 15,743 | Monocyte |
| CL:0000788 | naive B | 65,702 | B |
| CL:0000787 | memory B | 30,234 | B |
| CL:0000818 | transitional stage B | 29,889 | B |
| CL:0000980 | plasmablast | 3,754 | B |
| CL:0000934 | **CD4+ alpha-beta cytotoxic T** | 17,993 | CD4+ T (note: ID is CD4-specific, NOT CD8 — this was a bug in our initial map, fixed 2026-04-24) |
| CL:0000798 | gamma-delta T | 18,922 | Other (dropped) |
| CL:0000940 | mucosal invariant T (MAIT) | 8,835 | Other |
| CL:0000990 | conventional dendritic | 4,570 | Other |
| CL:0000784 | plasmacytoid dendritic | 1,897 | Other |
| CL:0001065 | innate lymphoid | 444 | Other |
| CL:0002489 | double negative thymocyte | 1,824 | Other |
| CL:0008001 | hematopoietic precursor | 1,812 | Other |
| CL:0000233 | platelet | 1,810 | Other |
| CL:0000232 | erythrocyte | 290 | Other |

## Harmonization applied

Filters applied in `src/data/harmonize_cohorts.py::load_cellxgene_cohort`:

1. Healthy only — passes (all 1.25M cells are `disease = normal`).
2. Adult only — passes (all donors ≥ 18).
3. Age not NaN — passes.
4. Cell type in canonical 5 — retains ~1.15M cells (drops gamma-delta, MAIT, DCs, platelets, etc.)

Output schema (written to `data/cohorts/integrated/{cell_type}.h5ad`):
- `.X` = raw counts (float32 CSR, passed through from `.raw.X`)
- `.obs`: `cohort_id='onek1k'`, `donor_id='onek1k:<id>'`, `age` (float), `age_precision='exact'`, `sex`, `assay='10x 3\' v2'`, `cell_type` (canonical)
- `.var`: `gene_symbol`, `ensembl_id` (from `feature_name` + var.index)

## Source decision — CellxGene vs. GEO RAW

Recorded 2026-04-24: **we use the CellxGene-curated h5ad, not the GEO RAW tar.** Rationale:
- GEO `GSE196830` ships 75 multiplexed pools as CSVs + genotype-assignment files. Demultiplexing would require running `souporcell` or similar; substantial work for no downstream benefit.
- The primary baseline (Phase 2) is the pre-trained sc-ImmuAging LASSO applied to holdouts — we do not retrain on OneK1K for the primary comparison. So minor QC/annotation divergence from the paper's internal version is acceptable.

GEO RAW retained as a fallback path; revisit if reviewers flag CellxGene cell-type annotations as divergent from the paper's. See `FUTURE_WORK.md` "OneK1K from GEO RAW (demultiplexed)" entry.

## Known issues / caveats

- **CL:0000934 is CD4, not CD8**, despite the similar-looking ID number. Our initial `CELLTYPE_CODE_TO_ONTOLOGY` mapping (in `src/baselines/score_pretrained_lasso.py`) had it on the CD8 side; the Task 1e sanity check caught the MAE anomaly and we fixed the map (commit before `19f0137`).
- **Platelet and erythrocyte contamination is small (~0.2%)** — dropped via the canonical-cell-type filter. Not a concern.
- **Single chemistry (10x 3' v2)** — makes this cohort a clean comparison partner with Stephenson (also 3') but introduces chemistry shift vs. Terekhova (5'). Tracked as Task 1f.

## Role in the study

**Primary training cohort.** Largest donor count (981); all cell types adequately powered for 10% MAE reduction detection at ρ = 0.8 per `data/detectability_floor.json`.

**Leakage status per foundation model** (from `data/leakage_audit.csv`):
- scGPT: **overlapping** (CellxGene Census 2023-05-15 included OneK1K)
- Geneformer: clean (V1 cutoff June 2021 predates OneK1K's April 2022 release; verified in Genecorpus-30M supp)
- scFoundation: clean (verified in Hao 2024 Nat Methods MOESM4+MOESM5)
- UCE: **overlapping** (CellxGene Census ~mid-late 2023)

So `loco_onek1k` is the primary fold for Geneformer + scFoundation comparisons, but scGPT/UCE rows on it must carry the training-set-overlap asterisk.

## Used in

- Task 1e: pre-trained sc-ImmuAging LASSO sanity check (5 cell types, 981 donors, R 0.53–0.77)
- Task 1c-v2: three-cohort harmonization (primary training cohort)
- Phase 3+ pilot (CD4+ T LoRA fine-tuning headline via Geneformer + scFoundation on loco_onek1k)
- Source of the symbol→Ensembl map used to align Terekhova's symbol-indexed var with Stephenson's Ensembl-indexed var
