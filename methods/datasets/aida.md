# AIDA — Asian Immune Diversity Atlas (Kock et al. 2025)

## Paper and accession

**Paper.** Kock et al., "Asian diversity in human immune cells," *Cell* 2025. DOI: [10.1016/j.cell.2025.02.017](https://www.cell.com/cell/fulltext/S0092-8674(25)00202-8).

**Accession**: CellxGene collection `ced320a1-29f3-47c1-a735-513c7084d508` — "Asian Immune Diversity Atlas (AIDA)". Two data freezes available:

| Dataset ID | h5ad | Size | Cells | Countries | Status |
|---|---|---:|---:|---|---|
| `c838aec3-03ef-4398-b882-0e3912abfff0` | `9deda9ad-6a71-401e-b909-5263919d85f9.h5ad` | 13.27 GB | 1,265,624 | Japan, Singapore, South Korea, Thailand, India | **v2 Data Freeze** ← we use |
| `b0e547f0-462b-4f81-b31b-5b0a5d96f537` | `a4be8aed-8456-4be7-9cb4-06ce76bdc0b4.h5ad` | 9.82 GB | 1,058,909 | Japan, Singapore, South Korea | v1 Data Freeze (superseded) |

## Access

Direct HTTPS via CellxGene, no auth. 13.27 GB download.

```
curl -L -C - --speed-limit 10000 --speed-time 120 --retry 10 --retry-delay 15 \
  -o 9deda9ad-6a71-401e-b909-5263919d85f9.h5ad \
  https://datasets.cellxgene.cziscience.com/9deda9ad-6a71-401e-b909-5263919d85f9.h5ad
```

Destination: `data/cohorts/raw/aida/9deda9ad-6a71-401e-b909-5263919d85f9.h5ad`.

**Network note.** CellxGene's S3+CloudFront occasionally stalls mid-transfer; the AIDA download stalled twice at ~1.1 GB during our run before succeeding on a third attempt with `--speed-limit / --speed-time / --retry-max-time 0` flags that auto-retry stalls. The URL is stable (HTTP 200 with `content-length: 14244724636` bytes) — retries just work. Use `-C -` for resume.

## Schema (CellxGene-curated)

Expected to follow the standard CellxGene schema:

- **Shape**: 1,265,624 cells × [var count, TBC]
- **`.obs`** will carry the standard CellxGene columns: `donor_id`, `cell_type` (+ `cell_type_ontology_term_id`), `assay` (+ id), `disease`, `sex`, `self_reported_ethnicity` (+ id) — this is the key column for ancestry stratification, `development_stage` (+ id), `is_primary_data`, `tissue` (+ id), `suspension_type`, and any AIDA-specific study metadata.
- **`.var`**: Ensembl IDs as `var_names`, `feature_name` with HGNC symbols.
- **`.raw.X`**: raw integer counts per CellxGene convention.

(The above is projected from other CellxGene releases; exact schema will be verified on first read of the on-disk file and this note tightened if anything diverges.)

## Role in the study — external ancestry holdout

AIDA is **not a training cohort**. It is the dedicated holdout for the ancestry-shift and age-axis-alignment readouts in Phase 4/5.

Per `roadmap/phase-1.md` success criterion: AIDA donors are split 50/50 before any model is trained, stratified by `(age_decile × self_reported_ethnicity)`:

- **Half A** (`ancestry_shift_mae_donors`): external holdout for criterion (a) generalization — compute per-FM per-cell-type MAE on this half's donors, compare against training-cohort LOCO folds.
- **Half B** (`age_axis_alignment_donors`): fitted targets for Phase 5 age-axis cosine-similarity analysis. Ridge-probe fit on embeddings of this half; the cosine between the AIDA-derived and European-cohort-derived age directions is the cross-ancestry alignment metric.

The split is written once to `data/aida_split.json` by `src/data/freeze_splits.py::build_aida_split` (seed = 0) and treated as immutable from that point forward.

### Stratification logic

- Age decile: `age // 10 * 10` (0, 10, 20, ..., 90)
- Self-reported ethnicity: as stored in CellxGene (expected 7 Asian population groups per the paper — Chinese, Indian, Japanese, Korean, Malay, Thai, plus one more)
- Within each `(decile, ethnicity)` stratum: shuffle donors (random seed 0), first half → ancestry_shift_mae, second half → age_axis_alignment. Odd counts get a coin flip.

The result is two donor sets balanced on age × ethnicity so both halves are independent samples from the same underlying ancestry distribution.

## Harmonization

AIDA is NOT harmonized into `data/cohorts/integrated/` alongside the training cohorts. It is kept as a raw h5ad + the `data/aida_split.json` assignment. Phase 4 evaluation code loads it directly from `data/cohorts/raw/aida/` and applies the split filter at scoring time.

## Known issues / caveats

1. **Chemistry heterogeneity** — the AIDA paper describes mixed 3'/5' chemistry across sites. Downstream FM evaluation must track `obs['assay']` per cell so the Phase 4 chemistry-stratified analysis can separate signal.
2. **Age-decile granularity at ethnicity strata** — with 7 ancestries × ~6–8 age deciles, some stratum cells will have only 1–3 donors; the split code handles 1-donor strata with a coin flip but expect minor imbalance in donor-count totals per half across rare (ancestry, decade) combinations.
3. **No immediate test-schema validation** — at time of writing, the download is in progress (~3.9 GB / 13.3 GB) and the h5ad has not been opened yet. The schema section above is projected from CellxGene convention; this note will be updated with actual observed obs/var columns as soon as the file completes.

## Leakage status (all `clean` per `data/leakage_audit.csv`)

- scGPT: clean — AIDA published 2025, postdates CellxGene Census 2023-05-15
- Geneformer: clean — postdates V1 June-2021 cutoff
- scFoundation: clean — postdates the Hao 2024 Nat Methods corpus
- UCE: clean — postdates UCE's mid-late 2023 Census snapshot

AIDA is leakage-clean for every FM by construction (2025 publication).

## Used in

- `src/data/freeze_splits.py::build_aida_split` → produces `data/aida_split.json` (two donor lists; stratification metadata)
- Phase 4: ancestry-shift MAE evaluation on `ancestry_shift_mae_donors`
- Phase 5: age-axis cosine-similarity analysis on `age_axis_alignment_donors`
- Phase 5: cross-ancestry generalization claims — the only cohort outside the European training corpus
