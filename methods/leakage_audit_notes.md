# Pretraining-corpus leakage audit — findings & methodology

**Date.** 2026-04-24.
**Output.** `data/leakage_audit.csv` (16 rows: 4 models × 4 cohorts).

## Summary — what this means for the primary LOCO folds

| LOCO fold | Clean models | Overlapping models | Unknown models |
|---|---|---|---|
| **loco_terekhova** (primary, 166 donors) | scGPT, Geneformer, scFoundation, UCE | — | — |
| **loco_onek1k** (primary, 981 donors) | — | scGPT, UCE | Geneformer, scFoundation |
| loco_stephenson (exploratory, 29 donors) | — | scGPT, UCE | Geneformer, scFoundation |
| AIDA (external holdout) | scGPT, Geneformer, scFoundation, UCE | — | — |

**Headline:** Terekhova is the only primary fold where *all four* foundation models are confirmed clean. For the OneK1K fold, scGPT and UCE results are confounded by training-set memorization and must be reported with a `leakage_flagged` annotation in Phase 3/4 result tables. AIDA is clean across the board (published 2025, postdates all checkpoints).

## Per-model pretraining-manifest source

- **scGPT (whole-human):** pretrained on 33M cells from CZ CELLxGENE Discover Census release `2023-05-15` (stated in Methods of Cui et al. 2024 Nature Methods, `bowang-lab/scGPT` README). No per-dataset manifest is published; inclusion is inferred by whether a CellxGene collection existed in that snapshot.
- **Geneformer (V1 checkpoint, `ctheodoris/Geneformer` HF, June 2021):** pretrained on Genecorpus-30M — 561 public datasets from GEO/SRA/HCA/SCP/EBI. Source list in Theodoris et al. 2023 Nature Supplementary Table 1 and HF dataset card `ctheodoris/Genecorpus-30M`. V1 is locked to pre-2022 data — all of our training cohorts except Stephenson and OneK1K are trivially post-cutoff.
- **scFoundation (May 2023 preprint, 2024 Nature Methods):** ~50M cells manually curated from GEO/HCA/SCP/EMBL-EBI. Source list in `DataSupplement1.xlsx` and `DataSupplement2.xlsx` in the `biomap-research/scFoundation` GitHub repo, also on Figshare record 24049200. Not CellxGene-based.
- **UCE (Nov 2023 bioRxiv v2):** ~36M cells, majority (>33M) from CZ CELLxGENE Census. Manifest in Extended Data Table 2 of the preprint. Census snapshot ~mid-to-late 2023.

## Confidence level by row

**High confidence (12/16 rows):**
- All `overlapping` rows for scGPT and UCE are strong: both models explicitly pretrain on CZ CellxGene Census, and both OneK1K (collection `dde06e0f`) and Stephenson (collection `ddfad306`) were in Census well before the pretraining snapshots. Without cohort filtering the Census release *would* include them.
- All `clean` rows for Terekhova: its distribution is Synapse-only (`syn49637038`) with no GEO/HCA/SCP/EBI/CellxGene ingestion. None of the four models' documented corpora reach Synapse.
- All `clean` rows for AIDA: 2025 publication postdates every checkpoint's snapshot.

**Medium confidence — `unknown` rows requiring human verification:**
- `Geneformer × OneK1K` — timing says cannot be in V1 (OneK1K Apr 2022 > V1 cutoff Jun 2021), but Genecorpus-30M Supplementary Table 1 was not opened directly (Nature page returned 403 during audit). Almost certainly `clean` by timing; one-line verification by opening the supplementary table.
- `Geneformer × Stephenson` — Stephenson (Apr 2021) is pre-cutoff, could be in Genecorpus-30M. Requires Supplementary Table 1 check.
- `scFoundation × OneK1K` — `DataSupplement1.xlsx` / `DataSupplement2.xlsx` in the scFoundation repo list the GEO accessions used, but were not retrievable via `WebFetch` (github blob not parsed). Human to open and `Ctrl-F "GSE196830"`.
- `scFoundation × Stephenson` — same, check for `E-MTAB-10026` in the data supplements.

## Reporting rule for Phase 3 / Phase 4

Each per-(model, cohort) LOCO row in `results/phase3/*.csv` and `results/phase4/*.csv` must carry a `leakage_status` column whose value is taken from this file. Tables in the paper/preprint:

- Clean rows are the primary result.
- Overlapping rows are reported with an asterisk and a footnote: *"this model was pretrained on the held-out cohort; result is a lower bound on true generalization"*.
- Unknown rows are reported but must be pushed to `clean` or `overlapping` before paper submission by the human-verification step above.

## Human verification TODOs

- [ ] Open Theodoris 2023 Nature Supplementary Table 1 → resolve Geneformer × {OneK1K, Stephenson}.
- [ ] Open `biomap-research/scFoundation/DataSupplement1.xlsx` + `DataSupplement2.xlsx` → resolve scFoundation × {OneK1K, Stephenson}.

Estimated effort: 15 min. Tracked as a standalone entry in `HUMAN_TASKS.md` (#4, to be added).
