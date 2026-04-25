# Phase 2 LOCO baseline table

**Artifact**: `results/baselines/loco_baseline_table.csv`. Assembled by `src/baselines/assemble_loco_baseline_table.py`.

## Model panel available from upstream sc-ImmuAging

The Phase-1 roadmap statement "Apply LASSO, RF, and PointNet pre-trained sc-ImmuAging clocks" is not fully achievable. After inspecting the `sc-ImmuAging` package (`data/scImmuAging/`):

| Model | File | Status |
|---|---|---|
| LASSO (`cv.glmnet`) | `data/all_model.RDS` — 5 cell-type-specific objects | ✅ shipped pretrained weights |
| Random Forest | `codes/RF.py` | Training script only — **no shipped weights** |
| PointNet | `codes/pointnet_unet.py` | Training script only — **no shipped weights** |

The R package `data/` directory contains five `cv.glmnet` serialized objects and nothing else (verified via `rdata.read_rds`). The RF and PointNet scripts reference `data/processed/*_RFmodel.sav` and TF checkpoints that are not distributed with the package. Reproducing them would require retraining on the paper's original training cohorts (two of which are EGA-controlled and unavailable to us — see `roadmap/phase-1.md` cohort-decisions section).

**Decision:** for Phase 2, the "sc-ImmuAging-pretrained" baseline is LASSO-only. The Phase-2 deep-learning comparator role is filled by **scAgeClock** (Xie 2026, npj Aging; pretrained weights shipped in `gangcai/scageclock`; see `results/baselines/scageclock_availability.md`), and the bulk-transcriptomics comparator role by Pasta (Salignon 2025).

## LOCO semantics for a pretrained external baseline

Because sc-ImmuAging LASSO was trained on five cohorts *other than* our three, "leave-one-cohort-out" reduces to "score the pretrained LASSO on each of our three cohorts as the held-out evaluation set". There is no per-fold LASSO retrain for this row of the table; every cell of the table is the same model (LASSO) evaluated on a different cohort. Sub-rows are per (cohort × cell type).

## Table structure

`loco_baseline_table.csv` carries 15 rows = 3 cohorts × 5 cell types, with the three Phase-4-style stratification columns on every row so downstream aggregation can filter without reloading the raw score files:

| Column | Meaning |
|---|---|
| `baseline` | `scImmuAging-pretrained` for all 15 rows (other baselines add their own rows in later tasks) |
| `training_cohorts` | `original-five` — signals the training-set asymmetry vs. our fine-tuned FMs (three-cohort) |
| `eval_cohort` | onek1k / stephenson / terekhova |
| `eval_chemistry` | 10x 3' v2 / 10x 3' / 10x 5' v2 |
| `cell_type` | CD4T, CD8T, MONO, NK, B |
| `n_donors`, `median_abs_err_yr`, `mean_abs_err_yr`, `pearson_r`, `pearson_p`, `mean_bias_yr` | standard LASSO metrics from `score_pretrained_lasso.py` with 100 pseudocells × 15 cells/donor |
| `leakage_status` | always `clean` for this row (LASSO's training cohorts do not overlap our eval cohorts) |
| `chemistry_match_to_baseline_training` | `match` for 10x 3'; `shifted` for 10x 5' v2 Terekhova |
| `detectability_flag` | per-row `powered` / `underpowered` from the Phase-1 ρ=0.8 floor (`data/detectability_floor.json`) |

## Headline observations

| eval_cohort | chemistry | best R / MAE (cell type) | worst R / MAE (cell type) |
|---|---|---|---|
| onek1k | 10x 3' v2 | **0.77 / 7.6y** (CD8T) | 0.53 / 10.7y (B) |
| stephenson | 10x 3' | **0.79 / 8.4y** (CD4T) | 0.18 / 14.6y (NK) |
| terekhova | 10x 5' v2 | **0.82 / 9.2y** (CD4T) | 0.08 / 15.0y (B) |

Three patterns visible in the assembled table:

1. **CD4T/CD8T are chemistry-robust** (R > 0.7 on all three cohorts) regardless of 10x 3' vs 5'. These two rows carry the "primary LOCO" headline for the LASSO baseline.

2. **MONO/NK/B degrade under chemistry shift** (Terekhova): LASSO MONO R drops from 0.71 (3') to 0.29 (5'); B collapses from 0.53 to 0.08. Captured in `detectability_flag` for reporting. See `methods/terekhova_chemistry_shift.md`.

3. **Stephenson is underpowered across the board** (29 donors, only 24 in CD4T/CD8T subsets) but internally consistent with OneK1K for the T-cell clocks, giving an independent cross-cohort replication signal at the 10x 3' chemistry. The R=0.18 on NK and 0.26 on B are consistent with the low-power regime (29 donors with 18 decade-precision ages).

## Strict-clean × chemistry-match × powered headline subset

Applying all three Phase-4 green filters:

- OneK1K × all 5 cell types: 5 cells (all green)
- Terekhova × {CD4T, NK, B} chemistry-shifted so not strict-clean
- Stephenson underpowered so not strict-clean

**Result: 5 strict-clean-AND-chemistry-match-AND-powered LASSO baseline cells** — the reference against which FM primary-fold improvements are measured in Phase 4. Combined with the inclusive-mode contrasts, this gives 15 data points total for the LASSO-vs-FM comparison.

## Phase 2 Task 2.3: scAgeClock baseline (added 2026-04-25)

scAgeClock GMA (Xie 2026) scored on the same 15 cohort × cell-type slices via `src/baselines/score_scageclock.py`. Inputs: harmonized h5ads → CP10k+log1p normalize on the full 48,968-gene matrix → reindex to scAgeClock's 19,234-gene vocabulary (96.9% gene-vocabulary overlap with our data) → prepend 4 categorical-feature columns (assay/cell_type/tissue=blood/sex with integer codes from scAgeClock's index TSVs) → CPU inference in 25K-cell chunks → per-donor median age. 30 minutes wall-clock for 2.75M cells.

**Headline observations:**

| Cohort | Best R cell type | Best R | Worst R | Systematic bias |
|---|---|---|---|---|
| OneK1K (overlapping) | CD8T | **0.59** | MONO 0.18 | −13 to −16y for 4/5 cell types; NK clean (−2y) |
| Stephenson (overlapping) | MONO | 0.38 | NK −0.15 | mixed; small N (24–29 donors) → noisy |
| Terekhova (clean) | CD8T | 0.24 | MONO 0.08 | −12 to −15y across all cell types |

The persistent ~−13y systematic bias is the dominant signal: scAgeClock predicts 13 years younger than truth on PBMC across cohorts. NK is the only cell type where bias is small (−2.3y on OneK1K), suggesting the bias is driven by cell-type-specific weights in the GMA model. The R values are markedly lower than LASSO's even on **leakage-overlapping training cells** — scAgeClock is a generalist trained across 400+ cell types and 40+ tissues, and its PBMC-specific accuracy is below a PBMC-specialist LASSO. This is informative for the Phase-4 narrative: a deep general-purpose clock does NOT automatically dominate a shallow specialist on a single tissue.

## Phase 2 Task 2.4: Pasta-REG baseline (added 2026-04-25)

Pasta-REG (Salignon 2025) scored on per-(donor × cell-type) pseudobulk profiles via `src/baselines/pseudobulk_for_pasta.py` + `src/baselines/score_pasta_reg.R`. Pseudobulk = sum of raw counts across all cells of a (donor, cell-type) shard; gene rownames stripped of Ensembl version suffix (only for `ENSG`-prefixed IDs; clone-style IDs like `AC000065.1` vs `AC000065.2` are kept distinct). Pasta filters to its 8,113-gene panel (intersection with our pseudobulk = 8,113 / 8,113 — full panel coverage), rank-normalizes per donor, and runs the REG `cv.glmnet` model.

**Headline observations:**

| Cohort | Best R cell type | Best R | MAE | Notes |
|---|---|---|---|---|
| OneK1K | CD4T | 0.60 | 24.2y | Large negative bias (−23 to −26y) drives high MAE |
| Stephenson | CD4T | 0.66 | 9.7y | T cells strong; MONO/NK collapse |
| Terekhova | **CD4T** | **0.78** | **8.0y** | **Beats LASSO (R=0.82 / MAE=9.2) on MAE; ties on R** |

Pasta's rank-normalization + bulk-trained weights make it remarkably **chemistry-invariant**: on Terekhova (10x 5'), Pasta CD4T R=0.78, CD8T R=0.75, B R=0.28 — all higher than LASSO's chemistry-degraded R values (CD4T 0.82, CD8T 0.73, B 0.08). **The 10x 5' Pasta-B-cell signal at R=0.28 is the chemistry-rescue baseline** that Phase 4 fine-tuned FMs need to beat.

Pasta's weakness is calibration: large mean biases on OneK1K (−23y) and Terekhova MONO (+20y). Median absolute error stays large on cohorts where bias dominates the residual.

## Strict-clean × chemistry-match × powered headline subset (3-baseline panel)

`results/baselines/loco_baseline_table.csv` now has 45 rows = 3 baselines × 3 cohorts × 5 cell types. The strict-clean filter (leakage_status=clean AND detectability_flag=powered) varies per baseline:

| Baseline | Strict-clean cells (eval_cohort × cell_type) |
|---|---|
| scImmuAging-LASSO | OneK1K × all 5 (chemistry-match), Terekhova × {CD4T, NK, B} (chemistry-shifted, included in chemistry-inclusive mode) |
| scAgeClock | Terekhova × {CD4T, NK, B} only (OneK1K & Stephenson overlapping; CD8T/MONO underpowered on Terekhova) |
| Pasta-REG | OneK1K × all 5 + Terekhova × {CD4T, NK, B} (Pasta has no per-cohort leakage; chemistry tag = bulk-vs-sc) |

**Per-cell minimum-MAE baseline** (the "best of 3" gate from the Phase-4 reporting policy):

| Cohort | CD4T | CD8T | MONO | NK | B |
|---|---|---|---|---|---|
| OneK1K | LASSO 9.4 | LASSO 7.6 | LASSO 7.9 | scAgeClock 11.5 | LASSO 10.7 |
| Stephenson | LASSO 8.4 | LASSO 11.4 | LASSO 10.7 | NK Pasta 11.4 | LASSO 12.1 |
| Terekhova | **Pasta 8.0** | **Pasta 7.6** | LASSO 12.8 | LASSO 12.5 | **Pasta 10.9** |

On the strict-clean Terekhova fold, Pasta is the best baseline for CD4T/CD8T/B and LASSO for NK/MONO. **The Phase-4 FM headline must clear these 5 best-baseline floors per cell type, not the LASSO floor alone.**
