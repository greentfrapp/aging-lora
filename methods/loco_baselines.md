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
