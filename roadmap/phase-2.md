# Phase 2 — Baseline establishment

## Goal

Before measuring whether foundation-model fine-tuning helps, this phase establishes the task-specific baselines the foundation models will be measured against. The primary approach uses the sc-ImmuAging pre-trained LASSO clocks (shipped in `data/scImmuAging/data/all_model.RDS`) applied directly to the project's LOCO holdout sets — this is the fastest path to a result and tests the most practically meaningful claim: can foundation models beat the published state-of-the-art with less training data? A training-matched LASSO retrain (same three cohorts as the foundation models) is scoped as a supplementary analysis, to be run after initial results are in hand.

### Baseline strategy decision (recorded 2026-04-24, panel revised 2026-04-25)

**Primary (option 2):** Apply pre-trained external clocks directly to our LOCO holdout cohorts. No retraining. The training-set asymmetry vs. our 3-cohort fine-tuned FMs must be stated explicitly in all result tables and the methods section. If FMs win against this panel, the claim is: *"foundation models outperform the published state-of-the-art using less training data."* If they lose, the asymmetry is a confound — proceed to option 1.

**Baseline panel (revised 2026-04-25 after upstream-availability check):**

| Baseline | Family | Source | Status |
|---|---|---|---|
| sc-ImmuAging pre-trained LASSO | classical, single-cell | `data/scImmuAging/data/all_model.RDS` (5 `cv.glmnet` objects) | ✅ shipped weights |
| ~~sc-ImmuAging pre-trained RF~~ | ~~ensemble, single-cell~~ | not available — only training script `data/scImmuAging/codes/RF.py` ships | dropped from primary; see `methods/loco_baselines.md` |
| ~~sc-ImmuAging pre-trained PointNet~~ | ~~deep-learning, single-cell~~ | not available — only training script ships | dropped from primary |
| **scAgeClock** (Xie 2026) | deep-learning, single-cell, attention | `data/scAgeClock/scageclock/data/trained_models/scAgeClock_GMA_model_state_dict.pth` (44 MB) | ✅ shipped weights — **fills the deep-learning slot** |
| **Pasta REG** (Salignon 2025) | classical-ish, bulk transcriptomics | R package `jsalignon/pasta`, REG model (absolute age) | ✅ shipped weights, R-only |
| ~~Pasta age-shift, Pasta CT46 classifier~~ | — | predict relative age / binary class — not directly MAE-comparable | dropped from primary |
| ~~Pasta retrained on our LOCO folds~~ | — | training code not in public repo; multi-day reimplementation | deferred to supplement |

**Supplementary (option 1, deferred):** Retrain LASSO on the same three cohorts as the foundation models, using `data/scImmuAging/codes/Lasso_training.R` as reference. Tracked in `results/baselines/loco_baseline_table.csv` as `[supplement — pending]`. Run after Phase 3 results if the primary comparison is inconclusive or flagged by reviewers.

## Success criteria

- **LASSO baseline complete (DONE 2026-04-24):** Pre-trained sc-ImmuAging LASSO scored on each LOCO holdout cohort. Per-cohort per-cell-type MAE, Pearson R, mean bias recorded in `results/baselines/loco_baseline_table.csv` (15 rows = 3 cohorts × 5 cell types) with leakage_status, chemistry_match_to_baseline_training, and detectability_flag stratification columns. Methods in `methods/loco_baselines.md`.
- **scAgeClock baseline complete:** scAgeClock applied to each LOCO holdout cohort; per-cohort per-cell-type MAE and Pearson R appended to `results/baselines/loco_baseline_table.csv` with `baseline=scAgeClock, training_cohorts=CELLxGENE-Census`. **Pretraining-corpus leakage audit completed and 4 new rows appended to `data/leakage_audit.csv`** (model=scAgeClock × {OneK1K, Stephenson, Terekhova, AIDA}) before the baseline rows are reported.
- **Pasta REG baseline complete:** Pasta-as-released REG model applied to per-(donor × cell-type) pseudobulk profiles; per-cohort per-cell-type MAE and Pearson R appended to `results/baselines/loco_baseline_table.csv`. Pasta age-shift and CT46 heads NOT reported in primary table (different metrics). Pseudobulk adaptation choices documented in `methods/pasta_pseudobulk_notes.md`.
- **Supplementary LASSO retrain (option 1) deferred:** tracked in `results/baselines/loco_baseline_table.csv` as `[supplement — pending]` until run.

## Tasks

- [x] **Task 2.1:** (Completed 2026-04-24, see commit dcd5763.) Apply pre-trained sc-ImmuAging LASSO to all three cohorts via the Python `rdata`-port pipeline in `src/baselines/score_pretrained_lasso.py`. 15-row table assembled by `src/baselines/assemble_loco_baseline_table.py`. The roadmap statement "LASSO/RF/PointNet" is reduced to LASSO-only because the upstream sc-ImmuAging package ships only `cv.glmnet` weights — RF.py and pointnet_unet.py are training scripts referencing files not in the public package. Documented in `methods/loco_baselines.md`.

- [ ] **Task 2.2:** Audit scAgeClock pretraining-corpus leakage. scAgeClock was trained on CELLxGENE Census; the version date determines whether OneK1K (CELLxGENE-curated 2022), Stephenson (CELLxGENE 2021), Terekhova (Synapse-only), and AIDA (2025) were included. Append 4 rows (one per cohort) to `data/leakage_audit.csv` with `model=scAgeClock` and the same `{clean, overlapping, unknown}` schema as the FM rows. Methodology section appended to `methods/leakage_audit_notes.md`. **This task gates Task 2.3.** Done when 4 new rows are committed and methodology is documented.

- [ ] **Task 2.3:** Apply scAgeClock to all three cohorts. Format the harmonized h5ads to scAgeClock's input schema (gene-symbol vocabulary, 19,234 protein-coding genes, 4 categorical-feature columns prepended). Run `scageclock.evaluation.prediction` per cohort with chunked inference (≤50K cells/batch to fit memory). Aggregate per-cell predictions to per-donor median age. Record results to `results/baselines/loco_baseline_table.csv` with `baseline=scAgeClock, training_cohorts=CELLxGENE-Census` and the three Phase-4 stratification columns. Done when all 15 cells (3 cohorts × 5 cell types) are populated.

- [ ] **Task 2.4:** Pasta REG baseline. Install R + `jsalignon/pasta` + dependencies. Pseudobulk our harmonized h5ads per (donor × cell type) by summing raw counts across cells. Convert pseudobulk matrices to ExpressionSet objects and call `adding_age_preds_to_pdata(..., REG=TRUE, Pasta=FALSE, CT46=FALSE)` to get absolute-age predictions. Append per-cohort per-cell-type MAE/R rows to `results/baselines/loco_baseline_table.csv` with `baseline=Pasta-REG, training_cohorts=Pasta-pretraining`. Document the pseudobulk-aggregation choice and any R-bridge issues in `methods/pasta_pseudobulk_notes.md`. **R-install effort is the main cost** — if the R bridge proves multi-day, defer Pasta to supplement. Done when Pasta-REG numbers are appended OR a deferral memo is written.

- [ ] **Task 2.5 (deferred):** Retrain sc-ImmuAging LASSO on our three cohorts using `Lasso_training.R` as reference; produces a training-matched comparison for the supplement. Run after Phase 3 results are available; only if the primary comparison is inconclusive or flagged by reviewers.

## References

```references
[
  {
    "title": "Single-cell immune aging clocks reveal inter-individual heterogeneity during infection and vaccination",
    "url": "https://www.nature.com/articles/s43587-025-00819-z",
    "authors": "Li et al.",
    "year": 2025,
    "venue": "Nature Aging"
  },
  {
    "title": "scAgeClock: a single-cell transcriptome-based human aging clock model using gated multi-head attention neural networks",
    "url": "https://www.nature.com/articles/s41514-026-00379-5",
    "authors": "Xie",
    "year": 2026,
    "venue": "npj Aging"
  },
  {
    "title": "Pasta, a versatile transcriptomic clock, maps the chemical and genetic determinants of aging and rejuvenation",
    "url": "https://www.biorxiv.org/content/10.1101/2025.06.04.657785v2",
    "authors": "Salignon, Tsiokou, Marques, Rodriguez-Diaz, Ang, Pietrocola, Riedel",
    "year": 2025,
    "venue": "bioRxiv"
  }
]
```
