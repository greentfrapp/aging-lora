# Phase 2 — Baseline establishment

## Goal

Before measuring whether foundation-model fine-tuning helps, this phase establishes the task-specific baselines the foundation models will be measured against. The primary approach uses the sc-ImmuAging pre-trained LASSO clocks (shipped in `data/scImmuAging/data/all_model.RDS`) applied directly to the project's LOCO holdout sets — this is the fastest path to a result and tests the most practically meaningful claim: can foundation models beat the published state-of-the-art with less training data? A training-matched LASSO retrain (same three cohorts as the foundation models) is scoped as a supplementary analysis, to be run after initial results are in hand.

### Baseline strategy decision (recorded 2026-04-24)

**Primary (option 2):** Apply the pre-trained sc-ImmuAging LASSO/RF/PointNet clocks directly to LOCO holdout donors. No retraining on our cohorts. The pre-trained models were trained on five cohorts (including two EGA cohorts we do not have); our foundation models are fine-tuned on three cohorts. This training-set asymmetry must be stated explicitly in all result tables and the methods section. If foundation models win, the claim is: *"foundation models outperform the published SOTA using less training data."* If they lose, the asymmetry is a confound — proceed to option 1.

**Supplementary (option 1):** Retrain LASSO on the same three cohorts as the foundation models, using `data/scImmuAging/codes/Lasso_training.R` as reference. This produces a training-matched comparison reported in the supplement. Scope: run after Phase 3 results are available and only if the primary comparison is inconclusive or flagged by reviewers.

## Success criteria

- Pre-trained sc-ImmuAging LASSO/RF/PointNet clocks applied to each LOCO holdout cohort; per-fold per-cell-type m.a.e. and Pearson R recorded in `results/baselines/loco_baseline_table.csv`. Training-set asymmetry noted in every table header.
- Pasta architecture retrained on PBMC pseudobulk (per-donor aggregated counts per cell type) on the same LOCO folds, reporting per-fold m.a.e. alongside the Pasta-as-released transfer baseline; any non-trivial pseudobulk adaptation decisions (gene-selection, sparsity handling) documented in `methods/pasta_pseudobulk_notes.md`.
- Supplementary LASSO retrain (option 1) deferred: tracked in `results/baselines/loco_baseline_table.csv` as empty columns flagged `[supplement — pending]` until run.

## Tasks

- [ ] Task: Apply pre-trained sc-ImmuAging clocks to LOCO holdout sets. Load `data/scImmuAging/data/all_model.RDS` and `all_model_inputfeatures.RDS`; convert each LOCO holdout cohort to a Seurat object with `donor_id` and `age` columns; run `PreProcess()` → `AgingClockCalculator()` → `Age_Donor()` for each of the five cell types; record per-fold per-cell-type m.a.e. and Pearson R in `results/baselines/loco_baseline_table.csv`. Label all rows `baseline=scImmuAging-pretrained, training_cohorts=original-five`. Done when all three LOCO folds × five cell types are populated.

- [ ] Task: [Supplementary — run after Phase 3] Retrain sc-ImmuAging LASSO on the project's three cohorts and LOCO splits. Using `data/scImmuAging/codes/Lasso_training.R` as reference, fit LASSO on each LOCO training fold (same three-cohort corpus as the foundation models) and evaluate on held-out cohort. Record results in `results/baselines/loco_baseline_table.csv` alongside the pre-trained baseline. This produces the training-matched comparison reported in the supplement. Done when all three LOCO folds × five cell types are populated with label `baseline=LASSO-retrained, training_cohorts=three-cohort`.

- [ ] Task: Verify or retrieve scAgeClock code availability. Check the scAgeClock paper (https://www.nature.com/articles/s41514-026-00379-5) supplementary materials and any linked GitHub repository for a publicly available implementation. If code is available, run scAgeClock on the same LOCO holdout sets (or apply author-released weights to non-overlapping donors and flag as a transfer check). If code is not available by the time Phase 2 completes, document this clearly in `results/baselines/scageclock_availability.md` and treat scAgeClock comparison as conditional on availability, with the project's primary criterion (a) measured only against sc-ImmuAging baselines. Done when code availability is confirmed or definitively ruled out.

- [ ] Task: Build the Pasta pseudobulk pipeline and retrain on LOCO splits. Aggregate per-cell-type per-donor count matrices to pseudobulk profiles. Retrain the Pasta architecture (https://github.com/jsalignon/pasta) on these pseudobulk LOCO training folds and evaluate on held-out cohorts, recording per-fold m.a.e. Also run Pasta-as-released (published weights applied out-of-the-box) to the same pseudobulk profiles as a transfer baseline. Document all adaptation choices (gene-intersection strategy, library-size normalization, sparsity floor) in `methods/pasta_pseudobulk_notes.md`. Done when both Pasta-retrained and Pasta-as-released per-fold m.a.e. values are recorded in `results/baselines/loco_baseline_table.csv`.

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
