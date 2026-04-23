# Phase 2 — Baseline reproduction

## Goal

Before measuring whether foundation-model fine-tuning helps, the project must establish reliable reproductions of the task-specific baselines on the frozen LOCO splits produced in Phase 1. This phase retrains sc-ImmuAging's LASSO, random forest, and PointNet heads on the project's own splits — not the paper's original splits — and adds the Pasta transcriptomic clock as a pseudobulk-style comparator. Matching the published sc-ImmuAging numbers (within tolerance) validates that preprocessing and donor assignments are consistent with the original work; any unexplained deviation must be diagnosed before proceeding to model comparisons.

## Success criteria

- LASSO m.a.e. on the sc-ImmuAging internal train/test split (864/217 donors) matches the published Extended Data Table 2 values within 15% for all five PBMC cell types (CD4+ T, CD8+ T, monocytes, NK, B cells). Deviations beyond 15% require diagnosis and a written explanation before proceeding.
- LOCO m.a.e. reported for LASSO, random forest, and PointNet across all five LOCO folds (and leave-one-chemistry-out), for all five cell types, using the frozen fold assignments from Phase 1. This is the primary comparison table that all foundation-model results in Phase 3–4 will be measured against.
- Pasta architecture retrained on PBMC pseudobulk (per-donor aggregated counts per cell type) on the same LOCO folds, reporting per-fold m.a.e. alongside the Pasta-as-released transfer baseline; any non-trivial pseudobulk adaptation decisions (gene-selection, sparsity handling) documented in `methods/pasta_pseudobulk_notes.md`.

## Tasks

- [ ] Task: Retrain sc-ImmuAging LASSO/RF/PointNet on the project's LOCO splits. Using the sc-ImmuAging codebase (https://github.com/CiiM-Bioinformatics-group/scImmuAging), fit each of the three model classes on each LOCO training fold and evaluate on the held-out cohort, using the top-2,000 HVGs selected on the training-fold union per scanpy `highly_variable_genes` defaults. Record per-fold per-cell-type median absolute error (m.a.e.) and Pearson R to `results/baselines/loco_baseline_table.csv`. Validate the internal split (non-LOCO 80/20) against Extended Data Table 2; flag any cell type where the discrepancy exceeds 15% for investigation before Phase 3 starts.

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
