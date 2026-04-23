# Phase 3 — CD4+ T pilot fine-tune & preprint

## Goal

With baselines in hand, this phase produces the minimal viable result: LoRA fine-tuning of scGPT and Geneformer on the CD4+ T cell LOCO folds, compared head-to-head against the Phase 2 baselines on identical splits. CD4+ T is chosen first because it has the strongest sc-ImmuAging internal R (≈0.91) and the largest per-cohort donor pool, making it the best-powered starting point. This phase also calibrates the true GPU-hours-per-run from measured timing (not the ≈4 h pilot estimate in the proposal), which gates the feasibility of Phase 4's full matrix. The phase ends with a bioRxiv preprint submission covering the CD4+ T LOCO result — the sprint-to-preprint mitigation for scoop risk.

## Success criteria

- scGPT and Geneformer LoRA (rank-16, attention + MLP layers, peft library) fine-tuned on all five CD4+ T LOCO training folds; per-fold LOCO m.a.e. and Pearson R recorded for both models and compared against Phase 2 LASSO/RF/PointNet numbers on the same folds.
- Measured GPU-hours per LoRA run (mean and SD across three random seeds) recorded in `compute/runtime_log.csv`; Phase 4 full-matrix time estimate updated from this measurement before Phase 4 begins.
- m.a.e.-detectability floor reviewed using CD4+ T pilot data: compute paired Wilcoxon power from observed per-donor absolute residuals; flag any CD4+ T fold that remains underpowered under the 80% power / α = 0.05 criterion and add it to the exploratory-only set.
- bioRxiv preprint posted within 10 weeks of project start; manuscript covers CD4+ T LOCO result for scGPT + Geneformer vs. sc-ImmuAging baselines, the leakage audit, and the frozen split design. At this point the paper is "minimal viable" regardless of whether criterion (a) is met.

## Tasks

- [ ] Task: Implement LoRA fine-tuning wrapper for scGPT and Geneformer. Using peft (LoRA rank-16 applied to attention + MLP projection layers), build training scripts that accept a cell-type label, a LOCO fold index, and a random seed; output a model checkpoint and per-donor age predictions on the held-out cohort. Run three seeds per (model, fold) combination for CD4+ T cells. Record wall-clock time and peak GPU memory per run. Done when reproducible per-fold predictions exist for all five CD4+ T LOCO folds, both models, three seeds.

- [ ] Task: Evaluate CD4+ T pilot results and update detectability floor. Compute per-fold LOCO m.a.e. (median absolute error) and Pearson R for scGPT-LoRA and Geneformer-LoRA; compare against Phase 2 LASSO baseline on the same folds. Run paired Wilcoxon power calculation on the observed per-donor absolute residuals (baseline vs. fine-tuned) per fold; update `data/loco_folds.json` to flag any fold where the 10% relative m.a.e. reduction is undetectable at 80% power. Write results to `results/phase3/cd4t_loco_table.csv`. Done when the results table is committed and all underpowered folds are flagged.

- [ ] Task: Prepare and submit bioRxiv preprint. Draft manuscript sections covering: (1) the LOCO split design and leakage audit; (2) CD4+ T LoRA fine-tune results (scGPT + Geneformer) vs. LASSO/RF/PointNet baselines; (3) GPU compute envelope and runtime calibration. Include the frozen split files and checkpoint hashes as supplementary data. Submit preprint to bioRxiv. Done when the bioRxiv DOI is confirmed (within 10 weeks of project start).

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
    "title": "scGPT: toward building a foundation model for single-cell multi-omics using generative AI",
    "url": "https://www.nature.com/articles/s41592-024-02201-0",
    "authors": "Cui et al.",
    "year": 2024,
    "venue": "Nature Methods"
  },
  {
    "title": "Transfer learning enables predictions in network biology (Geneformer)",
    "url": "https://www.nature.com/articles/s41586-023-06139-9",
    "authors": "Theodoris et al.",
    "year": 2023,
    "venue": "Nature"
  },
  {
    "title": "Zero-shot evaluation reveals limitations of single-cell foundation models",
    "url": "https://link.springer.com/article/10.1186/s13059-025-03574-x",
    "authors": "Kedzierska et al.",
    "year": 2025,
    "venue": "Genome Biology"
  }
]
```
