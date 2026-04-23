# Roadmap

## Phases

- [ ] **Phase 1 — Data ingestion & infrastructure.** See [phase-1](./roadmap/phase-1.md).
      Success: all five sc-ImmuAging cohorts processed to AnnData; leakage-audit table produced for all four foundation models × five cohorts; LOCO fold matrix frozen with ≥80-donor filter applied; AIDA 50/50 split frozen before any model is trained.

- [ ] **Phase 2 — Baseline reproduction.** See [phase-2](./roadmap/phase-2.md).
      Success: LASSO m.a.e. on the internal sc-ImmuAging split matches published Extended Data Table 2 values within 15% for all five cell types; LOCO m.a.e. recorded for LASSO, random forest, and PointNet across all five LOCO folds × five cell types in `results/baselines/loco_baseline_table.csv`; Pasta architecture retrained on PBMC pseudobulk produces per-cohort m.a.e. on the same LOCO folds.

- [ ] **Phase 3 — CD4+ T pilot fine-tune & preprint.** See [phase-3](./roadmap/phase-3.md).
      Success: scGPT and Geneformer LoRA fine-tunes report per-fold LOCO m.a.e. for CD4+ T cells; GPU-hours per run calibrated from measured pilot; m.a.e.-detectability floor reviewed using CD4+ T pilot data (paired Wilcoxon power from observed per-donor absolute residuals; any underpowered fold flagged to exploratory-only set); bioRxiv preprint posted within 10 weeks of project start (project start 2026-04-22; target ~2026-07-01).

- [ ] **Phase 4 — Full LOCO matrix, few-shot curve, and zero-shot transfer.** See [phase-4](./roadmap/phase-4.md).
      Success: LoRA result matrix complete for all four models × five cell types × surviving primary folds, with forest plot of per-fold Fisher-z Pearson R effect sizes produced via random-effects meta-analysis; AIDA ancestry-shift m.a.e. evaluated on the holdout half of the frozen AIDA split; few-shot downsampling curve produced for ≥2 cell types (using the best-performing LoRA foundation model from the primary LOCO matrix); zero-shot cell-type transfer R reported on ≥3 surviving joint LOCO folds; full fine-tune ablation complete on one model × one cell type × one fold.

- [ ] **Phase 5 — Biological readouts & benchmark harness release.** See [phase-5](./roadmap/phase-5.md).
      Success: age-axis cosine-similarity analysis complete for all FM × cell-type pairs with permutation p-values (N = 10,000) and AIDA cross-ancestry alignment readout; SHAP attribution table produced; in-silico perturbation run for the top-5 SHAP-attributed genes per cell type (≥1 foundation model), with direction of embedding shift toward younger/older phenotype reported; benchmark harness v1 released with frozen splits, checkpoint hashes, and scanpy/PyTorch/peft version pins.

## Cross-phase notes

1. **Leakage audit gates Phase 4 scope.** The Phase 1 leakage-audit table determines which (model, cohort) pairs are promoted to "primary" folds vs. "leakage-restricted" folds; the size and composition of the Phase 4 primary LOCO matrix are not known until that audit is complete.

2. **GPU-hours calibration gates Phase 4 scheduling.** Phase 3's measured per-run GPU cost is a hard prerequisite for booking Phase 4 compute; Phase 4 scheduling cannot be committed until Phase 3 reports this figure.

3. **AIDA split frozen in Phase 1; consumed across Phases 4 and 5.** The 50/50 AIDA ancestry-holdout split must be frozen before any model is trained (Phase 1). The ancestry-shift m.a.e. half is used for evaluation in Phase 4; the age-axis cosine-alignment half is used in Phase 5. Both phases depend on the frozen split produced in Phase 1.

4. **Phase 3 is the go/no-go gate for the primary claim.** If both scGPT and Geneformer LoRA fine-tunes fail to improve over the LASSO baseline by more than 5% relative m.a.e. on the CD4+ T LOCO primary folds (i.e., the LoRA-outperforms-LASSO claim appears null), the project pivots before Phase 4 begins: (a) Phase 4 is narrowed to the zero-shot transfer and few-shot curve experiments only (dropping the full 4-model × 5-cell-type sweep to the best single model); (b) the paper is repositioned as an evaluation study — "Do scRNA-seq foundation models improve immune aging clocks?" — with the null fine-tuning result as the headline. The preprint posted in Phase 3 already frames this as a possibility, so the pivot requires only a revision rather than a retraction.

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
    "title": "Large-scale foundation model on single-cell transcriptomics (scFoundation)",
    "url": "https://www.nature.com/articles/s41592-024-02305-7",
    "authors": "Hao et al.",
    "year": 2024,
    "venue": "Nature Methods"
  },
  {
    "title": "Universal Cell Embeddings: A Foundation Model for Cell Biology",
    "url": "https://www.biorxiv.org/content/10.1101/2023.11.28.568918v2",
    "authors": "Rosen, Roohani, Agrawal, Samotorcan, Tabula Sapiens Consortium, Quake, Leskovec",
    "year": 2023,
    "venue": "bioRxiv"
  },
  {
    "title": "Asian diversity in human immune cells (AIDA)",
    "url": "https://www.cell.com/cell/fulltext/S0092-8674(25)00202-8",
    "authors": "Kock et al.",
    "year": 2025,
    "venue": "Cell"
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
