# Roadmap

## Phases

- [x] **Phase 1 — Data ingestion & infrastructure.** See [phase-1](./roadmap/phase-1.md). *(Closed 2026-04-24.)*
      Three cohorts (OneK1K, Stephenson, Terekhova — Barreiro/Randolph dropped, tracked in `FUTURE_WORK.md`) harmonized to `data/cohorts/integrated/{B,CD4p_T,CD8p_T,Monocyte,NK}.h5ad` — 2.75M cells × 48,968 genes, raw integer counts (Terekhova reverse-normalized from log1p(CP10k) via metadata `nCount_RNA`; commit c220d92). Frozen immutable artifacts: `data/cohort_summary.csv`, `data/leakage_audit.csv` (16 rows, no `unknown`), `data/loco_folds.json`, `data/aida_split.json` (625 donors → 307/318), `data/detectability_floor.json` (ρ-sensitivity sweep), `data/checkpoint_hashes.txt` (all 4 FMs smoke-tested). Findings that reshape Phase 3/4: (a) `loco_terekhova` is the only leakage-clean fold across all 4 FMs — **replaces OneK1K as headline**; (b) Terekhova 10x 5' chemistry degrades the pre-trained LASSO cell-type-selectively (CD4T/CD8T robust; MONO/NK/B degrade; B collapses to R=0.08) — **reframed as Phase-4 chemistry-rescue secondary claim**; (c) detectability-floor + chemistry-shift stacking reduces the strict-clean headline cell count from 20 to ~10 (see `methods/terekhova_chemistry_shift.md`).

- [x] **Phase 2 — Baseline establishment.** See [phase-2](./roadmap/phase-2.md). *(Closed 2026-04-25 incl. all three pre-Phase-3 add-on tasks. Panel revised mid-phase: sc-ImmuAging ships pretrained LASSO only; see `methods/loco_baselines.md`.)*
      **75-row** `results/baselines/loco_baseline_table.csv` = 4 baselines (LASSO-pretrained, LASSO-retrained-3cohort, scAgeClock, Pasta-REG) × 3 training cohorts × 5 cell types + 3 baselines × AIDA × 5 cell types, each row tagged with `leakage_status`, `chemistry_match_to_baseline_training`, `detectability_flag`. **Per-cell minimum-MAE bar on the strict-clean Terekhova fold**: Pasta CD4T 8.0y, Pasta CD8T 7.6y, LASSO MONO 12.8y, LASSO NK 12.5y, Pasta B 10.9y. **Cross-ancestry headline cell: Pasta CD4T MAE=6.3y on AIDA — the lowest MAE in the entire 75-row matrix; Phase-3 FMs must clear 5.7y for a 10% win.** Three add-on findings: (i) AIDA scoring established the cross-ancestry baseline floor before Phase-3 fine-tuning; (ii) 3-cohort retrained LASSO is essentially equivalent to pretrained 5-cohort LASSO for CD4+T/CD8+T, so the FM win on these cell types cannot be attributed to "more training data"; (iii) empirical baseline-pair pairing-ρ is 0.06–0.35, far below the Phase-1 ρ=0.8 floor — Phase-3 will measure the actual baseline-vs-FM ρ to refine detectability flags. scAgeClock surprisingly weak: R values below LASSO even on its own training cohorts. RF/PointNet retraining and Pasta retraining both deferred (upstream training data unavailable / training code not in public repo).

- [ ] **Phase 3 — CD4+ T pilot fine-tune & preprint.** See [phase-3](./roadmap/phase-3.md).
      Success: scGPT and Geneformer LoRA fine-tunes report per-fold LOCO m.a.e. for CD4+ T cells; GPU-hours per run calibrated from measured pilot; m.a.e.-detectability floor reviewed using CD4+ T pilot data (paired Wilcoxon power from observed per-donor absolute residuals; any underpowered fold flagged to exploratory-only set); bioRxiv preprint posted within 10 weeks of project start (project start 2026-04-22; target ~2026-07-01).

- [ ] **Phase 4 — Full LOCO matrix, few-shot curve, and zero-shot transfer.** See [phase-4](./roadmap/phase-4.md).
      Success: LoRA result matrix complete for all four models × five cell types × surviving primary folds, with forest plot of per-fold Fisher-z Pearson R effect sizes produced via random-effects meta-analysis; AIDA ancestry-shift m.a.e. evaluated on the holdout half of the frozen AIDA split; few-shot downsampling curve produced for ≥2 cell types (using the best-performing LoRA foundation model from the primary LOCO matrix); zero-shot cell-type transfer R reported on ≥3 surviving joint LOCO folds; full fine-tune ablation complete on one model × one cell type × one fold.

- [ ] **Phase 5 — Biological readouts & benchmark harness release.** See [phase-5](./roadmap/phase-5.md).
      Success: age-axis cosine-similarity analysis complete for all FM × cell-type pairs with permutation p-values (N = 10,000) and AIDA cross-ancestry alignment readout; SHAP attribution table produced; in-silico perturbation run for the top-5 SHAP-attributed genes per cell type (≥1 foundation model), with direction of embedding shift toward younger/older phenotype reported; benchmark harness v1 released with frozen splits, checkpoint hashes, and scanpy/PyTorch/peft version pins.

## Cross-phase notes

1. **Leakage audit gates Phase 4 scope.** *(Resolved 2026-04-24.)* `data/leakage_audit.csv` is frozen: Geneformer is the only FM clean across all cohorts; scGPT/UCE overlap OneK1K+Stephenson via CellxGene Census; scFoundation overlaps Stephenson via HCA-Covid19PBMC. Phase 4 now uses a three-way stratification (`leakage_status` × `chemistry_match_to_baseline_training` × `detectability_flag`) with three aggregation modes — strict-clean headline, leakage-inclusive, chemistry-inclusive. See `methods/leakage_audit_notes.md` and `methods/terekhova_chemistry_shift.md`.

2. **GPU-hours calibration gates Phase 4 scheduling.** Phase 3's measured per-run GPU cost is a hard prerequisite for booking Phase 4 compute; Phase 4 scheduling cannot be committed until Phase 3 reports this figure.

3. **AIDA split frozen in Phase 1; consumed across Phases 4 and 5.** *(Frozen 2026-04-24 in `data/aida_split.json`: 625 donors → 307 ancestry_shift_mae + 318 age_axis_alignment, stratified over 35 age_decile × ethnicity strata.)* The ancestry-shift m.a.e. half is used for evaluation in Phase 4; the age-axis cosine-alignment half is used in Phase 5.

4. **Phase 3 is the go/no-go gate for the primary claim.** *(Updated 2026-04-25 after Phase-2 baseline numbers landed.)* The Phase-3 CD4+T pilot must clear **the per-cell minimum-MAE baseline from {LASSO, scAgeClock, Pasta-REG}** by ≥10% relative m.a.e. on the CD4+T LOCO primary folds. Per Phase-2 results, the targets are: OneK1K CD4T ≤8.5y (LASSO 9.4y); **Terekhova CD4T ≤7.2y (Pasta-REG 8.0y — the headline-tightening cell)**. Pre-Phase-2 success-probability estimate was ~80% against LASSO alone; revised post-Phase-2 estimate is **~50–60% against the per-cell minimum** because Pasta's chemistry-invariance pulls Terekhova MAE below LASSO. Three outcome modes: *headline win* (clears 10% margin → preprint posts the headline claim), *match* (within ±5% of best baseline → preprint reframes to "FMs match Pasta at lower compute on three cohorts" — still publishable), *loss* (FM loses to Pasta → Phase 4 narrows to zero-shot + few-shot only and the paper pivots to an evaluation study). The preprint posted in Phase 3 already frames the match/loss possibilities, so a pivot requires revision not retraction.

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
