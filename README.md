# Foundation-Model Fine-Tuning for Cell-Type-Specific Immune Aging Clocks

## Research Question

Do general-purpose single-cell foundation models (scGPT, Geneformer, scFoundation, UCE), when fine-tuned for continuous age regression, outperform the **strongest published immune-aging baselines** (sc-ImmuAging LASSO, scAgeClock, Pasta-REG) on per-PBMC-cell-type aging clocks — specifically when labelled donors are scarce and cohorts shift?

## What Success Looks Like

At least one of three orthogonal criteria must hold in at least one PBMC cell type:

- **(a) LOCO accuracy.** LoRA-fine-tuned foundation models reduce per-donor LOCO **median absolute error** by ≥10% relative vs. the **per-(cohort, cell type) minimum** of {sc-ImmuAging LASSO, scAgeClock, Pasta-REG} on the same matched splits. The "min-of-3" rule ensures FMs are credited for beating the *strongest* baseline per cell, not the weakest. Phase-2 numbers set the bars: e.g. **Terekhova CD4+ T must clear 7.2y** (Pasta 8.0y is the floor) and OneK1K CD4+ T must clear 8.5y (LASSO 9.4y).
- **(b) Few-shot crossover.** Foundation-model fine-tunes win below some labelled-donor threshold where pretraining-free baselines (LASSO, Pasta) win at full data, producing a publishable "when pretraining pays" curve. After Phase 2's chemistry-rescue finding (Pasta-B R=0.28 on Terekhova vs LASSO R=0.08), the B-cell few-shot test reads "do FMs add R *on top of* rank-normalized bulk?", not just "do FMs rescue where LASSO collapses."
- **(c) Zero-shot cell-type transfer.** Foundation-model embeddings trained on one cell type achieve Pearson R > 0.3 on a held-out cell type with jointly held-out donors **and beat Pasta-REG run directly on the target cell type's pseudobulk** (Pasta is cell-type-agnostic by construction — the criterion tests whether single-cell representation buys anything over rank-normalized bulk).

Concrete phase milestones (from ROADMAP):

| Phase | Status | Key deliverable |
|---|---|---|
| Phase 1 | ✅ closed 2026-04-24 | Three cohorts harmonized to integer counts (OneK1K + Stephenson + Terekhova; Barreiro dropped, see `FUTURE_WORK.md`); 20-row leakage-audit table (5 models × 4 cohorts); LOCO folds frozen with ρ=0.8 detectability floor; AIDA 50/50 split frozen |
| Phase 2 | ✅ closed 2026-04-25 | 45-row `results/baselines/loco_baseline_table.csv` (LASSO + scAgeClock + Pasta-REG × 3 cohorts × 5 cell types). Best-baseline-per-cell tables in `methods/loco_baselines.md` define the Phase-3/4 FM bar |
| Phase 3 | pending | LoRA-fine-tuned Geneformer + scFoundation + scGPT on CD4+ T LOCO; preprint by ~2026-07-01 |
| Phase 4 | pending | Full 4-FM × 5-cell-type LOCO matrix; AIDA ancestry-shift m.a.e.; few-shot curve (CD4+T + B); zero-shot cell-type transfer; full fine-tune ablation |
| Phase 5 | pending | Age-axis cosine-similarity analysis; SHAP attribution; in-silico perturbation; benchmark harness v1 |

A negative result (foundation-model fine-tunes strictly worse than the per-cell minimum baseline across every regime) is also a publishable outcome and is explicitly designed for. The "FMs beat the *strongest* published baseline" framing makes a null finding more publishable than the original "FMs beat LASSO" framing because the comparison panel is now a competitive set rather than a single weak target.

## Approach

This is a five-phase computational methods study requiring no wet-lab work or patient-data access. The training corpus is **1,176 healthy PBMC donors** assembled from three public cohorts (981 OneK1K Yazar 2022 via CellxGene; 29 Stephenson 2021 healthy controls via CellxGene; 166 Terekhova 2023 *Immunity* via Synapse). Four foundation models — scGPT (Cui et al., *Nature Methods* 2024), Geneformer (Theodoris et al., *Nature* 2023), scFoundation (Hao et al., *Nature Methods* 2024), and UCE (Rosen, Leskovec et al., bioRxiv 2023) — are fine-tuned with LoRA (rank-16, peft library) as continuous age regressors, one per PBMC cell type (CD4+ T, CD8+ T, monocytes, NK, B cells). Evaluation uses leave-one-cohort-out (LOCO) and leave-one-chemistry-out protocols; the low-data regime is tested via downsampling to match B- and NK-cell donor counts, where pretraining is hypothesised to pay most. The Phase-2 baseline panel is **{sc-ImmuAging LASSO (Li et al. 2025, pretrained), scAgeClock (Xie 2026, deep-learning, attention-based, CELLxGENE-Census-pretrained), Pasta-REG (Salignon 2025, rank-normalized bulk-transcriptomic)}**; sc-ImmuAging's RF and PointNet were dropped from the primary panel because the public package ships only the LASSO `cv.glmnet` weights (see `methods/loco_baselines.md`). Biological readouts — age-axis geometry in the foundation-model latent space, SHAP attribution, and in-silico perturbation — are delivered alongside the horse-race result. An external ancestry-shift evaluation uses AIDA (Kock et al., *Cell* 2025; 625 donors, 7 Asian population groups), split 50/50 and frozen before any model training. A leakage-audited benchmark harness (v1, frozen splits, checkpoint hashes, version-pinned dependencies, and the three reproducible Phase-2 baselines) is released with the preprint.

## Documents

| Document | Contents |
|---|---|
| [LANDSCAPE.md](./LANDSCAPE.md) | Related work, existing codebases, target benchmarks, data sources, and references — verified 2026-04-23 |
| [ROADMAP.md](./ROADMAP.md) | Five-phase execution plan with per-phase success criteria and cross-phase dependency notes |
| [FUTURE_WORK.md](./FUTURE_WORK.md) | Stretch goals and adjacent research directions beyond the current roadmap |
