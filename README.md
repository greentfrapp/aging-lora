# Foundation-Model Fine-Tuning for Cell-Type-Specific Immune Aging Clocks

## Research Question

Do general-purpose single-cell foundation models (scGPT, Geneformer, scFoundation, UCE), when fine-tuned for continuous age regression, outperform task-specific architectures (sc-ImmuAging's LASSO/RF/PointNet, scAgeClock) on per-PBMC-cell-type aging clocks — specifically when labelled donors are scarce and cohorts shift?

## What Success Looks Like

At least one of three orthogonal criteria must hold in at least one PBMC cell type:

- **(a) LOCO accuracy.** LoRA-fine-tuned foundation models reduce per-donor LOCO **median absolute error** by ≥10% relative vs. sc-ImmuAging retrained baselines on the same matched splits (primary metric, matching sc-ImmuAging's Extended Data Table 2 baseline).
- **(b) Few-shot crossover.** Foundation-model fine-tunes win below some labelled-donor threshold where pretraining-free baselines win at full data, producing a publishable "when pretraining pays" curve.
- **(c) Zero-shot cell-type transfer.** UCE/scGPT embeddings trained on one cell type achieve Pearson R > 0.3 on a held-out cell type (e.g., train CD4+ T, test B cells) with jointly held-out donors — where sc-ImmuAging cannot transfer by construction.

Concrete phase milestones (from ROADMAP):

| Phase | Key deliverable |
|---|---|
| Phase 1 | Five sc-ImmuAging cohorts in AnnData; leakage-audit table; LOCO fold matrix frozen (≥80-donor filter); AIDA 50/50 split frozen |
| Phase 2 | LASSO m.a.e. within 15% of published Extended Data Table 2 values; full LOCO baseline table (`results/baselines/loco_baseline_table.csv`) |
| Phase 3 | scGPT + Geneformer LoRA results for CD4+ T; bioRxiv preprint posted by ~2026-07-01 (within 10 weeks of project start) |
| Phase 4 | Full 4-model × 5-cell-type LOCO matrix; AIDA ancestry-shift m.a.e.; few-shot curve (≥2 cell types); zero-shot transfer (≥3 folds); full fine-tune ablation |
| Phase 5 | Age-axis cosine-similarity analysis with permutation p-values; SHAP attribution; in-silico perturbation (top-5 genes per cell type); benchmark harness v1 released |

A negative result (foundation-model fine-tunes strictly worse than LASSO/RF/PointNet across every regime) is also a publishable outcome and is explicitly designed for.

## Approach

This is a five-phase computational methods study requiring no wet-lab work or patient-data access. The training corpus is the 1,081-donor European PBMC scRNA-seq dataset from sc-ImmuAging (Li et al., *Nature Aging* 2025), assembled from five public GEO/Cell Atlas cohorts (GSE158055, GSE214534, GSE155673, Stephenson COVID-19 Cell Portal, OneK1K GSE196830). Four foundation models — scGPT (Cui et al., *Nature Methods* 2024), Geneformer (Theodoris et al., *Nature* 2023), scFoundation (Hao et al., *Nature Methods* 2024), and UCE (Rosen, Leskovec et al., bioRxiv 2023) — are fine-tuned with LoRA (rank-16, peft library) as continuous age regressors, one per PBMC cell type (CD4+ T, CD8+ T, monocytes, NK, B cells). Evaluation uses leave-one-cohort-out (LOCO) and leave-one-chemistry-out protocols; the low-data regime is tested via downsampling to match B- and NK-cell donor counts, where pretraining is hypothesised to pay most. Comparators are sc-ImmuAging's retrained LASSO/RF/PointNet and Pasta (Salignon et al., bioRxiv 2025) on identical splits. Biological readouts — age-axis geometry in the foundation-model latent space, SHAP attribution, and in-silico perturbation — are delivered alongside the horse-race result. An external ancestry-shift evaluation uses AIDA (Kock et al., *Cell* 2025; 619 donors, 7 Asian population groups), split 50/50 and frozen before any model training. A leakage-audited benchmark harness (v1, frozen splits, checkpoint hashes, version-pinned dependencies) is released with the preprint.

## Documents

| Document | Contents |
|---|---|
| [LANDSCAPE.md](./LANDSCAPE.md) | Related work, existing codebases, target benchmarks, data sources, and references — verified 2026-04-23 |
| [ROADMAP.md](./ROADMAP.md) | Five-phase execution plan with per-phase success criteria and cross-phase dependency notes |
| [FUTURE_WORK.md](./FUTURE_WORK.md) | Stretch goals and adjacent research directions beyond the current roadmap |
