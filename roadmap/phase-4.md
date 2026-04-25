# Phase 4 — Full LOCO matrix, few-shot curve, and zero-shot transfer

## Goal

Phase 3 produced the CD4+ T pilot with Geneformer + scFoundation + scGPT; this phase extends the fine-tuning sweep to **all four foundation models** (scGPT, Geneformer, scFoundation, UCE — UCE enters in Phase 4), **all five PBMC cell types**, and **all surviving primary LOCO folds**. It then runs the two regime-specific experiments that form the paper's secondary claims: the few-shot downsampling curve targeting the low-data B- and NK-cell regime (success criterion b), and the zero-shot cell-type transfer experiment with joint leave-one-donor × leave-one-cell-type holdout design (success criterion c). A single full fine-tune ablation (one model × one cell type × one fold) completes the LoRA-as-regularizer control. All results are aggregated via random-effects meta-analysis over per-fold effect sizes for the headline claim.

### Three-way stratified reporting policy (recorded 2026-04-24, updated after Phase-1 Task 1f)

Every result row in `results/phase4/full_loco_matrix.csv` carries three independent stratification columns. A row is **strict-clean** (paper headline) only when all three are green:

| Column | Values | Source | Green = |
|---|---|---|---|
| `leakage_status` | `clean` / `overlapping` | `data/leakage_audit.csv` | `clean` |
| `chemistry_match_to_baseline_training` | `match` / `shifted` | cohort → chemistry lookup: OneK1K + Stephenson = 10x 3' = `match`; Terekhova = 10x 5' = `shifted` (pre-trained LASSO was trained on 3' only — see `methods/terekhova_chemistry_shift.md`) | `match` |
| `detectability_flag` | `powered` / `underpowered` | `data/loco_folds.json` primary-flag, superseded by the `post_phase3_override` block produced by Phase 3's empirical-ρ measurement | `powered` |

Aggregation is reported in **three modes**, shown side by side:

- **Strict-clean pooled estimate** — random-effects meta-analysis restricted to rows where all three flags are green. This is the paper's **headline effect size**; for criterion (a) it determines whether the ≥10% relative MAE reduction claim holds.
- **Leakage-inclusive pooled estimate** — relax `leakage_status` only; keep `chemistry_match=match` and `detectability=powered`. Overlapping rows downweighted 1/2. Measures robustness of the headline to pretraining-corpus leakage.
- **Chemistry-inclusive pooled estimate** — relax `chemistry_match_to_baseline_training` only; keep clean and powered. Shifted rows downweighted 1/2. Directly tests the Phase-1 chemistry-rescue hypothesis: do FMs generalize across chemistries better than the pre-trained LASSO (which collapsed on 5' B cells in Task 1f)?

When any two modes disagree in direction or cross the 10% threshold, the paper reports the disagreement explicitly rather than picking one. `data/leakage_audit.csv` + `methods/leakage_audit_notes.md` is cited for `leakage_status`; `methods/terekhova_chemistry_shift.md` is cited for the chemistry-match dimension.

### Per-cell-type primary vs. exploratory assignments (Phase-1 ρ=0.8 floor + Phase-1 chemistry shift)

Combines `data/detectability_floor.json` (Phase 1 output, ρ=0.8 assumption — superseded per-row by Phase 3's `post_phase3_override`) with the chemistry-shift-selective R degradation observed for the pre-trained LASSO in Task 1f (`methods/terekhova_chemistry_shift.md`):

| Cell type | loco_onek1k (981 donors, 3') | loco_terekhova (166 donors, 5') | loco_stephenson (29 donors, 3') |
|---|:---:|:---:|:---:|
| CD4+ T | primary | primary (R=0.82 on LASSO — chemistry-robust) | exploratory |
| CD8+ T | primary | **exploratory** (166 < 180 detectability floor; but chemistry-robust per Task 1f, so may flip to primary if Phase 3 measures ρ ≥ 0.9) | exploratory |
| Monocyte | primary | **exploratory** (under-powered AND chemistry-degraded: LASSO R=0.29 on 5') | exploratory |
| NK | primary | primary for FMs (powered); LASSO exploratory (R=0.44 chemistry-degraded) | exploratory |
| B | primary | **chemistry-rescue target**: LASSO R=0.08 on 5' (collapsed). Primary for FMs if they recover R ≥ 0.5; if not, exploratory. | exploratory |

`loco_onek1k` × {scGPT, UCE} cells are `overlapping` — included in the leakage-inclusive pooled estimate only. `loco_terekhova` cells are all `clean` but all `shifted` — included in the chemistry-inclusive pooled estimate. The headline **strict-clean-AND-chemistry-match-AND-powered** cells are limited to `loco_onek1k` × {Geneformer, scFoundation} × {all 5 cell types} = **10 cells** (reduced from the original plan's 20). Phase 3's CD4+ T pilot adds a chemistry-inclusive entry for `loco_terekhova` × Geneformer/scFoundation.

### Baseline panel (revised 2026-04-25)

Phase 4 reports the FM full-LOCO matrix against the Phase-2 baseline panel **{LASSO, scAgeClock, Pasta-REG}**, NOT a single sc-ImmuAging baseline. Each (cohort × cell type) cell carries three baseline numbers; the FM's "headline win" is computed against the *minimum-MAE baseline per cell* (the "beat-best-baseline" rule from Phase 3). The `leakage_status` stratification column extends to scAgeClock rows: scAgeClock × {OneK1K, Stephenson} are `overlapping` (CELLxGENE Census), scAgeClock × {Terekhova, AIDA} are `clean` (recorded in `data/leakage_audit.csv` from the Phase 2 Task 2.2 audit). For `loco_terekhova`, all three baselines and all four FMs are leakage-clean → this is the only fold where the strict-clean meta-analysis includes scAgeClock as a comparator.

## Success criteria

- LoRA fine-tune result matrix complete for all four models × five cell types × surviving primary LOCO folds (≥3 seeds per cell); per-fold per-cell-type m.a.e. recorded in `results/phase4/full_loco_matrix.csv` with **three stratification columns** (`leakage_status`, `chemistry_match_to_baseline_training`, `detectability_flag`) on every row; **per-cell `vs_best_baseline_pct` column** giving relative MAE reduction vs. the minimum of {LASSO, scAgeClock, Pasta-REG} for that cell. The headline forest plot reports **three metrics side-by-side per fold** (median absolute error, mean bias, Pearson R) — single-metric reporting hides Pasta's high-bias-but-good-rank failure mode (e.g. Pasta on OneK1K has systematic −23y bias but R=0.60; MAE alone penalises it, R alone validates it; both panels needed). Clean vs overlapping rows rendered in distinct glyphs; match vs shifted chemistry rendered as complementary markers.
- Dual-regime few-shot downsampling curve produced for CD4+ T (chemistry-robust pre-trained baseline — LASSO R=0.82 on 5' Terekhova) and B cells (chemistry-collapsed pre-trained baseline — LASSO R=0.08 on 5' Terekhova; Pasta R=0.28 chemistry-rescue). On each cell type, run downsampling to 75%, 50%, 25%, 10% of training donors for **three** model lines: LASSO (count-based), Pasta-REG (rank-norm bulk; chemistry-invariant per Phase-2), and the best-performing LoRA foundation model. The 3-line plot tests the Phase-2 question "does single-cell FM pretraining add chemistry-invariance on top of rank-normalized bulk?" Crossover thresholds (if any) documented per cell type.
- Zero-shot cell-type transfer: at least three surviving joint LOCO folds (train cell type A on donor set A, evaluate cell type B on disjoint donor set B, both sets ≥80 donors); Pearson R and m.a.e. reported for each surviving fold. **Comparator: Pasta-REG predicting on the target cell type's pseudobulk directly** (Pasta is cell-type-agnostic by construction — its pseudobulk operates on whatever cells are passed in, so it gives a "free" zero-shot baseline). The interesting test is whether single-cell FM cross-cell-type embedding alignment beats Pasta's naive rank-norm-on-pseudobulk approach. The original "sc-ImmuAging cannot transfer by construction" framing remains a secondary contrast.
- Full fine-tune ablation complete on one foundation model × one cell type × one LOCO fold, reporting whether full fine-tuning materially changes ranking relative to LoRA; decision documented on whether to promote full fine-tune to primary reporting.

## Tasks

- [ ] Task: Extend LoRA fine-tuning to all models × cell types × LOCO folds. Using the training scripts from Phase 3, run LoRA rank-16 fine-tunes for scFoundation and UCE (in addition to the scGPT and Geneformer runs already complete) across all five PBMC cell types and all surviving primary LOCO folds, three seeds each. Schedule runs according to the Phase 3–calibrated GPU-hours estimate. Tag every row with the three stratification columns (`leakage_status` from `data/leakage_audit.csv`, `chemistry_match_to_baseline_training` from the cohort-chemistry lookup, `detectability_flag` from `data/loco_folds.json` + `post_phase3_override`). Write all per-fold results to `results/phase4/full_loco_matrix.csv`. Done when all surviving primary-fold cells are populated and the forest plot is generated in each of the three aggregation modes.

- [ ] Task: Run dual-regime few-shot downsampling curve (CD4+ T and B cells). For each cell type, downsample training donors to 75%, 50%, 25%, 10% of the full training set, and produce a **3-line plot per fold**: LASSO (count-based, chemistry-sensitive), Pasta-REG (rank-norm bulk, chemistry-invariant per Phase-2), and the best-performing LoRA foundation model. Pasta is included because Phase-2 revealed it as the actual chemistry-rescue baseline (Pasta-B R=0.28 on 5' Terekhova vs LASSO R=0.08); without Pasta in the few-shot plot the FM win on B cells is overstated. For B cells specifically, include a `loco_terekhova` (5' chemistry) arm alongside the `loco_onek1k` arm — the chemistry-rescue test now reads "FMs add R *on top of* Pasta's rank-normalization", not "FMs rescue where LASSO collapses." Identify any crossover points (sample size below which LASSO/Pasta beat the FM). Done when `results/phase4/fewshot_curves/{cd4t,bcell}_{onek1k,terekhova}.csv` are committed and crossover thresholds are documented in `methods/fewshot_analysis.md`.

- [ ] Task: Conditionally run Phase-1-deferred chemistry correction on Terekhova. Trigger condition: if the B-cell few-shot curve on `loco_terekhova` shows Geneformer/scFoundation R < 0.5 at the full-data level (i.e. FMs also collapse on 5' B cells), implement Harmony or ComBat correction keyed on `obs['assay']` for B cells; otherwise skip the correction entirely (FMs handle the chemistry shift without explicit correction — a stronger result than "correction helps"). Document the trigger decision in `methods/terekhova_chemistry_shift.md` as an addendum. Done when either the correction is implemented and `results/baselines/terekhova_chemistry_shift_corrected.csv` is committed, OR the trigger is recorded as not-fired with the supporting R values cited.

- [ ] Task: Run zero-shot cell-type transfer with joint holdout. For each surviving (source cell type, target cell type, cohort) triple with ≥80 donors on both sides of the donor split: train a ridge probe on frozen foundation-model embeddings of the source cell type on donor set A; evaluate age prediction on the target cell type on disjoint donor set B. Compute Pearson R and m.a.e. per fold. Apply the permutation null (N = 1,000 donor-label shuffles within each group) to establish significance. **Comparator on each fold: Pasta-REG run directly on donor set B's target cell type pseudobulk** (Pasta is cell-type-agnostic by construction; this gives a "free" zero-shot baseline that doesn't use cross-cell-type embedding alignment at all). The criterion-(c) win condition becomes "FM cross-cell-type transfer R > Pasta-on-target-cell-type R per fold." Record both numbers in `results/phase4/zero_shot_transfer.csv`; report aggregated effect sizes via random-effects meta-analysis across surviving folds. Done when at least three primary folds are evaluated against both the FM-cross-cell-type and the Pasta-on-target baselines.

- [ ] Task: Full fine-tune ablation and meta-analysis. Run full parameter fine-tuning (not LoRA) on one foundation model (the best-performing from the full LOCO matrix), one cell type (CD4+ T), one LOCO fold, three seeds. Compare full fine-tune vs. LoRA m.a.e. and document the finding in `results/phase4/full_finetune_ablation.md`; if full fine-tune changes the ranking across baselines by more than 5% relative m.a.e., escalate to primary reporting for that model. Separately, aggregate all primary LOCO folds via random-effects meta-analysis (Fisher-z R for criterion c; paired log-ratio of m.a.e. for criterion a) **vs. the per-cell minimum-MAE baseline** from {LASSO, scAgeClock, Pasta-REG}, in **all three** modes (strict-clean, leakage-inclusive, chemistry-inclusive — see "Three-way stratified reporting policy" above). The strict-clean mode of `loco_terekhova` includes scAgeClock as a clean comparator; on `loco_onek1k` strict-clean, scAgeClock rows are leakage-flagged and move to leakage-inclusive only. Produce the headline forest plot in `results/phase4/forest_plot.pdf` with FM-vs-best-baseline log-MAE-ratio per fold, clean-vs-overlapping glyphs distinguished, and match-vs-shifted chemistry rendered as complementary markers. Done when ablation is complete, all three meta-analysis modes are run against the per-cell-best baseline, and the forest plot is committed.

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
    "title": "Universal Cell Embeddings: A Foundation Model for Cell Biology (UCE)",
    "url": "https://www.biorxiv.org/content/10.1101/2023.11.28.568918v2",
    "authors": "Rosen, Roohani, Agrawal, Samotorcan, Tabula Sapiens Consortium, Quake, Leskovec",
    "year": 2023,
    "venue": "bioRxiv"
  },
  {
    "title": "Zero-shot evaluation reveals limitations of single-cell foundation models",
    "url": "https://link.springer.com/article/10.1186/s13059-025-03574-x",
    "authors": "Kedzierska et al.",
    "year": 2025,
    "venue": "Genome Biology"
  },
  {
    "title": "Discovering Candidate Anti-Aging Perturbations Using a Foundation Model for Gene Expression",
    "url": "https://www.mdpi.com/1422-0067/26/24/11977",
    "authors": "Tadevosyan, Efimov, Kriukov, Khrameeva",
    "year": 2025,
    "venue": "International Journal of Molecular Sciences"
  }
]
```
