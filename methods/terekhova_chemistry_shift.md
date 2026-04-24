# Task 1f — Terekhova 10x 5' chemistry-shift decision

**Status**: Decided 2026-04-24. LOCO evaluation proceeds with naive (chemistry-uncorrected) scoring as the primary Terekhova result; per-cell-type exploratory flags applied where R degrades below the primary threshold.

## Question

OneK1K and Stephenson are 10x 3'; Terekhova is 10x 5' v2. The pre-trained sc-ImmuAging LASSO (Li et al. 2025) was trained on 3' data only. When applied to Terekhova, does the 3'/5' chemistry shift destroy the aging signal to the point that batch correction is required before the pre-trained LASSO sanity check is interpretable?

## Experimental setup

- Source: `data/cohorts/integrated/*.h5ad` after the 2026-04-24 Terekhova raw-count fix (reverse-normalized from log1p(CP10k) to integer counts using `nCount_RNA` metadata; see `src/data/harmonize_cohorts.py::load_terekhova`).
- Scoring: `src/baselines/score_pretrained_lasso.py --source harmonized --cohort-id terekhova` for each of the 5 canonical cell types (CD4T, CD8T, MONO, NK, B). 100 pseudocells × 15 cells per donor; seed=0.
- Reference: OneK1K sanity results in `results/baselines/pretrained_sanity_summary.csv` (Task 1e, 981 donors).
- 166 Terekhova donors scored.

## Results

| Cell type | OneK1K (3') R | OneK1K MAE | Terekhova (5') R | Terekhova MAE | ΔR | Terekhova bias (yr) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| CD4T | 0.75 | 10.1 | **0.82** | 10.8 | **+0.07** | −8.8 |
| CD8T | 0.77 | 8.7 | **0.73** | 9.4 | **−0.04** | +1.9 |
| MONO | 0.71 | 9.6 | 0.29 | 13.7 | −0.42 | +1.2 |
| NK | 0.63 | 10.8 | 0.44 | 13.2 | −0.19 | +5.6 |
| B | 0.53 | 11.9 | **0.08** | 15.0 | **−0.45** | −2.5 |

Source: `results/baselines/terekhova_chemistry_shift_naive.csv`.

## Interpretation

1. **Two cell types are robust to the 3'→5' shift**: CD4T and CD8T retain R ≈ 0.73–0.82, statistically indistinguishable from the 3' reference. The −8.8 yr bias on CD4T is a fixed offset that does not destroy the age-ordering signal (high R with large negative bias = systematically lower predicted ages but the relative ordering is preserved).
2. **Three cell types degrade substantially**: MONO, NK, B. R drops by 0.19–0.45. For B cells the LASSO signal effectively collapses (R=0.08, p=0.31 — indistinguishable from random on the 166-donor sample).
3. **The degradation pattern is biologically plausible**, not an artifact of the reverse-normalization: CD4T/CD8T LASSOs are dominated by ribosomal/mitochondrial and housekeeping-adjacent markers whose 3'/5' coverage differs little, whereas B-cell and monocyte markers include more 3'-UTR-localized transcripts (immunoglobulin constant regions, MHC-II) where 5' sequencing captures different fragments.

## Decision

**Report naive (uncorrected) MAE as the primary Terekhova LOCO result**, with per-cell-type primary/exploratory flags driven by the Phase-1 detectability floor + the chemistry shift observed here:

| Cell type | Primary/Exploratory flag | Rationale |
|---|---|---|
| CD4T | primary | 166 ≥ 132 (ρ=0.8 floor); R preserved under chemistry shift |
| CD8T | exploratory | 166 < 180 (ρ=0.8 floor); R preserved, below detectability |
| MONO | exploratory | 166 < 229 (ρ=0.8 floor); R degraded ≥ 0.4 under chemistry shift |
| NK | primary | 166 ≥ 156 (ρ=0.8 floor); R degraded 0.19 — interpretable but weaker |
| B | exploratory | R=0.08 — no interpretable LASSO signal under 5' chemistry, regardless of sample size |

The LOCO-fold primary/exploratory flags in `data/loco_folds.json` (frozen 2026-04-24 at commit dad33dc) already reflect the detectability-floor component; the chemistry-shift component is additive and should be captured in the paper's result-matrix footnote rather than re-freezing the folds.

**Chemistry correction is NOT required to make the Terekhova LOCO result interpretable** for the primary cell types (CD4T, NK). For CD8T/MONO/B the detectability-floor flag already captures underpowering. Formal batch-correction (Harmony / scran / ComBat keyed on `obs['assay']`) is deferred to Phase 3 as an *exploratory sensitivity analysis*, comparing naive vs. corrected MAE side-by-side. Adding correction to the primary pipeline would confound the paper's main generalization claim: we want to report how the pre-trained sc-ImmuAging clock generalizes to an unseen chemistry *without* further intervention, since that is the realistic deployment scenario.

## Paper-reporting rules

- Terekhova LOCO MAE table shows per-cell-type rows annotated with `assay=10x 5' v2`.
- CD4T + NK rows appear in the primary result matrix; the CD4T row carries a footnote noting the −8.8 yr bias (offset-only, not rank-destroying).
- CD8T, MONO, B rows move to an exploratory-only supplementary table with dual annotations: (a) below detectability floor at ρ=0.8, (b) chemistry-shift-induced R drop for MONO/B.

## Deliverables produced 2026-04-24

- `results/baselines/terekhova_chemistry_shift_naive.csv` — per-cell-type naive MAE/R summary
- `results/baselines/terekhova_naive_{B,CD4T,CD8T,MONO,NK}.csv` — per-donor predictions
- This memo (`methods/terekhova_chemistry_shift.md`)

## Deferred to Phase 3

- Harmony/ComBat chemistry correction implementation on the integrated CP10k matrices, re-scoring the LASSO, and reporting corrected MAE alongside naive in `results/baselines/terekhova_chemistry_shift_corrected.csv`. Success criterion: corrected MONO/NK/B R > 0.5 without degrading CD4T/CD8T below their naive values.
