# Paper figures inventory

Every figure we might want, with: data source, current status, paper section it serves. Edit-in-place; mark status changes (planned → drafted → committed → polished).

Status legend:
- 📋 planned (concept only)
- ⏳ data exists, figure not drafted
- 🎨 figure drafted (e.g. saved PDF)
- ✅ committed to repo + cited in manuscript
- ⭐ headline figure

---

## Phase-2 result figures (data in hand; figures to draft)

### F1 ⭐ — 4-cohort × 4-baseline LOCO MAE+R heatmap

**Status:** ⏳ data exists, figure not drafted
**Section:** Results §2.2 (baseline panel)
**Data source:** `results/baselines/loco_baseline_table.csv` (75 rows)
**Description:** A 4-row × 5-column grid (one row per baseline, one column per cell type) of MAE values, with 4 mini-cells per cell (one per cohort). Color-coded by win/lose vs the per-cell minimum-MAE bar. Pearson R as overlaid annotation. The "Pasta CD4+T 6.3y on AIDA" cell highlighted as the lowest-MAE bar.
**Files to produce:** `results/phase3/fig_baseline_loco_heatmap.{csv,pdf}`

### F2 — Leakage audit table (rendered as a styled figure)

**Status:** ⏳ data exists in `data/leakage_audit.csv` (20 rows)
**Section:** Results §2.1 (benchmark design)
**Description:** 5 models × 4 cohorts grid, color-coded `clean` (green) / `overlapping` (orange). Footnotes for the HCA-Covid19PBMC mirror finding and the CELLxGENE Census build-vs-release date discovery. Useful for the introduction even if it duplicates a supplementary table.

### F3 — Empirical pairing-ρ bracket disclosure

**Status:** ⏳ data exists in `results/baselines/empirical_pairing_rho.csv`
**Section:** Methods + Results §2.1
**Description:** Per-cell-type horizontal bracket plot showing: Phase-1 planning ρ=0.8 (top tick) | Phase-2 baseline-pair ρ (median + min/max range) | placeholder for Phase-3 baseline-vs-FM ρ. Below each bracket, the corresponding required-N for the paired-Wilcoxon detectability floor.

## Phase-3 result figures (Phase-3 fills these in)

### F4 ⭐ — CD4+T tri-headline forest plot

**Status:** 📋 planned
**Section:** Results §2.3 (CD4+T tri-headline)
**Description:** Forest plot per FM (3 columns: Geneformer / scFoundation / scGPT) × per cell (3 rows: OneK1K / Terekhova / AIDA), showing FM MAE with 95% CI vs the per-cell minimum-MAE baseline. Win/match/loss color-coded. The 9-cell figure is the manuscript's primary headline.

### F5 ⭐ — Chemistry + ancestry robustness 3×3

**Status:** 📋 planned
**Section:** Results §2.4 (chemistry + ancestry robustness)
**Description:** 3-comparator (LASSO-pre, Pasta-REG, Geneformer-LoRA) × 3-context (3' OneK1K, 5' Terekhova European, 5' AIDA Asian) panel. Pearson R on y-axis. The figure asks: "does single-cell FM pretraining add chemistry-invariance + ancestry-invariance on top of rank-normalized bulk?"
**Files to produce:** `results/phase3/fig_cd4t_robustness_3x3.{csv,pdf}`

### F6 — Pasta calibration vs ranking diagnostic

**Status:** 📋 planned (cheap, data in hand)
**Section:** Discussion or supplement
**Description:** 4-panel scatter (one per cohort): Pasta-REG predicted age vs true age. Annotates R + bias + MAE. Visualizes why Pasta's high systematic bias on OneK1K (−23y) does not affect Pearson R (good ranking, bad calibration). Justifies the "report MAE + bias + R side-by-side" reporting policy.

## Phase-4 result figures

### F7 — Full LOCO matrix forest plot (4 FMs × 5 cell types × 3 LOCO folds + AIDA)

**Status:** 📋 planned
**Section:** Results §2.5 (full matrix)
**Description:** Per-fold log-MAE-ratio of each FM vs the per-cell minimum baseline. Three glyph styles: clean / leakage-overlapping / chemistry-shifted. Three metrics side by side per fold: MAE, bias, R (per Phase-2 caveat #5).

### F8 ⭐ — Few-shot dual-regime curve

**Status:** 📋 planned
**Section:** Results §2.5 (few-shot)
**Description:** Two panels (CD4+T, B cells), each showing 3 lines (LASSO / Pasta / best FM) of MAE vs n_donors_train at 100% / 75% / 50% / 25% / 10%. The B-cell panel additionally splits by chemistry context (3' OneK1K vs 5' Terekhova) since Pasta-rescue dominates the 5' regime.

### F9 — Zero-shot cell-type transfer matrix

**Status:** 📋 planned
**Section:** Results §2.6 (zero-shot transfer)
**Description:** 5×5 source-cell-type × target-cell-type heatmap. Per cell: FM Pearson R vs Pasta-REG-on-target Pearson R. Shows whether single-cell FM cross-cell-type embedding beats naive rank-norm-on-pseudobulk.

## Phase-5 result figures

### F10 — Age-axis cosine similarity heatmap (cell-type pairs, sex pairs, ancestry pairs)

**Status:** 📋 planned
**Section:** Results §2.7 (biological readouts)

### F11 — Top-50 SHAP gene overlap (FMs vs sc-ImmuAging LASSO vs iAge)

**Status:** 📋 planned
**Section:** Results §2.7

### F12 — In-silico perturbation: top-5 SHAP gene effect on predicted age

**Status:** 📋 planned
**Section:** Results §2.7

## Methods figures (supplementary)

### S1 — Cohort schematic (cohort × cell-type × donor count)

**Status:** ⏳ data in `data/cohort_summary.csv` + AIDA summary

### S2 — Detectability floor sensitivity-to-ρ curve (Phase-1 sensitivity sweep)

**Status:** ⏳ data in `data/detectability_floor.json::sensitivity_pairing_rho`

### S3 — Terekhova reverse-normalization verification

**Status:** ⏳ data in commit c220d92 (per-cell-type residual histograms)
**Description:** Histograms showing `expm1(X) * nCount_RNA / 10000` produces 100% integer counts within <0.01 tolerance. Justifies the data-integrity decision in `methods/terekhova_chemistry_shift.md`.
