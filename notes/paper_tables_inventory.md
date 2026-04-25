# Paper tables inventory

Every table we might want, with: data source, current status, paper section it serves. Edit-in-place.

Status legend:
- 📋 planned
- ⏳ data exists, table not formatted
- 🎨 table drafted (e.g. as CSV)
- ✅ committed to repo + cited in manuscript

---

## Headline tables

### T1 ⭐ — Cohort summary (4 cohorts × 5 cell types)

**Status:** ⏳ data in `data/cohort_summary.csv` + `data/cohorts/aida_eval/aida_summary.csv`
**Section:** Methods §4 (cohort sources) + supplementary
**Description:** Per-cohort: source/accession, n_donors, n_cells, age range (min/max/median), assay, ancestry, healthy/disease status, age_precision (exact/decade-bin), is_held_out (T for AIDA). Currently OneK1K + Stephenson + Terekhova in `cohort_summary.csv`; AIDA in `aida_summary.csv`. **Action**: merge into a single 4-cohort table with consistent columns.

### T2 ⭐ — Leakage audit table (5 models × 4 cohorts = 20 rows)

**Status:** ⏳ data in `data/leakage_audit.csv`
**Section:** Results §2.1 + Supplementary Methods
**Description:** Each row: (model, cohort_id, cohort_name, status ∈ {clean, overlapping}, evidence). Footnotes capture (a) the HCA-Covid19PBMC mirror (scFoundation × Stephenson) and (b) the Census release-vs-build date distinction (scAgeClock × AIDA). The full table goes in supplementary; main text cites a 2-row summary for the strict-clean cells we report.

### T3 ⭐ — Per-cell minimum-MAE baseline (best of 4)

**Status:** ⏳ data in `methods/loco_baselines.md` (rendered table); regenerable from `loco_baseline_table.csv`
**Section:** Results §2.2 (baseline panel)
**Description:** 4-cohort × 5-cell-type grid showing winning baseline (LASSO-pre / LASSO-retrained / scAgeClock / Pasta-REG) and its MAE for each cell. Pasta cells highlighted as the chemistry-rescue / ancestry-rescue baselines. AIDA CD4+T (Pasta 6.3y) flagged as the lowest-MAE cell.

## Phase-3 result tables

### T4 — CD4+T tri-headline FM-vs-baseline table

**Status:** 📋 planned (Phase-3)
**Section:** Results §2.3
**Description:** Per-FM × per-headline-cell × per-baseline (4 baselines). Each row: FM MAE, baseline minimum MAE, relative MAE reduction (%), 95% CI on the difference, win/match/loss classification. Companion to F4 forest plot.

### T5 — Phase-3 detectability disclosure

**Status:** 📋 planned (Phase-3)
**Section:** Methods §4 + Supplementary
**Description:** Per-cell-type table with 5 columns: Phase-1 planning ρ=0.8 + n_required, Phase-2 baseline-pair ρ + n_required, Phase-3 measured baseline-vs-FM ρ + n_required, current LOCO-fold donor count, primary/exploratory flag from `post_phase3_override`.

## Phase-4 result tables

### T6 — Full LOCO matrix (4 FMs × 5 cell types × 3 LOCO folds + AIDA, 3 seeds each)

**Status:** 📋 planned (Phase-4)
**Section:** Results §2.5
**Description:** The big result matrix. Per row: (model, cell_type, eval_cohort, seed, MAE, R, bias, leakage_status, chemistry_match, detectability_flag). Aggregated for the forest plot F7.

### T7 — Few-shot crossover thresholds

**Status:** 📋 planned (Phase-4)
**Section:** Results §2.5
**Description:** Per (cell_type, eval_cohort, FM): the donor count at which the FM line crosses the LASSO and Pasta lines. Shows which regimes FM pretraining buys you something.

### T8 — Zero-shot cell-type transfer R values

**Status:** 📋 planned (Phase-4)
**Section:** Results §2.6
**Description:** 5×5 source × target heatmap rendered as a table; FM R vs Pasta-on-target R per cell.

## Phase-5 result tables

### T9 — Top-50 SHAP genes per (FM, cell type)

**Status:** 📋 planned (Phase-5)
**Section:** Results §2.7
**Description:** Wide table (50 rows × ~20 columns: 4 FMs × 5 cell types). Each cell is a gene symbol + mean |SHAP|. Cross-referenced with sc-ImmuAging LASSO non-zero-coef genes and iAge cytokines.

### T10 — Age-axis cosine similarity matrix

**Status:** 📋 planned (Phase-5)
**Section:** Results §2.7
**Description:** Pairwise cosine similarities of (FM, cell_type, ancestry/sex) age-axis vectors. Permutation p-values per cell.

## Supplementary tables

### S1 — Reproducibility checksums

**Status:** ⏳ data in `data/checkpoint_hashes.txt`
**Description:** SHA-256 hashes of all 4 FM checkpoints + scAgeClock checkpoint + Pasta R-package version + cohort raw h5ad accessions + harmonized h5ad checksums.

### S2 — LOCO + AIDA frozen splits

**Status:** ⏳ data in `data/loco_folds.json` + `data/aida_split.json`
**Description:** Donor-id-level enumeration of train/test splits per LOCO fold and per-half of AIDA. Frozen since Phase 1; immutable.

### S3 — Phase-2 baseline panel full result table

**Status:** ✅ data in `results/baselines/loco_baseline_table.csv` (75 rows)
**Description:** The complete 4-baseline × 4-cohort × 5-cell-type matrix with all 14 columns (baseline, training_cohorts, eval_cohort, eval_chemistry, cell_type, n_donors, MAE/mean_MAE/R/p/bias, leakage_status, chemistry_match, detectability_flag).
