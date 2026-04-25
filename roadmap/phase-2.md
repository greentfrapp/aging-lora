# Phase 2 — Baseline establishment

## Goal

Before measuring whether foundation-model fine-tuning helps, this phase establishes the task-specific baselines the foundation models will be measured against. The primary approach uses the sc-ImmuAging pre-trained LASSO clocks (shipped in `data/scImmuAging/data/all_model.RDS`) applied directly to the project's LOCO holdout sets — this is the fastest path to a result and tests the most practically meaningful claim: can foundation models beat the published state-of-the-art with less training data? A training-matched LASSO retrain (same three cohorts as the foundation models) is scoped as a supplementary analysis, to be run after initial results are in hand.

### Baseline strategy decision (recorded 2026-04-24, panel revised 2026-04-25)

**Primary (option 2):** Apply pre-trained external clocks directly to our LOCO holdout cohorts. No retraining. The training-set asymmetry vs. our 3-cohort fine-tuned FMs must be stated explicitly in all result tables and the methods section. If FMs win against this panel, the claim is: *"foundation models outperform the published state-of-the-art using less training data."* If they lose, the asymmetry is a confound — proceed to option 1.

**Baseline panel (revised 2026-04-25 after upstream-availability check):**

| Baseline | Family | Source | Status |
|---|---|---|---|
| sc-ImmuAging pre-trained LASSO | classical, single-cell | `data/scImmuAging/data/all_model.RDS` (5 `cv.glmnet` objects) | ✅ shipped weights |
| ~~sc-ImmuAging pre-trained RF~~ | ~~ensemble, single-cell~~ | not available — only training script `data/scImmuAging/codes/RF.py` ships | dropped from primary; see `methods/loco_baselines.md` |
| ~~sc-ImmuAging pre-trained PointNet~~ | ~~deep-learning, single-cell~~ | not available — only training script ships | dropped from primary |
| **scAgeClock** (Xie 2026) | deep-learning, single-cell, attention | `data/scAgeClock/scageclock/data/trained_models/scAgeClock_GMA_model_state_dict.pth` (44 MB) | ✅ shipped weights — **fills the deep-learning slot** |
| **Pasta REG** (Salignon 2025) | classical-ish, bulk transcriptomics | R package `jsalignon/pasta`, REG model (absolute age) | ✅ shipped weights, R-only |
| ~~Pasta age-shift, Pasta CT46 classifier~~ | — | predict relative age / binary class — not directly MAE-comparable | dropped from primary |
| ~~Pasta retrained on our LOCO folds~~ | — | training code not in public repo; multi-day reimplementation | deferred to supplement |

**Supplementary (option 1, deferred):** Retrain LASSO on the same three cohorts as the foundation models, using `data/scImmuAging/codes/Lasso_training.R` as reference. Tracked in `results/baselines/loco_baseline_table.csv` as `[supplement — pending]`. Run after Phase 3 results if the primary comparison is inconclusive or flagged by reviewers.

## Success criteria

- **LASSO baseline complete (DONE 2026-04-24):** Pre-trained sc-ImmuAging LASSO scored on each LOCO holdout cohort. Per-cohort per-cell-type MAE, Pearson R, mean bias recorded in `results/baselines/loco_baseline_table.csv` (15 rows = 3 cohorts × 5 cell types) with leakage_status, chemistry_match_to_baseline_training, and detectability_flag stratification columns. Methods in `methods/loco_baselines.md`.
- **scAgeClock baseline complete:** scAgeClock applied to each LOCO holdout cohort; per-cohort per-cell-type MAE and Pearson R appended to `results/baselines/loco_baseline_table.csv` with `baseline=scAgeClock, training_cohorts=CELLxGENE-Census`. **Pretraining-corpus leakage audit completed and 4 new rows appended to `data/leakage_audit.csv`** (model=scAgeClock × {OneK1K, Stephenson, Terekhova, AIDA}) before the baseline rows are reported.
- **Pasta REG baseline complete:** Pasta-as-released REG model applied to per-(donor × cell-type) pseudobulk profiles; per-cohort per-cell-type MAE and Pearson R appended to `results/baselines/loco_baseline_table.csv`. Pasta age-shift and CT46 heads NOT reported in primary table (different metrics). Pseudobulk adaptation choices documented in `methods/pasta_pseudobulk_notes.md`.
- **Supplementary LASSO retrain (option 1) deferred:** tracked in `results/baselines/loco_baseline_table.csv` as `[supplement — pending]` until run.

## Tasks

- [x] **Task 2.1:** (Completed 2026-04-24, see commit dcd5763.) Apply pre-trained sc-ImmuAging LASSO to all three cohorts via the Python `rdata`-port pipeline in `src/baselines/score_pretrained_lasso.py`. 15-row table assembled by `src/baselines/assemble_loco_baseline_table.py`. The roadmap statement "LASSO/RF/PointNet" is reduced to LASSO-only because the upstream sc-ImmuAging package ships only `cv.glmnet` weights — RF.py and pointnet_unet.py are training scripts referencing files not in the public package. Documented in `methods/loco_baselines.md`.

- [x] **Task 2.2:** (Completed 2026-04-25, commit d7733aa.) scAgeClock pretraining-corpus leakage audit. 4 rows appended to `data/leakage_audit.csv` (now 5 models × 4 cohorts = 20 rows). scAgeClock × {OneK1K, Stephenson} = `overlapping` (CellxGene Census 2024-05-20 build); scAgeClock × {Terekhova, AIDA} = `clean` (Synapse-only and post-build-cutoff respectively). Methodology addendum appended to `methods/leakage_audit_notes.md`. Subtle finding: Census release LABEL date (2024-07-01) and BUILD date (2024-05-20) differ by 6 weeks; AIDA lands precisely in that gap, so the build date is the leakage-relevant cutoff.

- [x] **Task 2.3:** (Completed 2026-04-25, commit 770b39e.) scAgeClock applied to all three cohorts via `src/baselines/score_scageclock.py`. CPU inference at 25K cells/chunk; vocab match 96.9% (18,632 / 19,234 model genes mapped from our 48,968-gene harmonized data); 30 min wall-clock for 2.75M cells. 15 rows added to `loco_baseline_table.csv` with `baseline=scAgeClock`. Headline: persistent ~−13y systematic bias on PBMC across cohorts; R values markedly **below LASSO** even on leakage-overlapping training cells (best R=0.59 on OneK1K CD8T vs LASSO 0.77). A generalist deep clock trained across 400+ cell types does NOT automatically dominate a PBMC-specialist on a single tissue.

- [x] **Task 2.4:** (Completed 2026-04-25, commit 770b39e.) Pasta-REG baseline. R 4.5.3 already installed; `library(pasta)` loads cleanly after pulling Bioconductor deps `Biobase`, `GEOquery`, `biomaRt`. Pasta-as-released REG (calendar-age) model only — Pasta age-shift and CT46 heads dropped from primary panel (different metrics; not directly MAE-comparable). Pseudobulk per (donor × cell-type) via `src/baselines/pseudobulk_for_pasta.py`; R scoring via `src/baselines/score_pasta_reg.R`. Two integration gotchas captured in code comments: (i) Ensembl-version stripping must be limited to `ENSG`-prefixed IDs (clone-style identifiers like `AC000065.1` vs `AC000065.2` are distinct features and must keep their suffix), (ii) `pasta::predicting_age_score` calls `stats::predict + requireNamespace("glmnet")` which fails to dispatch `predict.cv.glmnet` under `Rscript`; bypass with direct `predict(cvfit_REG, ...)` after `library(glmnet)`. **Headline: Pasta CD4T R=0.78 / MAE=8.0y on Terekhova *beats LASSO* on MAE (R=0.82 / MAE=9.2y); Pasta B R=0.28 vs LASSO R=0.08 = chemistry-rescue baseline that Phase-4 FMs need to beat.** Pasta retraining and Pasta age-shift / CT46 heads remain deferred to supplement (training code not in public repo; multi-day reimplementation).

- [ ] **Task 2.5 (deferred):** *(Deferred to "post-Phase-3 if needed".)* Retrain sc-ImmuAging LASSO on our three cohorts using `Lasso_training.R` as reference. Superseded as a primary task by **Task 2.7** below, which brings the training-matched LASSO comparison forward into Phase 2 itself.

### Pre-Phase-3 add-ons (recorded 2026-04-25 after Phase-2 retrospective)

The Phase-2 retrospective surfaced three add-on tasks that strengthen the Phase-3 preprint and de-risk Phase 4. Doing them *before* Phase 3 starts is preferable: (a) AIDA is the Phase-4 ancestry-shift cohort and pipeline bugs are cheaper to find now than during FM training; (b) the training-matched LASSO comparison will be the first reviewer ask on the preprint; (c) the empirical pairing-ρ removes a planning assumption from the methods section.

- [x] **Task 2.6:** (Completed 2026-04-25, commit dbbce90.) Score LASSO + scAgeClock + Pasta-REG on AIDA. AIDA harmonized via per-cell-type streaming into `data/cohorts/aida_eval/` (1.27M raw cells → 1.05M cells across canonical 5; held back from FM training set). 15 AIDA rows appended to `loco_baseline_table.csv`. **Headline finding: Pasta-REG CD4+T MAE=6.3y / R=0.66 on AIDA is the lowest MAE in the entire 75-row baseline matrix** — Phase-3 FMs must clear 5.7y on AIDA CD4+T for a 10% win on the cross-ancestry headline cell. LASSO collapses on AIDA B (R=−0.03), same chemistry-collapse pattern as Terekhova B; chemistry-rescue baseline floor for B is now Pasta R=0.27 / 11.1y, not LASSO. AIDA-specific section appended to `methods/loco_baselines.md`.

- [x] **Task 2.7:** (Completed 2026-04-25, commit dbbce90.) Retrained sc-ImmuAging LASSO on our 3 cohorts (training-matched comparator) via `src/baselines/retrain_lasso_3cohort.py`. sklearn `LassoCV(cv=10)` on pseudocell-aggregated training data per LOCO fold; same upstream marker-gene panels for direct comparability. 15 rows added with `baseline=LASSO-retrained-3cohort`. **Headline finding: 3-cohort retrained LASSO is essentially equivalent to the pretrained 5-cohort LASSO for CD4+T and CD8+T across all eval cohorts** (Terekhova CD4T retrained R=0.81 vs pretrained R=0.82). Two failure modes flagged: OneK1K-out × B regularized to intercept-only (α=0.90, R=NaN) — the 195-donor Stephenson+Terekhova training set was too small + chemistry-mixed for the OneK1K B signal; OneK1K-out × MONO/NK had large negative bias (−15 to −17y) for the same reason. Pasta-REG remains the per-cell minimum-MAE baseline on Terekhova headline cells. The retrained LASSO is the **apples-to-apples comparator** to FM fine-tunes (same training corpus, same LOCO splits) and addresses the "is the FM win architectural or just a training-data effect?" reviewer concern.

- [x] **Task 2.8:** (Completed 2026-04-25, commit dbbce90.) Empirically measured pairing-ρ from baseline residuals via `src/baselines/empirical_pairing_rho.py`. **Per-cell empirical median ρ between baseline pairs is 0.06–0.35** (CD4T 0.23, CD8T 0.16, MONO 0.06, NK 0.28, B 0.35) — far below the Phase-1 ρ=0.8 planning value. Required donor-N at empirical ρ: 502–1,075 per cell type (vs Phase-1 floor 132–229). The Phase-1 detectability floor was 2–7× too optimistic. Caveat: this is baseline-PAIR ρ, a conservative LOWER BOUND on the Phase-3-measured baseline-vs-FM ρ; FMs and baselines should share more residual structure than two baselines do with each other. `data/detectability_floor.json` carries a new `post_phase2_empirical_rho` block alongside the Phase-1 fields (no overwrite). Phase-3 will measure the actual baseline-vs-FM ρ and append a `post_phase3_override` block. Methodology addendum in `methods/detectability_floor.md`.

## Phase 2 summary (final, 2026-04-25)

All seven primary + add-on tasks closed. The **75-row** `results/baselines/loco_baseline_table.csv` (4 baselines × 3 training cohorts × 5 cell types + 3 baselines × AIDA × 5 cell types) is now the Phase-3/4 reference.

**Updated per-cell minimum-MAE baseline (the "best of N" bar that FMs must beat in Phase 4):**

| Cohort | CD4T | CD8T | MONO | NK | B |
|---|---|---|---|---|---|
| OneK1K | LASSO-pre 9.4 | LASSO-pre 7.6 | LASSO-pre 7.9 | scAgeClock 11.5 | LASSO-pre 10.7 |
| Stephenson | LASSO-pre 8.4 | LASSO-pre 11.4 | LASSO-pre 10.7 | Pasta 11.4 | LASSO-pre 12.1 |
| Terekhova | **Pasta 8.0** | **Pasta 7.6** | LASSO-pre 12.8 | LASSO-pre 12.5 | **Pasta 10.9** |
| AIDA | **Pasta 6.3** ★ | scAgeClock 8.7 | scAgeClock 9.1 | scAgeClock 8.5 | scAgeClock 9.1 |

★ = lowest MAE in the entire 75-row baseline matrix; Phase-3 FMs must clear **5.7y on AIDA CD4+T** for a 10% win on the cross-ancestry headline cell. Bolded Terekhova cells are chemistry-rescue baselines (Pasta wins because rank-normalization makes it chemistry-invariant). The Phase-4 FM headline must clear all minimum-MAE floors per cell type, including Pasta and scAgeClock where they win, not just LASSO.

**Detectability floor caveat (Task 2.8):** the empirical baseline-pair ρ of 0.06–0.35 implies required-N of 502–1,075 per cell type — far above the Phase-1 ρ=0.8 floor of 132–229. Only OneK1K is unambiguously powered for any cell type under empirical ρ; Terekhova and AIDA are borderline. Phase-3 will measure the actual baseline-vs-FM ρ to refine these flags before any headline claim is published.

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
