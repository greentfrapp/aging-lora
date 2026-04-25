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

- [ ] **Task 2.6:** Score the three Phase-2 baselines (LASSO, scAgeClock, Pasta-REG) on AIDA. AIDA is downloaded as a CellxGene h5ad (`data/cohorts/raw/aida/9deda9ad-…h5ad`, 625 healthy donors split 307/318 in `data/aida_split.json`) but has not been run through the harmonization pipeline yet. Run `load_cellxgene_cohort` on AIDA into a separate output directory (`data/cohorts/aida_eval/{B,CD4p_T,CD8p_T,Monocyte,NK}.h5ad`, NOT mixed into the 3-cohort training h5ads, to honor the "frozen for FM evaluation" principle). Add `--integrated-dir` flag to `score_scageclock.py` and `pseudobulk_for_pasta.py` so they can score the AIDA subdir. Add 60 rows (3 baselines × 5 cell types × 4 cohorts) to `loco_baseline_table.csv`. Done when AIDA × {LASSO, scAgeClock, Pasta-REG} × 5 cell types are populated in the table; methods note appended to `methods/loco_baselines.md`.

- [ ] **Task 2.7:** Retrain sc-ImmuAging LASSO on our three cohorts (3-cohort, training-matched comparison). Port `data/scImmuAging/codes/Lasso_training.R` to Python (sklearn ElasticNet with α=1.0 = pure LASSO, or scikit-learn `LassoCV`; pseudocell aggregation per the upstream procedure). For each LOCO fold (train on the union of 2 cohorts, evaluate on the held-out 3rd), train one model per cell type. Append per-fold MAE/R rows to `loco_baseline_table.csv` with `baseline=LASSO-retrained-3cohort, training_cohorts=our-three-cohort`. This is the **methodologically-symmetric comparator** to FM fine-tuning (same training corpus, same LOCO splits) and addresses the "is the FM win architectural or just a training-data effect?" reviewer question. Done when 15 rows (3 LOCO folds × 5 cell types) are appended.

- [ ] **Task 2.8:** Empirically measure pairing-ρ from the Phase-2 baseline residuals and update the detectability floor. For each (cohort, cell type) cell, compute `cor(|err_LASSO|, |err_Pasta|)` and `cor(|err_LASSO|, |err_scAgeClock|)` per donor; report the median ρ per cell type in a 5-row table. Re-run `src/data/detectability_floor.py` with the empirical median ρ values to refine the per-cell paired-Wilcoxon detection threshold. Append a `post_phase2_empirical_rho` block to `data/detectability_floor.json` (preserving the Phase-1 ρ=0.8 planning value as a separate field — does not overwrite the frozen file). Document any per-cell-type primary/exploratory flag changes in a one-line addendum to `methods/detectability_floor.md`. Done when the empirical ρ table and the updated detectability floor are committed.

## Phase 2 summary (2026-04-25)

All four primary baseline tasks closed. The 45-row `results/baselines/loco_baseline_table.csv` (3 baselines × 3 cohorts × 5 cell types) is now the Phase-3/4 reference. Per-cell minimum-MAE baseline (the "best of 3" bar that FMs must beat in Phase 4):

| Cohort | CD4T | CD8T | MONO | NK | B |
|---|---|---|---|---|---|
| OneK1K | LASSO 9.4 | LASSO 7.6 | LASSO 7.9 | scAgeClock 11.5 | LASSO 10.7 |
| Stephenson | LASSO 8.4 | LASSO 11.4 | LASSO 10.7 | Pasta 11.4 | LASSO 12.1 |
| Terekhova | **Pasta 8.0** | **Pasta 7.6** | LASSO 12.8 | LASSO 12.5 | **Pasta 10.9** |

Bolded Terekhova cells are the chemistry-rescue baselines (Pasta wins because rank-normalization makes it chemistry-invariant). On the strict-clean Terekhova fold, Pasta is best for CD4T/CD8T/B and LASSO for NK/MONO. The Phase-4 FM headline must clear all 5 best-baseline floors per cell type, not just the LASSO floor.

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
