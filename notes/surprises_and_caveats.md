# Surprises and caveats

Findings that surprised us (positive or negative), and methodological caveats reviewers will care about. Edit-in-place. Each entry is one substantive paragraph; cross-references to `methods/`, `notes/research_journal.md`, or commits.

---

## Surprise 1: scAgeClock is *weaker* than LASSO on PBMC

scAgeClock (Xie 2026 *npj Aging*) is a 2026 deep-learning aging clock with multi-head attention, trained on millions of cells from CZ CELLxGENE Census across 400+ cell types and 40+ tissues. We expected it to dominate the pretrained linear LASSO on every cohort, especially on its own training cohorts (OneK1K, Stephenson). It does not. Best scAgeClock R on OneK1K = 0.59 (CD8T) vs LASSO 0.77; persistent ~−13y systematic bias across cohorts. **Implication for the paper's framing**: the FM-vs-baseline contrast is "specialist FMs beat both a specialist linear and a generalist deep model on a specific tissue," not "deep beats shallow." This makes the FM win claim sharper and more defensible. *`notes/research_journal.md` 2026-04-25; `methods/loco_baselines.md`.*

## Surprise 2: Pasta-REG (bulk-transcriptomic) beats LASSO on Terekhova and AIDA CD4+T

Pasta-REG, a 2025 bulk-transcriptomic aging clock (Salignon et al.) with rank-normalization on an 8,113-gene panel, beats the LASSO on the 5' chemistry cohorts: Terekhova CD4T MAE=8.0y vs LASSO 9.2y; AIDA CD4T MAE=6.3y vs LASSO 7.5y (the **lowest MAE in the entire 75-row baseline matrix**). Pasta's rank-normalization makes it chemistry-invariant *and* ancestry-invariant — properties that count-based methods (LASSO, FMs) must work harder to acquire. **Implication**: the FM headline must beat Pasta, not just LASSO. The Phase-3 success probability on Terekhova CD4+T dropped from ~80% (pre-Pasta) to ~50–60% (post-Pasta). The AIDA target (5.7y for 10% win) is the toughest in the matrix. *`notes/research_journal.md` 2026-04-25; `methods/loco_baselines.md`.*

## Surprise 3: 3-cohort retrained LASSO ≈ pretrained 5-cohort LASSO for CD4T/CD8T

A reviewer-anticipated control. The pretrained sc-ImmuAging LASSO was trained on 5 cohorts (1,081 donors); we trained on 3 (1,176 donors, different mix). For T cells, the two LASSOs are essentially equivalent (Terekhova CD4T retrained R=0.81 vs pretrained 0.82). **This decouples the "FM vs baseline" comparison from the "more training data vs less" confound.** Any FM win against LASSO-retrained-3cohort is unambiguously architectural (or due to better representation), not data-quantity. The retrained LASSO is the methodologically symmetric apples-to-apples comparator. *Commit dbbce90.*

## Surprise 4: Phase-1 detectability floor was 2–7× too optimistic

We assumed paired-test ρ=0.8 as a planning value (typical for aging clocks per literature). The measured baseline-pair ρ from Phase-2 outputs is 0.06–0.35 (CD4T 0.23, CD8T 0.16, MONO 0.06, NK 0.28, B 0.35). Required-N at empirical ρ jumps from 132–229 (planning) to 502–1,075 (empirical baseline-pair). **Implication**: only OneK1K is unambiguously powered for any cell type at the empirical-ρ extreme. Caveat: this is baseline-PAIR ρ; the actual baseline-vs-FM ρ (measured in Phase 3) is expected higher because FMs and baselines share more residual structure. The paper reports all three values (0.8 planning → 0.06–0.35 baseline-pair → Phase-3-measured) to bracket the truth.

## Caveat 1: leakage audits must use Census BUILD dates, not release-LABEL dates

CZ CELLxGENE Census labels each release with a date string (e.g., `2024-07-01`) but the actual *build* of that release happens up to 6 weeks earlier (here, 2024-05-20). For scAgeClock × AIDA, AIDA's CellxGene deposit was posted on the release-label date 2024-07-01 — *after* the build date — so AIDA is `clean` for scAgeClock. Using the release label as the cutoff would have incorrectly classified AIDA as `overlapping`. **The build date is the leakage-relevant cutoff.** This is a transferable lesson for any leakage audit against CZ CELLxGENE Census. *`methods/leakage_audit_notes.md` (scAgeClock addendum); `notes/research_journal.md` 2026-04-24.*

## Caveat 2: leakage audits must check ALL deposit mirrors, not just one accession

scFoundation × Stephenson was initially classified `clean` from a direct E-MTAB-10026 (ArrayExpress) accession search of Hao 2024 *Nat Methods* Supp Table 5. The same dataset is also deposited on the HCA portal as **HCA-Covid19PBMC**, and that deposit IS in scFoundation's training set. We caught it only by cross-referencing HCA Project IDs across the supplementary table. **Lesson**: leakage audits must cover all known deposit mirrors (HCA / ArrayExpress / CellxGene / GEO / Synapse), not the single most "canonical" accession. *`methods/leakage_audit_notes.md`.*

## Caveat 3: Terekhova source ships log1p(CP10k), not raw counts

The `all_pbmcs_rna.h5ad` from Synapse `syn51197006` (the only viable Terekhova source after the primary `syn56693935` corrupted) has no `.raw` and `.X` is log-normalized. A naive harmonization pipeline would silently mix log-normalized Terekhova with raw-count OneK1K/Stephenson, breaking both FM fine-tuning and LASSO scoring. We verified the normalization is invertible (`row_sum(expm1(X)) == 10000` exactly) and reverse-normalized using metadata `nCount_RNA` per cell. **Methodological warning for the paper**: `.raw` is not a CellxGene/Synapse universal — always verify by checking `(adata.X.data % 1 == 0).all()` before assuming integer counts. *`src/data/harmonize_cohorts.py::load_terekhova` + commit c220d92.*

## Caveat 4: sc-ImmuAging's RF and PointNet pretrained weights are NOT in the public package

The Li et al. 2025 *Nature Aging* sc-ImmuAging package ships **only the cv.glmnet LASSO weights** in `data/all_model.RDS`. The R/Python files `RF.py` and `pointnet_unet.py` are training scripts referencing `data/processed/*.sav` checkpoints that are not distributed. Reproducing them requires retraining on the original 5-cohort corpus (2 of which are EGA-controlled). The Phase-2 baseline panel was therefore revised to LASSO-only for the sc-ImmuAging entry, with scAgeClock filling the deep-learning slot and Pasta-REG the bulk-transcriptomic slot. **Disclosable methodological gap in the upstream paper that the FM benchmark needs to acknowledge.** *`methods/loco_baselines.md`.*

## Caveat 5: Pasta has high systematic bias despite good Pearson R

Pasta-REG predictions on OneK1K CD4+T have R=0.60 but mean bias = −23.3y; on Terekhova MONO bias = +20.5y. The model is well-calibrated for *ranking* but poorly calibrated for *absolute year prediction* on cohorts unlike its training distribution. **The forest plot in Phase 4 must report MAE + bias + R side-by-side** — single-metric reporting (MAE alone) penalizes Pasta unfairly relative to its actual aging-signal recovery, while R alone hides the calibration failure. *`roadmap/phase-4.md` reporting policy.*

## Caveat 6: Empirical baseline-pair ρ is a *lower bound* on baseline-vs-FM ρ

The Phase-2 ρ measurement is between BASELINE PAIRS (LASSO vs Pasta, etc.). The Phase-3 fine-tuned FMs are expected to share more residual structure with the baselines than two baselines do with each other (both predict the same chronological age signal from the same gene-expression data). So the empirical baseline-pair ρ (0.06–0.35) is a conservative lower bound; the actual baseline-vs-FM ρ (Phase 3 will measure it) likely sits between 0.4 and 0.7. The detectability floor reported in the preprint will use the Phase-3 measured value, with the bracketing ρ values disclosed in the methods section. *`methods/detectability_floor.md`.*

## Caveat 7: Stephenson's 18/29 healthy donors carry decade-precision ages

Stephenson's healthy controls have 11 exact-age donors and 18 decade-binned donors (e.g., "30-39"). We used decade midpoints (±5 yr precision) and flagged the rows via `obs['age_precision']`. **Stephenson per-cell-type R values (0.18–0.79) are consistent with the noisy-label regime; the cohort sits below the detectability floor for any cell type and is exploratory-only across all our analyses.** Not a wrinkle to hide; should be stated upfront in the methods. *`methods/datasets/stephenson.md`.*
