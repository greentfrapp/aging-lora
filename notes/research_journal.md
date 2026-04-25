# Research journal — chronological findings log

Append-only. New entries at the bottom. Each entry: 1-line headline + 2-5 sentence rationale + commit hash + cross-refs to `methods/` or `roadmap/`. See `README.md` for purpose.

---

## 2026-04-23: project scaffold + cohort decision

**Initial cohort plan (Case 1):** five sc-ImmuAging cohorts assumed available. The plan was to reproduce Li et al. 2025 *Nature Aging*'s LASSO/RF/PointNet baselines on the same 1,081-donor corpus, then fine-tune scGPT/Geneformer. Two of the five cohorts are EGA-controlled (EGAS00001005529, EGAS00001006990); applied for access but did not block on it. Public 3 cohorts: OneK1K (Yazar 2022 GSE196830), Stephenson (E-MTAB-10026), Terekhova (Synapse syn49637038).

## 2026-04-23: Barreiro/Randolph 2021 cohort dropped from primary

GEO `GSE162632` characteristics_ch1 contains no donor ages, and the release is genotype-multiplexed (mock + IAV cells from multiple donors per capture) with no demux files. Unblocking would require emailing the Barreiro lab + obtaining genotype VCFs + running Demuxlet/Vireo — weeks of work for a cohort already below the 80-donor LOCO-primary threshold. Tracked in `FUTURE_WORK.md`.

## 2026-04-24: Terekhova 2023 promoted from Case-3 to a primary cohort

166 healthy donors > 80-donor threshold → it becomes a primary LOCO fold alongside OneK1K. Continuous 25–85 yr age span. The Synapse primary tar (`syn56693935`) was corrupt at source; switched to the `syn51197006` all_pbmcs tarball as a fallback. *Roadmap: `roadmap/phase-1.md` Task 1c-v2; `methods/datasets/terekhova.md`.*

## 2026-04-24: pre-trained sc-ImmuAging LASSO sanity check on OneK1K (Task 1e)

Per-cell-type Pearson R: CD4T 0.75, CD8T 0.77, MONO 0.71, NK 0.63, B 0.53 — all positive and highly significant (p < 1e-70). MAE 7.6–10.7 yr, within 2× of the paper's Ext Data Table 2. Systematic −1.5 to −4.4 yr negative bias observed. Validates our Python `rdata` port of the cv.glmnet model. Commit dcd5763 (Phase 2). *`methods/pretrained_lasso_sanity_check.md`.*

## 2026-04-24: scFoundation × Stephenson leakage discovered via HCA mirror

Direct accession search for E-MTAB-10026 in scFoundation's training-cohort list missed the overlap. Cross-referencing **HCA Project IDs** in Hao 2024 *Nat Methods* Supp Table 5 row 81 surfaced the HCA-Covid19PBMC project ID — the same dataset deposited under a different identifier. **Methodological lesson: leakage audits must check all deposit mirrors (HCA / ArrayExpress / CellxGene / GEO), not just one canonical accession.** This is a paper-worthy finding for the reproducibility community. *`methods/leakage_audit_notes.md`.*

## 2026-04-24: Terekhova source ships log1p(CP10k), not raw counts (Task 1c)

The `all_pbmcs_rna.h5ad` from `syn51197006` has no `.raw` and `.X` is log-normalized (max ~7.5, non-integer). Without inversion, Terekhova cells in the harmonized output would be log-normalized while OneK1K/Stephenson are raw integers — breaks both FM fine-tuning and LASSO scoring. **Verified that `row_sum(expm1(X)) == 10000` exactly** and `expm1(X) * nCount_RNA / 10000` recovers integer counts at 100% <0.01 tolerance; all 1.9M cells have non-null `nCount_RNA` in the metadata CSV. Reverse-normalized via `load_terekhova` per-cell-type CSR transform. Commit c220d92.

## 2026-04-24: Terekhova 10x 5' chemistry shift is cell-type-selective (Task 1f)

Pre-trained LASSO on Terekhova (10x 5' v2) preserves R for CD4T (0.82) and CD8T (0.73); degrades for MONO (0.29) and NK (0.44); collapses for B (R=0.08). Decision: report naive (uncorrected) MAE as the primary Terekhova LOCO result; chemistry correction deferred to Phase 3 as exploratory. The B-cell collapse becomes a **chemistry-rescue target** for Phase 4 FMs. *`methods/terekhova_chemistry_shift.md`.*

## 2026-04-24: scAgeClock × cohort leakage audit (Task 2.2)

scAgeClock pretrained on CZ CELLxGENE Census version=`2024-07-01`, **built 2024-05-20**. The 6-week gap between release-label date and build date is the leakage-relevant cutoff: AIDA's CellxGene deposit posted 2024-07-01, postdating the build → AIDA × scAgeClock is `clean`. OneK1K (2022) and Stephenson (2021) are `overlapping`; Terekhova (Synapse-only) is `clean`. **Subtle finding worth a methods-section note**: leakage audits must use build dates, not release-label dates. Commit d7733aa.

## 2026-04-25: sc-ImmuAging ships pretrained LASSO ONLY (no RF/PointNet)

Inspected `data/scImmuAging/data/all_model.RDS` — 5 `cv.glmnet` objects keyed on cell type; nothing else. `RF.py` and `pointnet_unet.py` in `data/scImmuAging/codes/` are training scripts that reference `data/processed/*.sav` checkpoints **not distributed with the public package**. Reproducing them would require retraining on the paper's original 5 cohorts (2 of which are EGA-controlled). **Phase-2 panel revised** to LASSO-only as the upstream sc-ImmuAging baseline; scAgeClock fills the deep-learning slot, Pasta-REG fills the bulk-transcriptomic slot. *`methods/loco_baselines.md`.*

## 2026-04-25: Pasta-REG beats LASSO on Terekhova CD4+T (Task 2.4)

Pasta-REG MAE=8.0y / R=0.78 on Terekhova CD4+T, **lower MAE than the pretrained LASSO** (9.2y / 0.82). Pasta's rank-normalization makes it chemistry-invariant: where LASSO collapses on 5' B cells (R=0.08), Pasta survives (R=0.28). **Implication for FMs: the bar for the Phase-3 chemistry-shift headline is no longer LASSO 9.2y → Pasta 8.0y → 7.2y for the 10% win.** Commit 770b39e. *`methods/loco_baselines.md`.*

## 2026-04-25: scAgeClock weaker than LASSO on PBMC (Task 2.3)

scAgeClock — a 2026 deep-learning aging clock with attention, 19,234-gene input, CELLxGENE-Census-pretrained — performed *below* the pretrained LASSO on every cohort, including its own training cohorts (best R = 0.59 on OneK1K CD8T vs LASSO 0.77). Persistent ~−13y systematic bias on PBMC. **A generalist deep clock trained across 400+ cell types does NOT automatically dominate a PBMC-specialist linear model on a single tissue.** Useful framing for the paper: any FM win is a "specialist FM beats generalist FM + specialist linear" claim, not a "deep beats shallow" claim.

## 2026-04-25: Pasta-REG MAE=6.3y on AIDA CD4+T — lowest in entire baseline matrix (Task 2.6)

Scoring all 3 baselines on AIDA (4th cohort, 625 Asian-ancestry donors, 10x 5' v2) revealed Pasta-REG produces **MAE=6.3y / R=0.66 on AIDA CD4+T — the lowest MAE across the entire 75-row baseline matrix**. Pasta is both chemistry-invariant *and* ancestry-invariant by virtue of rank-normalization. **Phase-3 FM target on AIDA CD4+T is 5.7y for a 10% win — the toughest cell.** This becomes the cross-ancestry headline for the preprint. Commit dbbce90.

## 2026-04-25: 3-cohort retrained LASSO ≈ pretrained 5-cohort LASSO for CD4T/CD8T (Task 2.7)

`LassoCV` retrained on our 3-cohort corpus per LOCO fold delivers Terekhova CD4T R=0.81 (vs pretrained 0.82) and CD8T R=0.74 (vs 0.73). **For CD4T/CD8T, the FM-vs-LASSO gap CANNOT be attributed to "FMs had access to more cohorts"** — both retrained-LASSO and FMs see the same 3 cohorts. Two failure modes: OneK1K-out × B regularized to intercept-only (α=0.90, R=NaN); OneK1K-out × MONO/NK had large bias (the 195-donor + chemistry-mixed training set was too small for these cell types). Commit dbbce90.

## 2026-04-25: empirical baseline-pair ρ is 0.06–0.35, not 0.8 (Task 2.8)

Phase-1 detectability floor used ρ=0.8 as the planning value. Measured ρ between baseline-pair |error| vectors per cell type: CD4T 0.23, CD8T 0.16, MONO 0.06, NK 0.28, B 0.35. Required-N at empirical ρ: 502–1,075 per cell type (vs Phase-1 floor 132–229). **Phase-1 floor was 2–7× too optimistic.** Caveat: this is baseline-PAIR ρ, a conservative lower bound on Phase-3's baseline-vs-FM ρ; the truth is somewhere between 0.06–0.35 and 0.8. The preprint methods will report all three values to bracket the true figure. *`methods/detectability_floor.md`.*

## 2026-04-25: Phase-3 expanded to tri-headline structure

After Phase-2 closure, the Phase-3 preprint now reports three CD4+T headline cells: OneK1K (981 donors, 3' chemistry), Terekhova (166 donors, 5' chemistry), AIDA (595 donors, 5' Asian ancestry). Each cell scored with a "win/match/loss" classification against the per-cell minimum of 4 baselines (LASSO-pretrained, LASSO-retrained-3cohort, scAgeClock, Pasta-REG). Aggregate outcomes 3/3/2/1/0 wins drive the preprint headline framing (primary headline → strong primary → degraded → pivot). Commit 2662bf4. *`roadmap/phase-3.md`.*
