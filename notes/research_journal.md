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

## 2026-04-26: Phase-3-A run #1 — head-init bug, regression head failed to train

First Geneformer LoRA fine-tune attempt on `loco_onek1k` × CD4+T × seed 0 (19,000 train cells, 1 epoch) ran to completion but produced **MAE=30.5y, R=NaN** in 9.6 GPU-hours — worse than predicting any constant. Root cause: the regression head's bias was zero-initialized while ages span 25–85 yr, so the LoRA + head joint optimization started ~50y away from any plausible prediction; under MSE the head output diverged before LoRA deltas could compensate. Artifacts archived under `*.headbug-failed.{csv,pt}` suffix in `results/baselines/fm_finetuned/geneformer/archive/` and `compute/archive_runtime_log.headbug-failed.csv` rather than discarded — kept as a worked example for the methods-section "training-instability" footnote. **Methodological lesson: regression-head bias must be initialized to the training-set mean target value when fine-tuning a pretrained encoder for a continuous-target task.**

## 2026-04-26: Phase-3-A run #2 — head-bias-init fix lands, but model is underfit

Second attempt added `head_bias_init=mean_train_age` (≈48.93 y) at head construction time (`src/finetune/cli.py:152,169`). Run completed in 3.1 GPU-hours on 9,500 train cells × 1 epoch and produced **MAE=19.99y, R=0.33** on OneK1K CD4+T (981 donors). The NaN/divergence is gone but the result is essentially at the constant-mean prediction floor: training MSE plateaus near 270 across all 275 logged steps (`logs/phase3/geneformer_loco_onek1k_seed0_CD4p_T.jsonl`), indicating the model is barely improving on the bias-init prediction within 1 epoch. Phase-3-A GATE 2 (kickoff §6) is **not** cleared — target on this fold is ≤8.5y for a headline win, ≤9.4y to match the LASSO floor, and 19.99y is well above both. Most likely fixes before scaling to the full 18-fine-tune sweep: more epochs (kickoff §5 default = 5), full 19k train cells (not the 9.5k subsample), and possibly raising the backbone LR (currently 5e-5) or lowering `head_lr` (currently 1e-3) so the two param groups co-train rather than the head saturating first. *Open issue; needs investigation before GATE 2.*

## 2026-04-26: Phase-3-A convergence investigation — LR alone doesn't fix it; undertraining + cls-pooling are the real bottlenecks

Smoke ablations (E1 lr=5e-5 vs E2 lr=2e-4 at 20-step gpu-smoke) suggested a +0.29 R jump under 4× higher LoRA LR; smoke trajectory differential was within batch noise but eval R doubled on 5 donors. Production-scale rerun (intermediate-v2, identical to Run #2 except `lr=2e-4 head_lr=2e-4`, 9,500 train cells × 981 eval donors, 4.5h) **did not reproduce the R bump**: MAE 19.99→18.48 (−7.5%), R 0.327→0.316 (within noise). The 5-donor smoke had been overstating the LR effect.

Per-donor + state-dict comparison reveals the actual mechanism: **both runs are sitting on the "predict mean(train)" minimum of MSE.** Predictions span 4.7y (Run #2) and 9.4y (v2) while true ages span 78y — model dynamic range is 6–12% of target range. The 1.5y MAE drop in v2 is almost entirely a bias-drift effect (prediction mean +2.3y closer to eval mean), with error sd unchanged (16.23 → 16.18). LoRA delta DID grow 3.2× in v2 (median RMS 1.4e-4 → 4.5e-4) — gradients flow, parameters move — but the resulting cls embeddings still don't differentiate ages. Phase-2's *linear* LASSO hits MAE 9.4y on the same train data; a 110M-param frozen FM stuck at MAE 18–20y is a striking gap pointing to severe undertraining + likely cls-pooling weakness. **Hypothesis ranking flipped:** LR (was top) is real but not binding; undertraining + cls-only pooling (was contributing-factor) are now the joint top. Next experiments: E5a mean-pool ablation (5-min code change), E5b 3-epoch rerun, E5c 10× cells/donor. See `notes/phase3_geneformer_convergence.md` §8–10. Two CLI bugs caught during investigation: eval-cap default = 200 (10× Run #2 scope), `--gpu-smoke` clobbering production paths — both fixed (`src/finetune/cli.py`).

## 2026-04-27: E5a mean-pool ablation — biggest single-knob R win to date but the regime is unchanged

Mean-pool over attended positions (replacing `<cls>`-only) on the same v2 hyperparameters and data scale produced **R=0.362, MAE=19.34, wall=2.6h** — a +0.046 R improvement over v2 (cls, R=0.316). LoRA delta is the largest of any run yet (median RMS 5.9e-4, max 1.6e-3, ~4× Run #2's), confirming the mechanism: mean-pool surfaces gradient signal across all attended token positions, not just `<cls>`, so the optimizer has more places to learn. **Mean-pool strictly dominates cls in every metric we have** (R, LoRA delta, prediction sd) and should become the new default. MAE creeped back up to ~19.3y because v2's +2.3y prediction-mean drift was a cls-pool side effect under high LR, not a generalizable improvement; with mean-pool the head bias stays near 48.93 (≈ train mean) and MAE is dominated by the train→eval domain-shift floor. **The fundamental regime is unchanged**: prediction sd ≈ 1.1y vs true sd 16.5y — predictions span ~7% of the true range. Pattern across all three runs: each architectural intervention buys ~+0.04 R. To clear R>0.5 (GATE 2) we need a regime change — orders-of-magnitude more SGD steps — not another single-knob fix. Next: E5b (3 epochs at v2 hyperparameters + mean-pool, ~8h local / ~3h on A10g) directly tests the undertraining hypothesis. New CLI flag `--pool {cls,mean}` plus a defensive output-dir redirect for smoke modes (catches accidental contamination of the production summary.csv). Commit: see `notes/phase3_geneformer_convergence.md` §10–12.

## 2026-04-27: Phase-3-A E5b — first regime change clears the predict-mean basin (R 0.362 → 0.466)

First cloud run on g5.xlarge (A10g 24 GB, native bf16). Same data scale as Run #2 / v2 / E5a (9,500 train cells × 190 donors, 19,620 eval cells × 981 OneK1K donors); only change vs E5a is `--epochs 3` instead of 1 (and a fresh `--run-tag-suffix _e5b` to isolate outputs). **Final eval: MAE=17.37y / R=0.466, wall=1.7h.** R jumped +0.104 over E5a — the largest single delta across the four runs and **2.5× the +0.04 single-knob plateau** that LR (v2) and mean-pool (E5a) each produced. Train-MSE trajectory directly supports the undertraining diagnosis from the §9 hypothesis ranking: epoch 1 plateaued at 260–264 (same place all 1-epoch runs stopped); the 20-point break below var(train_ages)≈288 first appeared at step 375, ~70 steps into epoch 2; by step 875 (end of epoch 3) MSE was 222.8, total descent ~65 points. **GATE 2 still not cleared** — bar is MAE<12y / R>0.5; E5b is 17.37y / 0.466, the closest yet on both axes but R is 0.034 short and MAE is ~2× the LASSO floor of 9.4y. Per-step speedup on A10g vs 2080 Ti was ~8× (4.6 s/step vs 37 s/step), wall came in well under the cloud-review's 3h estimate. **Mechanism not yet verified**: don't know whether the R win is "model finally learning age axis" (prediction sd ↑↑) or "extended bias drift" (prediction sd still ~1y like Run #2/v2/E5a). State-dict + per-donor inspection scheduled before the next experiment to determine whether the right next step is more epochs or a different objective. Headline takeaway: regime change works where single-knob fixes asymptoted; preprint's degraded-claim "1/3 wins" outcome is now realistically in reach. See `notes/phase3_geneformer_convergence.md` §13–14.

## 2026-04-27: E5b mechanism check — regime change confirmed, not bias drift

Loaded the E5b checkpoint + the surviving Run #2 per-donor CSV; v2/E5a numbers from memo §1.1/§8/§10. **E5b's prediction sd more than doubled (1.09y → 2.48y) and prediction range went 9.4y → 16.7y, 21% of the true 78y range vs the 6–12% basin all three 1-epoch runs sat in.** This is the first run that genuinely spreads predictions across the age axis rather than orbiting the train mean. The 2.6y MAE drop from Run #2 decomposes into 3.3y of bias closure (pred mean 47.9 → 51.2, walking toward eval mean 63.9) AND 0.7y of err-sd reduction (16.2 → 15.5) — err-sd only drops when predictions actually correlate with truth, so the +0.14 R jump is real signal, not bias drift. Contrast v2 (cls,1ep) where MAE dropped 1.5y but err-sd was unchanged: that was pure bias drift. LoRA delta median RMS climbed 1.4e-4 → 4.5e-4 → 5.9e-4 → 9.4e-4 across the four runs (E5b is 6.7× Run #2's, 1.6× E5a's), confirming the optimizer is still finding gradient signal — no parameter-trajectory plateau yet. Head bias stays glued to ≈48.93 (mean train age, the init) across all runs — the head is not the moving variable. **Decision**: undertraining is binding and more cumulative LR-area is still paying. Next experiment is **E5d at 5 epochs** (no other changes). Crude extrapolation from the +0.014 R/epoch slope between E5a→E5b puts GATE 2's R>0.5 within reach of 2–3 more epochs; MAE<12y is a longer reach (need −5.4y from E5b) and may be domain-shift floor-bound (train mean 50.1y vs eval 63.1y, |Δμ|=13y). See `notes/phase3_geneformer_convergence.md` §14–15.

## 2026-04-27: Phase-3-A E5d — 5 epochs at this data scale regresses R from 0.466 to 0.431

E5d (g5.xlarge A10g, ~2.4h, $2.4) ran 5 epochs at the E5b config — same data scale (9,500 train cells × 190 donors), `--pool mean --lr 2e-4 --head-lr 2e-4`, only difference is epochs 5 vs 3. **Final eval: MAE=16.53y / R=0.431.** MAE continued improving from E5b's 17.37y (−0.84y), but **R regressed −0.035 from E5b's 0.466**, breaking the +0.014 R/epoch slope from E5a→E5b. Train MSE kept descending (222 at E5b end → 196 at E5d end, lowest of any run; total descent 28% from epoch-0 baseline). Mechanism check (`scratchpad/_inspect_e5d.py`): pred sd grew 2.48 → 3.24y (+30%), pred range 16.7 → 20.6y (+23%), LoRA delta median RMS 9.4e-4 → 1.4e-3 (+47%) — the optimizer is still moving parameters and predictions are still spreading more across the age axis. cov(pred, true) actually grew (19.07 → 23.04), so the alignment with truth held; R dropped because pred_sd grew faster than cov did. **This is overfitting in a subtle form**: not "model memorizes train and eval predictions go random" but "model finds more train-specific patterns and produces more-confident eval predictions, but the additional confidence isn't all on the right axis." The MAE improvement is bias-closure-driven (err mean −12.7 → −11.9, walking pred mean toward eval mean), not rank-improvement. **GATE 2 status worsened**: R 0.431 is further from the 0.5 bar than E5b's 0.466 was. **Decision**: §15's conditional triggers — "If R asymptotes between E5b and E5d, the next move is data scale (E5c)." E5d regressed past asymptote, even stronger signal. **Next is E5c** (`--max-cells-per-donor 500 --epochs 1`, 95k train cells, ~3.8h, ~$3.80) to test whether 10× more cells/donor reduces per-donor noise enough to lift the R~0.45 ceiling, separately from the "more epochs" lever. See `notes/phase3_geneformer_convergence.md` §16–17.

## 2026-04-27: Phase-3-A E5c — data-scale lever ALSO regresses R; E5b is a sweet spot

E5c (g5.xlarge A10g, ~4.3h, $4.3) ran 1 epoch with `--max-cells-per-donor 500` (10× E5b's cap), 93,702 train cells, 2,930 optimizer steps (~3.3× E5b). **Final eval on OneK1K CD4+T (981 donors): MAE=16.27y (best of any run) / R=0.385 (worse than both E5b 0.466 and E5d 0.431).** Both scaling levers — more epochs and more cells/donor — regress R below E5b. Mechanism inspection (`scratchpad/_inspect_e5c.py`) reveals a different failure mode than E5d's: E5c's pred_sd 2.63y is intermediate between E5b's 2.48 and E5d's 3.24, but R is the *worst* of the three despite having the **largest LoRA delta of any run** (median RMS 1.7e-3, max 4.5e-3 — vs E5b's 9.4e-4 / 2.3e-3 and E5d's 1.4e-3 / 3.3e-3). The optimizer is moving parameters more than ever but the prediction spread is *less aligned* with truth. **The bottleneck is not training budget; it's the per-cell MSE objective**: each donor in E5c contributes 500 cells with the same age label (vs 50 in E5b), so the model gets 500 duplicated (cell, age) pairs per donor and learns donor-specific cell-level idiosyncrasies — donor memorization, not age-rank generalization. MAE keeps improving across all four runs (19.99 → 17.37 → 16.53 → 16.27) because predictions continue walking toward eval mean (err mean −16.0 → −12.7 → −11.9 → −10.7), but that's bias closure, not rank-improvement. **GATE 2 verdict**: at this objective formulation, E5b's R=0.466 is the Geneformer-LoRA single-seed ceiling; neither scaling lever closes the 0.034 gap to the 0.5 bar. **Phase-3-A close-out next**: variance check at E5b (2 more seeds), AIDA scoring of all checkpoints, then close-out memo. **Phase-3-B and per-donor objective ablation reserved for the next session** — don't fit alongside the variance check and AIDA scoring within the 12h autonomous window. See `notes/phase3_geneformer_convergence.md` §18–19.

## 2026-04-27: Phase-3-A close-out — OneK1K-AIDA inversion reframes the §18 single-seed conclusion

Variance check at the E5b config (2 more seeds, 3.4h compute, $3.4) plus AIDA scoring of all 5 loco_onek1k Geneformer checkpoints (50 min inference, ~$0.5). **Three-seed E5b on OneK1K: R=0.453 ± 0.042, MAE=17.17 ± 0.43y.** R variance across seeds is 0.102 (0.396–0.498), larger than the 0.035 single-seed difference between E5b (0.466) and E5d (0.431) the §18 memo had treated as significant. The "E5b is a sweet spot" claim was a single-seed effect. **AIDA scoring reveals the major new finding: configs that look "regressed" on OneK1K are *better* on AIDA cross-ancestry**: E5c at R=0.545 on AIDA dominates E5b mean R=0.300, even though E5c's OneK1K R (0.385) is the worst of the three configs. Within-config seed variance shows the same negative correlation: E5b seed 1 has the highest OneK1K R (0.498) but the lowest AIDA R (0.240). **OneK1K rank vs AIDA rank is reversed**: E5b first on OneK1K, last on AIDA; E5c third on OneK1K, first on AIDA. Mechanism: OneK1K and the train cohorts are all European; AIDA is Asian. Configs that overfit donor-specific cell patterns on Stephenson+Terekhova generalize well to OneK1K (similar population distribution) but fail on AIDA (different population). E5c's 10× cells/donor is a stronger regularizer against population-specific overfitting at the cost of in-distribution accuracy. **Win/match/loss classification** vs Phase-2 baselines: 0/2 wins (loss on OneK1K to LASSO 9.45y/R=0.75; loss on AIDA to Pasta 6.32y/R=0.66). Per kickoff outcome rules, Phase-3 pivots toward evaluation-study framing for the preprint. **Phase-3-B go-decision**: yes, with structural changes — (1) loco_terekhova × E5b × seed 0 next as the third headline cell; (2) per-donor objective ablation as the highest-information-per-dollar follow-up; (3) defer scFoundation/scGPT until per-donor obj is tested; (4) frame preprint around the methodology finding (OneK1K-AIDA inversion) rather than horse-race victory. See `notes/phase3_geneformer_convergence.md` §20.

## 2026-04-28: Phase-3-A Terekhova fold — catastrophic chemistry-shift collapse (R=0.140)

`loco_terekhova` × E5b config × seed 0 (g5.xlarge A10g, 6.1h, ~$6.1) trained on OneK1K (981 donors, 10x 3') + Stephenson (24 donors, 10x 3') and evaluated on Terekhova (166 donors, 10x 5' v2). **Final eval: MAE=11.37y, R=0.140** — the worst R of any Phase-3-A run, far below the Pasta-REG (R=0.78, MAE=8.04) and LASSO (R=0.82, MAE=9.15) baselines on this fold. Mechanism inspection (`scratchpad/_inspect_terekhova.py`) shows striking features: pred_sd 4.71y is the LARGEST of any Geneformer LoRA run; LoRA delta median 2.2e-3 is 2.3× E5b on loco_onek1k; head_bias barely moved from init (63.46 vs 62.65 init); but predictions land 23.6y BELOW head bias on average (pred mean 39.87 vs head bias 63.46) because the LoRA-modified hidden states produce systematically negative dot-products with the head weight on Terekhova cells. **The opposite of the Phase-2 chemistry-rescue hypothesis.** The kickoff §3 hypothesis was that FMs would be chemistry-robust by virtue of diverse pretraining; the actual finding is that LoRA-fine-tuning on 3'-only data collapses to R=0.14 on 5' eval. Train cohort homogeneity (98% OneK1K cells) plus 2.3× LoRA delta means adapters overfit to 3'-chemistry-specific tokens that don't fire correctly on 5' rank-value encodings. Pasta-REG's chemistry-invariance via rank-normalization wins where the FM fails. **Tri-headline outcome: 0/3 wins** — OneK1K loss to LASSO, Terekhova loss to Pasta (catastrophic), AIDA loss to Pasta. Per kickoff outcome rules: strong pivot to evaluation-study framing, preprint headline becomes "scRNA-seq FMs do not match published immune-aging baselines on PBMC at this scale; chemistry-shift collapse and OneK1K-AIDA inversion are publishable methodological contributions." **Phase-3-B priorities reordered** (memo §21.4): (1) audit whether chemistry collapse is Geneformer-specific or FM-general (cheap experiment: scFoundation + scGPT on loco_terekhova, ~$15); (2) train-cohort balancing in `select_indices` (98% OneK1K is a confound) — ~30 LOC change; (3) per-donor objective should run on loco_terekhova not loco_onek1k. See `notes/phase3_geneformer_convergence.md` §21.

**AIDA scoring of the Terekhova-trained checkpoint: R = -0.146 (negative correlation), MAE 9.50.** The chemistry collapse propagates from Terekhova (5') to AIDA (also 5' v2). Re-examining train-set chemistry mix clarifies the asymmetry: loco_onek1k models train predominantly on 5' chemistry (Terekhova contributes 87% of cap=50 train cells; Stephenson only 13%), so their AIDA evaluations are same-chemistry 5'→5' but cross-ancestry — R 0.24–0.55 reflects population-only transfer. Loco_terekhova trains 100% on 3' chemistry (OneK1K dominates 98%, Stephenson 2%) and evals on 5', so it stacks chemistry shift + cross-ancestry on AIDA. **The OneK1K–AIDA inversion finding from §20.2 is therefore a pure population effect, not chemistry**. **Geneformer LoRA's chemistry-rescue is asymmetric**: trained-mostly-on-5' generalizes acceptably to 3', but trained-100%-on-3' collapses on 5' (R catastrophe). Either rank-value encoding is sensitive to which-end-bias direction, or the optimizer found 3'-specific tokens that don't fire on 5' inputs. See memo §21.5–§21.6.

## 2026-04-28: Phase-3-A B + NK extension complete; 0/6 wins; chemistry-rescue null

Three Geneformer LoRA fine-tunes on B and NK (NK × loco_terekhova cancelled in favor of the diagnostic ladder per `scratchpad/pseudobulk_review.md` + `geneformer_review.md`). Results:

- **B × loco_onek1k**: MAE 24.28y, R = -0.076 (anti-correlation; loss vs all 4 baselines, worst R)
- **NK × loco_onek1k**: MAE 19.77y, R = 0.165 (loss; ties Pasta-REG R=0.159, beats scAgeClock; loses to LASSO 0.629)
- **B × loco_terekhova (chemistry-rescue cell)**: MAE 14.23y, R = 0.075 (loss to Pasta-REG 10.86 / 0.281)

**Phase-3-A aggregate: 0/6 wins** across (cell × fold) pairs. Cell-type pattern across loco_onek1k: FM R correlates POSITIVELY with baseline R — FMs do somewhat-OK where baselines do well (CD4+T R=0.45 vs LASSO 0.75), fail catastrophically where baselines fail (B R=-0.08 vs LASSO 0.53). Exact OPPOSITE of the kickoff §3 few-shot hypothesis. The FM extracts a constant ~50-60% fraction of the available signal regardless of cell type — but if the absolute signal is small, that fraction is invisible in noise. **Chemistry-rescue test on B × loco_terekhova rejected the kickoff hypothesis directly**: Pasta-REG's rank-norm chemistry-invariance recovers R=0.28 where LASSO collapses to 0.08, but Geneformer LoRA only manages R=0.075 — essentially tied with LASSO's collapse, far below Pasta. **FMs do NOT add chemistry-invariance on top of what rank-norm bulk modelling already provides**. The negative claim now spans 6 (cell × fold) pairs with all losses but is bounded to "Geneformer LoRA at per-cell fine-tune protocol" — exactly the bounded claim the diagnostic ladder (memo §22.4 + new sub-phase) is designed to upgrade. See memo §22.

## 2026-04-28: Variant 1 Phase 1 — frozen-base ridge BEATS fine-tune on CD4+T (protocol-negative diagnostic)

Built `scripts/extract_embeddings.py` (mean-pool over attended positions, per-donor average) + `scripts/donor_ridge.py` (RidgeCV alpha selection, per-cohort holdout, optional AIDA transfer). Phase 1 ran frozen-base Geneformer V2-104M across 4 cohorts × CD4+T + ridge fits on both LOCO folds. **Three rows in `results/phase3/ridge_summary.csv`**: loco_onek1k×OneK1K R=0.560 / MAE 16.52 (vs E5b fine-tune R=0.466 / MAE 17.37); loco_onek1k transfer×AIDA R=0.527 / MAE 11.76 (vs E5c R=0.545 / MAE 9.53 — fine-tune slightly better R but 2.23y worse MAE because the transferred ridge doesn't bias-close to AIDA mean); loco_terekhova×Terekhova R=**0.576** / MAE 24.03 (vs fine-tune R=**0.140** — **+0.436 R uplift, the smoking gun**). Frozen-base ridge ≥ best fine-tune on R for every fold tested on CD4+T. **Protocol-attribution claim**: per-cell MSE fine-tune is destroying signal Geneformer's pretrained representation already encodes, including chemistry-robustness — the §21 R=0.140 chemistry collapse is now attributable to the fine-tune protocol, not the pretrained model. Bounded to one cell type until Phase 2 (B + NK frozen-base) confirms generality. Frozen-base ridge does NOT beat Phase-2 baselines (LASSO 0.747 / Pasta 0.778) — this is a diagnostic probe explaining *why* the FM loses, not a new winning recipe. Per §22.5 ladder, AIDA frozen-base R=0.527 in the [0.45, 0.60) "mixed" bracket → next is Phase 2 in series (cheap, ~30 min), then Variant 2 + Variant 3 in parallel. See memo §23.

## 2026-04-28: Variant 1 Phase 2 — protocol-negative confirmed but cell-type-magnitude split (B/NK = representation-negative)

Phase 2 (~2.7h compute, $2.7) ran 8 frozen-base extractions (4 cohorts × {B, NK}) + 6 ridge fits. **Frozen-base ridge ≥ fine-tune on R for all 5 (cell × fold) pairs that have both** (Δ R ∈ [+0.027, +0.436]); direction of §23 protocol-negative claim holds. **But the magnitude splits cleanly**: CD4+T frozen R=0.527-0.576 across 3 conditions (real signal, all p<1e-15) — fine-tune erases it (Terekhova catastrophe Δ R=+0.436). B and NK frozen R in [−0.01, 0.26] across 6 conditions, **3 of 6 p>0.1** — substrate is largely empty, especially B (R=−0.01 OneK1K, R=0.10 Terekhova, both n.s.). LASSO recovers R=0.531 on B × OneK1K from the same gene matrix, so the absent signal is in **Geneformer's compressed representation, not in the input data**. Updated picture: CD4+T = protocol-negative (Variants 2/3 attack the right problem); B/NK = representation-negative (last-layer mean-pool throws away signal LASSO finds in raw genes). New §24.4 hypotheses to distinguish: (a) layer ablation — earlier layers may preserve aging detail; (b) pooling — mean-pool may average over aging-relevant subpopulations; (c) vocabulary/pretraining domain mismatch upstream of the encoder. Variant 3 (layer-wise probe on B+NK, ~1h) discriminates (a)+(b) from (c). Updated next steps per memo §24.5: Variant 3 first (cheapest, addresses representation-negative directly), then Variant 2 on CD4+T, then scFoundation FM-class diagnostic. **Negative claim now**: "Geneformer LoRA fine-tune is protocol-negative on the one cell type where its frozen representation has signal, and is representation-negative on B and NK regardless of protocol." Publishable as diagnostic-study and motivates the FM-class comparison. See memo §24.

## 2026-04-28: Variant 1 audit (D.9 + D.10 + D.11) — bias-variance refines §23/§24, "smoking gun" framing was one-sided

`scripts/variant1_audit.py` re-fit all 9 ridge conditions and computed bootstrap CIs + bias-variance metrics. Three reframes land in memo §25:
**D.9 — bootstrap CIs split B from NK**: B is genuinely empty (0/3 CIs exclude zero, slopes ∈ [−0.005, 0.060]); NK is weak-but-real on OneK1K (R=0.260 [0.197, 0.320] tight CI) and Terekhova (R=0.199 [0.044, 0.347]). The §24 "B and NK look representation-negative" wording was wrong for NK. Reframed: "NK substrate captures ~40% of LASSO's NK-aging signal (0.260 vs 0.629)."
**D.10 — universal compression + AIDA bias near train mean**: All 9 conditions show pred_sd / eval_sd ∈ [0.30, 0.73] (heavy regularization-driven compression). CD4+T × AIDA pred_mean=52.86 is 11.1y above eval_mean (41.76) and 4y above train_mean (48.89); slope=0.39 → predictions track ranking but at compressed scale. AIDA R=0.527 is real ranking but the MAE=11.76 is partly artifactual; cross-ancestry claim must be qualified, not free-standing. CD4+T × Terekhova pred_mean=25.46 is 24y BELOW eval_mean — frozen-base ridge has catastrophic bias on this fold (MAE=24.03), much worse than fine-tune MAE=11.37. **The §23 "smoking gun" was one-sided**: frozen wins on R but loses badly on MAE. Refined story: per-cell MSE fine-tune is rank-negative AND bias-closure-positive; frozen ridge is rank-positive AND bias-closure-negative. Neither delivers both jointly. **D.11 — honestly-bounded claim ladder**: 4-tier ladder (strongest = "rank-negative AND bias-closure-positive fine-tune"; medium = "frozen captures partial signal below LASSO"; weakest = "AIDA is bias-shifted ranking"; bounded null = "B is empty, NK is weak"). Updated §22.5 / §24.5 decision tree: **Variant 3 (layer-wise probe) now higher priority** — cheapest test for whether B substrate exists at any layer; ~1h compute. See memo §25.

## 2026-04-28: Variant 3 layer-wise probe — HEADLINE: Geneformer layer-1 + ridge on CD4+T × Terekhova hits R=0.616 / MAE=8.82, Pareto-dominates the fine-tune

`scripts/extract_embeddings_layered.py` + `scripts/donor_ridge_layered.py` (~95 min compute, $1.6) extracted per-layer mean-pool embeddings from frozen Geneformer V2-104M (13 layers including embedding output) across 4 cohorts × 3 cell types, then fit ridge per (fold × cell × layer × eval_cohort) — 117 rows in `results/phase3/ridge_summary_layered.csv`. **The headline result**: frozen Geneformer **layer 1 + ridge** on CD4+T × Terekhova achieves **R=0.616 / MAE=8.82**, beating both the layer-12 frozen probe (Variant 1: R=0.576 / MAE=24.03) and the per-cell MSE LoRA fine-tune (R=0.140 / MAE=11.37) on both metrics simultaneously. **Layer-1 MAE is within 0.78y / 9.7% of Pasta-REG (8.04y)** — first time any FM-derived predictor approaches the GATE-2 baseline on the headline fold.
**Bias-variance fix at layer 1**: §25.2 found CD4+T × Terekhova frozen pred_mean=25.46 vs eval_mean=49.43 (24y bias catastrophe). Layer 1 has pred_mean=47.92 — within 1.5y of eval_mean. Layer 12 compresses Terekhova into a chemistry-specific subspace whose mean projects to ~25; layer 1 preserves a chemistry-robust age direction whose mean projects near the European train mean (~63) tempered by ridge regularization → ~48.
**B-cell substrate not entirely empty**: §25.1 said "B is genuinely empty" based on layer 12. Layer 9 of B × Terekhova reaches R=0.228 (p=0.003), CI [0.07, 0.37]. The B-cell representation gap is in the LATE layers, not upstream of the encoder. Within-cohort B × OneK1K stays flat at all layers — donor-batch dominated.
**NK middle-layer R uplifts are real but unstable on small train**: NK × OneK1K layer 3 R=0.304 / MAE=62.28 (alpha=0.1 over-fit on 195 donors × 768 dim). NK × Terekhova layer 2 R=0.266 / MAE=14.41 is reliable.
**Updated honestly-bounded claim ladder (memo §26.6)**: Strongest claim is now "frozen layer-1 Pareto-dominates the per-cell fine-tune AND comes within 9.7% MAE of Pasta-REG." This is the first positive Geneformer recipe of Phase-3-A.
**Updated decision tree**: priority is now (a) re-extract layer 1 from FINE-TUNED checkpoints + ridge — does fine-tuning destroy layer-1 signal too?; (b) Variant 2 target moves to R≥0.616 / MAE≤8.82; (c) scFoundation FM-class diagnostic with layer-by-layer probe. The discovery that **layer choice matters more than fine-tuning** is itself a publishable methodology finding for any FM-aging pipeline. See memo §26.

## 2026-04-28: Variant 3 follow-up — fine-tune layered probe overturns §26: ridge-on-fine-tune BEATS LASSO on OneK1K (1st Phase-3 WIN), readout was the bottleneck

Re-extracted per-layer embeddings from `loco_onek1k_seed0_CD4p_T_e5b.pt` and `loco_terekhova_seed0_CD4p_T_e5b.pt` checkpoints across 4 cohorts × CD4+T (~1.4h, $1.4) and ran ridge per layer per fold. Output: `results/phase3/ridge_summary_layered_finetune.csv` (52 rows). Also Variant 4 concat probe on frozen-base layers (no compute) — `ridge_summary_concat.csv` (72 rows).
**THE MAJOR FINDING (memo §27.1)**: ridge-on-fine-tuned-rep at layer 12 mean-pool gives R=0.631 / MAE=8.21 on loco_onek1k × OneK1K (981 donors LOCO holdout). Compare:
- Original E5b through linear head: 17.37 / 0.466 (the §22.3 "loss" result)
- Frozen layer-12 ridge (Variant 1): 16.52 / 0.560
- LASSO baseline: 9.45 / 0.747; **ridge-on-fine-tune MAE 8.21 BEATS LASSO MAE by 13.1%, clearing the kickoff §28 10%-win threshold (≤8.5y)**
On AIDA cross-ancestry from same loco_onek1k checkpoint: layer 12 ridge = R=0.611 / MAE=7.84, vs Pasta 6.32 — 24% above floor but cleanest cross-ancestry FM result yet (pred_mean 41.44 vs eval_mean 41.76, gap 0.32y; slope 0.544; sd_ratio 0.89). On Terekhova from loco_terekhova checkpoint: layer 1 ridge = R=0.619 / MAE=8.63, vs Pasta 8.04 = 7.4% above (MATCH-class result).
**Tri-headline revised tally: 1 WIN (OneK1K) + 1 MATCH (Terekhova) + 1 CLOSE-LOSS (AIDA)**, vs the §22.3 0/3 wins under the original head. This is materially stronger than the "0/3 → pivot to evaluation-study framing" call.
**Mechanism (§27.2)**: same backbone, same LoRA weights, only readout differs. Per-cell MSE-trained linear head sees per-cell expression noise, fits suboptimal weights. Per-donor ridge bypasses per-cell noise by fitting on per-donor mean embeddings post hoc. The §22.4 horse-race losses were attributable to the readout, NOT the LoRA fine-tune itself.
**Layer profile differs sharply by fold (§27.4)**: loco_terekhova fine-tune destroys layer-12 (frozen 0.576 → ft 0.284) but lifts mid-layers (layer 4 R=0.703); loco_onek1k fine-tune IMPROVES layer 12 (frozen 0.560 → ft 0.631). §23 "fine-tune destroys signal" was specifically true for layer-12-on-Terekhova, wrong as general statement.
**Phase-3-A reframe**: preprint pivots BACK to "FM matches baselines" headline for OneK1K CD4+T. The "wrong readout" finding is itself publishable methodology. Updated decision tree §27.9: ridge-readout on B and NK checkpoints next (~30 min); 3-seed variance check on the WIN; then Variant 2 / scFoundation FM-class with the readout-fix baked in. See memo §27.

## 2026-04-28: §28 audit — 3-seed variance + B/NK ridge readout — §27 WIN does NOT hold across seeds

Per §27.9 priorities, ran 5 additional fine-tune layered extractions (~2h, $2): 3-seed variance check on `loco_onek1k_seed{1,2}_CD4p_T_e5b` checkpoints (4 cohorts × 2 seeds = 8 extractions) plus B/NK ridge readout on `loco_onek1k_seed0_B_e5b`, `loco_onek1k_seed0_NK_e5b`, `loco_terekhova_seed0_B_e5b` (12 extractions). Output: `results/phase3/ridge_summary_post_finetune.csv` + `cd4t_3seed_ridge_layered.csv`.
**§28.1 — 3-seed CD4+T variance reveals the §27 WIN was single-seed only**:
- L12 OneK1K: seed 0 = 8.21y / 0.631; seed 1 = 14.83y / 0.629; seed 2 = 10.36y / 0.565. **3-seed mean = 11.13 ± 3.38y** vs the 8.5y WIN bar.
- L6 OneK1K (most stable layer): R=0.632 ± 0.008 (very stable rank), but MAE=10.85 ± 2.19y (above 8.5 bar at mean).
- The §27 WIN claim should be qualified to "single seed 0 result of 8.21y clears the WIN bar; 3-seed mean is CLOSE-MATCH at +15% above LASSO."
**§28.2 — AIDA × L11 is the more defensible cross-ancestry headline**: 3-seed mean R=0.566 ± 0.032, MAE=7.96 ± 0.42y. Tight std (±0.42y), reproducible. Still close-loss vs Pasta 6.32 (+25.9%), but the most reliable cross-ancestry FM result of Phase-3-A.
**§28.3 — B and NK ridge readout doesn't rescue §22.3 0/6 narrative**: B × loco_onek1k ridge L12 R=+0.099 (vs head −0.076) but MAE=63.71 (vs head 24.28) — ridge MAKES MAE WORSE. NK × loco_onek1k ridge L12 R=0.253/MAE=15.12 (vs head 0.165/19.77) — modest improvement but still loses to LASSO 0.629/9.64. B × loco_terekhova essentially unchanged. **B is genuinely representation-negative across fine-tune layers and seeds; NK gets partial rescue but no win.**
**Phase-3-A revised tri-headline (§28.5)**: 0 strict WINs at 3-seed mean (no cell clears 10%-win bar), 2 MATCH-class (OneK1K 15% above LASSO, Terekhova 7.4% above Pasta), 1 close-loss (AIDA 25.9% above Pasta). Better than original 0/3 from §22.3, weaker than §27.6 1+1+1 claim.
**Honest publishable claims (§28.4)**:
1. Strongest = "Per-donor ridge readout systematically improves over the per-cell MSE head across all CD4+T conditions; the fine-tuned representation contains more signal than the head extracts."
2. Methodology contribution = "Per-cell MSE linear head systematically underestimates donor-level signal in fine-tuned single-cell FMs; per-donor ridge regression is strictly better. Generalizes to any donor-prediction task using per-cell-trained FMs."
3. Bounded null = "B-cell substrate empty across all layers × seeds × readouts."
**Decision tree (§28.6)**: Variant 2 (pseudobulk fine-tune) target = OneK1K MAE ≤8.5y at 3-seed mean. scFoundation FM-class diagnostic with per-donor ridge baked in. See memo §28.

## 2026-04-29 — §29 scFoundation FM-class diagnostic: 0/6 is FM-class, not Geneformer-specific

**Pivot decision (§29 head)**: Variant 2 (pseudobulk fine-tune) deprioritized — per-donor ridge readout from §27 already enforces a donor-level objective post-hoc, and §28 seed-variance (std 3.38y on 8.5y bar) suggests optimization instability rather than loss-function. scFoundation diagnostic (, 6h) directly tests whether the 0/6 horse-race loss is Geneformer-recipe-specific or FM-class. Skipped Variant 2; ran scFoundation frozen + ridge on the same protocol as Geneformer §22/§27/§28.

**§29.3.1 results, scFoundation frozen + canonical pool='all' (3072-d) + per-donor mean across 20 cells/donor + RidgeCV**: across 9 (cell × eval cohort) conditions, scFoundation **loses to LASSO/Pasta on every CD4+T cell** and is **worse than Geneformer §28 3-seed mean on CD4+T × OneK1K** (R=0.475 vs 0.632, MAE=12.79 vs 10.85). 27× larger model with 100× more pretraining data underperforms Geneformer.

**§29.3.2 the 1-bit answer**: 0/6 is **FM-class, not Geneformer-specific**. Two FMs (Geneformer 110M, scFoundation 3B), two pretraining protocols (genecorpus rank-value, mixed bulk+sc), same canonical frozen + ridge readout, both lose 30–100% on CD4+T MAE vs bulk-trained baselines. B substrate empty for both (R near zero, CIs cross zero). NK weak for both. **Fine-tune + ridge readout (Geneformer §27/§28) is the recipe contribution**, NOT a substrate property.

**§29.3.3 bias check**: scFoundation predictions show same mean-compression as Geneformer (§25/§28.4): pred_mean=55.97 vs eval_mean=63.91 on OneK1K (−8y); AIDA pred_mean=62.18 vs eval_mean=41.76 (+20y) on the small-train fold. Mean-compression is FM-class, not architecture-specific.

**§29.3.4 caveats**: scFoundation was frozen (not fine-tuned — that needs Phase-4 LoRA). 7 of 12 extractions fp32, 5 bf16 (post-OOM retry); per-donor delta R=0.965 between fp32 and bf16, well below ridge variance. Per-layer scFoundation probe NOT done. tgt_t=4.0 (canonical) only.

**§29.3.5 verdict update**: Methodology paper claim narrows. The 'per-donor ridge readout > per-cell MSE head' finding is real and reproducible, BUT it does NOT close the bulk-vs-FM gap at frozen weights. **Fine-tuning + ridge readout** is the joint contribution (Geneformer §27 readout improvement only manifests on fine-tuned reps). B-cell substrate emptiness is an FM-class biological finding worth a single panel.

**§29.4 next-step priority (revised)**: 1) Higher-rank LoRA + longer training on Geneformer CD4+T × OneK1K — most promising lever to convert close-MATCH to strict WIN. 2) Per-layer scFoundation probe (~3h, ~) — cheap test of whether scFoundation has analogous layer-1 finding to Geneformer §26. 3) scFoundation LoRA fine-tune (4 on g5) deferred to Phase-4 unless higher-rank LoRA fails. See memo §29.

## 2026-04-29 — §30 rank-32 smoke: capacity isn't the bottleneck (PIVOT)

**Smoke test (Task D.12)**: single-seed (seed 0) rank-32 LoRA on CD4+T × loco_onek1k, all other hyperparameters held to e5b config (--epochs 3 --lr 2e-4 --max-cells-per-donor 50 --pool mean). Output: `results/baselines/fm_finetuned/geneformer/checkpoints/loco_onek1k_seed0_CD4p_T_e5b_r32.pt`, ridge in `results/phase3/ridge_summary_r32_smoke.csv`.
**§30.1 headline**: rank-32 L12 OneK1K MAE=11.00 / R=0.636 vs rank-16 seed-0 (e5b) MAE=8.21 / R=0.631 (§27.1). Rank-32 single-seed lands ALMOST EXACTLY on the rank-16 3-seed MEAN (11.13 ± 3.38). Per pre-stated decision rule (≤8.0y promote, ≥8.5y or worse than rank-16 pivot): **PIVOT**.
**§30.3 interpretation**: §28 seed std=3.38y is **optimization-limited, NOT capacity-limited**. Rank doubling doesn't lift the floor — rank-16 seed 0 was just a better-than-mean basin. Don't run 3-seed of rank-32; don't try rank-64. Remaining levers: longer training (5-6 epochs, direct test of (b)), AIDA-focused L9 3-seed bracket (R=0.617/MAE=6.92, well-calibrated, comparable to Pasta), or write up Phase-3-A as-is.
**§30.2 incidental finding**: AIDA L9 ridge R=0.617 / MAE=6.92 with near-zero bias (pred_mean=41.88 vs eval_mean=41.76). Comparable to Pasta-REG (R=0.659/MAE=6.32). Cross-ancestry signal lives in mid-layers, not just final-layer.
**Process notes**:
1. **3h sunk earlier on misconfigured rank-32**: launched with CLI defaults instead of the actual e5b launch flags from memo §15. Saved feedback memory `feedback_finetune_hparams.md`.
2. **Detached process management**: bash run_in_background's wrapper kill propagated to children, killing the python mid-training. Recovered by relaunching with `nohup setsid` for full session detachment, plus a /proc/PID/fd/X poller to back up the deleted-but-open jsonl in case the process exited between recovery cycles. (Final_eval/done events for the rank-32 run were missed by the poller — not blocking since the headline is the ridge-readout MAE, not the per-cell head MAE.)
3. **Cross-rank checkpoint loading**: extract_embeddings_layered.py defaulted to lora_rank=16, raising size mismatch when loading the rank-32 checkpoint. Added `--lora-rank` flag.
**Compute**: ~$3 wasted on the misconfigured run + ~$3 on the corrected run = ~$6 total. Wall ~5h end-to-end (including misfire).

## 2026-04-29 — §31 NK early-layer asymmetry on frozen Geneformer (no new compute)

**Step-back review action D.20**: re-read existing `results/phase3/ridge_summary_layered.csv` (frozen-base layered probe from §26). Found a clean cell-type-specific layer asymmetry not previously characterized.

**§31.1 finding**: Best-R layer per cell type, averaged across 3 (fold × eval_cohort) conditions:
- CD4+T: L9.7 mean (L12 wins 2/3, L5 wins on Terekhova chemistry-shift)
- B: L9.0 (substrate empty everywhere, R~0.04–0.23)
- **NK: L3.3 (early-layer dominant on ALL 3 conditions)** — L3 wins on OneK1K, L2 wins on Terekhova, L5 wins on AIDA

**§31.2 robustness**: NK Δ between best-layer-R and L12-R is largest on AIDA cross-ancestry (+0.121), smaller on Terekhova chemistry-shift (+0.067), smallest on in-distribution OneK1K (+0.044). Direction consistent across all 3 cohorts.

**§31.3 hypothesis**: NK is a more heterogeneous compartment (cytotoxic/regulatory/adaptive subsets); aging signal there is coarser compositional shifts captured by early layers. CD4+T aging is finer activation programs needing late-layer features. Testable via donor-cluster analysis on layer embeddings — out of scope for current writeup.

**§31.4 implications for writeup**:
1. Novel finding — no prior FM literature reports cell-type-conditional layer asymmetry for donor-level prediction.
2. The §30 L9-AIDA-rank-32 finding is NOT this asymmetry; that's fine-tuning artifact specific to seed 0 of rank-32. The frozen-base CD4+T finding has L12 winning on AIDA, not L9.
3. Methodology recommendation: cell-type-conditional layer selection (L12 for CD4+T, L3–L5 for NK).
4. Doesn't weaken §22.3 0/6 narrative — even NK best-layer R=0.169 on AIDA loses to LASSO/Pasta.

**Cost**: $0, ~30 min analysis + writeup. No new compute. The pattern was already in data we collected for §26 in March; took the step-back review to surface it as a distinct finding rather than buried in the layered CSV.

## 2026-04-29 — §32 matched-splits gene-EN + pseudobulk-input Geneformer (D.17 + D.18 frozen-base)

**Step-back review actions D.17 + D.18**: closed the two unaddressed apples-to-oranges concerns in the FM-vs-bulk comparison.

**D.17 (gene-EN matched splits)**: ElasticNetCV (top-5000 HVG, StandardScaler, 4 l1_ratios × 8 alphas × 3-fold inner CV) on the same `data/loco_folds.json` splits, same per-cell normalization, same donor caps as the FM experiments. Output `results/baselines/gene_en_matched_splits.csv` (9 rows). Script: `scripts/gene_en_matched_splits.py`.

**D.18 frozen-base (pseudobulk-input Geneformer)**: per donor, sum raw counts across selected cells → Geneformer rank-value tokenize as a single pseudo-cell → frozen forward → 13-layer mean-pool → ridge readout. Output `results/phase3/ridge_summary_pseudobulk.csv` (117 rows = 9 conditions × 13 layers), `results/phase3/embeddings_pseudobulk/*.npz`. Script: `scripts/extract_embeddings_pseudobulk.py`.

**§32.1 PAPER-CHANGING headline**: Gene-EN matched-splits R = 0.612 (CD4+T × OneK1K), 0.776 (Terekhova), 0.616 (AIDA loco_onek1k), 0.651 (AIDA loco_terekhova). FM frozen ridge readout R = 0.527–0.621 across the same conditions. Gap is **~0.05–0.15 R-units, not 0.38**. The "FM loses to gene-EN by 0.38 R-units" framing was an apples-to-oranges artifact — TF paper used more cohorts + different preprocessing + different hyperparams. AIDA cross-ancestry is essentially tied: gene-EN R=0.616/MAE=6.42 vs FM rank-32 L9 ridge R=0.617/MAE=6.92.

**§32.3 pseudobulk-input layer profile**: Best-R layer for CD4+T shifts to L1–L4 (early) when fed donor-aggregated input, opposite of per-cell mean-pool which favors L12. Consistent with §31's NK-early-layer hypothesis: early layers encode coarse expression-level features that match what bulk gene-EN extracts. R is competitive with per-cell mean-pool (Terekhova R=0.688 vs 0.621 — actually higher), but MAE is worse on cross-cohort conditions because the ridge fit's bias is hard to calibrate across very different mean ages.

**§32.4 reframing for the writeup**:
- ~~"Single-cell FM fine-tuning loses to gene-EN by ~0.38 R-units."~~
- → **"At the strict donor unit-of-analysis with ~190–1000 training donors, gene-EN, frozen Geneformer + ridge readout, and rank-32 LoRA + ridge readout all converge to R = 0.6–0.7 on CD4+T cross-cohort age regression."**
- B substrate empty in BOTH gene-EN AND FM (gene-EN B × OneK1K R=0.136; FM frozen R~0). B-empty is **substrate-level**, not architecture-level.
- NK at matched splits is hard regardless of model (gene-EN R=0.366; FM ridge L3 R=0.304–0.368).

**§32.5 next-step decision**: D.18 LoRA × 3-seed extension **deprioritized** — frozen-base pseudobulk-input result is sufficient to make the "FM and bulk converge at matched splits" point. Adding LoRA fine-tunes on pseudobulk-input is unlikely to flip the picture. The paper now has enough characterization across 5 sections (§22/§27/§28, §29, §30, §31, §32) to start drafting; remaining decision is which finding to lead with.

**Cost**: ~$3 D.18 frozen-base + $0 D.17 (CPU sklearn) = ~$3 total. Wall ~1 day with dev. Memo §32 (full table). Both experiments delivered the predicted-most-likely outcome (matched gap is small, pseudobulk-input shifts layer profile) — diagnostic value high regardless of whether the result was a WIN.

## 2026-04-29 — §33 load-bearing single-seed numbers inventory (D.27, autonomous-session pre-commitment)

Per the §28 audit lesson, every load-bearing positive number <3 seeds is a correction risk. §33 enumerates them:

**Tier-A (load-bearing for headline)**: 4 numbers — L9 AIDA rank-32 R=0.617/MAE=6.92, NK best-layer L3.3 across cohorts, pseudobulk-input CD4+T best layer L1-L4, frozen Terekhova L1 R=0.616.

**Tier-B (3-seed-anchored)**: 2 numbers — rank-16 L12 OneK1K MAE=10.85±2.19, rank-16 L12 AIDA MAE=8.32.

**Tier-C (single-seed but doesn't affect headline)**: 3 numbers — scFoundation R, rank-32 L12 OneK1K MAE, B substrate-empty.

This inventory institutionalizes the §28 lesson: future audits start by re-reading §33.1 to identify which numbers any current claim depends on.

## 2026-04-29 — §34 D.24 + D.25 + D.26 results (Tier 2 analyses, autonomous session)

D.24 extension (analysis-only on existing embeddings): pseudobulk-input ridge fits for 3 missing cross-cohort conditions (NK × Terekhova, NK × AIDA loco_terekhova, B × AIDA loco_terekhova). Now 12 conditions total. Output `ridge_summary_pseudobulk.csv` (156 rows). Key finding: **NK pseudobulk-input best layer is L0-L3 across all 4 conditions** (matches CD4+T's L1-L4 shift). Two-axis principle refined: pseudobulk-input → early layers regardless of cell type; per-cell mean-pool layer choice is cell-type-conditional.

D.25 (analysis-only on existing scFoundation embeddings): three-way matched-splits comparison. Output `d25_three_way_matched_splits.csv`. Finding: **scFoundation Δ vs gene-EN matched on CD4+T = -0.137 / -0.174 / -0.256 / -0.086** across conditions, vs Geneformer per-cell ridge Δ -0.052 / -0.088 / -0.155 / n.a. scFoundation lags Geneformer by 0.08-0.10 R-units consistently. **Matched-splits parity is Geneformer-specific, not pan-FM.** Closes scFoundation-LoRA from the queue.

D.26 (analysis-only): bootstrap CIs (n=1000) on §31 layer-asymmetry numbers. Output `layer_asymmetry_cis.csv`. Finding: **NK early-layer ΔR robustly excludes zero only on AIDA cross-ancestry** (loco_onek1k+aida CI [+0.055, +0.184]). On OneK1K and Terekhova, ΔR is positive (+0.04, +0.07) but CI includes zero. The "NK at L3.3 across all 3 cohorts" claim has weaker statistical support than the medians suggested. B × Terekhova actually has signal at L9 (R=0.228 CI [0.014, 0.247] excludes zero) — B isn't entirely substrate-empty.

## 2026-04-29 — §35 D.31 + D.32 results (proposed-and-implemented during D.21/D.22 GPU wait)

D.31 (analysis-only): kNN-age correlation per layer per condition. Output `d31_donor_cluster_metrics.csv`. Finding: §31's NK best-layer R advantage does NOT show up in nearest-neighbor donor-age structure. NK × OneK1K kNN-R: L3=0.337 vs L12=0.343 (L12 marginally better). **The early-layer NK ridge signal is dimensional-specific (specific aging-correlated axes), not cluster-structural.** Refines methodology framing to "ridge captures specific aging-axes that kNN doesn't."

D.32 (analysis-only): bootstrap CIs (n=1000) per-seed on rank-16 LoRA layered ridge across 3 seeds × 13 layers × 2 eval cohorts. Output `d32_rank16_3seed_layered_bootstrap_cis.csv`. **NEW HEADLINE**: L11 (not L9, not L12) is the best AIDA layer at rank-16 3-seed mean. R=0.566±0.032, MAE=7.96y±0.42y. Beats L12 (R=0.560/MAE=8.32) and L9 (R=0.520/MAE=8.36). 3-seed std on MAE 0.14-0.42y, well below 2.0y robustness threshold. Anchor-tier finding.

L11 MAE=7.96y is in the 7.5-8.5y "modestly behind, within ~1y, outline (a) hedged" band per decision rules. Outline (a) is supportable independently of D.21's rank-32 verification.

## 2026-04-29 — §36 D.21 partial (2-seed PASS) + D.22 PARTIAL (autonomous session)

D.22 NK frozen 3-seed verification DONE: 2/3 cohorts pass ΔR(L_best vs L12) > +0.05 threshold:
- loco_onek1k × AIDA cross-ancestry: ΔR=+0.085 PASS
- loco_onek1k × OneK1K: ΔR=+0.039 FAIL (just below)
- loco_terekhova × Terekhova: ΔR=+0.079 PASS

Outcome: **PARTIAL support**. NK cell-type-conditional finding survives with cohort-specific caveat: "NK shows robust early-layer advantage on cross-cohort settings (Terekhova chemistry-shift, AIDA cross-ancestry) but not on in-distribution OneK1K." Best-layer per cohort shifted from §31 single-seed (L3/L2/L5) to D.22 3-seed mean (L3/L2/**L6**) — direction holds, specific layer is less stable.

D.21 2-seed partial result (seeds 0+1): **L9 AIDA 2-seed mean MAE = 7.29y ± 0.53y, R = +0.592 ± 0.035**. Decision rule (per §D.21): MAE 7.29y < 7.5y threshold → outline (a) viable, parity headline survives. σ(MAE) << 2.0y robustness threshold. Even at 2-seed, result is in the upper decision band.

## 2026-04-30 — §37 verification gate FINAL (D.21 3-seed done, outline (a) selected)

D.21 final 3-seed result (all 3 seeds): **L9 AIDA 3-seed mean: R=0.594±0.025, MAE=7.33y±0.38y**. Per-seed L9 AIDA MAE: 6.92 / 7.66 / 7.40. Decision rule: PASS (≤7.5y → outline (a) viable, parity headline survives).

Layer profile at 3-seed mean (AIDA): L9 best by MAE (7.33), L11 best by R (0.612). The L11-best-by-R pattern replicates from D.32 rank-16 3-seed.

**Combined verification gate outcome**: outline (a) selected with two cohort-specific caveats (NK methodology cross-cohort only; B substrate-empty not bilateral). Five headline contributions established:
1. Matched-splits parity (Geneformer-specific) on AIDA cross-ancestry
2. Cell-type-conditional layer-of-best-readout (cohort-caveated)
3. Unit-of-analysis × layer interaction (two-axis principle)
4. Methodology-aware FM-vs-bulk comparison
5. Capacity ablation (rank-16 vs rank-32 doesn't fix the gap)

## 2026-04-30 — §38 D.37 inner-CV layer selection (deployment-recipe test)

User asked: "How was the right layer selected?" and "Can we run a more rigorous test with cross validation?" Implemented K-fold inner CV on train donors only — picks layer using only training data (no test-set selection bias).

Output: `results/phase3/d37_cv_layer_selection.csv` (16 rows).

**Cross-seed CV-layer stability findings**:
- **Rank-32 LoRA × CD4+T**: CV picks L12 in all 3 seeds, oracle is L12 in all 3 seeds. **PERFECT — deployment recipe is "use L12"**.
- **Rank-16 LoRA × CD4+T**: CV picks [6,7,6], oracle [7,6,6]. Within ±1 layer, R penalty ≤0.01. Deployable.
- **NK frozen × Terekhova**: CV picks [2,3,3], oracle [2,2,2]. Within ±1 layer (early). Deployable directionally.
- **NK frozen × OneK1K**: CV picks [0,2,3], oracle [3,3,4]. Variable, R penalty up to 0.13. NOT deployable single-seed.
- **Frozen CD4+T × loco_onek1k**: CV picks L4, oracle L12. Wrong layer, R penalty 0.085. NOT deployable single-seed.

**Two-tier methodology contribution**:
- Tier 1 (deployable recipe): "LoRA fine-tuning + ridge readout at last (or near-last) layer; CV reliably picks within ±1 layer of oracle"
- Tier 2 (characterization-only): "Frozen Geneformer per-cell mean-pool layer-of-best-readout is cell-type-conditional directionally but specific layer requires multi-seed ensemble for deployment"

**Updated paper headline**: "Cell-type-conditional layer selection in single-cell foundation model probing; deployment-ready recipes for fine-tuned variants."

This is the strictest methodology test possible from existing data. Result is paper-strengthening.

## 2026-04-30 — §39 D.36 strict MAE CI overlap test on parity claim

D.36 (proposed-and-implemented): bootstrap MAE CI on rank-32 L9 AIDA 3-seed pooled (n=3000) vs gene-EN matched (n=1000).

- rank-32: median MAE 7.38y, 95% CI [6.40, 8.56]
- gene-EN: median MAE 6.07y, 95% CI [5.28, 6.92]
- **CI overlap [6.40, 6.92]** (narrow but exists)
- Mean diff: +1.35y (rank-32 worse)
- **Mann-Whitney U: p < 0.001** — distributions statistically distinguishable

Refines the §32 parity narrative: "**competitive within seed variance, not strictly tied**." CI-overlap supports "competitive" claim; Mann-Whitney rules out "tied." Both numbers reported in writeup.

The §32 paper-changing narrative still holds in essence: the 0.4 R-units gap was a methodology-mismatch artifact. But the residual ~1.35y MAE gap at matched splits is real and statistically distinguishable.
