# Phase 3 — CD4+ T pilot fine-tune & preprint

## Goal

With baselines in hand, this phase produces the minimal viable result: LoRA fine-tuning of **Geneformer, scFoundation, and scGPT** on the CD4+ T cell LOCO folds, compared head-to-head against the Phase 2 baselines on identical splits. CD4+ T is chosen first because it has the strongest sc-ImmuAging internal R (≈0.91) and the largest per-cohort donor pool, making it the best-powered starting point. This phase also calibrates the true GPU-hours-per-run from measured timing (not the ≈4 h pilot estimate in the proposal), which gates the feasibility of Phase 4's full matrix. The phase ends with a bioRxiv preprint submission covering the CD4+ T LOCO result — the sprint-to-preprint mitigation for scoop risk.

### Model-panel decision (recorded 2026-04-24 after leakage audit)

The Phase 3 pilot launches with **three** models, not the original two, driven by the completed leakage audit (`data/leakage_audit.csv`, `methods/leakage_audit_notes.md`):

| Role | Model | Why |
|---|---|---|
| **Lead (headline)** | **Geneformer** | Only foundation model `clean` across *all* three training cohorts + AIDA. Carries the paper's primary "true-holdout generalization" claim. |
| Secondary (primary-fold contrast) | **scFoundation** | `clean` on OneK1K and Terekhova (the two primary LOCO folds). `overlapping` on Stephenson via HCA-Covid19PBMC ingestion; Stephenson rows carry the caveat footnote. Worth the extra LoRA run because it's the second "true-holdout" comparator on the 981-donor primary fold. |
| Comparator | **scGPT** | `overlapping` on OneK1K and Stephenson (CellxGene Census). `clean` only on Terekhova. Kept in the panel because it is the most-cited single-cell FM and readers expect a scGPT row; all non-Terekhova scGPT rows go into the leakage-flagged analysis pathway. |
| *Deferred to Phase 4* | UCE | Same overlap pattern as scGPT (CellxGene Census); adds no new info relative to scGPT at the Phase 3 pilot. Enters in Phase 4 for the full matrix. |

**Headline primary fold: `loco_terekhova` (166 donors, 10x 5' v2), the only fold that is leakage-`clean` for all four FMs.** `loco_onek1k` is a secondary primary fold — cleanest for Geneformer and scFoundation; scGPT/UCE rows on it are leakage-flagged. This is the inverse of the original plan, which had OneK1K as the main LOCO table.

### Chemistry-robustness framing (recorded 2026-04-24 after Task 1f)

Phase 1 Task 1f (`methods/terekhova_chemistry_shift.md`) established that the pre-trained LASSO's 3'→5' transfer is cell-type-selective: CD4T and CD8T retain R ≈ 0.7–0.8 on Terekhova; MONO/NK/B degrade substantially, with B cells collapsing to R = 0.08. The Phase 3 pilot uses CD4+ T — which is chemistry-robust for LASSO — so the pilot's FM-vs-LASSO contrast on `loco_terekhova` is a *clean* FM benefit test, not a chemistry-rescue test. This is deliberate: the chemistry-rescue story (Can FMs recover where LASSO fails?) is sharper on B and NK cells and is reserved for Phase 4's few-shot curve, where it becomes a secondary headline rather than a pilot distraction.

For CD4+ T itself, the Phase 3 pilot produces a **3-comparator × 3-chemistry-context** subfigure (LASSO-pretrained vs Pasta-REG vs Geneformer-LoRA × {3' OneK1K, 5' Terekhova European, 5' AIDA Asian}) so the preprint headline shows chemistry-generalization AND ancestry-generalization behaviour in a single composite figure. The baselines are already scored for all three contexts (Phase-2 Tasks 2.1, 2.4, 2.6); only the FM-on-5' and FM-on-AIDA numbers are new.

### Tri-headline preprint structure (revised 2026-04-25 after Phase-2 add-ons)

With Phase 2 closed (75-row `results/baselines/loco_baseline_table.csv` covering 4 baselines × 3 training cohorts × 5 cell types + AIDA × 3 baselines × 5 cell types), the preprint headline expands from one cell to **three CD4+ T headline cells**, each with explicit win/match/loss criteria computed against the **per-cell minimum of FOUR baselines: {LASSO-pretrained, LASSO-retrained-3cohort, scAgeClock, Pasta-REG}**:

| Headline cell | Best baseline / MAE | 10%-win FM target | Story if FM wins |
|---|---|---|---|
| **OneK1K CD4+T** (981 donors, 10x 3') | LASSO-pretrained 9.4y | ≤8.5y | "FMs match-or-beat the published 5-cohort LASSO at 3-cohort training" |
| **Terekhova CD4+T** (166 donors, 10x 5') | Pasta-REG 8.0y | ≤7.2y | "FMs beat the chemistry-invariant rank-norm bulk model under chemistry shift" |
| **AIDA CD4+T** (595 donors, 10x 5' v2 + Asian ancestry) | Pasta-REG **6.3y** | **≤5.7y** ★ | "FMs beat the strongest published baseline on the cross-ancestry headline cell" |

★ AIDA CD4+T's Pasta floor (6.3y) is the lowest MAE in the entire 75-row baseline matrix; this is the toughest cell. AIDA is leakage-`clean` for all 4 FMs (Phase-1 audit) AND for scAgeClock (Phase-2 Task 2.2 audit) AND for Pasta (bulk pretraining), making it the only cell where every comparator + every FM is strict-clean simultaneously.

The Terekhova bar is materially tighter than the original "beat LASSO 9.2y" gate. The AIDA bar is tighter still. Pre-Phase-2 estimate of beating-best-baseline by 10% on Terekhova CD4T was ~80%; revised post-Phase-2 estimate is **~50–60% on Terekhova** and **~30–40% on AIDA** (Pasta-REG is rank-norm-invariant + ancestry-invariant in a way that's hard to beat).

**Headline classification rules (per cell):**

- *Win*: FM clears the 10% margin against the per-cell best baseline. Cell-level claim: "FMs outperform the strongest published baseline."
- *Match*: FM matches the best baseline within ±5%. Cell-level claim: "FMs match Pasta-REG (or LASSO-pretrained) at lower training cost on three cohorts."
- *Loss*: FM loses to the best baseline. Cell-level: null finding for that cell.

**Aggregate preprint outcomes** based on count of cell-level wins across the three headline cells:

- **3/3 wins** → headline: "FMs outperform the strongest published baselines across chemistry shift and ancestry shift on the CD4+T LOCO-pilot."
- **2/3 wins** → headline: "FMs outperform on [winning cells]; [losing cell] is a Pasta-tie/loss" (still a strong primary result).
- **1/3 wins** → degraded claim: "FMs match-or-beat published baselines on [winning cell]; deeper investigation needed for ancestry/chemistry generalization."
- **0/3 wins** → pivot to evaluation-study framing: "Do scRNA-seq FMs improve on rank-normalized bulk-transcriptomic clocks for PBMC age prediction?" with the null finding as the headline.

The 4th comparator (LASSO-retrained-3cohort) is the **methodologically symmetric apples-to-apples baseline** to the FM fine-tunes — both see the same 3 cohorts. Including it in the gate directly addresses the "FM win is just from more training data" reviewer concern. Phase-2 data shows LASSO-retrained ≈ LASSO-pretrained for CD4+T/CD8+T, so the per-cell minimum bar is unchanged for those cells, but explicitly listing both LASSOs in the comparator panel pre-empts the criticism.

### Phase-3-A status (updated 2026-04-28: ridge readout converts 0/3 → 0+2+1 at 3-seed mean; methodology contribution is the headline)

**UPDATE 2026-04-28** (memo §27 + §28).
Variant 3 follow-up extracted per-layer mean-pool embeddings from `loco_onek1k_seedX_CD4p_T_e5b.pt` (seeds 0/1/2) and `loco_terekhova_seed0_CD4p_T_e5b.pt`, refit ridge per layer, ran 3-seed variance check. **Revised tri-headline at 3-seed mean**:
- **OneK1K CD4+T**: ridge L6 R=0.632 ± 0.008, MAE=10.85 ± 2.19y vs LASSO 9.45/0.747 → **CLOSE-MATCH (+15%)**. Single-seed best (seed 0 L12 = 8.21y) cleared the strict ≤8.5y WIN bar but does not generalize.
- **Terekhova CD4+T**: ridge L1 single-seed = R=0.619 / MAE=8.63 vs Pasta 8.04/0.778 → **MATCH** (+7.4%, awaits 3-seed validation).
- **AIDA CD4+T**: ridge L11 3-seed mean R=0.566 ± 0.032, MAE=7.96 ± 0.42y vs Pasta 6.32/0.659 → **close-loss** (+25.9%, most reproducible cross-ancestry FM result of the phase).

Aggregate revised: **0 strict WINs + 2 MATCH-class + 1 close-loss** at 3-seed mean (better than the §22.3 0/3 horse-race tally; weaker than the optimistic single-seed §27 1+1+1 read).

The "horse-race loss" verdict from §22.4 was substantially attributable to the per-cell MSE linear head readout: same backbone + same LoRA weights + per-donor ridge readout reduces median MAE by 1.7–9.2y across CD4+T conditions. **The publishable methodology contribution is "per-cell MSE head systematically underestimates donor-level signal in fine-tuned single-cell FMs; per-donor ridge is strictly better"** — independent of whether strict WIN bars are cleared. B and NK do NOT rescue under ridge readout (B remains representation-empty across layers × seeds; NK gets partial improvement, no win).

### Phase-3-A status (recorded 2026-04-26, superseded above)

Phase-3-A smoke (single Geneformer LoRA fine-tune on `loco_onek1k` × CD4+T × seed 0) is in progress; **GATE 2 not yet cleared**. Investigation chain so far (full diagnosis in `notes/phase3_geneformer_convergence.md`; chronological summary in `notes/research_journal.md` 2026-04-26 entries):

| Run | Train cells | Epochs | LR (backbone / head) | Wall | MAE | R | Status |
|---|---|---|---|---|---|---|---|
| #1 (`*.headbug-failed`) | 19,000 | 1 | 5e-5 / 1e-3 | 9.6 h | 30.5 y | NaN | Head bias zero-init → divergence; archived |
| #2 | 9,500 | 1 | 5e-5 / 1e-3 | 3.1 h | 19.99 y | 0.33 | Bias init = mean train age (48.93); train MSE plateau ≈ 270 — underfit |
| Intermediate v2 | 9,500 | 1 | 2e-4 / 2e-4 | 4.5 h | 18.48 y | 0.32 | 4× LR; LoRA delta grew 3.2× but R unchanged at 981-donor scale — bias drift, not signal |

State-dict + per-donor analysis on Runs #2 and v2 reveals both runs are sitting on the "predict mean(train)" minimum of MSE: prediction sd is 0.9–1.1y while true age range spans 78y. The 1.5y MAE improvement in v2 is bias-drift, not age-axis-signal. Phase-2's *linear* LASSO hits MAE 9.4y on the same train data — a 110M-param frozen FM stuck at MAE 18–20y points to **severe undertraining + likely cls-pooling weakness** as the binding constraints, not LR. Next experiments planned in the convergence memo: E5a mean-pool ablation (5-min code change), E5b 3-epoch rerun, E5c 10× cells/donor.

## Success criteria

- Geneformer, scFoundation, and scGPT LoRA (rank-16, attention + MLP layers, peft library) fine-tuned on all CD4+ T LOCO training folds; per-fold LOCO m.a.e. and Pearson R recorded for all three models and compared against the **Phase 2 panel (LASSO-pretrained + LASSO-retrained-3cohort + scAgeClock + Pasta-REG)** numbers on the same folds. Result rows carry the three Phase-4 stratification columns (`leakage_status` from `data/leakage_audit.csv`, `chemistry_match_to_baseline_training`, `detectability_flag`).
- **Beat-best-baseline gate evaluated** with the **per-cell min-of-4** rule: for each (cohort, cell type) cell, compute relative MAE reduction of each FM vs. the *minimum* of {LASSO-pretrained, LASSO-retrained-3cohort, scAgeClock, Pasta-REG} MAEs on that cell. Headline win on a cell requires the relative reduction to be ≥10% against this minimum-MAE baseline. The per-cell win/match/loss classification rule (above) is recorded for each (cohort, cell type) cell and aggregated across the three headline cells (OneK1K, Terekhova, AIDA) for the preprint's primary table. Including LASSO-retrained-3cohort as the 4th comparator addresses the "FM win is just from more training data" concern: the retrained LASSO sees the same 3-cohort corpus as the FMs.
- **AIDA scoring (added 2026-04-25):** every fine-tuned FM checkpoint produced in this phase is also scored on AIDA CD4+T (595 donors, 10x 5' v2, Asian ancestry) — one extra inference call per fine-tune, no additional training. AIDA is held back from training but scored at evaluation time to give the preprint a cross-ancestry headline cell *before* Phase 4 begins. AIDA × FM rows are leakage-`clean` for all 4 FMs (Phase-1 audit) so they enter the strict-clean meta-analysis directly. **AIDA pipeline note:** the trained checkpoint from each LOCO fold is applied to AIDA without retraining; AIDA donor splits from `data/aida_split.json` are honored (the ancestry_shift_mae half is used here; the age_axis_alignment half is reserved for Phase 5).
- **Chemistry+ancestry-robustness subfigure produced for CD4+ T**: 3-comparator × 3-chemistry-context panel showing Pearson R for {LASSO-pretrained, Pasta-REG, Geneformer LoRA} × {3' OneK1K, 5' Terekhova European, 5' AIDA Asian}. The 3×3 panel simultaneously tests chemistry robustness (3'→5' transfer) AND ancestry robustness (European→Asian transfer) in one figure. Pasta-REG is the chemistry-invariant + ancestry-invariant baseline (Phase-2 Pasta-B R=0.28 on Terekhova vs LASSO R=0.08; Pasta CD4T R=0.66 on AIDA vs LASSO R=0.65). The subfigure asks: "does single-cell FM pretraining add chemistry-invariance + ancestry-invariance *on top of* what rank-normalized bulk modelling already provides?" Committed as `results/phase3/cd4t_robustness_3x3.csv` + `fig_cd4t_robustness_3x3.pdf`. Headline element of the preprint.
- Measured GPU-hours per LoRA run (mean and SD across three random seeds) recorded in `compute/runtime_log.csv`; Phase 4 full-matrix time estimate updated from this measurement before Phase 4 begins.
- m.a.e.-detectability floor reviewed using CD4+ T pilot data: compute paired Wilcoxon power from observed per-donor absolute residuals (empirical baseline-vs-FM ρ measured from CD4+ T pilot). The preprint methods reports **all three ρ values bracketing the truth**: (i) Phase-1 planning ρ=0.8 (overoptimistic, n_required=132–229); (ii) Phase-2 empirical baseline-pair ρ=0.06–0.35 (conservative lower bound, n_required=502–1,075); (iii) the Phase-3-measured baseline-vs-FM ρ which is expected to fall between these. The post_phase3 ρ value drives the per-cell-type primary/exploratory flags via a `post_phase3_override` block in `data/loco_folds.json` (preserves Phase-1 planning flags as baseline; supersedes them for Phase 4 reporting).
- bioRxiv preprint posted within 10 weeks of project start; manuscript's headline figure is **the CD4+ T tri-cell result** (OneK1K + Terekhova + AIDA) for Geneformer + scFoundation + scGPT vs. **the per-cell minimum-MAE of {LASSO-pretrained, LASSO-retrained-3cohort, scAgeClock, Pasta-REG}**, with the full 20-row leakage-audit table (5 models × 4 cohorts), the frozen split design, the 3×3 chemistry+ancestry robustness subfigure, and the bracketed detectability-ρ disclosure. At this point the paper is "minimal viable" regardless of whether criterion (a) is met on all three headline cells.

## Tasks

- [ ] Task: Implement LoRA fine-tuning wrapper for Geneformer, scFoundation, and scGPT. Using peft (LoRA rank-16 applied to attention + MLP projection layers), build training scripts that accept a cell-type label, a LOCO fold index, and a random seed; output a model checkpoint and per-donor age predictions on the held-out cohort. Run three seeds per (model, fold) combination for CD4+ T cells on the **two primary LOCO folds** (`loco_onek1k`, `loco_terekhova`); Stephenson LOCO is exploratory-only and can be deferred. Record wall-clock time and peak GPU memory per run. Done when reproducible per-fold predictions exist for CD4+ T × {Geneformer, scFoundation, scGPT} × {loco_onek1k, loco_terekhova} × 3 seeds.

- [ ] Task: Score every fine-tuned FM checkpoint on AIDA CD4+T. For each (model, LOCO-fold, seed) checkpoint produced by the previous task, run an additional inference pass on `data/cohorts/aida_eval/CD4p_T.h5ad` filtered to the `ancestry_shift_mae` donor half from `data/aida_split.json` (307 donors). Append per-donor predictions to `results/phase3/cd4t_aida_predictions.csv` and per-(model, source-fold, seed) summary rows to `results/phase3/cd4t_aida_summary.csv`. The AIDA evaluation has no extra training cost — it's just inference. AIDA is leakage-clean for all 4 FMs + scAgeClock + Pasta, so its rows enter the strict-clean meta-analysis directly. Done when AIDA CD4+T predictions exist for all 18 (3 FMs × 2 source folds × 3 seeds) checkpoints.

- [ ] Task: Evaluate CD4+ T pilot results and update detectability floor. Compute per-fold LOCO m.a.e. (median absolute error) and Pearson R for each of the three FMs **on each of the three headline cells** (OneK1K, Terekhova, AIDA). Compare against the **Phase 2 panel (LASSO-pretrained + LASSO-retrained-3cohort + scAgeClock + Pasta-REG) on the same cells**. For each cell, compute relative MAE reduction of each FM vs. the minimum of the four baseline MAEs (the "beat-best-baseline" min-of-4 gate); classify the cell win/match/loss; aggregate to a 3-cell tri-headline outcome. Compute the *empirical* per-donor absolute-error correlation ρ between each baseline and each fine-tuned model (the Phase 1 floor assumed ρ=0.8; Phase 2 measured baseline-pair ρ ≈ 0.06–0.35; Phase 3 measures the actual baseline-vs-FM value). Run paired Wilcoxon signed-rank test per cell; append a `post_phase3_override` block to `data/loco_folds.json` with the updated primary/exploratory flags implied by the empirical baseline-vs-FM ρ (do NOT overwrite the frozen Phase-1 flags). Write results to `results/phase3/cd4t_loco_table.csv` with `leakage_status`, `chemistry_match_to_baseline_training`, `vs_best_baseline_pct`, and `headline_outcome ∈ {win, match, loss}` columns per (model, cell). Done when the table is committed, the `post_phase3_override` block is populated, and the methods section reports the bracketed ρ disclosure (Phase-1 planning ρ=0.8, Phase-2 baseline-pair ρ=0.06–0.35, Phase-3 measured ρ).

- [ ] Task: Produce CD4+T chemistry+ancestry-robustness 3×3 subfigure. Assemble the 3-comparator × 3-chemistry-context panel (LASSO-pretrained vs Pasta-REG vs Geneformer LoRA × {3' OneK1K, 5' Terekhova European, 5' AIDA Asian}), reusing the already-scored LASSO and Pasta numbers in `results/baselines/loco_baseline_table.csv` plus the new FM-on-AIDA scores from the previous task. **Pasta-REG is the chemistry-invariant + ancestry-invariant comparator** (Phase-2: Pasta-B R=0.28 on Terekhova vs LASSO R=0.08; Pasta CD4T R=0.66 on AIDA vs LASSO R=0.65). The 3×3 figure simultaneously tests two robustness dimensions in one panel: "does single-cell FM pretraining add chemistry-invariance + ancestry-invariance *on top of* what rank-normalized bulk modelling already provides?" Write `results/phase3/cd4t_robustness_3x3.csv` and render `fig_cd4t_robustness_3x3.pdf`. Done when the PDF is committed and cited in the preprint headline figure list.

- [ ] Task: Prepare and submit bioRxiv preprint. Draft manuscript sections covering: (1) the LOCO split design and the full 20-row leakage audit (5 models × 4 cohorts; highlight the HCA/ArrayExpress mirror lesson — scFoundation × Stephenson was missed by direct-accession search and recovered via HCA Project ID cross-reference); (2) the **CD4+T tri-headline result** with per-cell win/match/loss classification — OneK1K (3-cohort vs 5-cohort training-data control), Terekhova (chemistry-shift headline), AIDA (cross-ancestry headline) — vs. the **min-of-4 baseline panel** (LASSO-pretrained, LASSO-retrained-3cohort, scAgeClock, Pasta-REG); (3) the 3×3 chemistry+ancestry robustness subfigure; (4) the bracketed detectability-ρ disclosure (Phase-1 ρ=0.8 → Phase-2 ρ=0.06–0.35 → Phase-3 measured); (5) GPU compute envelope and runtime calibration. Include the frozen split files (including the `post_phase3_override` block), the AIDA `ancestry_shift_mae` donor half used here, and checkpoint hashes as supplementary data. Submit preprint to bioRxiv. Done when the bioRxiv DOI is confirmed (within 10 weeks of project start).

## Phase-3-A B-cell extension (added 2026-04-28 after CD4+T close-out)

**Why pull B-cell forward from Phase 4.** The Phase-3-A CD4+T tri-headline closed 0/3 wins (memo §20–§21). The "FMs do not match published baselines" claim is bounded to the cell with the *strongest* baselines. B-cell is the inverse case: Pasta-REG R=0.28 (Terekhova) and 0.26 (AIDA) are very weak; LASSO collapses on Terekhova B (R=0.08, the chemistry-rescue cell originally framed as Phase 4's secondary headline). If the Phase-3-A negative result on CD4+T is to stand as a bounded null, B-cell numbers in the same window are required so the negative claim is "0/N across cells with strong AND weak baselines," not "0/3 on the cell with strongest baselines only." Conversely if FMs *win* on B-cell chemistry-rescue, that single cell flips the preprint headline from null to "FMs rescue chemistry-shift collapse where rank-norm bulk is the only competitive baseline" — exactly the kickoff §3 hypothesis.

### Per-cell baseline floor (from `results/baselines/loco_baseline_table.csv`)

| B-cell fold | Best baseline / MAE / R | Min-of-4 MAE | FM target (10% win) |
|---|---|---|---|
| OneK1K | LASSO 10.66y / R=0.531 | 10.66y | ≤ 9.59y |
| Terekhova | Pasta-REG 10.86y / R=0.281 | 10.86y | ≤ 9.78y |
| AIDA | scAgeClock 9.10y / R=0.218 (best MAE); Pasta R=0.265 (best R) | 9.10y | ≤ 8.19y |

### Success criteria

- Geneformer LoRA at the **Phase-3-A E5b config** (3 epochs, `--max-cells-per-donor 50`, `--pool mean`, `--lr 2e-4 --head-lr 2e-4`, seed 0) trained on `loco_onek1k` × B and `loco_terekhova` × B. Per-fold LOCO m.a.e. and Pearson R recorded in `results/baselines/fm_finetuned/geneformer/summary.csv`.
- AIDA × B scoring of both checkpoints via `scripts/score_aida.py` against `data/cohorts/aida_eval/B.h5ad`, restricted to `ancestry_shift_mae` donors.
- Per-cell win/match/loss classification appended to the tri-cell-tri-headline result. **Win threshold**: relative MAE reduction ≥ 10% vs the per-cell min-of-4 baseline; OR R clearly exceeding the best baseline R by > 0.10 (single-seed for now; multi-seed deferred to Phase-3-B).

### Tasks

- [ ] Task B.1: Train Geneformer LoRA at E5b config on `loco_onek1k` × B × seed 0. Train cohorts: Stephenson + Terekhova (~190 donors × 50 cells = 9,500 train cells). Eval: OneK1K B (~981 donors). Expected wall ~1.4h on A10g, $1.4 on-demand.
- [ ] Task B.2: Train Geneformer LoRA at E5b config on `loco_terekhova` × B × seed 0. Train cohorts: OneK1K + Stephenson (~1,005 donors × 50 cells = 50,250 train cells). Eval: Terekhova B (~166 donors). Expected wall ~6h on A10g, $6 on-demand. **This is the chemistry-rescue cell** — Pasta R=0.28 vs LASSO R=0.08; sharpest test of "do FMs add chemistry-invariance on top of rank-norm bulk?"
- [ ] Task B.3: AIDA-score both checkpoints on AIDA × B (293 ancestry_shift_mae donors). Inference only, ~10 min each. Done when both rows appear in `results/phase3/aida_summary.csv`.
- [ ] Task B.4: Tri-headline classification updated to include B-cell rows; documented in `notes/phase3_geneformer_convergence.md` § (next available); committed with seed-0 caveat.

## Phase-3-A NK-cell extension (added 2026-04-28 after CD4+T close-out)

**Why pull NK-cell forward.** Same rationale as B: NK has weaker baselines than CD4+T and was originally part of the kickoff few-shot crossover panel (B + NK). Phase-1 chemistry-shift findings showed NK degrades on 5' chemistry (LASSO R=0.44, vs CD4T R=0.82) — a more moderate degradation than B's collapse but still a clearer FM-vs-baseline contrast than CD4+T.

### Per-cell baseline floor

| NK-cell fold | Best baseline / MAE / R | Min-of-4 MAE | FM target (10% win) |
|---|---|---|---|
| OneK1K | LASSO 9.64y / R=0.629 | 9.64y | ≤ 8.68y |
| Terekhova | LASSO 12.48y / R=0.440 | 12.48y | ≤ 11.23y |
| AIDA | scAgeClock 8.49y / R=0.204 (best MAE); Pasta R=0.258 (best R) | 8.49y | ≤ 7.64y |

### Success criteria

Same shape as B-cell extension: 2 LoRA fine-tunes + 2 AIDA inferences, win/match/loss classification per cell, single-seed for Phase-3-A scope.

### Tasks

- [ ] Task NK.1: Train Geneformer LoRA at E5b config on `loco_onek1k` × NK × seed 0. Expected wall ~1.4h, $1.4.
- [ ] Task NK.2: Train Geneformer LoRA at E5b config on `loco_terekhova` × NK × seed 0. Expected wall ~6h, $6.
- [ ] Task NK.3: AIDA-score both checkpoints on AIDA × NK (~293 donors). Inference only.
- [ ] Task NK.4: Tri-headline classification updated to include NK-cell rows.

### Combined budget for B + NK extension

4 fine-tunes (~14.8h GPU, ~$15) + 4 AIDA inferences (~30 min, ~$0.5) = **~$15.5 / ~15.4h wall sequential** on g5.xlarge A10g. Total Phase-3-A spend with B + NK additions ≈ $36–37 (CD4+T spend was ~$21).

## Phase-3-A protocol-and-FM-class diagnostic (added 2026-04-28 after B+NK extension partial results)

**Why pivot from per-cell fine-tuning.** B × loco_onek1k and NK × loco_onek1k completed mid-extension and produced more catastrophic losses than CD4+T (B R=−0.076 anti-correlation, NK R=0.165). Two scratchpad reviews (`scratchpad/pseudobulk_review.md`, `scratchpad/geneformer_review.md`) argue the per-cell fine-tune protocol confounds four variables — model class, feature space, unit of analysis, training objective — and the negative result so far cannot distinguish among them. The reviews propose a diagnostic ladder that disentangles these in increasing rigor.

The full B + NK extension was originally 4 runs; **NK × loco_terekhova is cancelled** in favor of the diagnostic ladder. The compute saved (~$6, ~6h) plus diagnostic compute (~$3 for Variant 1) yields decisively more information per dollar.

### Diagnostic ladder

#### Variant 1 — mean-pool existing embeddings, fit ridge at donor level (cheapest, ~$3 / half-day)

Reuse all existing Geneformer checkpoints. For each (checkpoint × eval cohort × cell type):
1. Run inference, extract per-cell embeddings at the regression-head input layer (mean-pooled `last_hidden_state`).
2. Average across cells per donor → 768-dim vector per donor.
3. Fit ridge regression of donor age on these per-donor mean embeddings (nested 3-fold CV on training cohorts).
4. Evaluate Pearson R + MAE on holdout.

Run **two versions** per cell × fold:
- *Frozen-base mean-pool*: use the unmodified Geneformer V2 backbone (skip LoRA load). Tells us what the FM's pretrained embeddings encode.
- *Fine-tuned mean-pool*: load each LoRA + head checkpoint, take embeddings at the same layer. Tells us whether fine-tuning preserved or destroyed age-relevant structure.

#### Variant 2 — pseudobulk-input fine-tune (medium, ~$15 / 1 day), conditional on Variant 1 mid-positive

Compute per-donor pseudobulk (log1p mean expression) for each (cohort × cell type), tokenize as Geneformer rank-encoding, fine-tune LoRA at per-donor MSE on age. Tests whether matching input tokenization to the donor unit closes the residual gap. Watch for tokenization-OOD (pseudobulk rank distribution may differ from single-cell pretraining distribution) as a side finding.

#### Variant 3 — layer-wise frozen probe (~$8 / 1 day), conditional on Variant 1 result

Frozen Geneformer base, no fine-tuning. Pass each donor's pseudobulk through the model, extract activations at every transformer layer (1–12). Mean-pool per layer per donor; fit ridge per layer. Plot held-out R vs layer depth. Three diagnostic outcomes:
- Monotonic-with-depth: standard pattern; fine-tune from last layer.
- Peak at middle layers: age signal compressed out by deeper cell-type-specialization layers; fine-tune from middle.
- Flat / near-zero at all layers: pretrained representation does not encode age-relevant structure.

Most reviewer-defensible diagnostic.

### Decision tree from Variant 1 result

- **Variant 1 frozen-base ridge R ≥ 0.65 on AIDA CD4+T**: protocol diagnosis settled. Per-cell head was destroying signal that frozen embeddings actually contained. Run Variant 3 to localize where the signal lives, write up. Skip Variant 2.
- **Variant 1 R = 0.45–0.60**: mixed; run Variant 2 (pseudobulk-input fine-tune) and Variant 3 (layer-wise probe) in parallel.
- **Variant 1 R ≤ 0.40 even from frozen base**: signal genuinely absent in pretrained representation. Run Variant 3 first to bracket the negative, then pivot to FM-class diagnostic (below).

### Variant 1 results landed (2026-04-28) — three audit follow-ups required

Phase 1 (CD4+T, memo §23) and Phase 2 (B + NK, memo §24) ran. Results:
- CD4+T frozen-base R = 0.527–0.576 across 3 cohort/eval conditions → falls in the 0.45–0.60 *mixed* bracket.
- B and NK frozen-base R in [−0.013, 0.260], with 3 of 6 conditions p > 0.10 → *cell-type-dependent failure mode*: CD4+T is protocol-negative (signal exists, fine-tune destroys it); B and NK look representation-negative.

`scratchpad/variant_1_review.md` flags **three weaknesses in the §23/§24 framing that must be audited before invoking Variant 2 or 3**:

1. **B/NK statistical claims are imprecise.** "3 of 6 conditions p > 0.1" doesn't characterize the other 3 — NK × OneK1K (R=0.260, p=1.2e-16) and NK × Terekhova (R=0.199, p=0.010) are *real but weak* signal, not absent. The "substrate contains no signal" wording overreaches; the correct claim is "substrate contains weak signal, materially less than LASSO extracts." Need exact R, 95% bootstrap CI, p for all 6 B/NK conditions before settling the framing.
2. **CD4+T frozen-base does NOT match strong baselines.** Frozen R=0.527–0.576 vs LASSO 0.747 / Pasta 0.778 is a real gap. The §23 "smoking gun" framing accidentally suggested otherwise. Story should be **"frozen-base recovers what fine-tuning destroys"**, not "frozen-base matches the strong baseline."
3. **AIDA cross-ancestry result needs a bias-variance audit.** R=0.527 on AIDA from a ridge trained on European cohorts is the most striking number in the table. Phase-3-A AIDA scoring already documented that fine-tune predictions get pulled toward the European training mean. A frozen-base ridge has a different bias profile, and the "ancestry generalization" might partly be that frozen-base predictions are compressed toward the global mean and get credit on AIDA's age distribution. Need pred_sd vs eval_sd, pred range vs eval range, before relying on the cross-ancestry headline.

### Tasks (audit follow-ups before Variant 2/3)

- [ ] Task D.9 — **Statistical characterization of all B/NK frozen-base conditions.** Re-load the 6 (cell × eval cohort) ridge predictions, compute exact R, 95% bootstrap CI (n=1000 resamples), p-value, and pred_sd vs eval_sd. Append a table to memo §24 distinguishing "weak signal" (e.g. NK × OneK1K R=0.260) from "absent signal" (e.g. B × OneK1K R=−0.013). Update §24.2 wording from "substrate is empty" to the more accurate "weak relative to LASSO." ~1h.
- [ ] Task D.10 — **AIDA bias-variance audit.** For each of the three AIDA frozen-base ridge predictions (CD4+T, B, NK), compute (i) pred_sd vs AIDA-eval_sd, (ii) pred range vs eval range, (iii) regress pred on true age and report slope (slope < 1 = compression toward mean), (iv) compare AIDA pred mean to OneK1K train age mean (gap ≈ 0 = bias-toward-train-mean explanation). If CD4+T × AIDA pred slope is ≪ 1 and pred sd ≪ eval sd, the R=0.527 is a compression artifact and the cross-ancestry claim must be retracted. Append findings to memo §23 + §24. ~1h analysis-only.
- [ ] Task D.11 — **Reframe §23/§24 narrative based on D.9 + D.10.** Update the convergence memo's claim hierarchy: replace any "frozen-base matches" or "FM signal absent" wording with calibrated equivalents. Add a comparison panel (frozen-base R vs LASSO R vs Pasta R) per cell type. Outcome: an honestly-bounded claim ladder that survives a hostile review. ~1h doc-only.

These three audit tasks are blocking for Variants 2 and 3 because:
- The Variant 3 (layer-wise probe) interpretation depends on whether B/NK is "no signal" or "weak signal" — the layer plot looks for a peak above frozen-mean-pool baseline, and the baseline value matters.
- The Variant 2 (pseudobulk fine-tune) outcome interpretation depends on whether the CD4+T target is "match frozen R=0.55" or "match LASSO R=0.75" — different bars motivate different follow-ups.

If D.10 finds the AIDA R=0.527 *is* a compression artifact, the §22.5 decision-tree branch (which placed AIDA in the [0.45, 0.60) "mixed" bracket) re-evaluates downward, possibly into the "Variant 3 first, FM-class diagnostic next" bucket — which would re-prioritize scFoundation/scGPT (D.7, D.8) ahead of Variant 2 (D.5).

### Honestly-bounded claim ladder (post-D.9–D.11 target)

Per critique #2 of `scratchpad/variant_1_review.md`, the Variant 1 result is a **diagnostic finding, not a competitive recipe**. The reframed claim ladder D.11 lands should look like:

| Tier | Claim | Evidence required |
|---|---|---|
| Strongest | "Per-cell MSE LoRA fine-tuning destroys CD4+T age signal that the same Geneformer encoder already encodes." | Frozen R=0.527–0.576 vs fine-tune R=0.140–0.466; +0.436 R Terekhova uplift; D.9 sanity-check on CIs. |
| Medium | "Geneformer's pretrained mean-pool representation captures CD4+T age signal *partially* — substantially below LASSO/Pasta on raw genes (0.527 vs 0.747–0.778)." | Side-by-side panel of frozen R vs LASSO R vs Pasta R per cell type. |
| Weakest (conditional on D.10) | "Frozen-base ridge generalizes to AIDA at R=0.527 / MAE 11.76 — *if* predictions span the AIDA age range and pred_sd ≈ eval_sd." | D.10 bias-variance audit. If pred sd is compressed, retract to "AIDA MAE comparable to OneK1K but R inflated by central-tendency bias." |
| Bounded null | "B and NK frozen-base ridge extracts measurably weak signal (NK × OneK1K R=0.260, p<1e-15) but materially less than LASSO on raw genes (LASSO 0.531–0.629). The representation gap is upstream of the encoder mean-pool, not protocol-induced." | D.9 exact R + 95% CI per condition; LASSO comparison row. |

### FM-class diagnostic (parallel or after Variant 1)

If Variant 1 indicates Geneformer-specific weakness (or even if not, for completeness): build LoRA wrappers for scFoundation and scGPT, run a 3×3 matrix on CD4+T × loco_onek1k:
- **3 FMs**: Geneformer (done), scFoundation, scGPT. Skip UCE (contrastive cell-identity objective likely opposite of what aging needs).
- **3 protocols**: per-cell fine-tune (E5b config), Variant 1 frozen-mean-pool ridge, Variant 1 fine-tuned-mean-pool ridge.

Outcome interpretation (per `scratchpad/geneformer_review.md`):
- scFoundation per-cell ≫ Geneformer per-cell: rank-vs-magnitude objective is the bottleneck. scFoundation's read-depth-conditional reconstruction preserves quantitative magnitude; switch to it.
- All FMs fail per-cell, all match gene-EN at frozen-mean-pool: unit-of-analysis dominates; FM class is irrelevant in this set.
- All FMs fail at all protocols: structural negative — current single-cell FMs as a class don't encode salient adult-PBMC age signal at any reasonable readout.

Pragmatic note: scFoundation × Stephenson is `overlapping` per leakage audit (HCA-Covid19PBMC ingestion). For diagnostic phase only, accept asymmetric cohort coverage and flag as methods limitation.

### Tasks

- [x] Task D.1 (done 2026-04-28): Build `scripts/extract_embeddings.py` — takes Geneformer checkpoint (LoRA + head, or `--frozen-base` flag), cohort, cell type; runs inference; writes per-donor mean-pool 768-dim vectors to `results/phase3/embeddings/{cohort}_{cell_type}_{run_tag}.npz`.
- [x] Task D.2 (done 2026-04-28): Build `scripts/donor_ridge.py` — `.npz` → ridge (nested 3-fold CV alpha selection) → Pearson R + MAE on holdout, optional `--also-eval-aida`. Writes to `results/phase3/ridge_summary.csv`.
- [x] Task D.3 (done 2026-04-28, **frozen-base only**): Variant 1 Phase 1 (CD4+T) + Phase 2 (B + NK) frozen-base across 4 cohorts × {CD4+T, B, NK} + ridge fits on both LOCO folds with AIDA transfer. 12 ridge rows in `results/phase3/ridge_summary.csv`. **Fine-tuned-mean-pool extraction (originally part of D.3) is deferred** — the audit tasks D.9–D.11 must land first to confirm the bias-variance picture before paying the ~$3 to extract from each fine-tuned checkpoint.
- [x] Task D.4 (done 2026-04-28): Tabulated in memo §23 (CD4+T) and §24 (B+NK). Branch landed in [0.45, 0.60) mixed bracket → Variants 2 + 3 in parallel — *but contingent on D.9–D.11 audit outcomes*.
- [~] Task D.5 (**SKIPPED 2026-04-29**, see §29 pivot): Variant 2 (pseudobulk fine-tune) was originally elevated by §28.6 but is subsumed by §27's ridge-readout finding — per-donor ridge on the fine-tuned representation already enforces a donor-level objective post-hoc. Pseudobulk would force ~1 example/donor at training time, which likely *worsens* the §28 seed-variance problem (std=3.38y on the 8.5y bar) rather than fixing it. Compute redirected to D.7.
- [x] Task D.6 (done 2026-04-28): Variant 3 — `scripts/extract_embeddings_layered.py` + `donor_ridge_layered.py` + `donor_ridge_layered_finetune.py` + `donor_ridge_layered_post_finetune.py`. Outputs in `results/phase3/ridge_summary_layered*.csv`. See memo §26 / §27 / §28.
- [x] Task D.7 (done 2026-04-29, see §29.3): scFoundation FM-class diagnostic at frozen-base + per-donor ridge readout. **Result: 0/6 is FM-class, NOT Geneformer-specific.** scFoundation 3B params at canonical pool='all' (3072-d) loses to LASSO/Pasta on every CD4+T condition (MAE 30–100% above bulk baselines), and is *worse than Geneformer §28 3-seed mean* on CD4+T × OneK1K (R=0.475 vs 0.632, MAE=12.79 vs 10.85). B substrate empty across both FMs (R near zero, CIs cross zero); NK weak across both. Fine-tune + ridge readout (Geneformer §27/§28) is the recipe contribution; frozen FMs of any tested scale lose 30–100% to bulk baselines on donor-level age. Output: `results/phase3/ridge_summary_scfoundation.csv` (9 rows), embeddings in `results/phase3/embeddings_scfoundation/`. ~$2.5 compute, ~2.5h wall.
- [~] Task D.8 (deprioritized): scGPT LoRA wrapper. Run only if scFoundation diagnostic returns an interesting (positive or negative) signal that benefits from triangulation.

### Phase-3-B follow-ups (post-§29 review, 2026-04-29)

- [x] Task D.12 (done 2026-04-29, **NEGATIVE → PIVOT**, see §30): Geneformer rank-32 LoRA, 1 seed, CD4+T × loco_onek1k. Result: L12 OneK1K MAE=11.00 / R=0.636 vs rank-16 seed-0 MAE=8.21 / R=0.631. **Rank-32 single-seed lands on rank-16 3-seed mean (11.13 ± 3.38).** Capacity doubling does NOT lift the §28 close-MATCH floor — the seed std=3.38y is optimization-limited, not capacity-limited. Incidental finding: AIDA L9 ridge R=0.617 / MAE=6.92 with near-zero bias (comparable to Pasta-REG 0.659/6.32). Output: `results/phase3/ridge_summary_r32_smoke.csv`, embeddings in `results/phase3/embeddings_layered/*r32_alllayers.npz`, checkpoint `results/baselines/fm_finetuned/geneformer/checkpoints/loco_onek1k_seed0_CD4p_T_e5b_r32.pt`. Compute ~$3, wall ~100min training + ~30min extraction.
- [ ] Task D.13 (conditional on D.12): scFoundation 3-seed bracket (frozen) — re-run extraction with seeds 1, 2 for cell-sampling variance check on §29.3 (existing seed=0 stays). Tests whether the §29 negative result is robust. ~$2 (cell sampling only, no training), ~5h wall. Defensive symmetry with §28's 3-seed bracket on Geneformer.
- [ ] Task D.14 (conditional on D.12 → D.13): scFoundation LoRA × 3 seeds × 2 LOCO folds × CD4+T. Apples-to-apples FM-class fine-tune test. ~$24 on g5, ~1 day wall. Run only if D.13 confirms the negative AND if higher-leverage Geneformer levers exhausted.
- [ ] Task D.15 (conditional, far): Geneformer full FT (no LoRA) × 3 seeds × 2 folds × CD4+T. The "more parameters, careful regularization" ceiling test. ~$30–50, ~1.5 days wall. Last resort if neither rank-32 nor longer training closes the §28 close-MATCH to a strict WIN.
- [ ] Task D.16 (proposed 2026-04-29, post-D.12 pivot): Geneformer LoRA + longer training (rank-16 × 5–6 epochs × 1 seed smoke). Direct test of the optimization-limited hypothesis from §30.3. If R=0.631 / MAE=8.21 from rank-16 seed-0 was an under-converged seed, longer training tightens std across seeds and lifts the floor. If not, MAE=10.85 ± 2.19 is a real ceiling and we write up Phase-3-A. Compute ~$3, ~2h wall. Same hyperparams as e5b except `--epochs 5` or `--epochs 6`, run-tag `_e5b_e6` or similar.

### Phase-3-B step-back review (2026-04-29, see scratchpad/step_back_review.md)

After §30's PIVOT we re-read existing layered-ridge CSVs (`ridge_summary_layered.csv`, `ridge_summary_post_finetune.csv`, `ridge_summary_layered_finetune.csv`, `ridge_summary_r32_smoke.csv`) and discovered:

1. **The L9-beats-L12-on-AIDA finding from §30.2 is rank-32 seed-0 specific, not structural.** Frozen-base CD4+T → AIDA: L12 wins R (0.527 vs L9 0.466) and MAE (11.76 vs 13.09). Rank-16 3-seed mean: L12 wins R (0.560 vs L9 0.520), L12 ≈ L9 on MAE (8.32 vs 8.36). The single-seed rank-32 L9 MAE=6.92 is the outlier. Candidate-3 headline (cross-layer asymmetry as paper lead) is *not supported* without a 3-seed verification on rank-32 — and indirect evidence suggests it would regress.
2. **Real cross-layer finding emerges**: NK-relevant signal lives in EARLY frozen-base layers (L2-L5 win R across all 3 cohorts; L0 wins MAE on OneK1K) while CD4+T-relevant signal lives at L12. B substrate empty everywhere. This is a cell-type-specific layer asymmetry visible *without fine-tuning*, supported by existing data, and writeup-worthy as a small panel.
3. **§22 / §23 numbers are slightly outdated**: §26's frozen L1 on Terekhova (R=0.616, MAE=8.82) sets the upper end of frozen-base CD4+T; previous "0.527–0.576" range underweights this.
4. **Apples-to-oranges concern with TF-paper gene-EN baseline**: their gene-EN R=0.83 LOCO + 0.77 AIDA used different splits, preprocessing, hyperparams. Our FM-loses-by-X-R-units claim's *magnitude* is uncertain until gene-EN runs on matched splits.
5. **Pseudobulk-input was skipped in §29** with the argument that ridge-readout already enforces donor objective post-hoc. The step-back review correctly notes this is *not* the same comparison: ridge-after-mean-pool aggregates *after* the FM; pseudobulk-input feeds aggregated input *to* the FM. The TF gene-EN baseline operates on log1p-mean pseudobulk — apples-to-apples requires matching that input shape.

### New Tier-1 tasks (load-bearing for writeup)

- [x] Task D.17 (done 2026-04-29, see §32): **Gene-EN baseline on FM-matched splits + preprocessing**. ElasticNetCV (top-5000 HVG, StandardScaler, 4 l1_ratios × 8 alphas × 3-fold CV) on loco_onek1k + loco_terekhova × CD4+T/B/NK. Output `results/baselines/gene_en_matched_splits.csv` (9 rows). **PAPER-CHANGING**: gene-EN matched R = 0.61 / 0.78 / 0.62 / 0.65 across CD4+T conditions (vs TF paper R=0.83 LOCO + R=0.77 AIDA). The "FM loses by 0.38 R-units" framing was an apples-to-oranges artifact; matched gap is 0.05–0.15 R-units. Cross-ancestry AIDA: gene-EN R=0.616/MAE=6.42 vs FM rank-32 L9 ridge R=0.617/MAE=6.92 — essentially tied. Memo §32, journal 2026-04-29.
- [x] Task D.18 frozen-base (done 2026-04-29, see §32): **Pseudobulk-input Geneformer + ridge readout** across CD4+T/B/NK × all 3 eval cohorts × 13 layers. Output `results/phase3/ridge_summary_pseudobulk.csv` (117 rows), `scripts/extract_embeddings_pseudobulk.py`. Pseudobulk-input shifts best-R layer to L1–L4 (early) for CD4+T, opposite of per-cell mean-pool which favors L12 — when fed donor-aggregated input, the FM behaves more like a bulk model. Ridge R competitive with per-cell mean-pool (sometimes higher: Terekhova R=0.688 vs 0.621), MAE worse on cross-cohort. **D.18 LoRA × 3-seed extension DEPRIORITIZED** — frozen-base result sufficient for "FM and bulk converge at matched splits" point. Memo §32.6, journal 2026-04-29.
- [~] Task D.19 (**SUBSUMED by D.21** post-§32 reframed review): L9 AIDA 3-seed verification on rank-32. Originally Tier 2 after the step-back review demoted it; the §32 matched-splits parity claim re-elevated it to Tier 1 because the L9 R=0.617 / MAE=6.92 vs gene-EN matched R=0.616 / MAE=6.42 tie depends on this single seed-0 number. Re-promoted as D.21 with explicit decision rules. Track work under D.21.
- [x] Task D.20 (done 2026-04-29, see §31): NK-early-layer asymmetry analysis. Confirmed from existing `results/phase3/ridge_summary_layered.csv`: NK-relevant signal lives in L2–L5 of frozen Geneformer across all 3 eval cohorts (best layers L3 OneK1K, L2 Terekhova, L5 AIDA; mean L3.3); CD4+T at L12 (mean L9.7); B substrate empty (mean L9.0 but R<0.23). Δ between best-layer R and L12 R for NK is largest on cross-ancestry AIDA (+0.121), suggesting early-layer NK features generalize better than late-layer ones. Methodology recommendation: cell-type-conditional layer selection. ~$0, no new compute. Memo §31, journal 2026-04-29.

### Phase-3-B reframed-review tasks (2026-04-29, post-§32, see scratchpad/reframed_review.md)

After §32 invalidated the central FM-loses-by-0.38-R-units claim, the paper's center of gravity shifted from "structural negative on FM aging prediction" to "methodology contribution + matched-splits comparison." The reframed-review proposed promoting cell-type-conditional layer selection to headline status; the critique flagged that several load-bearing positive numbers are still single-seed and that prior framings have been wrong twice in this conversation. The conclusion: **verify single-seed numbers and close the single-cell-type axes (matched-splits, pseudobulk-input) before committing to any framing.** Pre-commit decision rules so the next 3-seed result doesn't trigger another reactive reframing.

#### Tier 1 — verification (load-bearing, ~$40, 2–3 days)

- [x] Task D.21 (DONE 2026-04-30, see §37): **L9 AIDA rank-32 LoRA × 3-seed verification — DECISION-RULE PASS.** 3-seed mean L9 AIDA: R=0.594±0.025, MAE=7.33y±0.38y. Per-seed MAE: 6.92/7.66/7.40. σ(MAE)=0.38y << 2.0y robustness threshold → anchor-tier. **Decision: PASS** (≤7.5y band → outline (a) viable, parity headline survives). L11 best by R (0.612), L9 best by MAE (7.33). Memo §36, §37; journal 2026-04-30.
- [x] Task D.22 (DONE 2026-04-29, see §36): **NK frozen 3-seed verification — PARTIAL support.** ΔR(L_best vs L12) at 3-seed mean: AIDA cross-ancestry +0.085 PASS, OneK1K +0.039 FAIL (just below +0.05), Terekhova chemistry-shift +0.079 PASS. **Decision: 2/3 PARTIAL.** NK cell-type-conditional finding survives with cohort-specific caveat (cross-cohort yes, in-distribution no). Best-layer per cohort shifted from §31 single-seed (L3/L2/L5) to D.22 3-seed mean (L3/L2/**L6**) — direction holds, specific layer less stable. Memo §36; journal 2026-04-29.
- [x] Task D.23 (DONE 2026-04-29, see §34): **Matched-splits gene-EN on NK and B — B-empty FAILED bilateral.** Extended to 12 conditions. NK: 0.366/0.236/0.422/0.244 (in 0.30-0.50 band). B: 0.136/0.126/**0.321**/0.168 — B × Terekhova exceeds 0.20 threshold. **Decision: B substrate-empty NOT bilateral**; revert to "B mostly weak in both methods, with chemistry-shift exception (Terekhova R=0.321 that FM frozen probe doesn't capture)." Memo §34; journal 2026-04-29.
- [x] Task D.24 (DONE 2026-04-29, see §34.1): **Pseudobulk-input on NK and B — TWO-AXIS principle SUPPORTED.** NK pseudobulk-input best layer L0-L3 across all 4 conditions. B pseudobulk-input scattered (substrate-weak). **Decision**: pseudobulk-input → early layers regardless of cell type; per-cell mean-pool layer choice is cell-type-conditional. Two-axis principle supported. Memo §34.1; journal 2026-04-29.

#### Tier 2 — DONE 2026-04-29

- [x] Task D.25 (DONE 2026-04-29, see §34.2): **scFoundation frozen-ridge in matched-splits framing — Geneformer-specific parity confirmed.** scFoundation Δ vs gene-EN matched on CD4+T: -0.137/-0.174/-0.256/-0.086. vs Geneformer per-cell ridge Δ: -0.052/-0.088/-0.155/n.a. scFoundation lags Geneformer by 0.08-0.10 R-units. **Decision**: matched-splits parity is Geneformer-specific, not pan-FM. Closes scFoundation-LoRA from queue. Memo §34.2; journal 2026-04-29.
- [x] Task D.26 (DONE 2026-04-29, see §34.3): **Bootstrap CIs on §31 layer-asymmetry numbers — NK claim narrowed to AIDA.** NK ΔR(best vs L12) bootstrap CI excludes zero only on AIDA cross-ancestry (CI [+0.055, +0.184]). On OneK1K and Terekhova, CIs include zero. **Decision**: cell-type-conditional layer claim has weaker statistical support than medians suggested; robust only on AIDA cross-ancestry pre-3-seed verification. Memo §34.3; journal 2026-04-29.

#### Non-compute action items — ALL DONE 2026-04-29

- [x] Task D.27 (DONE 2026-04-29, see memo §33): Load-bearing single-seed numbers inventory. Tier-A (4 numbers, currently load-bearing), Tier-B (2 numbers, 3-seed-anchored), Tier-C (3 numbers, single-seed but not headline-relevant).
- [x] Task D.28 (DONE 2026-04-29, see `notes/paper_outline_drafts.md`): Two-tier paper outline drafts (a methodology-led, b comparison-led) with decision-rule selection table.
- [x] Task D.29 (DONE 2026-04-29, see `notes/decision_rules_phase3.md`): Pre-committed decision bands for D.21-D.24 in writing BEFORE runs landed. Institutionalizes the §28 lesson.
- [x] Task D.30 (DONE 2026-04-29, scratchpad note): NatComms venue speculation retracted from `scratchpad/reframed_review.md` with explicit retraction note. Honest floor is *Genome Biology* / *Bioinformatics* methods.

#### Reframed-review verification gate — RESOLVED 2026-04-30 (outline (a) selected)

| Verification | Outcome | Decision band |
|---|---|---|
| D.21 (rank-32 L9 AIDA 3-seed MAE) | **7.33y** | ≤7.5y PASS |
| D.22 (NK ΔR > +0.05 across 3 cohorts) | **2/3** PASS | PARTIAL support |
| D.23 (B-empty < 0.20 R bilateral) | FAIL | NOT bilateral |
| D.24 (NK pseudobulk best layer in L0-L4) | YES (L0-L3) | two-axis SUPPORTED |
| D.25 (scFoundation matched-splits) | LAGS by 0.08-0.10 | parity Geneformer-specific |
| D.26 (NK ΔR bootstrap CI) | Only AIDA excludes 0 | NK methodology AIDA-specific |

**Outline (a) METHODOLOGY-LED selected** with two cohort-specific caveats:
1. NK cell-type-conditional layer claim: cross-cohort only (not in-distribution)
2. B substrate-empty claim: not bilateral (Terekhova chemistry exception)

Compute spent: ~$18-20 GPU. Wall: ~11 hours. Writing begins now.

#### Phase-3-B reframed-review-extension tasks (proposed-and-implemented 2026-04-29/30)

After Tier 1 verification gate resolved, the autonomous session implemented additional follow-ups during GPU waits and after seed 2 landed:

- [x] Task D.31 (DONE 2026-04-29, see §35.1): **Donor-cluster mechanistic analysis** on §31 layer asymmetry. kNN-age correlation per layer per condition. Finding: §31's NK best-layer R advantage is dimensional-specific (specific aging-correlated axes), not cluster-structural. NK × OneK1K kNN-R: L3=0.337 vs L12=0.343 (L12 marginally better). Refines methodology framing.
- [x] Task D.32 (DONE 2026-04-29, see §35.2): **Bootstrap CIs on rank-16 LoRA 3-seed layered ridge.** L11 (not L9 or L12) is best AIDA layer at rank-16 3-seed mean: R=0.566±0.032, MAE=7.96y±0.42y. NEW anchor-tier headline. 3-seed std 0.14-0.42y << 2.0y robustness threshold.
- [x] Task D.33 (DONE 2026-04-29, see `notes/paper_draft_v0.md`): **First-pass paper draft v0** with Methods + Results section stubs covering §3.1-§3.7. All existing results filled in, pending tasks hedged. Outline (b) initially used as default; switched to outline (a) post-D.21 verification.
- [x] Task D.34 (DONE 2026-04-29, see `notes/methodology_diffs_vs_tf_paper.md`): **Methodology diff vs TF paper.** Decomposes TF's R=0.83 vs our matched R=0.61: more training cohorts (+0.08-0.15) + pseudocell augmentation (+0.05-0.10) + preprocessing (+0.02-0.05). Both numbers correct in their own framing.
- [x] Task D.35 (DONE 2026-04-29, see `results/phase3/paper_numbers_unified.csv`): **Unified paper-numbers CSV** with 120 rows covering all method × cell × eval × seed combos. Single reference for paper-writing.
- [x] Task D.36 (DONE 2026-04-30, see §39): **Strict MAE CI overlap test on parity claim.** Bootstrap rank-32 L9 3-seed (n=3000) vs gene-EN matched (n=1000): rank-32 [6.40, 8.56], gene-EN [5.28, 6.92], overlap [6.40, 6.92], Mann-Whitney p<0.001. Refines parity to "competitive within seed variance, distributions distinguishable."
- [x] Task D.37 (DONE 2026-04-30, see §38): **Inner-CV layer selection (deployment-recipe test).** K-fold CV on train donors only, then evaluate at CV-selected layer on holdout. **Findings**: rank-32 LoRA CV picks L12 in all 3 seeds (PERFECT deployability); rank-16 LoRA within ±1 layer of oracle; frozen Geneformer per-cell mean-pool layer selection NOT robustly deployable single-seed. Methodology contribution refined into two tiers: deployable recipe for fine-tuned variants, characterization-only for frozen-base.

Compute for D.31-D.37: ~$0 (all analysis-only on existing data). Wall: ~3 hours of session time.

#### Open follow-ups (proposed 2026-04-30, post-D.37 limitations §38.5)

D.37 documented three open limitations of the inner-CV layer-selection methodology:
1. Single-seed CV picks variable layers for frozen-base — does ensembling stabilize?
2. Within-train K-fold CV doesn't capture cohort batch effects — does cohort-holdout CV agree?
3. Argmax(CV-R) is a point estimate — what's the layer-choice uncertainty?

E.1–E.4 address these. Pre-committed decision rules baked into each task description so post-hoc rationalization is harder (the §28 lesson, applied prospectively).

- [x] **Task E.1 (DONE 2026-04-30, see §40.1): Multi-seed modal-layer ensemble.** Modal-layer agreement with oracle: 2/4 conditions (rank-32 PERFECT 3/3; rank-16 GOOD 2/3; NK frozen × loco_onek1k 0/3; NK frozen × loco_terekhova 0/3). Decision: ensembling helps marginally per global rule, but pattern is method-stratified — works for fine-tuned variants, fails for frozen NK. Output: `results/phase3/e1_modal_layer_ensemble.csv` (4 rows).

- [x] **Task E.2 (DONE 2026-04-30, see §40.3): Cohort-holdout inner CV.** Agreement with K-fold CV: 7/32 = 21.9% (well below 50% threshold). Decision: <50% → "layer choice is cohort-specific, not generalizable." Mechanism: with only 2 train cohorts, cohort-holdout CV is fundamentally unstable — Stephenson-as-inner-val vs Terekhova-as-inner-val produce picks 5+ layers apart. Methodological null result with clear scope (would need ≥3 train cohorts to be informative). Output: `results/phase3/e2_cohort_holdout_cv.csv` (32 rows).

- [x] **Task E.3 (DONE 2026-04-30, see §40.4): Bootstrap CIs on layer-selection itself.** N=200 bootstraps × 16 conditions × 5 folds × 13 layers ≈ 208k ridge fits. Wall: ~70 min (loco_terekhova conditions ~30× slower than loco_onek1k due to 5× larger train). Decision and per-condition results in §40.4. Output: `results/phase3/e3_bootstrap_layer_selection.csv`.

- [x] **Task E.4 (DONE 2026-04-30, see §40.2): End-to-end ensemble deployment test.** R-penalty drops: rank-32 trivially 0 (already at oracle); rank-16 89.9% (ensemble works); NK frozen onek1k −62.7% (ensemble HURTS); NK frozen terekhova −22.3% (slight hurt). Decision: ensemble deployment recommended only for rank-16 (1/4); for frozen-base, ensembling actively reduces accuracy because modal layer differs from each seed's local-noise CV pick. Output: `results/phase3/e4_ensemble_deployment.csv` (4 rows).

#### E.1-E.4 outcome summary

The four stress tests collectively confirm and refine the §38 two-tier finding rather than overturning it:

- **Tier 1 (fine-tuned variants)**: single-seed K-fold CV picks oracle reliably; ensembling/bootstrap don't add value because there's no instability to fix. Deployment recipe: **single-seed K-fold CV + ridge readout at the CV-selected (typically last) layer**. Confirmed across rank-16 and rank-32 LoRA × 3 seeds.

- **Tier 2 (frozen-base layer selection)**: directional regime claim survives (NK reads early L0-L4, CD4+T reads late L9-L12), but specific layer choice is unstable. Modal-layer ensembling actively hurts (E.4); cohort-holdout CV is uninformative on 2-cohort folds (E.2); bootstrap layer-selection (E.3) does provide tighter recipe in some conditions but not universally. Honest paper framing: **characterization-only at single-seed CV, with bootstrap-derived layer selection (E.3) as a stronger but more compute-intensive alternative**.

The "did you select layer on the test set?" reviewer challenge is fully addressed:
- For fine-tuned variants: **No** — K-fold CV on train donors picks the same layer as the oracle.
- For frozen-base: **No, but** the recipe is post-hoc-only; we report directional regimes, not specific layer numbers, as the methodology contribution.

### Phase-3-B priority order — Tier 1 + extensions COMPLETE 2026-04-30

Done in autonomous session 2026-04-29/30:
1. ✅ D.27 / D.29 / D.30 (pre-commitments)
2. ✅ D.21–D.24 (Tier 1 verifications)
3. ✅ D.25 / D.26 (Tier 2 robustness)
4. ✅ D.28 paper outline drafts (outline (a) selected)
5. ✅ D.31–D.37 (proposed-and-implemented extensions)

Open:
6. **E.1–E.4** (proposed but not run): methodology-strengthening follow-ups. Bundle E.1+E.2+E.4 recommended (~1.5-2h, $0).
7. **D.13 / D.16** — D.13 (scFoundation 3-seed) and D.16 (Geneformer longer training) remain as defensive ablations only if reviewers ask. **D.19 subsumed by D.21**.
8. **D.14 / D.15** — scFoundation LoRA, full FT — back burner. The §32 reframing + D.25 (scFoundation lags Geneformer) weakens the case for either; capacity-overfitting closeout is no longer load-bearing for the matched-splits headline.
9. **Writing**: paper_draft_v0.md is the starting point. Convert to full draft after E.1+E.2+E.4 (or skip directly to writing if user prefers).

### Compute envelope

| Block | Compute | Wall |
|---|---|---|
| Variant 1 frozen-base (D.1–D.4) | **~$3 spent** | done 2026-04-28 |
| Variant 1 audit (D.9–D.11) | $0 (analysis-only) | done 2026-04-28 |
| Variant 3 frozen + post-finetune (D.6) | **~$5 spent** | done 2026-04-28 |
| ~~Variant 2 (D.5)~~ | ~~$15~~ | **skipped 2026-04-29 — §29; resurrected as D.18 with corrected protocol after step-back review** |
| scFoundation diagnostic (D.7) | **~$3 spent** | done 2026-04-29 |
| Rank-32 smoke (D.12) | **~$3 spent** | done 2026-04-29 |
| Step-back review tasks (D.17, D.18, D.20) | **~$3 spent** | done 2026-04-29 |
| Reframed-review Tier 1 verification (D.21–D.24) | **~$18 spent** | done 2026-04-30 |
| Reframed-review Tier 2 + non-compute (D.25–D.30) | **~$0 spent** | done 2026-04-29 |
| Reframed-review extensions (D.31–D.37) | **~$0 spent** | done 2026-04-30 |
| Open follow-ups (E.1–E.4) | ~$0 conditional | ~1.5-3h conditional |
| Phase-3-B defensibility ablations (D.13–D.16) | ~$15–60 conditional | ~2–4 days conditional |

### Cancelled from B+NK extension

- ~~Task NK.2 (NK × loco_terekhova × E5b × seed 0)~~ — redundant with B × loco_terekhova for the chemistry-shift hypothesis; ~$6 compute redirected to Variant 1.
- Task NK.4 partial — NK rows in tri-headline classification still updatable from the loco_onek1k + AIDA scoring already completed.

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
    "title": "Zero-shot evaluation reveals limitations of single-cell foundation models",
    "url": "https://link.springer.com/article/10.1186/s13059-025-03574-x",
    "authors": "Kedzierska et al.",
    "year": 2025,
    "venue": "Genome Biology"
  }
]
```
