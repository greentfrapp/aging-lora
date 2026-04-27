# Phase-3-A autonomous-window summary (2026-04-27)

User went offline ~12:30 UTC, returns ~00:30 UTC. This file is a TL;DR of what happened during the 12-hour autonomous window. Full details are in `notes/phase3_geneformer_convergence.md` §13–20 and `notes/research_journal.md` (entries dated 2026-04-27).

## What ran

Five Geneformer LoRA fine-tunes on g5.xlarge A10g + AIDA scoring of all five checkpoints + one in-progress run:

| Run | Config | OneK1K MAE | OneK1K R | AIDA MAE | AIDA R | Wall | Cost |
|---|---|---|---|---|---|---|---|
| E5b seed 0 | 3ep cap=50 mean | 17.37 | 0.466 | 10.39 | 0.311 | 1.7h | $1.7 |
| E5d seed 0 | 5ep cap=50 mean | 16.53 | 0.431 | 9.40 | 0.441 | 2.4h | $2.4 |
| E5c seed 0 | 1ep cap=500 mean | 16.27 | 0.385 | 9.53 | **0.545** | 4.3h | $4.3 |
| E5b seed 1 | 3ep cap=50 mean | 17.57 | **0.498** | 10.18 | 0.240 | 1.7h | $1.7 |
| E5b seed 2 | 3ep cap=50 mean | 16.56 | 0.396 | 10.60 | 0.350 | 1.7h | $1.7 |
| **In progress** Terekhova × E5b seed 0 | 3ep cap=50 mean | — | — | — | — | ~6h | ~$6.0 |

**Total compute: ~14.4h training + 50min inference, ~$15 spent so far. Terekhova run brings total to ~$21 (under the $30 autonomous cap).**

E5b 3-seed mean on OneK1K: **R=0.453 ± 0.042, MAE=17.17 ± 0.43y**.

## Key findings

### 1. Variance is bigger than thought (forces §18 reinterpretation)

E5b R varies 0.396–0.498 across 3 seeds (range 0.102, σ=0.042). The §18 single-seed claim that "E5b is a sweet spot" relied on R differences smaller than this seed-variance. E5d (R=0.431, single seed) is within 1σ of E5b's mean. **Cannot distinguish "more epochs hurts" from seed noise without 3 seeds at E5d too.** Phase-3-B should multi-seed every config it scores.

### 2. OneK1K–AIDA inversion (the headline)

**Configs that look "regressed" on OneK1K are *better* on AIDA cross-ancestry.**

| Run | OneK1K rank | AIDA rank |
|---|---|---|
| E5b mean | 1 | 4 |
| E5d single | 2 | 2 |
| E5c single | 3 | 1 |

Within-config seed variance shows the same negative correlation: E5b seed 1 has the best OneK1K R (0.498) and the worst AIDA R (0.240). Mechanism: OneK1K + train cohorts (Stephenson + Terekhova) are all European; AIDA is Asian. Configs that overfit donor-specific cell patterns on the European train cohorts generalize well to OneK1K (similar population) but fail on AIDA (different population). E5c's 10× cells/donor is a stronger regularizer against population-specific overfitting at the cost of in-distribution accuracy.

This is publishable methodology in its own right — different scale levers favor different generalization regimes — and reframes the preprint from "FMs beat baselines" toward "FM scaling levers trade off in-distribution accuracy vs cross-ancestry transfer."

### 3. Win/match/loss vs Phase-2 baselines: 0/2

| Cell | FM best | Min baseline | FM bar (10% win) | Outcome |
|---|---|---|---|---|
| OneK1K CD4T | E5b mean R=0.453, MAE=17.17 | LASSO 9.45y / R=0.75 | ≤8.50y | **loss** (FM loses on both axes) |
| AIDA CD4T | E5c R=0.545, MAE=9.53 | Pasta-REG 6.32y / R=0.66 | ≤5.69y | **loss** (close on R, far on MAE) |

Per kickoff outcome rules: 0/2 → pivot to evaluation-study framing for the preprint. Not a null finding (E5c on AIDA cleanly beats scAgeClock R=0.30) but not a horse-race win either.

## What did NOT run (deferred, awaiting user authorization)

1. **scFoundation × loco_onek1k** — was originally on the Phase-3-B short list but I deferred it. Per the §20.4 recommendation, want to test per-donor objective on Geneformer (already-instrumented) before committing $20+ on three FMs running the suspected-suboptimal per-cell objective.
2. **scGPT × loco_onek1k** — same reasoning.
3. **Per-donor objective ablation** — the highest-information-per-dollar follow-up. Would test whether the §18/§20.2 finding that per-cell MSE causes donor-level memorization is correct. Estimated 2h dev (modify `train_loop.py` to aggregate predictions per donor before MSE; add donor IDs to batches) + 1.7h test run. Skipped autonomously because code changes carry bug risk; want user review.
4. **3-seed multi-seed at E5d and E5c configs** — would resolve §1 above but adds 4 runs × 1.7-3.8h = 8-15h compute beyond the autonomous window.
5. **loco_stephenson fold** — exploratory-only per the Phase-3 plan; not headline.

## Files committed during the window

5 commits on `origin/main` (47df4ee → 3bea055):

- 47df4ee — E5b results + repaired checkpoint hashes + memo §13/journal
- 29bb94a — E5b mechanism check + memo §14/§15
- 6b383bb — E5d results + memo §16/§17/journal
- 1020375 — `scripts/score_aida.py` + AIDA donor-namespace fix
- 03263c5 — E5c results + memo §18/§19/journal
- 2d224bb — E5b variance check (seeds 1+2) results
- 3bea055 — Phase-3-A close-out memo §20 + AIDA scoring of all 5 checkpoints

## Phase-3-B recommendation

**Go, with these changes from the original kickoff:**

1. **Run loco_terekhova × E5b × seed 0 first** (currently in progress as of this writing — likely complete by the time you read this). Provides the third headline cell (after OneK1K and AIDA).
2. **Defer scFoundation + scGPT until per-donor objective is tested.** Test per-donor on Geneformer first; if it lifts the AIDA/OneK1K trade-off, propagate to other FMs.
3. **Multi-seed every reported config.** §1 above showed seed variance is larger than naive single-seed comparisons assume. Phase-3-B should report 3-seed mean ± std for any (FM, fold, config) cell that appears in the preprint.
4. **Frame preprint around the methodology finding.** §2 above is more interesting than the baseline-beating story would have been if any config had won. Use the OneK1K–AIDA inversion + per-cell-vs-per-donor objective tradeoff as the methodology contribution.

## Live state

- Terekhova run ID: bash background `b507pgji2`. Logs: `logs/phase3/terekhova_seed0_console.log` + `logs/phase3/geneformer_loco_terekhova_seed0_CD4p_T_e5b.jsonl`. Output checkpoint will land at `results/baselines/fm_finetuned/geneformer/checkpoints/loco_terekhova_seed0_CD4p_T_e5b.pt`.
- Monitor task ID: `bzyq1w8a9` (persistent).
- AIDA scoring script: `scripts/score_aida.py` — usage in its docstring.
- AIDA results: `results/phase3/aida_summary.csv` and `results/phase3/aida_per_donor/`.
- Mechanism inspection scripts: `scratchpad/_inspect_e5{b,c,d}.py` (gitignored).
