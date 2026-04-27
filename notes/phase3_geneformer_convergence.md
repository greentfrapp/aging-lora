# Phase-3-A Geneformer convergence investigation

**Status:** open. Investigation started 2026-04-26 after Phase-3-A Run #2 (`results/baselines/fm_finetuned/geneformer/summary.csv`) returned MAE=19.99y / R=0.33 on `loco_onek1k` × CD4+T — well above the 9.4y LASSO floor and 8.5y headline target.

## 1. Forensic readout of Run #2 artifacts

### 1.1 Saved checkpoint

`results/baselines/fm_finetuned/geneformer/checkpoints/loco_onek1k_seed0_CD4p_T.pt` (LoRA + head only, 122 keys):

| Group | Init | Final RMS | Verdict |
|---|---|---|---|
| `lora_A` (60 mats, [16,768]) | Kaiming, ~0.02 RMS | **0.020** (median) | Did not move from init |
| `lora_B` (60 mats, [768,16]) | **zero** (PEFT default) | **1.0e-3** (median) | Moved off zero, but tiny |
| `head.weight` ([1,768]) | normal(std=0.02) | 0.027 | ~unchanged from init |
| `head.bias` ([1]) | **48.93** (mean train age) | 48.93 | Held at init |

**Implied LoRA delta** on each weight matrix: (α/r) · B · A = 2 · B · A.
With A_rms ≈ 2e-2 and B_rms ≈ 1e-3 over rank 16, the delta has RMS ≈ 2 · 2e-2 · 1e-3 · √16 ≈ **8e-5**. Original BERT attention weights have RMS ≈ 1/√768 ≈ 0.036, so **LoRA's contribution is ~0.2% of the underlying weight magnitude.** LoRA is effectively still off.

### 1.2 Training trace

`logs/phase3/geneformer_loco_onek1k_seed0_CD4p_T.jsonl`:

- Config: epochs=1, batch=8, grad_accum=4 → ~296 optimizer steps; lr=5e-5 (backbone/LoRA), head_lr=1e-3; warmup_pct=0.10; weight_decay=0.01; bf16=True; seq_len=2048.
- `train_mse_running` over 11 logged step-windows: 270, 282, 282, 282, 278, 279, 264, 265, 274, 268, 258. **Plateau ≈ 270 with no downward trend.**
- Final eval: MAE=19.99, R=0.33, n_donors=981.

### 1.3 Train vs eval distributions (`data/cohorts/integrated/CD4p_T.h5ad`)

| Set | n_cells | n_donors | μ_age | σ_age | var | range |
|---|---|---|---|---|---|---|
| Train (Stephenson+Terekhova) | 923,581 | 190 | **50.08** | 16.99 | **288.58** | 21–81 |
| Eval (OneK1K) | 624,570 | 981 | **63.08** | 16.38 | 268.19 | 19–97 |

Two facts to anchor on:

1. **Train MSE plateau ≈ var(train_ages).** 270 ≈ 288.58 with mild head-weight contribution explaining the 18-point gap. Model is predicting near-constant on train.
2. **OneK1K eval distribution is shifted +13y vs. train.** A perfect mean-of-train predictor on OneK1K has **MAE ≈ |63 − 50| = 13y** before any dispersion penalty. The remaining ~7y of the observed 20y MAE comes from the head's ±0.5y prediction range failing to track ages spanning 19–97.

This is a real LOCO domain shift, but it is *not* a free pass: the headline goal is to produce per-cell age predictions that beat the LASSO MAE floor of 9.4y on OneK1K, and Phase-2 baselines are all trained on the same Stephenson+Terekhova data and clear that bar. So the FM has to learn a transferable age signal — not just the train-mean.

## 2. Environment

```
torch                 2.11.0+cu126
device capability     (7, 5)            # 2080 Ti, Turing
torch.cuda.is_bf16_supported  True       # software-supported on Turing (emulated)
transformers          5.6.1
peft                  0.19.1
```

Hardware-level bf16 (native tensor cores) starts at compute capability 8.0 (Ampere). On Turing the autocast path falls back to a slower bf16 codepath that still has bf16's 7-bit mantissa. This is a relevant precision concern — see §3 hypothesis (D).

## 3. Hypotheses, ranked by evidence weight

### (A) LoRA learning rate too low + far too few steps. **Highest confidence.**

- LoRA's standard fine-tuning LR is **1e-4 to 5e-4** (often quoted as 5–10× higher than full fine-tuning of the same backbone). The kickoff doc's 5e-5 was the full-FT BERT default.
- 1 epoch × ~296 optimizer steps × peak LR 5e-5 × ~0.5 area-under-warmup-decay ≈ **7e-3 cumulative LR.** That is far below typical LoRA total-LR budgets (5e-2 to 5e-1).
- Direct evidence: LoRA-A unchanged from init, LoRA-B at RMS 1e-3 (started at 0). The optimizer literally has not had enough cumulative LR to grow LoRA-B to functional magnitudes.

**Test:** raise lr to 2e-4, run 3 epochs.

### (B) Weight decay 0.01 fighting LoRA growth. **Medium confidence.**

- AdamW applies decoupled weight decay to *all* trainable params, including LoRA-A and LoRA-B. With LoRA-A at ~0.02 RMS, the weight-decay step at lr=5e-5 pulls A back toward 0 by ~5e-5 · 0.01 · 0.02 = 1e-8/step — small in absolute terms but proportional to A magnitude.
- Many community LoRA recipes set `weight_decay=0` on LoRA params (or apply WD only to head/bias-free layers).
- This compounds (A): low LR + non-zero WD makes the LoRA trajectory even more constrained.

**Test:** drop WD on LoRA params (keep on head).

### (C) head_lr=1e-3 saturates the head before LoRA can learn. **Medium confidence.**

- Head has 769 params (768 weight + 1 bias). At lr=1e-3 with grad-accum batch 32 over MSE on 50y-scale targets, the head adapts in ~10 steps to whatever signal exists in the cls embedding.
- Once the head is "tuned" against a near-constant cls embedding (since LoRA hasn't moved), the head's gradient w.r.t. LoRA is small (head.weight ≈ 0.027). The model gets stuck at a head-only minimum.
- Asymmetric LR ratio of 20× (head 1e-3 vs LoRA 5e-5) is much larger than typical recipes (≤5×).

**Test:** drop head_lr to 2e-4 (matched to a raised LoRA LR), or use a single LR for all trainable params.

### (D) bf16 numerical noise on tiny LoRA-B gradients. **Lower confidence but plausible.**

- LoRA-B starts at zero. Its gradient flows back through A (RMS 0.02), so LoRA-B's grad magnitude is on the order of `activation × A × output_grad ≈ 1 × 0.02 × O(loss/target_scale)`.
- With bf16's relative precision ≈ 1/128 ≈ 8e-3, a per-element grad component below ~8e-3 × |dominant grad| can be quantized to zero. LoRA-B's first few thousand updates may be partially absorbed into bf16 noise.
- On Turing this is exacerbated because the bf16 path is software-emulated, not tensor-core accelerated, so there's no speed benefit either way.
- The `torch.cuda.amp.GradScaler`-style grad scaling does **not** apply to bf16 autocast (only fp16); we have no protection against bf16 grad underflow here.

**Test:** re-run with `--no-bf16` (fp32 autocast).

### (E) Insufficient epochs. **High confidence as a contributing factor; not the root cause.**

- Kickoff §5 default = 5 epochs. Run #2 used `epochs=1` (CLI default is 3, so this was an override).
- Even if (A)–(C) were fixed, 1 epoch on 9,500 cells is a thin training budget.

**Test:** baked into the test for (A).

### (F) gradient checkpointing × `use_reentrant=False` × peft. Low confidence — sanity flag only.

- Some transformers/peft version combinations have hit subtle bugs where embedding-grad-requires + reentrant=False silently zeros adapter gradients on the first attention block. Easy to rule in/out by inspecting the per-layer LoRA-B norms.
- From §1.1, all 60 LoRA-B matrices are non-zero with similar magnitude → gradient *is* flowing through every layer. This hypothesis is largely ruled out.

## 4. Proposed experiment matrix

Run order goes cheap → expensive. Each step has a stop/go criterion; only proceed if cheaper steps don't already explain the plateau. All experiments use `--gpu-smoke` (the existing 5-min CLI mode that runs ~250 train cells × 20 steps) unless stated otherwise.

### E1. Reproducibility GPU-smoke at current hyperparameters [~5 min, baseline]

```
uv run python -m src.finetune.cli --fm geneformer --fold loco_onek1k --seed 0 \
    --gpu-smoke
```

Expected: train_mse plateau near var(train_ages_subset). Confirms baseline before changing anything.

### E2. Bump LoRA LR to 2e-4 [~5 min]

```
... --gpu-smoke --lr 2e-4 --head-lr 2e-4
```

**Stop criterion:** train_mse_running drops below 200 (i.e., visibly below var(train_ages) ≈ 250 on the smoke subset) within the 20 logged steps. If so, hypothesis (A) is the dominant cause.

### E3. fp32 autocast at current LR [~5 min]

```
... --gpu-smoke --no-bf16
```

**Stop criterion:** if E2 was conclusive, skip. Otherwise: same loss-drop criterion. If fp32 helps where bf16 didn't, hypothesis (D) is in play.

### E4. Combined fix: LR=2e-4, no WD on LoRA, fp32 [~5 min]

Requires a code change in `src/finetune/train_loop.py` to split LoRA params into a no-decay group. Keep this branch local until E2/E3 results inform the final config.

### E5. Full-budget run [~3–4 GPU-h]

Pick the winning config from E2–E4. Run with kickoff §5 defaults: epochs=3 (or 5), full-cohort train cells (no `--max-cells-per-donor 50`), seq_len=2048, bf16 only if E3 said it's safe.

```
uv run python -m src.finetune.cli --fm geneformer --fold loco_onek1k --seed 0 \
    --epochs 3 --lr 2e-4 --head-lr 2e-4 --max-cells-per-donor 100 \
    [--no-bf16 if E3 ruled it in]
```

**GATE 2 criterion (from kickoff §6):** validation MAE < 12y on OneK1K CD4+T (i.e., beats trivial mean predictor + meaningful per-donor rank correlation R > 0.4). Once cleared, scale to the full 18-fine-tune sweep.

## 5. Domain-shift caveat for downstream MAE interpretation

On `loco_onek1k`, the train→eval mean shift is +13y. The Phase-2 baselines (LASSO 9.4y, etc.) clear this domain shift, so this is a "the model must learn a transferable signal" story, not a "your eval is unfair" story. But the headline-target for Phase-3 was set against those baselines and is reachable; the FM just has to do the same thing they're doing. The point of recording this here is so we don't burn cycles trying to drive train MSE to zero — even an oracle on train will only get to MAE ≈ 13y on eval if it doesn't learn the age axis itself.

## 6. Coordination with the parallel session

The user has unrelated scAgeClock-on-`ren_covid19` processes running CPU-side (`scripts/phase_3/run_scaging.py` + `scripts/phase_3/infer_scageclock.py`). nvidia-smi confirms GPU is idle aside from desktop apps, so GPU smoke tests above are safe to run without disrupting them.

## 7. E1–E2 results (2026-04-26, 20-step gpu-smoke)

Both runs: `--gpu-smoke` (250 train cells / 10 donors → 125 eval cells / 5 OneK1K donors), seed=0, 20 optimizer steps, log_every=2, bf16 autocast on.

| step | E1 train_mse (lr=5e-5, head_lr=1e-3) | E2 train_mse (lr=2e-4, head_lr=2e-4) | Δ |
|---|---|---|---|
| 2  | 128.8 | 128.8 |  0.0 |
| 4  | 150.9 | 150.2 | −0.7 |
| 6  | 138.3 | 138.2 | −0.1 |
| 8  | 139.1 | 138.1 | −1.0 |
| 10 | 155.8 | 155.5 | −0.3 |
| 12 | 140.3 | 138.4 | −1.9 |
| 14 | 159.3 | 158.7 | −0.6 |
| 16 | 117.1 | 117.9 | +0.8 |
| 18 | 110.5 | 107.8 | −2.7 |
| 20 | 140.1 | 138.0 | −2.1 |
| **eval MAE** | **30.74** | **30.83** | +0.09 |
| **eval R** | **0.327** | **0.615** | **+0.288** |

Findings:

1. **Train-loss trajectories are nearly indistinguishable.** Per-batch noise (110–160 range) dwarfs any LR-driven differential over 20 steps at smoke scale. Smoke MSE alone cannot distinguish hypotheses.
2. **Eval Pearson R nearly doubles under lr=2e-4** (0.33 → 0.62) on identical 5 donors / identical seed / identical init / identical data ordering. **This is the smoking gun for hypothesis (A): LoRA learns more under 4× higher LR within the same step budget.** R from the cheap-LR run (E1) matched Run #2's R at production scale (0.327), so smoke→prod scaling of R is preserved.
3. **MAE stays flat (~30.7 in both)**, dominated by the bias offset between the smoke's 10-donor train mean (42.0y, head bias init) and 5-donor OneK1K eval mean. Smoke MAE is saturated by domain shift, not by LoRA quality. **R is the right metric at smoke scale.**

Caveat: 5 eval donors is a noisy estimator of R; the +0.29 R jump is suggestive but not statistically tight. Combined with hypothesis (A)'s mechanism story (LoRA delta RMS ~8e-5 → too small to bend the loss surface), I'm confident enough to commit to lr=2e-4 as the new default and move on without running more smoke seeds.

Hypothesis (D) bf16 was not isolated by E1/E2 but its effect is bounded above by what's left after fixing (A). Will test in E5 (full-budget) by toggling `--no-bf16` only if E5 underperforms.

## 8. Intermediate-v2 results (2026-04-26, production scale)

Run with `lr=2e-4 head_lr=2e-4` at Run #2 data scale (9,500 train cells × 1 epoch, full 981-donor OneK1K eval, bf16) — apples-to-apples vs Run #2, only the optimizer changed. Wall=16,327s (≈4.5h, slowed by parallel scAgeClock CPU contention). Two false-start CLI bugs caught and fixed during this experiment:

- **Eval cap default**: first launch left `--eval-max-cells-per-donor` at the CLI default (200), giving 195,559 eval cells vs Run #2's 19,620. Killed and restarted with `--eval-max-cells-per-donor 20`. Lesson: CLI default should match Run #2's scope, not be 10× wider.
- **`--gpu-smoke` clobbering production paths**: smoke runs were appending to the production summary.csv / JSONL / checkpoint. Fixed: `gpusmoke_` prefix + `--run-tag-suffix STR` arg + skip-checkpoint-on-smoke.

### Headline numbers

| | Run #2 (lr=5e-5, head_lr=1e-3) | v2 (lr=2e-4, head_lr=2e-4) | Δ |
|---|---|---|---|
| MAE (y) | 19.99 | 18.48 | −1.51 (−7.5%) |
| Pearson R | 0.327 | 0.316 | −0.011 (within noise) |

The smoke-scale +0.29 R bump from E2 **did not survive at 981-donor scale**. R is essentially identical to Run #2; MAE dropped a real-but-modest 1.5y. So lr=2e-4 alone does not move the needle in any meaningful way — hypothesis (A) was overstated by the 5-donor smoke.

### Mechanism (per-donor predictions + LoRA state-dict, `scratchpad/_compare_run2_vs_v2.py`)

| | Run #2 | v2 |
|---|---|---|
| pred mean | 47.89 | 50.23 |
| **pred sd** | **0.89** | **1.13** |
| pred range | 45.8 → 50.5 (Δ=4.7y) | 43.2 → 52.6 (Δ=9.4y) |
| true range | 19 → 97 (Δ=78y) | 19 → 97 (Δ=78y) |
| err mean (offset) | −16.02 | −13.69 |
| err sd | 16.23 | 16.18 |
| **LoRA delta RMS (median)** | **1.4e-4** | **4.5e-4 (3.2× ↑)** |
| LoRA delta RMS (max) | 3.8e-4 | 1.1e-3 |
| head.weight RMS | 0.0273 | 0.0216 |
| head.bias | 48.927 | 48.926 |

Three observations that change the diagnosis:

1. **The model's effective dynamic range is 6–12% of the target's range** in both runs. Predictions fan out over 4.7y / 9.4y while ages span 78y. The model is sitting on the "predict mean(train)" minimum of MSE.
2. **The 1.5y MAE improvement in v2 is almost entirely a bias-drift effect** (mean prediction moved +2.3y closer to eval mean), not an age-axis-signal improvement. Error sd is unchanged (16.23 → 16.18).
3. **LoRA delta DID grow 3.2× in v2** — gradients flowed, parameters moved, the optimizer is functioning. But the resulting cls embeddings still don't differentiate ages enough for the (smaller, head_lr=2e-4) head to spread its predictions. The bottleneck isn't gradient magnitude.

This is the classic LoRA-undertraining pattern. Phase-2's *linear* LASSO hits MAE 9.4y on the same train data; a 110M-param frozen FM stuck at MAE 18–20y is a striking gap that points to the model being unable to escape the mean-prediction basin within 296 steps.

## 9. Revised hypothesis ranking (post-v2)

1. **Severe undertraining + cls-pooling weakness.** *Was #5; now joint top.* 296 optimizer steps × 110M-param frozen backbone is well below typical LoRA recipes (5–20k steps). And `<cls>`-only pooling over rank-value-encoded gene tokens may not localize age signal — mean-pooling over the attended tokens is a known fix.
2. **Original LR (A).** Still relevant — v2 LoRA delta grew 3× — but not the binding constraint by itself.
3. **Weight decay on LoRA (B), head/backbone LR mismatch (C).** Secondary refinements.
4. **bf16 numerical noise (D).** Looking less likely — LoRA delta moved cleanly in fp32-equivalent magnitudes.
5. **Gradient checkpointing × peft bug (F).** Already ruled out.

## 10. E5a results (2026-04-27, mean-pool ablation)

Same data scale as Run #2 / v2 (9,500 train cells × 1 epoch, full 981-donor OneK1K eval, lr=2e-4 head_lr=2e-4 bf16). Only change: `<cls>`-only pooling → mean-pool over attended positions, via new `--pool {cls,mean}` flag in `src/finetune/cli.py` and `src/finetune/lora_wrappers/geneformer.py`. Wall=9,354s (≈2.6h, fastest of the three production runs — parallel scAgeClock CPU load lifted by the time E5a launched).

| | Run #2 (cls, lr=5e-5) | v2 (cls, lr=2e-4) | **E5a (mean, lr=2e-4)** |
|---|---|---|---|
| MAE (y) | 19.99 | 18.48 | 19.34 |
| **R** | 0.327 | 0.316 | **0.362** ← best |
| pred mean | 47.89 | 50.23 | 48.99 |
| pred sd | 0.89 | 1.13 | 1.09 |
| **LoRA delta RMS (median)** | 1.4e-4 | 4.5e-4 | **5.9e-4** ← largest |
| LoRA delta RMS (max) | 3.8e-4 | 1.1e-3 | **1.6e-3** ← largest |

Findings:

1. **Mean-pool is the largest single-knob R improvement so far** (+0.046 vs v2). Surfaces gradient signal across all attended token positions, not just `<cls>` — LoRA delta is the biggest yet (median 5.9e-4, max 1.6e-3, ≈4× Run #2). Strictly dominates cls in every metric we have.
2. **MAE creeps back up to ~19.3y** because the v2 +2.3y bias-drift effect is gone — that drift was a side effect of cls-pool's tighter prediction range under high LR, not a generalizable improvement. With mean-pool the head bias stays near 48.93 (≈ train mean), and MAE = |eval_mean − pred_mean| + dispersion ≈ 14.9 + 4.4 ≈ 19.3y, dominated by the train→eval domain shift floor.
3. **The fundamental regime hasn't changed.** Prediction sd 1.09y vs true sd 16.5y — predictions span ~7% of the true range. R=0.362 is still well below GATE 2's R>0.5 threshold; Phase-2's *linear* LASSO at MAE=9.4y still beats the FM by a wide margin. The model is still stuck in the "predict near-mean(train)" basin.

**Pattern across the three runs**: each architectural intervention buys ~+0.04 R. To clear R>0.5 (need ~+0.14 more) we need a **regime change** — many more optimizer steps — not another single-knob fix. That's E5b (3 epochs) and E5c (10× cells/donor) territory.

## 11. Updated plan

Now down to two experiments. Both use the new `--pool mean` default (E5a confirmed it dominates cls):

- **E5b — 3 epochs at Run #2 data scale, mean-pool** [no code change, ~8h local / ~3h on A10g cloud]. Direct test of the undertraining hypothesis. Cumulative LR-area goes 3×, optimizer-step count goes 3×.
- **E5c — `--max-cells-per-donor 500`, 1 epoch, mean-pool** [no code change, 10× more train cells per epoch, ~8h local]. Tests undertraining via dataset size rather than epochs.

E5b is preferred next: same data sees more SGD steps, isolates the "more training" variable cleanly. E5c also tests "more data" but adds a confound (different cells sampled per donor each step).

**GATE 2 criterion (unchanged):** validation MAE < 12y on OneK1K CD4+T, R > 0.5 on the full 981-donor eval. Once cleared, scale to the full 18-fine-tune sweep (Phase-3-B) with mean-pool baked in.

## 12. CLI changes committed during investigation

- `--run-tag-suffix STR`: appended to run_tag; isolates outputs for ablation runs.
- `--log-every INT`: previously hardcoded at 25; now CLI-configurable so short smokes produce step traces.
- `--gpu-smoke` now adds `gpusmoke_` prefix to run_tag and skips checkpoint save (was previously sharing paths with real runs).
- `--pool {cls,mean}`: pooling over backbone hidden states for the regression head (new in E5a). Default still `cls` for back-compat with Run #2 / v2; should be flipped to `mean` after Phase-3-A closes.
- Smoke modes (`--smoke`, `--gpu-smoke`) now defensively redirect `--output-dir` from the production `results/baselines/fm_finetuned` to `results/phase3_smoke` if the user kept the production default. Prevents accidental contamination of the production `summary.csv` (which happened once during E5a code-validation; reverted in the same commit).

## 13. E5b results (2026-04-27, 3 epochs + mean-pool on A10g)

First cloud run (g5.xlarge, A10g 24 GB, native bf16). Same data scale as Run #2 / v2 / E5a (9,500 train cells × 190 donors, 19,620 eval cells × 981 OneK1K donors), `--pool mean --lr 2e-4 --head-lr 2e-4 --epochs 3 --max-cells-per-donor 50 --eval-max-cells-per-donor 20 --run-tag-suffix _e5b`. Wall=5,994s (~1.7h) — A10g delivered ~4.6 s/step vs the 2080 Ti's ~37 s/step (8× per-step speedup; net 3× wall after 3× steps).

| | Run #2 (cls, lr=5e-5, 1ep) | v2 (cls, lr=2e-4, 1ep) | E5a (mean, lr=2e-4, 1ep) | **E5b (mean, lr=2e-4, 3ep)** |
|---|---|---|---|---|
| MAE (y) | 19.99 | 18.48 | 19.34 | **17.37** ← best |
| **R** | 0.327 | 0.316 | 0.362 | **0.466** ← best |
| Wall | 3.1h (2080 Ti) | 4.5h (2080 Ti) | 2.6h (2080 Ti) | **1.7h (A10g)** |

Train-MSE-running trajectory (logs/phase3/geneformer_loco_onek1k_seed0_CD4p_T_e5b.jsonl):

| step | MSE | epoch | note |
|---|---|---|---|
| 125 | 264.6 | 0 | near var(train_ages)≈288 (matches Run #2/v2 epoch-1 plateau) |
| 250 | 260.5 | 0 | ~end of epoch 1 — same plateau Run #2/v2/E5a stopped at |
| 375 | 239.9 | 1 | **first 20-pt drop below var(train) — basin escape begins in epoch 2** |
| 500 | 236.1 | 1 | descent slowing |
| 650 | 232.3 | 2 | |
| 775 | 247.3 | 2 | bounces (batch noise; Run #2 trace had similar swings 258–282) |
| 875 | 222.8 | 2 | lowest yet, total descent ~65 points |

Findings:

1. **+0.104 R over E5a — biggest single jump of the four runs.** Pattern across the prior three runs was ~+0.04 R per single-knob fix (LR 5e-5→2e-4: −0.011; cls→mean: +0.046). E5b is the first true regime change (3× SGD steps) and bought 2.5× the single-knob delta. Confirms the §9 hypothesis that undertraining is the binding constraint, not LR or pooling alone.
2. **GATE 2 still not cleared.** Target was MAE<12y, R>0.5. E5b is 17.37y / 0.466 — closest yet on both axes; R is 0.034 below the bar. Headline LASSO floor on this fold is 9.4y, so the FM is still ~2× over the linear baseline at this MAE.
3. **Train-MSE trajectory matches the diagnosis exactly.** Epoch 1 plateaued at 260–264 (same as 1-epoch runs ending at ~270). Epoch 2 dropped 20 points — the model only began escaping the predict-mean basin once it had >1 epoch of cumulative LR-area. By epoch 3 end, MSE 222 is ~23% below var(train_ages).
4. **Mechanism not yet verified.** Have per-donor predictions and the trainable-state checkpoint, but haven't yet checked whether the +0.104 R is "model finally learning the age axis" (prediction sd ↑↑, LoRA delta ↑↑) vs "extended bias drift" (prediction sd ~1y like Run #2/v2/E5a, MAE drop driven by mean-shift toward eval mean). Pending §14 inspection.

## 14. E5b mechanism inspection (2026-04-27)

Loaded `results/baselines/fm_finetuned/geneformer/checkpoints/loco_onek1k_seed0_CD4p_T_e5b.pt` + the Run #2 / E5b per-donor CSVs (only Run #2 unsuffixed CSV survived locally; v2 / E5a numbers carried forward from §1.1, §8, §10). Script: `scratchpad/_inspect_e5b.py` (gitignored).

| run | pred sd (y) | pred range (y) | err mean | err sd | LoRA Δ RMS median | LoRA Δ RMS max | head_w RMS | head_b | R | MAE |
|---|---|---|---|---|---|---|---|---|---|---|
| Run #2 (cls, 1ep) | 0.89 | 4.7 | −16.0 | 16.2 | 1.4e-4 | 3.8e-4 | 0.0273 | 48.927 | 0.327 | 19.99 |
| v2 (cls, 1ep) | 1.13 | 9.4 | −13.7 | 16.2 | 4.5e-4 | 1.1e-3 | 0.0216 | 48.926 | 0.316 | 18.48 |
| E5a (mean, 1ep) | 1.09 | — | — | — | 5.9e-4 | 1.6e-3 | — | — | 0.362 | 19.34 |
| **E5b (mean, 3ep)** | **2.48** | **16.7** | **−12.7** | **15.5** | **9.4e-4** | **2.3e-3** | **0.0234** | **48.927** | **0.466** | **17.37** |

Three findings settle the §13.4 open question — **E5b is a regime change, not bias drift**:

1. **Prediction sd more than doubled** (1.09y → 2.48y). All three 1-epoch runs sat at ~1y sd regardless of LR or pooling — the bias-drift basin. E5b is the first run to genuinely spread predictions across the age axis. Range went 9.4y → 16.7y, 21% of true range vs the basin's 6–12%.

2. **MAE improvement is partly real signal, not pure bias closure.** The 2.6y MAE drop from Run #2 decomposes into ~3.3y of bias closure (pred mean 47.9 → 51.2, walking toward eval mean 63.9) and a 0.7y err-sd reduction (16.2 → 15.5). Err-sd only drops when predictions actually correlate with truth — corroborated by the +0.14 R jump. Contrast v2's MAE drop, where err-sd was unchanged (pure bias drift); E5b is mixed.

3. **LoRA continues to grow.** Median delta RMS climbed 1.4e-4 → 4.5e-4 → 5.9e-4 → 9.4e-4 across the four runs. 6.7× Run #2's. The optimizer is still finding gradient signal — no plateau evidence in the parameter trajectory.

Head bias stays glued to ≈48.93 (mean train age, init value) across all runs — head bias is not the moving variable; the LoRA-driven hidden-state representation is.

## 15. Next experiment: more epochs

§14 picks the "model is actually spreading predictions" branch from §13.4: pred sd doubled, LoRA delta climbed 1.6× E5a, R climbed 2.5× the single-knob plateau. Undertraining is the binding constraint and more cumulative LR-area is still paying. Crude extrapolation from the +0.014 R/epoch slope (E5a→E5b: 0.362→0.466 in 2 extra epochs) puts GATE 2's R>0.5 within reach of 2–3 more epochs.

- **E5d — 5 epochs at the E5b config, mean-pool, A10g** [no code change, ~3h, ~$3 on-demand]. Same data scale (`--max-cells-per-donor 50`, ~9,500 train cells/epoch), `--lr 2e-4 --head-lr 2e-4 --pool mean --epochs 5`. If R clears 0.5 at 5 epochs, GATE 2 is the next thing to verify (MAE<12y is a longer reach: need −5.4y from E5b vs the 2.6y three epochs delivered). If R asymptotes between E5b and E5d, the next move is data scale (E5c) rather than more epochs. Either outcome is informative.
- **E5e — 7 epochs (conditional)**: only if E5d's R shows the +0.05/2-epochs trend continuing. Otherwise we shift to E5c or to a different objective.

MAE 12y is still the hard part: even if R climbs to 0.6, MAE-bound by the train→eval domain shift floor (Stephenson+Terekhova mean 50.1y vs OneK1K 63.1y; perfect-rank predictor on train would still hit |Δμ|=13y eval MAE absent good age-rank generalization). The 2/3-headline preprint outcome (R>0.5 wins) is still in play; the 3/3 (MAE<8.5y) requires the model to substantially correct for the domain shift, not just rank-order within it.

## 16. E5d results (2026-04-27, 5 epochs + mean-pool on A10g)

Same data scale and hyperparameters as E5b; only change is `--epochs 5`. Wall=8,693s (~2.4h, scaled cleanly from E5b's 1.7h × 5/3 + fixed eval). Final eval on OneK1K CD4+T (981 donors): **MAE=16.53y / R=0.431**. Train-MSE-running continued descending to 196 at step 1475 (lowest of any run, total descent ~75 points from epoch-0 baseline 271).

| run | pred sd (y) | pred range | err mean | err sd | LoRA Δ RMS median | LoRA Δ RMS max | head_w RMS | head_b | R | MAE |
|---|---|---|---|---|---|---|---|---|---|---|
| Run #2 (cls, 1ep) | 0.89 | 4.7 | −16.0 | 16.2 | 1.4e-4 | 3.8e-4 | 0.0273 | 48.927 | 0.327 | 19.99 |
| v2 (cls, 1ep) | 1.13 | 9.4 | −13.7 | 16.2 | 4.5e-4 | 1.1e-3 | 0.0216 | 48.926 | 0.316 | 18.48 |
| E5a (mean, 1ep) | 1.09 | — | — | — | 5.9e-4 | 1.6e-3 | — | — | 0.362 | 19.34 |
| E5b (mean, 3ep) | 2.48 | 16.7 | −12.7 | 15.50 | 9.4e-4 | 2.3e-3 | 0.0234 | 48.927 | 0.466 | 17.37 |
| **E5d (mean, 5ep)** | **3.24** | **20.6** | **−11.9** | **15.38** | **1.4e-3** | **3.3e-3** | **0.0254** | **48.926** | **0.431** | **16.53** |

Findings — the §15 "more epochs continue paying" extrapolation **fails between epochs 3 and 5**, but in a more subtle way than naive overfitting:

1. **R regressed from 0.466 → 0.431** while MAE continued improving (17.37 → 16.53). Train MSE kept descending (222 → 196). Predictions are getting *more* confident (pred sd 2.48 → 3.24, pred range 16.7y → 20.6y) and the err mean is closer to zero, but per-donor age correlation got *looser*.

2. **Not classic overfitting (model wasn't just memorizing train).** Decomposition: cov(pred, true) actually grew from 19.07 to 23.04 between E5b and E5d — the model's predictions are still aligned with truth, the alignment is just less *tight per unit of prediction-variance*. Pearson R = cov / (sd_pred × sd_true): pred_sd grew 30%, cov only grew 21%, so R drops. The additional learning is real on train but the prediction-spread it produces on eval doesn't scale proportionally with truth.

3. **MAE-vs-R divergence is bias-closure-driven.** Continued bias closure (err mean −12.7 → −11.9) shifts predictions ~1y closer to eval mean, dropping MAE. Err sd dropped marginally too (15.50 → 15.38). The MAE improvement is real but doesn't reflect rank-improvement; R drops because the *spread* of the more-confident predictions is partially off-axis.

4. **LoRA still moving, head still parked.** LoRA delta median RMS climbed to 1.4e-3 (the largest yet, +47% over E5b). Head bias stays at 48.93 (mean train age, init value) across all five runs — head is not the moving variable. The optimizer found 47% more parameter movement between epochs 4-5; the *quality* of that movement, not its magnitude, is what regressed.

GATE 2 status: **R bar moved further away** (E5b 0.466 was closer to 0.5 than E5d's 0.431). MAE 16.53y is still 4.5y over the 12y bar. The "more epochs" track has run its course at this data scale.

## 17. Next experiment: data scale (E5c)

§15's conditional triggers: "If R asymptotes between E5b and E5d, the next move is data scale (E5c)." E5d went past asymptote into regression — even stronger signal that the 9,500-cells × 190-donors data slice is the binding constraint, not training budget.

- **E5c — `--max-cells-per-donor 500`, 1 epoch, mean-pool, A10g** [no code change, ~3.8h, ~$3.80 on-demand]. 95,000 train cells/epoch (10× E5b/E5d), ~2,968 optimizer steps (~3.3× E5b's 888, ~2× E5d's 1,480). Tests whether more cells per donor reduces the per-donor age estimate's noise enough to break through the R~0.45 ceiling, separately from the "more epochs" lever that just played out. CLI: `--epochs 1 --lr 2e-4 --head-lr 2e-4 --pool mean --max-cells-per-donor 500 --eval-max-cells-per-donor 20 --run-tag-suffix _e5c`.
- **Variance check (deferred)**: 3 seeds at the *winning* config. Both E5b's +0.104 R jump and E5d's −0.035 R regression are single-seed measurements. Will run multi-seed once we've identified the best base config so the variance characterization is on the right config, not on a pre-final one.

If E5c also asymptotes around R=0.45, the next thread is reconsidering objective: per-donor median targets (one prediction per donor) rather than per-cell MSE may be a better fit — or auxiliary tasks (e.g., predicting cell-type-shared age signal across cell types).
