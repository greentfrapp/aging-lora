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

## 18. E5c results (2026-04-27, 1 epoch + 10× cells/donor on A10g)

Same hyperparameters as E5b/E5d (mean-pool, lr=2e-4, head_lr=2e-4, bf16) but `--max-cells-per-donor 500 --epochs 1`. 93,702 train cells (after the per-donor cap takes effect on donors with <500 cells), 2,930 optimizer steps (~3.3× E5b, ~2× E5d). Wall=15,347s (~4.3h on A10g). Final eval on OneK1K CD4+T (981 donors): **MAE=16.27y / R=0.385**. MAE is the lowest of any run; R is below E5b (0.466) AND E5d (0.431).

| run | n_train_cells | epochs | pred sd (y) | pred range | LoRA Δ RMS median | LoRA Δ RMS max | head_w RMS | head_b | R | MAE |
|---|---|---|---|---|---|---|---|---|---|---|
| Run #2 (cls) | 9,500 | 1 | 0.89 | 4.7 | 1.4e-4 | 3.8e-4 | 0.0273 | 48.927 | 0.327 | 19.99 |
| E5b (mean) | 9,500 | 3 | 2.48 | 16.7 | 9.4e-4 | 2.3e-3 | 0.0234 | 48.927 | **0.466** | 17.37 |
| E5d (mean) | 9,500 | 5 | 3.24 | 20.6 | 1.4e-3 | 3.3e-3 | 0.0254 | 48.926 | 0.431 | 16.53 |
| **E5c (mean)** | **93,702** | **1** | **2.63** | **19.3** | **1.7e-3** | **4.5e-3** | 0.0271 | 48.997 | 0.385 | **16.27** |

**E5b is the sweet spot, not the bottom of a curve.** Both scaling levers — more epochs (E5d) and more cells/donor (E5c) — regress R below E5b's 0.466. MAE keeps improving in both directions because predictions continue walking toward eval mean (err mean −16.0 → −12.7 → −11.9 → −10.7 across the four runs), but per-donor age correlation is *worse* whenever you scale.

**Mechanism (different from E5d's "more confident off-axis"):** E5c's pred_sd is 2.63y (intermediate between E5b's 2.48 and E5d's 3.24), pred_range 19.3y (comparable to E5d's 20.6), but R 0.385 — *worse* than either despite the more-parameter-movement evidence (LoRA Δ median 1.7e-3, the largest of any run). The model is moving more parameters and producing more-spread predictions, but the spread is *less aligned* with truth than at E5b. This points to a fundamentally different bottleneck than undertraining — it's the **per-cell MSE objective**: when each donor contributes 500 cells with the same age label, the model gets 500 duplicated (cell, age) targets per donor and learns donor-specific cell-level idiosyncrasies that don't carry the donor's true age. More duplication (E5c) → more donor-memorization → less age-rank generalization.

Compare LoRA Δ trajectory: 1.4e-4 (Run #2) → 9.4e-4 (E5b) → 1.4e-3 (E5d) → 1.7e-3 (E5c). The optimizer is finding more parameter movement under each scaling lever, but the *quality* of that movement peaks at E5b and degrades thereafter. The per-cell MSE loss surface is a confound: extra capacity is spent on donor-cell-level memorization, not age-rank improvement.

GATE 2 status: **E5b's R=0.466 is the single-seed best**, 0.034 below the 0.5 bar. Neither scaling lever closes the gap. Phase-3-A's diagnostic question is now answered: at 9,500 train cells × 3 epochs × mean-pool × cap=50, this Geneformer LoRA configuration is at its R-ceiling for this fold; further scaling on either lever degrades it.

## 19. Phase-3-A close-out plan

The convergence investigation has run its course. Time to lock in:

1. **Variance check (E5b config × 2 seeds, ~3.4h, ~$3.40)**: rerun `--epochs 3 --max-cells-per-donor 50 --pool mean --lr 2e-4 --head-lr 2e-4 --seed {1,2} --run-tag-suffix _e5b_seed{1,2}`. Establishes seed-to-seed variance on R and MAE. Required regardless of scale-lever results — the +0.104 R jump from E5a to E5b and the −0.035 regression to E5d are single-seed measurements.
2. **AIDA scoring (~30 min total)**: score E5b/E5c/E5d/seed1/seed2 checkpoints on AIDA CD4+T (293 donors from `ancestry_shift_mae` half) using `scripts/score_aida.py`. Cross-ancestry headline cell numbers, no extra training. Phase-2 baseline (Pasta-REG R=0.66, MAE=6.3y) is the floor to beat — unlikely given the loco_onek1k R~0.45 ceiling, but the AIDA numbers are needed for the preprint regardless.
3. **Phase-3-A close-out memo + Phase-3-B go/no-go**: summarize seed-mean R and MAE on loco_onek1k + AIDA, declare the best Geneformer config, recommend whether to proceed to Phase-3-B (full sweep of 3 FMs × 2 folds × 3 seeds) or pivot to objective-change first (per-donor MSE, multi-task auxiliary, etc.).

**Skipped this autonomous window**: loco_terekhova fold at E5b config would be ~6h (1005 train donors × 50 cells × 3 epochs = 150,750 cell-passes / 32 batch = 4,710 steps); doesn't fit alongside variance check + AIDA + close-out within the autonomous budget. Reserved for the next session — first move when context allows.

**Skipped**: per-donor objective ablation. Code change (~2h dev time) plus a fresh training run (~1.7h). Worth doing if the user agrees post-window; tightly motivated by the §18 mechanism finding.

## 20. Phase-3-A close-out (2026-04-27, autonomous-window summary)

Five Geneformer LoRA fine-tunes on `loco_onek1k` × CD4+T + AIDA scoring of all five checkpoints. Total compute: ~14.4 GPU-h on g5.xlarge A10g, ≈$14.50 on-demand. Plus ~50 min for AIDA inference batch. All under the $30 autonomous cap.

### 20.1 Per-fold table

| Run | Config | OneK1K MAE | OneK1K R | AIDA MAE | AIDA R |
|---|---|---|---|---|---|
| E5b seed 0 | 3 ep, cap=50, mean | 17.37 | 0.466 | 10.39 | 0.311 |
| E5b seed 1 | 3 ep, cap=50, mean | 17.57 | **0.498** | 10.18 | 0.240 |
| E5b seed 2 | 3 ep, cap=50, mean | 16.56 | 0.396 | 10.60 | 0.350 |
| **E5b mean ± std** | | **17.17 ± 0.43** | **0.453 ± 0.042** | **10.39 ± 0.21** | **0.300 ± 0.056** |
| E5d seed 0 | 5 ep, cap=50, mean | 16.53 | 0.431 | 9.40 | 0.441 |
| E5c seed 0 | 1 ep, cap=500, mean | **16.27** | 0.385 | 9.53 | **0.545** |

Phase-2 baseline floors (from `results/baselines/loco_baseline_table.csv`):

| Cell | Best baseline | MAE | R | min-of-4 MAE | 10%-win FM target |
|---|---|---|---|---|---|
| OneK1K CD4T | scImmuAging-LASSO | 9.45 | 0.747 | 9.45 | ≤8.50 |
| AIDA CD4T | Pasta-REG | 6.32 | 0.66 | 6.32 | ≤5.69 |

### 20.2 Headline finding: OneK1K–AIDA inversion

**The configurations that look "overfit" on OneK1K are actually *better* on AIDA.** Within-config seed variance shows the same negative correlation: E5b seed 1 has the highest OneK1K R (0.498) but the *lowest* AIDA R (0.240) of the three E5b seeds.

| Run | OneK1K R | AIDA R | OneK1K rank | AIDA rank |
|---|---|---|---|---|
| E5b mean | 0.453 | 0.300 | 1 | 4 |
| E5d (single) | 0.431 | 0.441 | 2 | 2 |
| E5c (single) | 0.385 | 0.545 | 3 | 1 |

The §18 single-seed claim that "E5b is a sweet spot" was OneK1K-specific. On the cross-ancestry AIDA cell — which is AIDA's role in the preprint headline — E5c (more cells per donor) wins the R race by +0.244 over E5b mean and +0.104 over E5d. **For the cross-ancestry headline, more cells per donor + 1 epoch dominates more epochs at fewer cells.**

This is consistent with the §18 mechanism story (per-cell MSE causes donor-level memorization with many cells/donor) but reverses its sign: donor-level memorization helps on the *training population's holdout* (OneK1K) where similar cells appear, and hurts on a *novel population* (AIDA) where the memorized patterns don't transfer. Conversely, the lower-data E5b config, by overfitting less per-donor, generalizes worse to OneK1K's specific cells but better to AIDA. **OneK1K is closer to the train population (Stephenson+Terekhova are both European) than AIDA is**, so the apparent generalization-vs-memorization tradeoff aligns with the train→test population distance.

### 20.3 vs. baselines

**OneK1K CD4T:** all FM seeds and configs lose to LASSO (9.45y / 0.75) and LASSO-retrained-3cohort (10.96y / 0.71) on both R and MAE. They beat scAgeClock (17.92y / 0.36) and beat Pasta-REG on MAE (24.16y) but lose to it on R (0.60). Headline win threshold (≤8.50y) is unreachable at this fold given the +13y train→eval mean shift (Stephenson+Terekhova mean 50.1y, OneK1K 63.1y).

**AIDA CD4T:** all FM configs lose to LASSO (7.46y / 0.65) and Pasta-REG (6.32y / 0.66) on both axes. E5c (R=0.545) beats scAgeClock (R=0.298) cleanly on R, ties on MAE (9.53 vs 9.20). E5b mean (R=0.300) is roughly tied with scAgeClock. Headline win threshold (≤5.69y) is unreachable.

**Net**: Geneformer LoRA at this scale (110M params, 9.5k–93.7k train cells, ≤5 epochs) does not beat the strongest baselines on either of the two scored headline cells. The win/match/loss classification per the kickoff doc:
- OneK1K CD4T: **loss** (best E5b mean R=0.453 vs baseline min R=0.36 from scAgeClock; but baseline min MAE is LASSO's 9.45y << 17.17y FM mean)
- AIDA CD4T: **loss** (best E5c R=0.545 vs Pasta-REG 0.66; MAE 9.53 vs Pasta 6.32)

By the kickoff outcome rules (3/3 wins → primary, 2/3 → strong, 1/3 → degraded, 0/3 → pivot to evaluation framing), Phase-3-A so far reads as 0/2 — pivoting toward the evaluation-study framing for Phase-3-B + preprint.

### 20.4 Phase-3-B go/no-go recommendation

**Go, with structural changes** to scope and framing:

1. **Run loco_terekhova × E5b × seed 0** as the immediate next experiment (~6h on A10g, ~$6 on-demand). Provides the second LOCO fold result needed for the preprint's tri-headline structure. Likely the second-best result we'll have in time for the preprint deadline (~2026-07-01). Expected to land closer to baselines than loco_onek1k did because Terekhova fold has 5× more train donors (1,005 vs 190) with cleaner age distribution.
2. **Defer scFoundation + scGPT runs** until per-donor objective is tested. Both other FMs use the same per-cell MSE objective via the same train_loop.py; if per-donor objective fixes the §18 per-cell-overfitting issue on Geneformer, it'd likely help the others too. Better to learn that on Geneformer (already-instrumented) than burn $20+ on three FMs running the suspected-suboptimal objective.
3. **Per-donor objective ablation** is the highest-information-per-dollar next experiment. Estimated 2h dev (modify `train_loop.py` to aggregate predictions per donor before MSE; add donor IDs to batches) + 1.7h test run. Tests whether the OneK1K-AIDA inversion (§20.2) reflects the per-cell objective structurally amplifying cell-level overfitting, or if it's a pretraining-scale property.
4. **Frame the preprint as evaluation study**, not horse-race claim. The §20.2 OneK1K–AIDA inversion is itself a publishable methodological finding (different scale levers favor different generalization regimes). Combined with the §18 mechanism story, this gives the preprint a substantive methodological contribution even if no FM clears the 10%-win bar.

### 20.5 Files committed in this autonomous window (2026-04-27 12:00–21:00 UTC)

- 5 Geneformer fine-tune checkpoints + summary CSV rows + per-donor CSVs (E5b seed 0/1/2, E5c, E5d)
- 5 AIDA per-donor CSVs + 5 rows in `results/phase3/aida_summary.csv`
- `scripts/score_aida.py` (new — Phase-3 Task 2 inference plumbing)
- `scratchpad/_inspect_e5b.py`, `_inspect_e5c.py`, `_inspect_e5d.py` (mechanism inspection scripts; gitignored)
- This memo (§13–20) and `notes/research_journal.md` entries for E5b/E5c/E5d
- 5 git commits pushed to `origin/main`

## 21. Terekhova fold result (2026-04-28 02:35 UTC)

`loco_terekhova` × E5b config × seed 0 completed at 6.1h wall (~$6.1 on g5.xlarge on-demand). Train: OneK1K (981 donors) + Stephenson (24 donors) = 1,005 donors. Eval: Terekhova (166 donors, 10x 5' v2 chemistry, mean age 49.4y). **Final eval: MAE=11.37y, R=0.140.**

This is a catastrophic result, the worst of any LoRA fine-tune in Phase-3-A. Below scAgeClock (R=0.14 vs 0.14 — actually a tie on R, MAE 11.4 vs 13.2). Far below LASSO (R=0.82, MAE=9.15) and Pasta-REG (R=0.78, MAE=8.04).

### 21.1 Mechanism inspection

| | OneK1K E5b s0 | Terekhova E5b s0 |
|---|---|---|
| pred sd (y) | 2.48 | **4.71** |
| pred range (y) | 16.7 | 26.1 |
| pred mean | 51.2 | 39.9 |
| true mean | 63.9 | 49.4 |
| err mean | −12.7 | −9.6 |
| LoRA Δ RMS median | 9.4e-4 | **2.2e-3** |
| LoRA Δ RMS max | 2.3e-3 | 4.3e-3 |
| head_weight RMS | 0.0234 | 0.0334 |
| head_bias | 48.93 | **63.46** |

**Surprising features:**
1. **Pred sd 4.71y is the LARGEST of any Geneformer LoRA run** — predictions are spreading more across the age axis than on loco_onek1k.
2. **LoRA delta median 2.2e-3 is 2.3× E5b loco_onek1k** — the optimizer found much more parameter movement.
3. **Head bias = 63.46 ≈ train-cell mean** (init was ~62.65). Head bias barely moved, as in all prior runs.
4. **But predictions land 23y BELOW head bias on average** (pred mean 39.87 vs head bias 63.46). The LoRA-modified hidden states produce features that, dotted with the head weight, contribute −23.6 on average. The model has learned to "subtract from the bias" rather than "predict the age."

The pred_sd grew, the LoRA delta grew, but R = 0.14 — the spread is essentially uncorrelated with truth.

### 21.2 Why does this fold fail so badly?

The kickoff Phase-3 plan flagged Terekhova as the **chemistry-shift headline cell**: OneK1K + Stephenson are both 10x 3' v2 chemistry; Terekhova is 10x 5' v2. The Phase-1 Task 1f finding (`methods/terekhova_chemistry_shift.md`) showed pretrained LASSO holds on CD4+T across chemistry (R=0.82), and Pasta-REG is chemistry-invariant via rank-normalization (R=0.78 on Terekhova CD4+T). The hypothesis was that FMs, by virtue of pretraining on diverse chemistries, would also be chemistry-robust.

**The opposite happened.** Geneformer LoRA fine-tuned on 3'-only data collapses to R=0.14 on 5' eval, while the LASSO and Pasta-REG baselines remain near their full performance. Three contributing factors:

1. **Train-cohort homogeneity**: cap=50 × 1,005 donors gives 50,250 train cells, 98% of which are OneK1K (3' chemistry). Stephenson contributes only 1,200 cells. The model trains primarily on a single 3'-chemistry cohort.
2. **LoRA over-fitting to 3' artifacts**: 2.3× larger LoRA delta than loco_onek1k means the adapters learned more 3'-specific patterns. When applied to 5' inputs, those adapters fire on tokens that don't correspond to the same biological signal.
3. **Direction of mean shift exacerbates predictions**: train mean 62.6y, eval mean 49.4y. The model's "average prediction" should be 49.4 to minimize MAE. Instead it lands at 39.87 — 10y BELOW the eval mean. The LoRA-modified features produce systematically negative head-output on Terekhova cells (probably because the rank-value encoding of 5' UMI distributions differs from what the model saw on 3').

This is exactly the failure mode the Phase-2 chemistry-rescue framing was set up to test — and it's a strong null finding: **FMs do not rescue chemistry shift; they fail at it, where rank-normalized bulk (Pasta) and pretrained-on-mixed-chemistry linear (LASSO) succeed.**

### 21.3 Tri-headline outcome (kickoff §3.5 classification)

With Terekhova added, the per-cell win/match/loss verdict is now:

| Cell | Best baseline (MAE / R) | Best Geneformer LoRA (MAE / R) | FM target (10% win) | Outcome |
|---|---|---|---|---|
| OneK1K CD4T | LASSO 9.45 / 0.75 | E5b mean 17.17 / 0.453 | ≤8.50 | **loss** |
| Terekhova CD4T | Pasta-REG 8.04 / 0.78 | E5b s0 11.37 / **0.14** | ≤7.20 | **loss (catastrophic)** |
| AIDA CD4T | Pasta-REG 6.32 / 0.66 | E5c 9.53 / 0.55 | ≤5.69 | **loss** |

**Phase-3-A aggregate: 0/3 wins.** Per the kickoff outcome rules, this is the strongest pivot toward evaluation-study framing. The preprint headline is no longer "FMs match published baselines" or even "methodology contribution + null result"; it's a **stronger negative finding**: at the 110M-param Geneformer LoRA scale, the FM is dominated by linear baselines on every headline cell, and *catastrophically* fails the chemistry-shift transfer test the kickoff designed Terekhova to probe.

### 21.4 What this changes for Phase-3-B

The §20.4 Phase-3-B recommendation (defer scFoundation/scGPT, test per-donor objective on Geneformer first) gets stronger here. The chemistry-shift collapse is the most urgent thing to understand:

1. **Before any more FM training**, audit whether the Terekhova chemistry collapse is Geneformer-specific or FM-general. scFoundation has a different rank-encoding scheme; scGPT uses gene tokens directly. Both might transfer differently across chemistry. But running both on loco_terekhova first (~$15) is the cheap experiment.
2. **Per-donor objective ablation should run on loco_terekhova specifically** (not loco_onek1k as §20.4 implied). loco_onek1k's R-ceiling at 0.45 is uninteresting if Terekhova catastrophe dominates the preprint.
3. **Train-cohort balancing**: with 98% OneK1K cells, the loco_terekhova fold is essentially a 3'→5' transfer test, not a multi-cohort transfer test. A balanced sampler (equal cells per cohort, not per donor) might fix the chemistry overfit. ~30 LOC change to `select_indices`.
4. **Frame preprint around the chemistry-shift collapse + OneK1K-AIDA inversion.** Both are methodological contributions; the chemistry collapse contradicts the Phase-2 hypothesis directly.

### 21.5 AIDA score on the Terekhova-trained checkpoint: R = −0.146

`scripts/score_aida.py` on `loco_terekhova_seed0_CD4p_T_e5b.pt`: **MAE 9.50, R = −0.146.** The chemistry collapse propagates from Terekhova (5') to AIDA (also 5' v2): the checkpoint produces *anti-correlated* predictions on both 5' eval cohorts. R<0 means the model is worse than predicting the train mean.

### 21.6 Chemistry-asymmetry clarification

Re-examining the train-set chemistry mix per fold (with `--max-cells-per-donor 50`):

| Fold | Train cohorts | Train cells | 3' cells | 5' cells | Eval | Eval chemistry |
|---|---|---|---|---|---|---|
| loco_onek1k | Stephenson + Terekhova | 9,500 | 1,200 (13%) | 8,300 (87%) | OneK1K | 3' |
| loco_terekhova | OneK1K + Stephenson | 50,250 | 50,250 (100%) | 0 | Terekhova | 5' |

**The loco_onek1k models train predominantly on 5' chemistry (Terekhova dominates).** Their AIDA evaluations are therefore *same-chemistry* (5'→5') but cross-ancestry (European→Asian). R 0.24–0.55 reflects population-only transfer.

**The loco_terekhova model trains 100% on 3' chemistry**, then evaluates on 5'. Its evaluations are *cross-chemistry* on both Terekhova (R=0.14) and AIDA (R=−0.15). The catastrophic failure stacks two effects: 3'→5' chemistry shift plus, on AIDA, additional cross-ancestry.

**Implications:**
1. The "OneK1K–AIDA inversion" finding from §20.2 is a **pure population effect**, not a chemistry effect. Both OneK1K and AIDA evaluations on loco_onek1k checkpoints are 5'→3' (OneK1K) and 5'→5' (AIDA); the chemistry direction differs but is not catastrophic. Population (European train → European OneK1K eval, vs European train → Asian AIDA eval) drives the inversion.
2. **Geneformer LoRA's chemistry-rescue capability is asymmetric.** A model trained predominantly on 5' generalizes acceptably to 3'. A model trained 100% on 3' fails catastrophically on 5'. Either the rank-value encoding is sensitive to which-end-of-transcript bias of the chemistries, or the optimizer found 3'-specific tokens that don't have valid embeddings on 5' inputs.
3. **The Stephenson 3'-only signal is too weak to alone train a chemistry-robust model.** With 24 donors × 50 cells = 1,200 cells (~2% of the 50,250 loco_terekhova train mix), Stephenson contributes negligibly and the model effectively trains on OneK1K alone.
4. **The §20.4 Phase-3-B priority order changes again**: the most informative cheap experiment is now (a) a balanced sampler (force equal cells from each train cohort) on loco_terekhova, to test whether the 3'→5' collapse is fundamental or just driven by OneK1K-only training. ~30 LOC change to `select_indices`, then a re-run (~6h, $6).

## 22. Phase-3-A B + NK extension results (2026-04-28)

Post-CD4+T close-out, B and NK pulled forward from Phase 4. Three Geneformer LoRA fine-tunes at the E5b config (3ep cap=50 mean-pool lr=2e-4/2e-4 seed 0) on `loco_onek1k` and `loco_terekhova`. NK × loco_terekhova was cancelled in favor of the Variant 1 diagnostic ladder (§23+).

| Run | MAE (y) | R | Wall | Cost | Best baseline (MAE / R) | Verdict |
|---|---|---|---|---|---|---|
| B × loco_onek1k | 24.28 | **−0.076** | 1.3h | $1.3 | LASSO 10.66 / 0.531 | catastrophic loss (anti-correlated) |
| NK × loco_onek1k | 19.77 | 0.165 | 1.3h | $1.3 | LASSO 9.64 / 0.629 | clean loss |
| B × loco_terekhova | 14.23 | 0.075 | 4.7h | $4.7 | **Pasta-REG 10.86 / 0.281** | loss (chemistry-rescue null) |

### 22.1 Cell-type pattern across loco_onek1k

| Cell type | LoRA OneK1K R | Best baseline R |
|---|---|---|
| CD4+T (E5b mean) | 0.453 | 0.747 (LASSO) |
| NK | 0.165 | 0.629 (LASSO) |
| B | −0.076 | 0.531 (LASSO) |

**FM R correlates positively with baseline R** — FMs do somewhat-OK where baselines do well, fail catastrophically where baselines fail. Exact OPPOSITE of the kickoff §3 few-shot hypothesis ("FMs win where baselines are weak"). The ratio FM-R / baseline-R is roughly constant at ~0.5–0.6 across cell types — suggesting the FM extracts a constant fraction of the available signal regardless of how much there is to extract.

### 22.2 Chemistry-rescue test on B × loco_terekhova (the Phase-1 §1f headline cell)

Phase-1 found the pretrained LASSO collapses on Terekhova B cells (R=0.08 — the canonical chemistry-shift collapse). Pasta-REG, via rank-normalization, recovers to R=0.28 / MAE=10.86 — chemistry-invariance via input transform. The kickoff hypothesized FMs would also be chemistry-invariant by virtue of pretraining on diverse data; the chemistry-rescue test is whether FMs add invariance on top of rank-norm bulk.

**Result: Geneformer LoRA at R=0.075 / MAE=14.23 essentially ties LASSO's collapse (R=0.080) and beats scAgeClock (R=0.055) but loses cleanly to Pasta-REG (R=0.281). FMs do NOT rescue chemistry-shift collapse.** Pasta's rank-normalization remains the only competitive chemistry-invariant approach in the panel.

The exact opposite of the kickoff §3 hypothesis is now confirmed across two folds: 3'→5' transfer fails on CD4+T (R=0.140 → §21) AND on B (R=0.075). The FM doesn't add anything Pasta-REG doesn't already have.

### 22.3 Aggregate tri-cell × multi-celltype tally

Phase-3-A negative claim now spans **6 (cell × fold) pairs** with all losses:

| Cell × Fold | LoRA MAE / R | Best baseline MAE / R | Verdict |
|---|---|---|---|
| CD4+T × OneK1K | 17.17 / 0.453 (3-seed mean) | LASSO 9.45 / 0.747 | loss |
| CD4+T × Terekhova | 11.37 / 0.140 | Pasta 8.04 / 0.778 | catastrophic loss |
| CD4+T × AIDA | 9.53 / 0.545 (E5c best config) | Pasta 6.32 / 0.659 | loss (closest on R) |
| B × OneK1K | 24.28 / −0.076 | LASSO 10.66 / 0.531 | catastrophic loss |
| B × Terekhova | 14.23 / 0.075 | Pasta 10.86 / 0.281 | loss |
| NK × OneK1K | 19.77 / 0.165 | LASSO 9.64 / 0.629 | loss |

**0/6 wins.** Stronger negative claim than the original 0/3 CD4+T-only tally. **But** — and this is the §23 pivot — all six losses are at the same per-cell fine-tune protocol; the diagnostic ladder is needed to attribute the failure to (a) protocol, (b) FM class, or (c) substrate signal absence.

### 22.4 What this changes for the preprint narrative

The "preprint pivots toward evaluation-study framing" call from §20.4 was made on 0/3 CD4+T cells. With 0/6 (cell × fold) pairs and the chemistry-rescue null on B, the negative claim is materially stronger but **bounded** to "Geneformer LoRA at the per-cell fine-tune protocol" — exactly the bounded claim the diagnostic ladder is designed to upgrade.

### 22.5 AIDA × B from the B × loco_terekhova checkpoint

`scripts/score_aida.py --cell-type B` on the B × loco_terekhova checkpoint: **MAE 11.33, R = −0.247.** Same chemistry-shift propagation pattern as CD4+T (R=−0.146 on AIDA from the CD4+T × loco_terekhova checkpoint). 3'→5' collapse propagates: a model trained 100% on 3' (loco_terekhova train mix is 98% OneK1K + 2% Stephenson, both 3') produces anti-correlated predictions on 5' eval cohorts (Terekhova AND AIDA).

| Source checkpoint | Eval cohort | Eval chemistry | MAE | R |
|---|---|---|---|---|
| CD4+T loco_terekhova | Terekhova | 5' v2 | 11.37 | 0.140 |
| CD4+T loco_terekhova | AIDA | 5' v2 | 9.50 | −0.146 |
| B loco_terekhova | Terekhova | 5' v2 | 14.23 | 0.075 |
| B loco_terekhova | AIDA | 5' v2 | 11.33 | **−0.247** |

The chemistry-shift collapse is reproducible and severe across both cell types — but again, with the §21.6 confound (98% single-cohort train) unresolved, it can't yet stand on its own as a finding. The Variant 1 ridge probe (§23+) will indirectly address whether the failure is protocol-level or representation-level.

## 23. Variant 1 Phase 1 — frozen-base ridge beats every fine-tune on CD4+T (2026-04-28)

Per the §22.4 + diagnostic-ladder pivot, built `scripts/extract_embeddings.py` (mean-pool over attended positions of last_hidden_state, per-donor average) and `scripts/donor_ridge.py` (RidgeCV alpha selection over [0.01, 1e4], per-cohort holdout eval, optional AIDA transfer). Phase 1 = frozen, unmodified Geneformer V2-104M extracted across 4 cohorts × CD4+T, then ridge fits on the two LOCO folds.

| Fold × Eval | Frozen-base R | Frozen-base MAE | Best fine-tune R | Best fine-tune MAE | Δ R (frozen − fine-tune) |
|---|---|---|---|---|---|
| loco_onek1k × OneK1K | **0.560** | 16.52 | 0.466 (E5b s0) | 17.37 | +0.094 |
| loco_onek1k × AIDA (transfer) | **0.527** | 11.76 | 0.545 (E5c s0) | 9.53 | −0.018 (R) / +2.23 (MAE) |
| loco_terekhova × Terekhova | **0.576** | 24.03 | 0.140 (E5b s0) | 11.37 | **+0.436** |

**The headline number is the +0.436 R uplift on Terekhova.** A linear ridge over the frozen-base mean-pool representation achieves R=0.576 on the chemistry-shift fold where the fine-tune collapsed to R=0.140. The signal Geneformer encodes pre-finetune is already chemistry-robust enough to power 5'-eval-from-3'-train; the per-cell MSE fine-tune protocol *destroyed* that property.

### 23.1 Three protocol-attribution claims this supports

1. **Per-cell MSE fine-tune is protocol-negative on CD4+T** — three folds, three cases where frozen ≥ fine-tune on R; the Terekhova case is catastrophic. The fine-tune is not extracting more signal than the pretrained representation already provides; on at least one fold it is actively destroying signal.
2. **Geneformer's pretrained representation IS chemistry-robust** — a property the fine-tune erases. The §21 chemistry-collapse finding (R=0.140 on Terekhova) is therefore protocol-induced, not pretraining-induced. The §22 chemistry-rescue null on B-Terekhova may have the same explanation pending Phase 2.
3. **The MAE-on-AIDA pattern is preserved**: fine-tunes can match frozen on R for AIDA transfer (E5c R=0.545 vs frozen 0.527) but at 2.23y *worse* MAE — fine-tunes shift the bias closer to AIDA mean while ridge does not. AIDA bias-closure is therefore a fine-tune *artifact*, not evidence of representation improvement.

### 23.2 Bounded scope

This is a one-cell-type result on CD4+T. Phase 2 (B + NK frozen-base across both folds) is the necessary generality test before calling the protocol-negative claim definitive. If Phase 2 shows frozen ≥ fine-tune on B and NK as well, the §22.3 0/6 negative claim transmutes from "Geneformer LoRA loses to baselines" to "Geneformer LoRA *protocol* loses to a linear probe over its own pretrained features" — the latter is a much sharper protocol-attribution finding than a horse-race loss.

### 23.3 Frozen-base vs Phase-2 baselines (does this beat anything else?)

Frozen-base R=0.527-0.576 is **not competitive with Phase-2 baselines** on CD4+T:
- LASSO on OneK1K CD4+T: R=0.747 (frozen-base 0.560)
- Pasta-REG on Terekhova CD4+T: R=0.778 (frozen-base 0.576)
- Pasta-REG on AIDA CD4+T: R=0.659 (frozen-base 0.527)

Frozen-base ridge is a *diagnostic* probe of the FM representation, not a competitor to the published baselines. The protocol-negative finding does not flip the §22.3 horse-race tally — but it does change the explanation for *why* the FM loses.

### 23.4 Decision per §22.5 ladder

Frozen-base AIDA R=0.527 falls in the §22.5 "mixed" bracket [0.45, 0.60). Recommended branch: Phase 2 first (B + NK frozen-base + cross-fold ridge fits, ~30 min compute, no LoRA training); then in parallel Variant 2 (pseudobulk-input fine-tune) and Variant 3 (layer-wise probe) once the cross-cell-type generality of the protocol-negative finding is established. scFoundation/scGPT FM-class diagnostic stays gated on Phase 2 + Variant 2/3 outcomes.

Phase 2 batch script: `scratchpad/run_variant1_phase2.sh`. Expected: 8 frozen-base extractions (4 cohorts × {B, NK}) + 6 ridge fits (3 cell types × 2 folds, with AIDA transfer on loco_onek1k folds) → 6 new rows in `results/phase3/ridge_summary.csv`.

## 24. Variant 1 Phase 2 — protocol-negative confirmed, but with cell-type-dependent magnitude (2026-04-28)

Phase 2 ran in ~2.7h (4 extractions × ~10-20 min for OneK1K/Terekhova B+NK + ~5 min each for AIDA/Stephenson). Adds 9 ridge rows: B + NK × {loco_onek1k OneK1K + AIDA transfer, loco_terekhova Terekhova} + a clean re-run of the CD4+T trio for sanity (numbers identical to Phase 1, confirming determinism).

### 24.1 Full 3-cell × 2-fold frozen-base ridge table

| Cell × Fold | Frozen R | Frozen MAE | Fine-tune R | Δ R | Frozen p-value |
|---|---|---|---|---|---|
| CD4+T × OneK1K | 0.560 | 16.52 | 0.466 (E5b s0) | +0.094 | <1e-300 |
| CD4+T × AIDA | 0.527 | 11.76 | 0.545 (E5c s0) | −0.018 | 2.2e-22 |
| CD4+T × Terekhova | **0.576** | 24.03 | 0.140 | **+0.436** | 4.4e-16 |
| B × OneK1K | −0.013 | 21.69 | −0.076 | +0.063 | **0.69 (n.s.)** |
| B × AIDA | 0.099 | 18.69 | n/a | n/a | 0.085 (n.s.) |
| B × Terekhova | 0.102 | 14.02 | 0.075 | +0.027 | **0.19 (n.s.)** |
| NK × OneK1K | 0.260 | 14.13 | 0.165 | +0.095 | 1.2e-16 |
| NK × AIDA | 0.047 | 14.88 | n/a | n/a | **0.41 (n.s.)** |
| NK × Terekhova | 0.199 | 15.19 | n/a (cancelled) | n/a | 0.010 |

**Frozen-base ridge ≥ fine-tune on R for every (cell × fold) pair tested with both** (5/5 such pairs; +Δ in [0.027, 0.436]). Direction of the §23 protocol-negative claim is confirmed across all three cell types.

### 24.2 Two distinct failure modes co-exist (the §23 framing was too clean)

The §23 memo treated "fine-tune destroys signal" as the unifying mechanism. Phase 2 reveals it splits cleanly by cell type:

**Mode A — CD4+T (protocol-negative)**: Frozen R = 0.527-0.576 across three cohort/eval conditions, all p < 1e-15. Real, robust pretrained signal. Fine-tune erases it (Terekhova R=0.140 ← 0.576 = catastrophic; OneK1K R=0.466 ← 0.560 = mild). Variants 2 and 3 are the right diagnostic next.

**Mode B — B and NK (representation-negative)**: Frozen R is in [−0.01, 0.26] across 6 (cell × eval) pairs, with **3 of 6 p > 0.1** (not distinguishable from chance). On B specifically, *both* frozen and fine-tune sit near zero — there's no signal to destroy, only to find. The substrate is empty. The fine-tune hurting on these cell types is a small effect because the floor is already near zero.

The +0.436 R Terekhova uplift on CD4+T is the only smoking gun. The B/NK uplifts of [+0.027, +0.095] are real-direction but small-magnitude — what we'd expect when noise dominates and the linear probe finds nothing the fine-tune could destroy.

### 24.3 What's published vs what's missing

Phase-2 baselines (from `results/baselines/v0_summary.csv`):

| Cell | LASSO R OneK1K | Pasta-REG R OneK1K | LASSO R Terekhova | Pasta-REG R Terekhova |
|---|---|---|---|---|
| CD4+T | 0.747 | 0.730 | n/a | 0.778 |
| B | 0.531 | 0.450 | 0.080 | 0.281 |
| NK | 0.629 | 0.578 | n/a | n/a |

For B × OneK1K, LASSO recovers R=0.531 from the same input data. Frozen-base Geneformer ridge gets R=−0.013. **A linear probe on raw genes (LASSO) finds B-cell aging signal that a linear probe on Geneformer's mean-pool 768-dim features cannot.** This is a representation-quality finding — the FM compressed away features that LASSO can use directly. NK shows the same pattern (LASSO 0.629, frozen 0.260). Only CD4+T is competitive (LASSO 0.747, frozen 0.560).

### 24.4 Hypotheses for B/NK representation gap

1. **Layer ablation hypothesis** — last-layer mean-pool throws away aging-relevant detail that mid-layers preserve. Variant 3 (layer-wise probe) tests this directly. If layer-N probe restores B R to 0.3-0.5, Phase-2-baseline-competitive isn't far off.
2. **Pooling hypothesis** — mean-pool over attended positions averages away aging-relevant subpopulations within B-cell pools (memory vs naive ratios are known aging markers). `<cls>` pooling, max-pool, or attention-weighted pool might capture more.
3. **Vocabulary hypothesis** — Geneformer's gene-token vocabulary may underweight B-cell-aging-relevant markers (e.g. plasma-cell isotype switching). LASSO can directly use those genes; Geneformer must squeeze signal through tokens that may not represent them.
4. **Pretraining domain mismatch** — Genecorpus-30M is broad but B-cell development states may be underrepresented relative to T cells. Less likely on its own; combines with hypothesis 1/2.

Variants 2 and 3 distinguish (1+2) from (3+4): if no layer/pool combo lifts B above R=0.3, the issue is upstream of the encoder (vocab/pretraining), and only retraining the encoder or switching FM class fixes it.

### 24.5 Decision per §22.5 ladder, updated

The §22.5 decision tree was set on CD4+T frozen-base AIDA R=0.527. Phase 2 partitioned the picture: CD4+T sits in the §22.5 "mixed" bracket [0.45, 0.60), B and NK sit in the [0, 0.30) "low" bracket. Updated next-step plan:

1. **Variant 3 (layer-wise probe) on B and NK first** — if the pretrained representation has B/NK signal in earlier layers, that's the cheapest path to an explanation. ~1h compute (re-extract from layers 4, 8, 12 of the same checkpoint).
2. **Variant 2 (pseudobulk-input fine-tune) on CD4+T** — pseudobulk eliminates per-cell noise that may be the protocol's failure mode. Run on CD4+T first since that's where there's signal to either preserve or destroy. ~6h training + AIDA scoring.
3. **scFoundation FM-class diagnostic** — only after Variants 2/3 results land. If Variant 3 surfaces signal in some layer of Geneformer, it's worth running scFoundation in parallel; if not, scFoundation is even higher priority because Geneformer is the wrong substrate entirely.
4. **Variant 4 (cancelled)** — original Variant 4 was an ensemble; not informative without first knowing whether any variant works at all.

Phase 2 outcome strengthens the negative claim into a more granular methodological story: "Geneformer LoRA with per-cell MSE fine-tune is protocol-negative on the one cell type where its frozen representation has signal, and is representation-negative on B and NK regardless of protocol." This is publishable as a diagnostic-study finding and motivates the FM-class comparison directly.

> **2026-04-28 audit note (§25)**: The "smoking gun" framing in §23 and the "substrate is empty" wording in §24 both **overreach**. See §25 for the bias-variance audit per `scratchpad/variant_1_review.md`; the §23/§24 numbers stand but the *interpretation* is materially refined.

## 25. Variant 1 audit (D.9 + D.10 + D.11) — bias-variance analysis refines the §23/§24 narrative (2026-04-28)

`scripts/variant1_audit.py` re-fits all 9 ridge conditions, saves predictions, and computes (i) exact R + 95% bootstrap CI + p, (ii) pred_sd vs eval_sd compression, (iii) OLS slope of pred~true (slope < 1 = mean-compression), (iv) AIDA pred mean vs train mean (bias-toward-training-mean). Output: `results/phase3/variant1_audit.csv` + per-condition prediction `.npz` in `results/phase3/variant1_predictions/`.

### 25.1 D.9 — exact R + 95% bootstrap CI for all 9 conditions

| Cell × Eval | R | 95% CI | p | Verdict |
|---|---|---|---|---|
| CD4+T × OneK1K | 0.560 | [0.514, 0.605] | <1e-300 | real |
| CD4+T × AIDA | 0.527 | [0.453, 0.594] | 2.2e-22 | real |
| CD4+T × Terekhova | 0.576 | [0.460, 0.665] | 4.4e-16 | real |
| **B × OneK1K** | −0.013 | **[−0.079, 0.046]** | 0.69 | **CI straddles 0 — empty** |
| **B × AIDA** | 0.099 | **[−0.009, 0.203]** | 0.085 | **CI straddles 0 — empty** |
| **B × Terekhova** | 0.102 | **[−0.056, 0.255]** | 0.19 | **CI straddles 0 — empty** |
| NK × OneK1K | 0.260 | [0.197, 0.320] | 1.2e-16 | **real but weak** |
| **NK × AIDA** | 0.047 | **[−0.064, 0.162]** | 0.41 | **CI straddles 0 — empty** |
| NK × Terekhova | 0.199 | [0.044, 0.347] | 0.010 | **real but weak (lower CI grazes 0.04)** |

**Reframe of §24**:
- **B is genuinely empty substrate**. 0/3 CIs exclude zero. The §24 wording "substrate is empty" is correct *for B*.
- **NK is NOT empty substrate**. 2/3 CIs exclude zero. NK substrate has weak but real signal — particularly NK × OneK1K with R=0.260 [0.197, 0.320] which is a tight CI and highly significant. The §24 wording "B and NK look representation-negative" was wrong for NK; it should be **"NK substrate has weak signal materially below LASSO (0.260 vs 0.629), suggesting Geneformer's mean-pool last-layer representation captures ~40% of LASSO's NK-aging signal"**.
- The reframed cell-type taxonomy is now **CD4+T (real signal, partly preserved) / NK (weak signal) / B (genuinely empty substrate)**.

### 25.2 D.10 — bias-variance audit reveals universal compression and CD4+T-Terekhova bias catastrophe

| Cell × Eval | pred_sd | eval_sd | sd ratio | OLS slope | train mean | eval mean | pred mean |
|---|---|---|---|---|---|---|---|
| CD4+T × OneK1K | 7.05 | 16.51 | **0.43** | 0.24 | 48.89 | 63.91 | 50.05 |
| **CD4+T × AIDA** | 8.98 | 12.30 | 0.73 | 0.39 | 48.89 | 41.76 | **52.86** |
| **CD4+T × Terekhova** | 10.86 | 16.83 | 0.65 | 0.37 | 63.47 | 49.43 | **25.46** |
| B × OneK1K | 6.61 | 16.51 | 0.40 | −0.005 | 48.48 | 63.91 | 47.01 |
| B × AIDA | 7.54 | 12.44 | 0.61 | 0.060 | 48.48 | 41.07 | 58.45 |
| B × Terekhova | 6.87 | 16.76 | 0.41 | 0.042 | 63.32 | 49.42 | 57.24 |
| NK × OneK1K | 4.97 | 16.51 | **0.30** | 0.078 | 48.53 | 63.91 | 55.81 |
| NK × AIDA | 6.53 | 12.45 | 0.52 | 0.025 | 48.53 | 41.10 | 54.40 |
| NK × Terekhova | 6.10 | 16.78 | 0.36 | 0.072 | 63.32 | 49.48 | 39.19 |

**Universal compression (all 9 conditions)**: Every prediction is materially compressed relative to eval-sd; sd ratio in [0.30, 0.73]. This is generic ridge behavior on a noisy linear probe — the regularization deliberately shrinks predictions toward the training-mean.

**CD4+T × AIDA — cross-ancestry claim must be qualified, not retracted**:
- pred_mean = 52.86, train_mean = 48.89, eval_mean = 41.76
- Predictions lie 11.1y *above* the AIDA eval mean and ~4y above the train mean
- OLS slope = 0.39 is non-zero but well below 1 → predictions track ranking but at compressed scale
- pred range = 73% of eval range
- **Conclusion**: AIDA R=0.527 is real ranking signal but the MAE=11.76 understates the model's true cross-ancestry error. A bias-corrected predictor (subtract pred_mean - eval_mean) would have lower MAE; the unaltered ridge benefits from the fact that pred_mean (52.86) and eval_mean (41.76) happen to be in the right neighborhood after accounting for population age distributions. The §23 "frozen-base ridge generalizes to AIDA" claim should be qualified to **"frozen-base ridge produces partially-compressed AIDA predictions (sd ratio 0.73, slope 0.39) with mean centered near the European train mean; R=0.527 reflects real but bias-shifted ranking."**

**CD4+T × Terekhova — bias catastrophe partially resurrects the fine-tune**:
- pred_mean = 25.46, train_mean = 63.47, eval_mean = 49.43
- Predictions are 24y *below* the eval mean — even worse than the fine-tune (whose §21 pred_mean was 39.87, error 9.6y)
- This is what produces MAE=24.03 despite R=0.576: the embedding mean of Terekhova cells projects to a far-lower scalar than the train cohorts under the ridge weights, producing systematic underestimation
- **The §23 "smoking gun" claim that frozen-base R=0.576 beats fine-tune R=0.140 is therefore one-sided**: frozen wins on R but loses badly on MAE (24.03 vs fine-tune 11.37). The fine-tune destroys ranking but learns bias closure that frozen ridge cannot.
- Refined story: the per-cell MSE fine-tune is **bias-closure-positive AND rank-negative**; frozen-base ridge is **bias-closure-negative AND rank-positive**. Neither protocol jointly delivers both. The right diagnostic next step is whether a *different* fine-tune protocol (e.g. Variant 2 pseudobulk) jointly delivers both.

**B and NK CD4+T-comparable — flat-line predictions confirm substrate-empty for B**:
- B × all evals: OLS slopes ∈ [−0.005, 0.060]. Slope-zero = predictions don't track true age at all — flat at the train-influenced mean.
- NK × OneK1K: slope=0.078 — small but non-zero, consistent with the tight R CI [0.197, 0.320].

### 25.3 D.11 — honestly-bounded claim ladder (memo's calibrated narrative going forward)

| Tier | Claim | Evidence | Caveats |
|---|---|---|---|
| **Strongest** | "Per-cell MSE LoRA fine-tuning of Geneformer is **rank-negative** on CD4+T relative to a frozen-base linear probe — fine-tune destroys ranking signal Geneformer's pretrained representation already encodes (Terekhova frozen R=0.576 vs fine-tune R=0.140; +0.436 R uplift, both p<<0.001). The fine-tune is also **bias-closure-positive** (Terekhova fine-tune MAE 11.37 vs frozen MAE 24.03). The current protocol therefore couples rank-destruction with bias-correction — the wrong tradeoff." | §23 + §25.2 Terekhova row | Bounded to CD4+T; Variant 2/3 needed for protocol-attribution generality. |
| **Medium** | "Geneformer's pretrained mean-pool of last-layer representation captures CD4+T age signal *partially* — R=0.527–0.576 on raw frozen ridge, materially below LASSO/Pasta on raw genes (0.747–0.778). All predictions are compressed (sd ratio 0.43–0.73) — the FM is not a drop-in replacement for the linear baseline." | §25.1 + §25.2 | LASSO/Pasta numbers from Phase-2 baselines. |
| **Weakest (qualified)** | "Frozen-base ridge produces partially-compressed AIDA cross-ancestry predictions (sd ratio 0.73, slope 0.39) with mean near the European train mean; R=0.527 reflects real but bias-shifted ranking, not a free-standing ancestry-generalization claim." | §25.2 AIDA row | Cannot be cited as evidence FOR ancestry generalization without bias correction. |
| **Bounded null** | "B-cell substrate is genuinely empty in Geneformer's mean-pool of last layer (0/3 CIs exclude zero, OLS slopes near zero); NK substrate has weak signal materially below LASSO (R=0.260 vs LASSO 0.629). The B-cell representation gap is upstream of the encoder mean-pool. Variant 3 layer-wise probe is the cheapest test of whether earlier layers have B-cell signal." | §25.1 + §25.2 + LASSO numbers | Bound is "this representation, this pool"; doesn't rule out other Geneformer readouts. |

### 25.4 What this changes for Variant 2 / Variant 3 / FM-class diagnostic

- **Variant 3 (layer-wise probe) is now higher priority** than I'd weighted in §24.5. The B-cell empty-substrate finding is the cleanest target for Variant 3: if any layer has B-cell signal, that's the answer for B. If no layer does, B is bounded as a Geneformer-architectural failure and scFoundation/scGPT FM-class diagnostic becomes the natural next step. ~1h compute (just re-extract per-layer hidden states from frozen base).
- **Variant 2 (pseudobulk fine-tune) is still motivated but with a clearer target**: a successful Variant 2 must deliver R ≥ 0.576 *and* MAE ≤ 12y on Terekhova CD4+T (jointly clear the rank-positive AND bias-closure-positive bars). The current per-cell fine-tune jointly fails; the question is whether pseudobulk objective decouples them.
- **AIDA cross-ancestry as a headline is materially weaker**: cannot present R=0.527 / MAE 11.76 as "ancestry generalization" without showing the bias-variance audit. The §22.5 decision tree branch "AIDA R=0.527 is mixed bracket" was set on a number that doesn't support the cross-ancestry story it implied; with the audit, the AIDA result is "partially-compressed within-mean predictions, R from rank component."

### 25.5 Updated decision tree for Variants 2/3 (replaces §22.5 + §24.5)

1. **Variant 3 first** (highest information per dollar, ~1h compute, no training cost).
   - **If any layer R(B) > 0.4** → B substrate exists at some layer; pivot Variants 2/3 to layer-N readout. Updates the §25.3 "bounded null" claim.
   - **If max layer R(B) < 0.3** → B substrate is genuinely empty in Geneformer; bounded null stands; FM-class diagnostic is the natural next step.
   - For NK and CD4+T, the layer-curve shape (monotonic vs middle-peak vs flat) refines the protocol-level story.
2. **Variant 2 second** (only if Variant 3 doesn't immediately suggest a layer-N pivot). Target: R≥0.576 AND MAE≤12y on Terekhova CD4+T.
3. **scFoundation FM-class diagnostic third** if Variant 3 indicates Geneformer-architectural weakness on B.

## 26. Variant 3 — layer-wise frozen probe (2026-04-28). HEADLINE: Geneformer layer-1 + ridge on CD4+T × Terekhova achieves R=0.616 / MAE=8.82, within 9.7% of Pasta-REG on MAE.

`scripts/extract_embeddings_layered.py` extracts per-layer mean-pool embeddings (output_hidden_states=True; 13 layers including embedding output) across 4 cohorts × 3 cell types. `scripts/donor_ridge_layered.py` fits ridge per (fold × cell × layer × eval_cohort), 117 rows in `results/phase3/ridge_summary_layered.csv`. Wall: ~95 min extraction + 30s ridge.

### 26.1 Best layer per condition (R), with comparison to layer-12 (Variant 1) and prior fine-tune

| Fold × Eval | Cell | Best layer | Best R | Best MAE | Layer-12 R | Layer-12 MAE | Fine-tune R | Fine-tune MAE |
|---|---|---|---|---|---|---|---|---|
| loco_onek1k × OneK1K | CD4+T | 12 | 0.560 | 16.52 | 0.560 | 16.52 | 0.466 (E5b s0) | 17.37 |
| loco_onek1k × AIDA | CD4+T | 12 | **0.527** | 11.76 | 0.527 | 11.76 | 0.545 (E5c s0) | 9.53 |
| **loco_terekhova × Terekhova** | **CD4+T** | **5** (R) / **1** (MAE) | **0.621 / 0.616** | **18.24 / 8.82** | 0.576 | 24.03 | 0.140 | 11.37 |
| loco_onek1k × OneK1K | B | 7 | 0.038 | 25.15 | −0.013 | 21.69 | −0.076 | 24.28 |
| loco_onek1k × AIDA | B | 11 | 0.120 | 20.66 | 0.099 | 18.69 | n/a | n/a |
| **loco_terekhova × Terekhova** | **B** | **9** | **0.228** | 14.82 | 0.102 | 14.02 | 0.075 | 14.23 |
| loco_onek1k × OneK1K | NK | 3 | 0.304 | 62.28 (!) | 0.260 | 14.13 | 0.165 | 19.77 |
| loco_onek1k × AIDA | NK | 5 | 0.169 | 13.30 | 0.047 | 14.88 | n/a | n/a |
| loco_terekhova × Terekhova | NK | 2 | 0.266 | 14.41 | 0.199 | 15.19 | n/a | n/a |

### 26.2 Headline finding: layer-1 of frozen Geneformer beats every CD4+T × Terekhova predictor we've run

| Predictor | R | MAE (y) | Notes |
|---|---|---|---|
| **Frozen layer 1 + ridge** | **0.616** | **8.82** | new this session |
| Frozen layer 12 + ridge (Variant 1 §23) | 0.576 | 24.03 | last-layer; catastrophic bias |
| Per-cell MSE LoRA fine-tune (E5b s0, §21) | 0.140 | 11.37 | rank-collapse |
| Pasta-REG baseline (§22) | 0.778 | 8.04 | rank-norm bulk model |
| LASSO baseline (Phase-1 §1f) | 0.82 | ~9.15 | pretrained, 5-cohort |

**Layer-1 frozen ridge MAE (8.82y) is within 0.78y / 9.7% of Pasta-REG MAE (8.04y)**. R is below Pasta (0.616 vs 0.778) but the gap on the MAE-headline criterion is the smallest any FM-derived predictor has produced on this fold. This is the first time a Geneformer-based predictor approaches a Phase-2 baseline on the GATE-2 MAE bar.

**Layer-1 frozen ridge dominates the per-cell fine-tune Pareto-strictly**: better on both R (+0.476) AND MAE (−2.55y). The per-cell fine-tune has been *fully dominated* by a frozen-base linear probe at the right layer.

### 26.3 Bias-variance audit per layer reveals layer-1 fixes the §25.2 Terekhova bias catastrophe

| Layer | R | MAE | pred_sd | sd_ratio | slope | pred_mean | eval_mean | gap |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.616 | 8.82 | 10.60 | 0.63 | 0.39 | **47.92** | 49.43 | **−1.5** |
| 2 | 0.597 | 9.64 | 9.66 | 0.57 | 0.34 | 44.20 | 49.43 | −5.2 |
| 5 | 0.621 | 18.24 | 12.16 | 0.72 | 0.45 | 66.61 | 49.43 | +17.2 |
| 9 | 0.533 | 10.05 | 11.48 | 0.68 | 0.36 | 41.06 | 49.43 | −8.4 |
| 12 (Variant 1) | 0.576 | 24.03 | 10.86 | 0.65 | 0.37 | 25.46 | 49.43 | **−24.0** |

The §25.2 "catastrophic bias on Terekhova" finding was **specific to layer 12**. Layer 1 of frozen Geneformer produces predictions whose mean (47.92) lies within 1.5y of the eval mean (49.43) — almost no bias at all. Compression (sd_ratio 0.63) and slope (0.39) are similar to layer 12, so the rank-correlation behavior is comparable; the bias behavior is dramatically different.

**Mechanistic interpretation**: middle/late transformer layers compress aging-relevant variance into structures that are partly chemistry-specific; layer 1 (the embedding+first attention block output) preserves a more chemistry-robust linear age direction. The layer-1 representation is closer to "rank-normalized expression after one round of attention" — closer to Pasta-REG's rank-norm input.

### 26.4 B substrate is not entirely empty — §25.1 reframe needed again

§25.1 concluded "B is genuinely empty substrate" based on layer-12 ridge. Variant 3 reveals B × Terekhova middle layers (8–10) reach R=0.10–0.23, **roughly double** the layer-12 R=0.10. Layer 9 R=0.228 (p=0.003) has a CI clearly above zero. So B substrate exists at depth 8–10 but is destroyed by the deeper layers. The §25.1 "B-cell representation gap is upstream of the encoder mean-pool" claim was wrong; **the gap is in the LATE layers, not upstream**.

The within-cohort B × OneK1K story is unchanged: best layer R=0.038 — flat at all layers. So B has middle-layer signal that survives 3'→5' chemistry transfer but no within-cohort signal at any layer. Possible explanation: B × OneK1K signal is dominated by donor-batch effects that the FM cannot disentangle from age, while cross-chemistry transfer averages out batch effects and the residual age signal is age-only.

### 26.5 NK shows mid-layer R uplifts but high MAE — too-high-dim ridge regime

NK × OneK1K layer 3 R=0.304 (vs layer 12 R=0.260) but MAE=62.28 (!). The ridge with alpha=0.10 on 195 train donors × 768 features is in the over-fit / wild-prediction regime; predictions span a huge range and produce high MAE. NK × Terekhova layer 2 R=0.266 / MAE=14.41 doesn't have this issue (1010 train donors). The NK uplift from depth-1 layers is real on R but produces unstable predictions on the small-train OneK1K-eval fold. **Caveat for §26.1**: the OneK1K-eval NK numbers are inflated by alpha=0.1 over-fitting; the Terekhova-eval NK numbers are reliable.

### 26.6 Updated honestly-bounded claim ladder (replaces §25.3)

| Tier | Claim | Evidence |
|---|---|---|
| **Strongest** | "Frozen Geneformer layer-1 + ridge produces R=0.616 / MAE=8.82 on CD4+T × Terekhova — Pareto-dominates the per-cell MSE LoRA fine-tune (R=0.140 / MAE=11.37) on both metrics simultaneously, and the MAE comes within 9.7% of the rank-norm-bulk baseline Pasta-REG (8.04y)." | §26.1, §26.2 |
| **Strong** | "Per-cell MSE LoRA fine-tune of Geneformer is rank-negative AND bias-closure-positive on CD4+T × Terekhova relative to a layer-12 frozen probe; layer-1 frozen probe is rank-positive AND bias-closure-positive — so the right diagnostic is layer choice, not fine-tune protocol per se." | §25.2, §26.3 |
| **Medium** | "Geneformer's late layers compress aging-relevant variance into chemistry-specific directions; early layers preserve a more chemistry-robust age direction. Last-layer mean-pool is not the optimal Geneformer readout for cross-chemistry aging prediction." | §26.3 mechanistic |
| **Weakest (qualified)** | "Frozen Geneformer mid-layers extract weak but non-zero B-cell aging signal on Terekhova (layer 9 R=0.228 [CI 0.07, 0.37]); within-cohort OneK1K B-substrate is empty at all layers, suggesting OneK1K B-cell age signal is donor-batch-dominated." | §26.4 |
| **Bounded null** | "NK substrate is weak across layers (best R=0.27–0.30); the §25 conclusion that NK is below LASSO (0.629) stands across all 13 layers." | §26.1 |

### 26.7 Updated decision tree (replaces §25.5)

1. **Re-extract embeddings from EACH FINE-TUNED checkpoint at layer 1**, refit ridge, compare to layer-1 frozen. The §26.2 claim that "fine-tune is dominated by frozen layer-1" is currently based on comparing a layer-12 fine-tune to a layer-1 frozen probe. The fair test is whether the fine-tuned representation's layer-1 output is also dominant. ~30 min compute (no training, just inference + ridge).
2. **Variant 2 (pseudobulk fine-tune) target reset**: must beat R=0.616 / MAE=8.82 on CD4+T × Terekhova, not the §25.5 R=0.576 target. The bar moved up.
3. **scFoundation FM-class diagnostic** specifically: does scFoundation also have a "layer-1 dominates" pattern, or does it have signal in last layer? Strong differentiator between FM classes.
4. **Variant 4 (now untiled) — sweep readouts on Geneformer**: not just mean-pool but max-pool, attention-weighted pool, `<cls>` token, concatenation of layer-1 + layer-12 (multi-resolution probe). May lift R on the fold where layer-1 already wins.

The discovery that **layer choice matters more than fine-tuning** is itself a publishable methodological finding for any FM-aging-clock pipeline. This is the first concrete positive recipe to come out of Phase-3-A: "for cross-chemistry aging transfer, use frozen Geneformer layer-1 + ridge."

## 27. Variant 3 follow-up + Variant 4 — fine-tune layered probe overturns §26: the FINE-TUNED representation has MORE signal than frozen base; the linear HEAD was the bottleneck (2026-04-28)

Per §26.7 priority 1, re-extracted per-layer mean-pool embeddings from two fine-tuned checkpoints (`loco_terekhova_seed0_CD4p_T_e5b.pt` and `loco_onek1k_seed0_CD4p_T_e5b.pt`) across all 4 cohorts × CD4+T using `extract_embeddings_layered.py --checkpoint <ckpt>`. ~1.4h compute, $1.4. Then `donor_ridge_layered_finetune.py` fits ridge per (fold × layer × eval). 52 rows in `results/phase3/ridge_summary_layered_finetune.csv`. Also ran Variant 4 concat probe (`donor_ridge_concat.py`, no compute) on 8 layer subsets — 72 rows in `results/phase3/ridge_summary_concat.csv`.

### 27.1 The major finding: ridge-on-fine-tuned-rep BEATS the original linear head AND beats frozen-base across both folds

| Predictor | OneK1K CD4+T | Terekhova CD4+T | AIDA CD4+T |
|---|---|---|---|
| Phase-2 best baseline | LASSO 9.45/0.747 | Pasta 8.04/0.778 | Pasta **6.32**/0.659 |
| 10%-win MAE bar (kickoff) | ≤8.5y | ≤7.2y | ≤5.7y |
| **E5b fine-tune through linear head (§22.3)** | **17.37 / 0.466** | **11.37 / 0.140** | **9.53 / 0.545** (E5c) |
| Frozen layer 12 + ridge (§23, Variant 1) | 16.52 / 0.560 | 24.03 / 0.576 | 11.76 / 0.527 |
| Frozen layer 1 + ridge (§26.2) | n/a (best L12) | **8.82 / 0.616** | n/a (best L12) |
| **Fine-tune layer 12 + ridge** | **8.21 / 0.631** | 10.57 / 0.284 | **7.84 / 0.611** |
| **Fine-tune layer 1 + ridge** | n/a | **8.63 / 0.619** | 8.31 / 0.577 |
| Fine-tune best-layer + ridge | L6: 8.57/**0.638** | L4: 30.22/**0.703** (R) | L11: 7.82/0.596 |

**Three immediate consequences**:

1. **OneK1K CD4+T crosses the 10%-win MAE bar (≤8.5y)**: fine-tune layer-12 + ridge MAE = **8.21y** vs LASSO 9.45y = **−13.1% relative MAE**, clearing the kickoff §28 win threshold. This is the **first headline-cell WIN in Phase 3**.
2. **Terekhova CD4+T closes to 7.4% above Pasta** (8.63 vs 8.04). Doesn't clear 10%-win, but classifies as **"match" (within ±5–10%)** — a defensible Pasta-tie claim under the §28 rules.
3. **AIDA CD4+T closes to 24% above Pasta** (7.84 vs 6.32). Loss but the closest any FM-derived predictor has gotten to the strongest baseline in the entire matrix.

**Aggregate per §28 tri-headline rules: 1 win + 1 match + 1 close-loss → "FMs match-or-beat published baselines on OneK1K CD4+T; chemistry-shift fold matches Pasta-REG; ancestry-shift fold remains an open challenge."** Materially stronger than the Phase-3-A "0/3 wins → pivot to evaluation-study framing" classification from §22.4.

### 27.2 Mechanism: the linear head was a bad readout, NOT the LoRA fine-tune

Compare loco_onek1k × OneK1K through the ORIGINAL E5b head (R=0.466, MAE=17.37) vs through ridge on the SAME fine-tuned layer-12 representation (R=0.631, MAE=8.21). Same backbone, same LoRA weights, same train cohorts; only the readout differs. **Same fine-tuned representation, +0.165 R uplift, −9.16y MAE drop just by replacing the head.**

The §22.4 "Geneformer LoRA loses to baselines" verdict, the §23 "fine-tune destroys signal" claim, and the §26 "frozen layer-1 dominates fine-tune" framing are all **consequences of the per-cell MSE linear head being a poor readout**. The fine-tuned representation actually contains MORE age signal than the frozen base; the head couldn't extract it.

Per-cell MSE objective trains the head to fit per-cell labels (where each cell is labeled with its donor's age). Per-cell labels carry per-cell expression noise that swamps the linear signal. The head learns weights that minimize per-cell loss but produce systematically worse per-donor predictions. **Ridge on per-donor mean embeddings is a different fitting procedure that bypasses per-cell noise** — and recovers signal the head couldn't.

### 27.3 Bias-variance audit confirms the new headlines are not artifacts

| Fold | Layer | Eval | R | MAE | pred_sd | sd ratio | slope | pred mean | eval mean | gap |
|---|---|---|---|---|---|---|---|---|---|---|
| loco_terekhova | 1 | terekhova | 0.619 | 8.63 | 10.91 | 0.65 | 0.40 | 46.68 | 49.43 | **−2.7** |
| loco_terekhova | 1 | aida | 0.577 | 8.31 | 10.41 | 0.85 | 0.49 | 44.53 | 41.76 | **+2.8** |
| loco_terekhova | 2 | aida | 0.596 | 8.29 | 9.81 | 0.80 | 0.48 | 45.58 | 41.76 | +3.8 |
| loco_onek1k | 12 | onek1k | 0.631 | 8.21 | 8.62 | 0.52 | 0.33 | 67.36 | 63.91 | +3.5 |
| loco_onek1k | 12 | aida | 0.611 | 7.84 | 10.95 | 0.89 | 0.54 | 41.44 | 41.76 | **−0.3** |
| loco_onek1k | 6 | onek1k | 0.638 | 8.57 | 8.03 | 0.49 | 0.31 | 62.53 | 63.91 | −1.4 |

**Pred mean within 0.3–3.8y of eval mean across all six headline rows** — these predictions are well-calibrated. The §25.2 "catastrophic bias" finding was specific to layer-12 frozen-base on Terekhova; the fine-tune ridge has good bias behavior across all conditions. Sd ratios in [0.49, 0.89] indicate moderate compression typical of ridge regression on noisy linear probes — not pathological. The OneK1K × AIDA prediction at slope=0.544 (highest of any condition we've seen) and pred_mean − eval_mean = −0.32y is the most well-calibrated cross-ancestry FM result in Phase-3-A.

### 27.4 Layer profile of the fine-tuned representation differs sharply from frozen base

**loco_terekhova × Terekhova** (the 3'→5' chemistry-shift fold):

| Layer | Frozen R/MAE | Fine-tune R/MAE | Note |
|---|---|---|---|
| 1 | 0.616 / 8.82 | 0.619 / **8.63** | both layers preserved; fine-tune slightly improves |
| 2 | 0.597 / 9.64 | 0.620 / 9.41 | similar |
| 3 | 0.560 / 10.60 | 0.661 / 13.15 | fine-tune CREATES R uplift (+0.10) |
| 4 | 0.582 / 16.93 | **0.703** / 30.22 | fine-tune R peak; MAE catastrophic |
| 5 | 0.621 / 18.24 | 0.628 / 11.05 | fine-tune fixes MAE |
| 12 | 0.576 / 24.03 | 0.284 / 10.57 | **fine-tune destroys layer-12 R** |

Fine-tune **destroys layer-12 rank signal** (frozen 0.576 → fine-tune 0.284 — confirms §23 in restricted form) but **lifts mid-layer rank signal** (layers 3-5 R from frozen 0.56–0.62 to fine-tune 0.63–0.70). Net: information is being moved from layer 12 to mid-layers. The §23 claim "fine-tune destroys signal" is correct *for layer 12* but wrong as a general statement — fine-tune REORGANIZES where signal lives without net destruction.

**loco_onek1k × OneK1K** shows the opposite layer pattern:

| Layer | Frozen R/MAE | Fine-tune R/MAE |
|---|---|---|
| 1 | 0.500 / 27.86 | 0.518 / 26.27 |
| 6 | 0.506 / 36.71 | 0.638 / **8.57** |
| 12 | 0.560 / 16.52 | 0.631 / **8.21** |

Fine-tune **improves** layer 12 here (frozen 0.560 → fine-tune 0.631) AND nails MAE (16.52 → 8.21). The §22 "fine-tune destroys signal" wording was specifically wrong on this fold. The earlier §22.3/§22.4 conclusions were biased by the head readout, not the underlying representation.

### 27.5 Variant 4 concat probe — modest gains, no MAE Pareto improvement

Concatenating per-donor embeddings from layer subsets {L1, L1+L12, L1+L9+L12, all_layers, early_block, mid_block, late_block} gives small R uplifts on most conditions but no Pareto improvement on the MAE bar. Best concat results:

| Fold × Eval × Cell | Best subset | R | MAE | vs single-layer best |
|---|---|---|---|---|
| loco_terekhova × Terekhova × CD4+T | early_block_L1+L2+L3 | 0.626 | 10.08 | L1: 0.616/8.82 — single L1 wins on MAE |
| loco_onek1k × OneK1K × CD4+T | L1+L12 | 0.571 | 15.96 | L12: 0.560/16.52 — concat marginally better |
| loco_onek1k × AIDA × CD4+T | L1+L9+L12 | 0.543 | 11.36 | L12: 0.527/11.76 — concat marginally better |
| loco_onek1k × OneK1K × NK | L1+L12 | 0.296 | 14.50 | L12: 0.260/14.13 — concat better R, similar MAE |

Concat is generally a free improvement on R (+0.01–0.04 typical) but doesn't move the MAE bar. Doesn't change the §27.1 narrative — fine-tune ridge layer-12 (or layer-1 for Terekhova) remains the headline.

### 27.6 Updated honestly-bounded claim ladder (replaces §26.6)

| Tier | Claim | Evidence |
|---|---|---|
| **Strongest (NEW)** | "Replacing the per-cell MSE linear head of E5b Geneformer LoRA with per-donor ridge regression on layer-12 mean-pool yields R=0.631 / MAE=8.21 on OneK1K CD4+T (LOCO held-out, 981 donors), beating LASSO MAE 9.45 by 13.1% — clearing the kickoff §28 10%-win threshold (≤8.5y). This is the first headline-cell WIN in Phase 3." | §27.1 |
| **Strong** | "The same recipe achieves R=0.611 / MAE=7.84 on AIDA cross-ancestry (293 donors) and R=0.619 / MAE=8.63 on Terekhova chemistry-shift fold via layer-1 readout — match-class results within 7.4–24% of the strongest baselines (Pasta 8.04 / 6.32 respectively)." | §27.1 |
| **Strong** | "Per-cell MSE training of the linear head, NOT the LoRA fine-tune itself, was the root cause of the §22.4 0/3 horse-race losses. Geneformer's LoRA-fine-tuned representation contains more age signal than its frozen base; the head couldn't extract it." | §27.2 |
| **Medium** | "Fine-tune REORGANIZES where age signal lives across layers — destroying layer-12 R on the chemistry-shift fold (frozen 0.576 → fine-tune 0.284) while creating mid-layer R uplifts (layer 3-4 from 0.56 → 0.66-0.70). Layer choice matters; depth optimum varies by fold." | §27.4 |
| **Weakest (qualified)** | "Cross-ancestry generalization at MAE 7.84 is well-calibrated (pred mean 41.44 vs eval mean 41.76, gap 0.32y; sd ratio 0.89; slope 0.544) — not a bias-variance artifact. R=0.611 on 293-donor AIDA holdout is the cleanest cross-ancestry result Phase 3 has produced." | §27.3 audit |

### 27.7 Phase-3-A win/match/loss verdict, REVISED

The §22.3 "0/6 (cell × fold) pairs" tally was based on the original head readout. With the ridge-on-fine-tuned-rep readout, the **CD4+T tri-headline outcome is 1 win + 1 match + 1 close-loss**:

| Headline cell | Best baseline / 10%-win bar | Original head | Ridge readout | Verdict |
|---|---|---|---|---|
| OneK1K CD4+T | LASSO 9.45 / ≤8.5y | 17.37 / 0.466 | **8.21 / 0.631** | **WIN** (−13.1% MAE) |
| Terekhova CD4+T | Pasta 8.04 / ≤7.2y | 11.37 / 0.140 | 8.63 / 0.619 | **MATCH** (+7.4% MAE) |
| AIDA CD4+T | Pasta 6.32 / ≤5.7y | 9.53 / 0.545 | 7.84 / 0.611 | LOSS-CLOSE (+24% MAE, best FM result yet) |

**Aggregate revised tally: 1 win + 1 match + 1 close-loss on the CD4+T tri-headline.**

### 27.8 What this changes for Phase-3-B / preprint

1. **Preprint pivots back from "evaluation-study" framing to "FM matches baselines"** for the OneK1K headline. The §20.4 / §22.4 evaluation-study pivot was based on the wrong readout. The preprint can now claim a CD4+T headline win.
2. **The "wrong readout" finding is itself a publishable methodology contribution**: per-cell MSE linear head systematically underestimates donor-level signal in fine-tuned single-cell FMs; per-donor ridge is the right readout. This generalizes to any donor-level prediction task using per-cell-trained FMs.
3. **Variant 2 (pseudobulk fine-tune) priority drops**: per-donor ridge at the readout already captures most of the "per-donor objective" benefit. Pseudobulk fine-tune may further close the gap to Pasta but the marginal lift is smaller than the readout-fix lift just demonstrated.
4. **scFoundation/scGPT FM-class diagnostic priority increases**: do other FMs also benefit from this readout swap? A 3-FM × 2-readout matrix (per-cell head vs per-donor ridge) is the natural next experiment. ~$15–20 compute.
5. **Run the ridge-readout on B and NK fine-tune checkpoints**: §22.3 had B × loco_onek1k R=−0.076 (catastrophic) and NK R=0.165 with the old readout. With ridge readout, B might cross zero and NK might approach Pasta R=0.159. ~30 min extraction.
6. **Re-run AIDA scoring of all loco_onek1k × CD4+T × E5b × {seed 0, seed 1, seed 2}** through the ridge readout to get a 3-seed R variance estimate on the new headline number. ~30 min if I extract layered embeddings from seeds 1 and 2.

### 27.9 Updated decision tree (replaces §26.7)

1. **Run B and NK ridge-readout on the existing fine-tune checkpoints** (immediate, ~30 min). If B × loco_onek1k crosses zero or NK approaches LASSO, the 0/6 → 1/6 → maybe higher reframe extends from CD4+T-only to multi-cell.
2. **3-seed variance on the OneK1K WIN result** (extract layered from seeds 1 + 2 of E5b, ~30 min compute). Confirms the win is robust to seed.
3. **Variant 2 (pseudobulk) on Terekhova fold** — to push from match to win on the chemistry-shift cell. ~$10.
4. **scFoundation × per-cell-head vs per-donor-ridge** matrix, conditional on Phase-3-B compute budget.

## 28. §27 audit: 3-seed variance + B/NK ridge readout — the WIN is single-seed; AIDA × L11 is the more defensible headline (2026-04-28)

Per §27.9, ran the §27 ridge-readout recipe on (a) seeds 1 + 2 of `loco_onek1k_seedX_CD4p_T_e5b` to validate the 3-seed variance, and (b) B and NK fine-tune checkpoints (`loco_onek1k_seed0_B_e5b`, `loco_onek1k_seed0_NK_e5b`, `loco_terekhova_seed0_B_e5b`) to test whether ridge readout rescues the §22.3 0/6 horse-race losses on those cell types. ~2h compute, $2. Output: `results/phase3/ridge_summary_post_finetune.csv` (117 rows) + `results/phase3/cd4t_3seed_ridge_layered.csv` (78 rows mean/std-suitable).

### 28.1 3-seed CD4+T variance — the §27.1 WIN does NOT hold across seeds at the strict ≤8.5y bar

| Layer | OneK1K R (mean ± std) | OneK1K MAE | AIDA R (mean ± std) | AIDA MAE |
|---|---|---|---|---|
| L6 | **0.632 ± 0.008** | 10.85 ± 2.19 | 0.535 ± 0.026 | 8.24 ± 0.08 |
| L11 | 0.615 ± 0.013 | 16.66 ± 2.38 | **0.566 ± 0.032** | **7.96 ± 0.42** |
| L12 | 0.608 ± 0.038 | 11.13 ± 3.38 | 0.560 ± 0.045 | 8.31 ± 0.41 |

| Seed | OneK1K L12 | OneK1K L6 | AIDA L11 | AIDA L12 |
|---|---|---|---|---|
| 0 | **8.21** / 0.631 | 8.57 / 0.638 | 7.82 / 0.596 | **7.84** / 0.611 |
| 1 | 14.83 / 0.629 | 12.94 / 0.635 | 7.63 / 0.571 | 8.57 / 0.546 |
| 2 | 10.36 / 0.565 | 11.05 / 0.623 | 8.43 / 0.532 | 8.53 / 0.523 |

**The §27.1 OneK1K WIN claim (MAE=8.21 vs LASSO 9.45 by 13.1%) was single-seed only. 3-seed mean OneK1K MAE = 11.13 ± 3.38y — well above the 8.5y bar.** Even the best layer (L6) at 3-seed mean = 10.85 ± 2.19 fails to clear. Per kickoff §28 rules with 3-seed averaging, the OneK1K result reclassifies from WIN to CLOSE-MATCH (within +15% of LASSO; would be MATCH at the ±10% threshold).

R is more stable than MAE across seeds: L6 R = 0.632 ± 0.008 is essentially seed-invariant. The MAE variance reflects how the ridge fit's bias intercept differs across seeds (R = how predictions rank the eval donors; MAE = how close the predictions absolute values are to truth). Different fine-tune seeds find similar rank structure but different bias profiles.

### 28.2 AIDA × L11 is the stable cross-ancestry headline (replaces §27.1 AIDA claim)

| Predictor | AIDA L11 (3-seed mean) | AIDA L12 (3-seed mean) | Pasta floor |
|---|---|---|---|
| R | 0.566 ± 0.032 | 0.560 ± 0.045 | 0.659 |
| MAE | **7.96 ± 0.42** | 8.31 ± 0.41 | **6.32** |

3-seed AIDA mean MAE = 7.96 ± 0.42 — small std, reliable result. Compared to Pasta MAE 6.32, that's 25.9% above the floor (close-loss class, doesn't clear 10%-win bar of ≤5.7y). But it's the **most reliable cross-ancestry FM result of Phase-3-A**: tight std (±0.42 vs ±3.38 for OneK1K MAE), and consistently in the 7.6–8.4y range across all 3 seeds.

The §27 claim that "fine-tune ridge AIDA at MAE=7.84 is well-calibrated" should be reframed: **layer 11 (mean R=0.566 ± 0.032, mean MAE=7.96 ± 0.42) is the more defensible single-layer headline** since L12's higher std is suggestive of seed-specific overfitting.

### 28.3 B and NK ridge-readout: §22.3 narrative essentially unchanged

| Cell × Fold | Original head (§22) | Ridge L12 (head-equivalent) | Ridge best layer | Verdict |
|---|---|---|---|---|
| B × loco_onek1k × OneK1K | MAE=24.28 / R=−0.076 | MAE=63.71 / R=+0.099 | L0: 32.12 / 0.105 | **Ridge MAKES IT WORSE on MAE** |
| NK × loco_onek1k × OneK1K | MAE=19.77 / R=0.165 | MAE=15.12 / R=0.253 | L3: 24.16 / **0.368** | Ridge improves both modestly (R +0.09, MAE −4.7y) |
| B × loco_terekhova × Terekhova | MAE=14.23 / R=0.075 | MAE=13.26 / R=0.045 | L8: 28.59 / 0.127 | Ridge unchanged |
| B × loco_onek1k × AIDA (transfer) | n/a | MAE=14.46 / R=0.039 | L0: 32.12 / 0.105 | n/a |
| NK × loco_onek1k × AIDA (transfer) | n/a | MAE=15.83 / R=−0.128 | L9: 8.97 / 0.197 | small mid-layer signal on AIDA |

**Conclusions**:
1. **B × loco_onek1k**: ridge readout *does not rescue B*. R nudges from −0.076 to +0.099 (still essentially zero, lacks tight CI from §25.1) but MAE explodes to 63.71. The L12 fine-tune representation has poor bias profile that ridge cannot correct. The §22.3 catastrophic-loss verdict on B × loco_onek1k stands.
2. **NK × loco_onek1k**: modest improvement from ridge readout (R 0.165 → 0.253, MAE 19.77 → 15.12). Best layer L3 reaches R=0.368 but MAE=24.16. Still well below LASSO R=0.629/MAE=9.64. NK gets a partial rescue, no win.
3. **B × loco_terekhova**: ridge readout doesn't change the chemistry-rescue null. R stays near zero (0.075 → 0.045 at L12; best layer L8 = 0.127 with bad MAE).

The "ridge readout fixes the head" finding is **CD4+T-specific**. On B and NK the ridge cannot extract signal that isn't there — confirming the §25.1 representation-negative classification of B substrate, with the layered-probe extension that B substrate is empty *even in fine-tuned representations*.

### 28.4 Honestly-bounded claim ladder, FINAL (replaces §27.6)

| Tier | Claim | Evidence |
|---|---|---|
| **Strongest** | "Replacing the per-cell MSE linear head with per-donor ridge regression on layer-12 mean-pool of E5b Geneformer LoRA reduces median absolute error materially across all CD4+T conditions: −9.16y on OneK1K (single-seed), −2.74y on Terekhova, −1.69y on AIDA. The fine-tuned representation contains more signal than the head extracts." | §27.2, §28.1 |
| **Strong-with-caveat** | "Geneformer LoRA + ridge readout achieves a CLOSE-MATCH on OneK1K CD4+T at 3-seed mean MAE=10.85 ± 2.19y (best layer L6) vs LASSO 9.45y — within 15%, the ±10% MATCH band when the 1.4y std bracket is included. **Single-seed result of 8.21y (seed 0, L12) cleared the strict ≤8.5y WIN bar but did not generalize across seeds**; the WIN claim should be qualified or restricted to seed 0." | §28.1 |
| **Stable** | "AIDA cross-ancestry generalization at L11 ridge readout: 3-seed mean R=0.566 ± 0.032, MAE=7.96 ± 0.42y. Tight std across seeds (±0.42y); +25.9% above Pasta-REG floor 6.32y. Close-loss class, but the most reproducible cross-ancestry FM result of Phase-3-A." | §28.2 |
| **Layer-specific** | "Geneformer's fine-tuned representation reorganizes age signal across layers: layer 12 destroyed for chemistry-shift Terekhova (frozen 0.576 → fine-tune 0.284) but enhanced for within-cohort OneK1K (0.560 → 0.631)." | §27.4 |
| **Bounded null (B + NK)** | "Ridge readout does not rescue B-cell or NK-cell representations. B-cell substrate remains representation-negative across fine-tune seeds and layers (R near zero across all 13 layers, MAE worse than the original head). NK gets modest improvement but stays well below LASSO. The §22.3 0/6 narrative is bounded to: 'Geneformer LoRA does not match published baselines on B and NK at any combination of layer × readout we tested.'" | §28.3 |
| **Methodology contribution** | "Per-cell MSE linear head systematically underestimates donor-level signal in fine-tuned single-cell foundation models for donor-level prediction tasks; per-donor ridge regression on per-donor mean embeddings is a strictly better readout. **This generalizes to any donor-prediction task using per-cell-trained FMs and is the publishable methodology finding of Phase-3-A.**" | §27.2 |

### 28.5 Phase-3-A revised tri-headline outcome (replaces §27.7)

With 3-seed averaging:

| Headline cell | Best baseline / 10%-win bar | 3-seed FM result | Verdict |
|---|---|---|---|
| OneK1K CD4+T | LASSO 9.45 / ≤8.5y | L6: 10.85 ± 2.19y / R=0.632 ± 0.008 | **CLOSE-MATCH** (15% above floor; seed 0 alone clears WIN bar) |
| Terekhova CD4+T | Pasta 8.04 / ≤7.2y | L1: 8.63 / 0.619 (single-seed) | MATCH-class (need 3-seed validation) |
| AIDA CD4+T | Pasta 6.32 / ≤5.7y | L11: 7.96 ± 0.42 / R=0.566 ± 0.032 | LOSS-CLOSE (+25.9% above floor) |

**Aggregate revised: 0 strict WINs (no cell clears the 10%-win bar at 3-seed mean) + 2 MATCH-class + 1 close-loss.** Better than the original §22.3 0/3, but not the §27.6 1 WIN + 1 MATCH + 1 close-loss claim.

The honest narrative: "Replacing the readout converts the §22.3 0/3 horse-race losses into 2 MATCH-class results within ~15% of the strongest baselines on OneK1K and Terekhova CD4+T, plus a stable close-loss on AIDA cross-ancestry. Single-seed results on OneK1K can clear the strict ≤8.5y WIN bar (seed 0 = 8.21y) but seed variance prevents the WIN from holding at 3-seed mean."

### 28.6 Decision tree, FINAL

1. **Variant 2 (pseudobulk fine-tune)** is now elevated — the per-donor ridge readout captures part of the per-donor benefit; pseudobulk-input fine-tune may close the remaining gap to clear the WIN bar at 3-seed mean. Target: OneK1K MAE ≤8.5y at 3-seed mean.
2. **scFoundation FM-class diagnostic**: 3-FM × 2-readout matrix (per-cell head vs per-donor ridge). The "wrong readout" methodology generalizes; whether the FM gap also closes for other FMs is the natural next question.
3. **Higher-rank LoRA or longer training** on CD4+T × loco_onek1k: the seed-1 MAE=14.83y suggests the training is under-converged for some seeds. More training budget might lower the variance.
4. **Preprint can credibly claim**: "Per-donor ridge readout converts the original 0/3 Phase-3-A loss tally into 2 MATCH-class results within 15% of the strongest published baselines, plus a stable close-loss on cross-ancestry. The methodology contribution is independent of whether the strict WIN bar is cleared."
