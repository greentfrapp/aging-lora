# Phase 3 kickoff plan (Option C, drafted 2026-04-25)

Pre-execution plan. No compute spent yet. Pin down scope, GPU constraints, code structure, hyperparameters, and risk register before any LoRA fine-tune runs. Decision points that need user buy-in are flagged **DECIDE**.

---

## 1. Recap of Phase-3 success criteria

From `roadmap/phase-3.md` (revised 2026-04-25 with tri-headline + AIDA + min-of-4):

1. LoRA fine-tune Geneformer + scFoundation + scGPT on CD4+T LOCO folds (OneK1K-out, Terekhova-out) × 3 seeds = **18 fine-tunes**
2. AIDA inference on every fine-tuned checkpoint = 18 extra eval passes (no training)
3. Per-cell "beat-best-baseline" classification against min-of-4 baselines (LASSO-pre, LASSO-retrained, scAgeClock, Pasta-REG)
4. 3-comparator × 3-chemistry-context **3×3 robustness subfigure** (LASSO + Pasta + Geneformer-LoRA × {3' OneK1K, 5' Terekhova, 5' AIDA Asian})
5. Measured GPU-hours per LoRA run → updates Phase-4 estimate
6. Detectability floor with **bracketed ρ disclosure** (Phase-1 ρ=0.8 / Phase-2 baseline-pair ρ=0.06–0.35 / Phase-3 measured baseline-vs-FM ρ)
7. bioRxiv preprint posted within 10 weeks of project start (target ~2026-07-01)

## 2. Hardware reality check: 2080 Ti, 11 GB VRAM

This is the binding constraint. 2080 Ti is compute-capability 7.5 (Turing); Geneformer, scGPT can train cleanly; scFoundation and UCE are at risk.

| FM | Params | fp32 weights | Training memory (LoRA, fp16, batch=32 cells) | Verdict on 2080 Ti |
|---|---|---|---|---|
| Geneformer | 110M (V2 base) | 440 MB | ~3 GB peak | ✅ fits comfortably |
| scGPT | ~50M (whole-human) | 200 MB | ~2 GB peak | ✅ fits comfortably |
| scFoundation | 3B (xtrimo) | 12 GB | OOM as-is. Needs LoRA-only + gradient checkpointing + activation offloading. Best case ~9–10 GB peak. | ⚠️ tight; may not fit |
| UCE | 1.4B (33L 1280-dim) | 5.6 GB | ~8–9 GB peak with LoRA-only + checkpointing | ⚠️ tight; should fit but risky |

Roadmap says **Phase 3 model panel = Geneformer + scFoundation + scGPT** (UCE deferred to Phase 4). The 2080 Ti makes scFoundation the hard case.

**DECIDE-A**: scFoundation Phase-3 plan. Three options:
- **A1**: Run scFoundation training on this 2080 Ti with aggressive memory tricks (LoRA-only-trainable + 4-bit quantized base + gradient checkpointing + bf16 activations). High risk of OOM; likely needs micro-batches (1–4 cells) + grad accumulation. Wall-clock per fine-tune: 8–24h on 2080 Ti.
- **A2**: Defer scFoundation to a cloud GPU run (A100 80GB, ~$3/h × ~2h × 6 fine-tunes = ~$40). Drops scFoundation from the local-only Phase-3 sweep but retains it for the preprint.
- **A3**: Drop scFoundation from Phase 3 entirely; Phase-3 panel becomes Geneformer + scGPT (the two that fit comfortably). scFoundation moves to Phase 4 alongside UCE.

**My recommendation**: A2 if you have a credit-card willingness for ~$40 of cloud spend; A3 otherwise. A1 is high engineering effort + reliability risk for marginal benefit (the 2080 Ti will be the bottleneck either way).

## 3. PyTorch reinstall

Current: `torch 2.11.0+cpu`. Need a CUDA build. 2080 Ti CC 7.5 + driver 560.94 supports CUDA 12.x.

**Recommended command** (run by user; uv-managed):

```
uv pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

After install, verify:
```
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
Expected: `True NVIDIA GeForce RTX 2080 Ti`.

**Caveat**: torch 2.11.0 is post-2.5/2.6 (the 2026 stable line). May not have CU121 wheels yet. Fallback: pin to `torch==2.6.*` or `torch==2.5.*` which definitely have wheels for CC 7.5. The pyproject.toml currently does not pin torch version — we should pin it once CUDA is working.

**DECIDE-B**: do you want me to attempt the install, or do you prefer to do it manually? (I'll need permission to modify the venv either way.)

## 4. Per-FM implementation plan

### 4.1 Geneformer (lead, headline)

- **Source**: `data/scAgeClock/scageclock/...` — wait, that's scAgeClock. Geneformer comes from HuggingFace `ctheodoris/Geneformer` (smoke-tested in Phase 1 at `save/geneformer/`).
- **Input format**: rank-value-encoded gene tokens (top-N genes by rank-normalized expression). Tokenizer in the repo.
- **LoRA target modules**: `query`, `value` projections in BertSelfAttention; optionally `intermediate.dense` and `output.dense` in BertOutput. Default-aggressive: r=16, alpha=32, target = `["query", "value", "intermediate.dense", "output.dense"]`.
- **Regression head**: 1-layer linear on top of CLS-token embedding (or mean-pooled cell embeddings). Single output unit, MSE loss.
- **Hyperparameters (defaults)**: lr=5e-5 (LoRA), epochs=5, batch=32 cells, gradient clipping 1.0, warmup 10%, AdamW.
- **Per-donor aggregation at inference**: predict per cell, take median per donor. Match the LASSO pseudocell-aggregation conceptually.

### 4.2 scGPT (comparator)

- **Source**: `save/scGPT_human/best_model.pt` (smoke-tested in Phase 1).
- **Input format**: gene-aware tokenization (gene IDs + value bins); their `prepare_data` pipeline.
- **LoRA targets**: same template — attention `Wqkv` projection (scGPT uses fused QKV; PEFT supports targeting `Wqkv` directly), MLP `fc1`+`fc2`. r=16, alpha=32.
- **Regression head**: linear on cell embedding (their `forward` produces a per-cell embedding via attention pooling).
- **Hyperparameters**: lr=5e-5, epochs=5, batch=32 cells. Same as Geneformer.

### 4.3 scFoundation (secondary; pending DECIDE-A)

- **Source**: `save/scFoundation/models.ckpt` (from user; smoke-tested in Phase 1).
- **Input format**: zero-padded gene expression vector + cell-context tokens (T/S/B variants).
- **LoRA targets**: per their FoundationDataLoader spec, attention layers in their performer-style transformer. Need to inspect repo to find module names.
- **Regression head**: linear on the `cell` token embedding from their gene/cell/rde 3-head output.
- **Hyperparameters**: lr=2e-5 (smaller for the larger backbone), epochs=3, batch=4–8 cells (2080 Ti memory). With grad accumulation 8x, effective batch=32.

### 4.4 Common code structure

```
src/finetune/
├── __init__.py
├── lora_wrappers/
│   ├── __init__.py
│   ├── geneformer.py           # build_geneformer_lora(checkpoint_path, **lora_kwargs)
│   ├── scgpt.py
│   └── scfoundation.py         # if A1 or A2
├── data_loader.py              # AnnData → FM-specific input format
├── train_loop.py               # FM-agnostic training loop (lr scheduler, eval, ckpt)
├── eval_runner.py              # per-donor aggregation + MAE/R + AIDA pass
└── cli.py                      # `uv run python -m src.finetune.cli --fm geneformer --fold loco_onek1k --seed 0`
```

Each `build_X_lora` returns a `(model, tokenizer_or_collator, regression_head)` tuple with consistent interface.

### 4.5 Output structure (matches the reorg from `26fd981`)

```
results/baselines/fm_finetuned/
├── geneformer/
│   ├── summary.csv                                     # all (model, fold, seed, eval_cohort, cell_type) rows
│   ├── per_donor/
│   │   └── {fold}_{seed}_{eval_cohort}_{ct}.csv
│   └── checkpoints/
│       └── {fold}_{seed}.pt                            # LoRA delta-weights only (small)
├── scgpt/
└── scfoundation/                                        # if A1/A2
```

## 5. Hyperparameter pinning (DECIDE-C)

I recommend defaults below; flag any disagreement.

| Hyperparameter | Default | Rationale |
|---|---|---|
| LoRA rank | 16 | Roadmap-specified |
| LoRA alpha | 32 | Standard 2x rank |
| LoRA targets | attention QKV + MLP fc1/fc2 | "Attention + MLP layers" per roadmap; aggressive |
| LoRA dropout | 0.05 | PEFT default |
| Learning rate | 5e-5 (Geneformer/scGPT), 2e-5 (scFoundation) | Standard for LoRA on transformers |
| Epochs | 5 (Geneformer/scGPT), 3 (scFoundation) | Match published downstream-task fine-tuning |
| Batch size | 32 (Geneformer), 32 (scGPT), 4 (scFoundation) | 2080 Ti budget |
| Effective batch (with grad accum) | 32 | Constant across FMs |
| Optimizer | AdamW(weight_decay=0.01) | Standard |
| LR schedule | linear warmup 10% → linear decay | Standard |
| Gradient clipping | 1.0 | Standard |
| Loss | MSE on per-cell predicted age | Regression objective |
| Per-donor aggregation | median of per-cell predictions | Matches LASSO/Pasta convention |
| Seeds | {0, 1, 2} | 3 seeds × 2 folds × N FMs = N×6 runs |
| Mixed precision | bf16 (Geneformer/scGPT), bf16 + grad checkpointing (scFoundation) | 2080 Ti supports bf16 since CUDA 11.x |

## 6. Step-by-step plan (with go/no-go gates)

### Phase-3-pre (this kickoff plan): no compute

- ✅ Hardware probe done (2080 Ti, 11 GB)
- ✅ Phase-3 kickoff plan written (this doc)
- **GATE 1**: user signs off on this doc + DECIDE-A (scFoundation A1/A2/A3) + DECIDE-B (torch reinstall) + DECIDE-C (hyperparameters)

### Phase-3-A (smoke test, 1 fine-tune): ~4–8 GPU-hours

Goal: validate the wrapper + measure GPU-hours-per-fine-tune empirically.

1. Reinstall torch with CUDA support (DECIDE-B)
2. Implement `src/finetune/lora_wrappers/geneformer.py` + `data_loader.py` + `train_loop.py` + `cli.py`
3. Run **one** Geneformer LoRA fine-tune: `--fm geneformer --fold loco_onek1k --seed 0` on CD4+T cells from Stephenson + Terekhova → eval on OneK1K
4. Sanity check: validation MAE < 12y, training stable (no NaN loss, no OOM)
5. **GATE 2**: empirical GPU-hours/run × 18 ≤ acceptable budget? Sane MAE? Greenlight full sweep.

### Phase-3-B (full LoRA sweep): ~30–60 GPU-hours

Sequential on the single 2080 Ti, since each fine-tune saturates the card.

- 3 seeds × 2 folds × (Geneformer + scGPT) = 12 fine-tunes
- + 6 scFoundation fine-tunes (DECIDE-A: locally with memory tricks / cloud / skip)
- Schedule via a simple loop or `make` targets — not parallel, not Slurm scope.

### Phase-3-C (AIDA inference): ~3 GPU-hours

For each of 18 trained checkpoints, score on AIDA CD4+T's `ancestry_shift_mae` half (307 donors). One `--cohort-id aida --integrated-dir data/cohorts/aida_eval` flag through the eval runner.

### Phase-3-D (analysis + figures): ~2 days CPU

- Aggregate per-(model, fold, seed) → per-(model, fold) means + 95% bootstrap CIs
- Compute baseline-vs-FM ρ per cell type → update detectability floor
- Generate forest plot F4 + 3×3 robustness F5
- Append to `results/baselines/loco_baseline_table.csv` via `assemble_loco_baseline_table.py` extension

### Phase-3-E (preprint drafting): 1–2 weeks

Outside this autonomous-execution scope. Hand off to manuscript writing per `notes/paper_outline.md`.

## 7. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| 2080 Ti OOM on scFoundation despite tricks | Medium | scFoundation drops from Phase 3 | DECIDE-A A2/A3 fallback; document scope cut in preprint |
| LoRA wrapper bugs on a specific FM | Medium | Smoke test catches; lose hours not days | GATE 2 |
| Validation MAE doesn't beat baseline minimum | Medium | Reframe paper to "evaluation study" per roadmap fallback | Roadmap already plans this; preprint headline degrades but stays publishable |
| FM training instability (NaN loss) | Low | Hyperparameter retune cycle | Standard mitigations (lr ↓, grad clip ↓, fp32 fallback) |
| Driver/CUDA wheel mismatch on 2080 Ti | Low | Torch reinstall fails → debug | DECIDE-B; 2080 Ti is well-supported |
| Wall-clock for 18 fine-tunes blows past timeline | Medium | Cut seed count to 2, or scope down to 1 fold | Acceptable degradation |

## 8. Decisions needed before Phase-3-A starts

- **DECIDE-A**: scFoundation strategy (A1 local + tricks / A2 cloud A100 / A3 drop). My recommendation: **A2** (~$40 cloud spend) if budget allows; **A3** otherwise.
- **DECIDE-B**: torch reinstall. My recommendation: I attempt it, you authorize the venv modification.
- **DECIDE-C**: hyperparameters per §5. My recommendation: accept defaults; we can tune later if needed.
- **DECIDE-D**: which day-of-week to start? Phase-3-A is ~4–8 hours of GPU + 1 day of code. Phase-3-B is several days of monitored compute. Want a specific window?

## 9. Out-of-scope for Phase 3

- Other cell types (CD8+T, MONO, NK, B) — Phase 4
- UCE — Phase 4
- Full fine-tune (vs LoRA) ablation — Phase 4
- Few-shot downsampling curve — Phase 4
- Zero-shot cell-type transfer — Phase 4
- Pasta retraining or RF/PointNet retraining — supplementary, deferred

---

Once you signal go on the four DECIDEs, I move to Phase-3-A.
