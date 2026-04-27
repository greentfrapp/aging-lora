# Phase-3 cloud compute review (EC2 GPU instances)

Reviewing AWS EC2 GPU instance options for Phase-3 fine-tunes, balancing cost ($/h, $ per experiment) against time-to-result and VRAM headroom for upcoming workloads (scFoundation in particular). Reference baseline: local RTX 2080 Ti (Turing CC 7.5, 11 GB VRAM, no native bf16).

## 1. Workload requirements per FM

| FM | Params | fp32 weights | LoRA training memory (bf16, batch=8, seq=2048) | VRAM floor |
|---|---|---|---|---|
| Geneformer-V2 | 110M | 440 MB | ~10 GB peak (measured on 2080 Ti) | 11 GB ✅ |
| scGPT (whole-human) | ~50M | 200 MB | ~6–8 GB peak | 11 GB ✅ |
| scFoundation (xtrimo) | 3B | 12 GB | ~9–10 GB with 4-bit base + LoRA + grad-ckpt; **at risk on 11 GB** | 16 GB minimum, 24 GB safe |
| UCE | 1.4B | 5.6 GB | ~8–9 GB with LoRA + grad-ckpt | 11 GB tight, 16 GB comfortable |

**Throughput-relevant features:**
- Native **BF16 tensor cores**: starting at compute-cap 8.0 (Ampere). Turing emulates bf16 in software — slow.
- **Memory bandwidth**: attention with seq_len=2048 + grad-ckpt produces ~20 GB/forward of memory traffic per layer × 12 layers; this is bandwidth-sensitive.
- **VRAM**: scFoundation effectively gates everything — 11 GB is at-risk, 24 GB is safe.

## 2. Relevant EC2 instance families

Single-GPU instances (multi-GPU is overkill at our LoRA scale; see §5):

| Instance | GPU | Arch | VRAM | BW | BF16 native | On-demand $/h | Spot $/h (typical) |
|---|---|---|---|---|---|---|---|
| **g4dn.xlarge** | T4 | Turing 7.5 | 16 GB | 320 GB/s | ❌ emulated | $0.526 | ~$0.16 |
| **g5.xlarge** | A10g | Ampere 8.6 | 24 GB | 600 GB/s | ✅ | $1.006 | ~$0.30–0.40 |
| **g6.xlarge** | L4 | Ada 8.9 | 24 GB | 300 GB/s | ✅ + FP8 | $0.805 | ~$0.25–0.35 |
| **g6e.xlarge** | L40S | Ada 8.9 | 48 GB | 864 GB/s | ✅ + FP8 | $1.861 | ~$0.55–0.75 |
| p3.2xlarge | V100 | Volta 7.0 | 16 GB | 900 GB/s | ❌ (FP16 only) | $3.06 | ~$0.92 |
| p4d.24xlarge | 8× A100 40 GB | Ampere 8.0 | 320 GB | 1555 GB/s/GPU | ✅ | $32.77 | ~$10 |

**Notes:**
- **G4dn (T4)** has the same Turing limitation as the local 2080 Ti — emulated bf16, no speedup vs local. Only useful as a free local-equivalent if you're capacity-constrained, not for time savings.
- **G5 (A10g)** is the sweet spot for our workloads: native bf16, 24 GB VRAM (handles scFoundation), 600 GB/s bandwidth.
- **G6 (L4)** is cheaper than G5 ($0.805 vs $1.006) and has FP8, but **half the memory bandwidth** (300 vs 600 GB/s). For seq_len=2048 transformer training with grad-ckpt, attention is bandwidth-bound — L4 is roughly 1.3–1.5× slower than A10g on this exact workload despite being a newer architecture. Net: g5 is ~$0.20/h more but ~30% faster, so often cheaper end-to-end.
- **G6e (L40S)** at 48 GB only matters if a future workload needs >24 GB, which Phase-3 doesn't.
- **P3 (V100)** has no bf16 (only fp16) and is 3× the price of g5; obsolete for our purposes.
- **P4d/P5 (A100/H100)** are massively overkill — minimum 8-GPU instance, no single-GPU offering.

## 3. Per-experiment cost & wall estimates

### 3.1 Phase-3-A pilot (Geneformer)

Baseline: 1 epoch × 9,500 train cells × 296 steps takes 3.1 h on the local 2080 Ti (Run #2). A10g is ~2–3× faster on this workload (native bf16 + same memory bandwidth). L4 is ~1.5–2× faster (native bf16 but bandwidth-limited).

| Experiment | Steps | 2080 Ti wall | A10g wall | A10g $ on-demand | A10g $ spot | L4 $ on-demand |
|---|---|---|---|---|---|---|
| **E5b** (3 epochs) | 888 + eval | ~10 h | ~4 h | **$4.00** | $1.40 | $4.30 |
| **E5c** (10× cells, 1 ep) | 2,968 + eval | ~30 h | ~12 h | $12.00 | $4.20 | $13.00 |
| Single seed full Phase-3-A (3 ep × full data) | ~9,000 + eval | ~90 h | ~35 h | $35 | $12 | $38 |

E5b is the clear short-term action item; on-demand A10g costs ≈ $4 vs ~10 h on the local 2080 Ti.

### 3.2 Phase-3-B full sweep (after Phase-3-A closes)

3 FMs × 2 folds × 3 seeds = 18 fine-tunes. Per kickoff §6 estimate at "30–60 GPU-hours" total on local; A10g cuts that to ~12–25 GPU-h.

| Strategy | A10g hours | $/h | Total $ |
|---|---|---|---|
| Sequential g5.xlarge **on-demand** | 25 | 1.006 | **$25** |
| Sequential g5.xlarge **spot** | 25 | 0.35 | **$9** |
| Parallel 4× on g5.12xlarge on-demand (same total, 6 h wall) | 6 × 4 GPUs | 5.672 (instance) | $34 |
| Parallel 4× on g5.12xlarge spot | 6 × 4 GPUs | 1.50 | $9 |

**Sequential single-GPU is more $-efficient than 4-GPU parallel** for our scale (LoRA fine-tunes don't benefit from data parallel on this size). Spot gives 3× cost reduction — viable if we add mid-run checkpointing (currently checkpoint is end-of-training only; spot interruption loses the run).

### 3.3 scFoundation (deferred to Phase-3-B + Phase-4)

Per kickoff doc §2 DECIDE-A: scFoundation needs ≥16 GB safe, 24 GB comfortable. Local 2080 Ti is at-risk.

- 6 fine-tunes (1 fold × 3 seeds × 2 folds) × ~4 h on A10g × $1.006/h = **~$24 on-demand, ~$8 spot**.
- Alternative kickoff §2 estimate was "$40 on A100" — A10g is cheaper because LoRA on a 3B model still fits in 24 GB and we don't need A100's compute density.

## 4. Storage and network

- **Code + small artifacts**: <1 GB. Trivial on any instance.
- **FM checkpoints** (`save/`): ~5 GB total (Geneformer 440 MB, scGPT 200 MB, scFoundation 12 GB at fp32 — but compressed checkpoint is ~6 GB, UCE 5.6 GB). Pre-stage on S3 once, sync to EBS at instance launch.
- **Harmonized data** (`data/cohorts/integrated/*.h5ad`): need to check size — probably 5–15 GB for all 5 cell types. Stage on S3.
- **EBS gp3** at $0.08/GB-month: 100 GB × $0.08 / 30 = ~$0.27/day idle. Negligible vs compute.
- **Network egress**: zero if we don't pull artifacts back to local until done. Push final per-donor CSVs + summary CSVs (small) at end.

Practical setup: one g5.xlarge spawned for each experiment, EBS volume cloned from a "ready" snapshot with code + data + FM checkpoints pre-staged. ~5 min boot to first step.

## 5. Spot vs on-demand strategy

| Run type | Recommendation | Reasoning |
|---|---|---|
| E5b, E5c (single 4–12 h runs, irreplaceable) | **On-demand** | $4–12 cost is small; spot interruption loses 4+ h of work without mid-run ckpt. |
| Phase-3-B sweep (18 × ~3 h independent runs) | **Spot, with retry** | Per-run loss is ≤3 h; aggregate over 18 absorbs spottiness; $9 vs $25 saving. Add a `--resume-from-step` option to `train_loop.py` to bound the loss further. |
| Phase-4 ablation (single run) | On-demand | Ditto E5b reasoning. |
| Headline Phase-3 results we can't afford to delay | On-demand | Risk-adjusted. |

## 6. Recommendations

1. **For E5b (next experiment)**: launch on **g5.xlarge on-demand** (~$4, ~4 h). Faster turnaround than local 2080 Ti's ~10 h, no spot risk for an irreplaceable diagnostic run.

2. **For Phase-3-B full sweep** (after Phase-3-A closes): default to **g5.xlarge spot** with a `--resume-from-step` capability added to `train_loop.py`. Total ~$9–15 vs ~$25–35 on-demand.

3. **For scFoundation (Phase-3-B / Phase-4)**: g5.xlarge on-demand for the first fine-tune to validate the memory profile; if it fits, batch the remaining 5 on spot. Total ~$15–25.

4. **Skip G6 (L4)** for our workload despite the lower hourly rate — A10g's 2× memory bandwidth wins on net throughput for seq_len=2048 attention with grad-checkpointing.

5. **Skip multi-GPU instances** (g5.12xlarge etc.). LoRA on 110M–3B-param backbones doesn't benefit from data parallel at our 32-effective-batch scale; sequential single-GPU is cheaper.

6. **Pre-stage** code + data + FM checkpoints on S3 in a "phase3-ready" prefix; bake into a launch script that pulls them at boot. 5-min boot beats 30-min download per launch.

## 7. Mid-run checkpointing (prerequisite for spot)

`src/finetune/train_loop.py` currently saves the LoRA + head state only at end-of-training. To use spot safely, add:
- Save trainable state every N optimizer steps to a designated path.
- On `--resume-from-checkpoint PATH` flag, load the saved state and continue from `global_step`.
- Spot interruption loses ≤ N steps of work.

Implementation: ~30 LOC in `train_loop.py`. Worth doing before Phase-3-B begins.
