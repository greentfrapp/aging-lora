"""Phase-3 LoRA fine-tune CLI.

Usage:
    uv run python -m src.finetune.cli \
        --fm geneformer \
        --fold loco_onek1k \
        --seed 0 \
        --cell-type "CD4+ T" \
        --max-cells-per-donor 500 \
        --eval-max-cells-per-donor 200

Add --smoke to run a 1-minute CPU-friendly validation pass.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data_loader import GeneformerVocab, TokenizedAnnData, select_indices
from .lora_wrappers.geneformer import build_geneformer_lora, trainable_param_summary
from .train_loop import TrainConfig, train


def _slug(s: str) -> str:
    return s.replace("+", "p").replace(" ", "_").replace("/", "_")


def _load_fold(fold_id: str) -> dict:
    folds = json.loads(Path("data/loco_folds.json").read_text())["folds"]
    for f in folds:
        if f["fold_id"] == fold_id:
            return f
    raise ValueError(f"unknown fold {fold_id}; have: {[f['fold_id'] for f in folds]}")


def _h5ad_for_celltype(cell_type: str) -> Path:
    return Path("data/cohorts/integrated") / f"{_slug(cell_type)}.h5ad"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fm", choices=["geneformer"], required=True)
    p.add_argument("--fold", required=True, help="fold_id from data/loco_folds.json")
    p.add_argument("--cell-type", default="CD4+ T")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--gradient-checkpointing", action="store_true", default=True)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--max-cells-per-donor", type=int, default=500)
    p.add_argument("--eval-max-cells-per-donor", type=int, default=200)
    p.add_argument("--max-train-steps", type=int, default=None)
    p.add_argument("--eval-every", type=int, default=99,
                   help="run mid-training eval every N epochs (default: only final eval)")
    p.add_argument("--log-every", type=int, default=25,
                   help="log a train-step row every N optimizer steps")
    p.add_argument("--device", default=None, help="cuda / cpu / auto")
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--no-bf16", dest="bf16", action="store_false")
    p.add_argument("--smoke", action="store_true",
                   help="tiny CPU-friendly run for code validation")
    p.add_argument("--gpu-smoke", action="store_true",
                   help="small GPU run (~5 min) to validate full CLI path before real run")
    p.add_argument("--output-dir", default="results/baselines/fm_finetuned")
    p.add_argument("--run-tag-suffix", default=None,
                   help="optional suffix appended to run_tag; isolates output paths "
                        "for ablation/smoke runs (e.g. '_e2_lr2e-4')")
    args = p.parse_args()

    if args.smoke:
        args.epochs = 1
        args.batch_size = 2
        args.grad_accum = 1
        args.seq_len = 256
        args.max_cells_per_donor = 4
        args.eval_max_cells_per_donor = 4
        args.max_train_steps = 3
        args.device = args.device or "cpu"
        args.bf16 = False  # CPU bf16 compute can be flaky
        # Smoke also restricts donors (handled below).
    elif args.gpu_smoke:
        # ~5 min validation on GPU. Real config (seq=2048, b=8) but tiny data
        # so we can confirm the full CLI path runs end-to-end.
        args.epochs = 1
        args.batch_size = 8
        args.grad_accum = 1
        args.seq_len = 2048
        args.max_cells_per_donor = 25
        args.eval_max_cells_per_donor = 25
        args.max_train_steps = 20

    if args.device in (None, "auto"):
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    fold = _load_fold(args.fold)
    h5ad = _h5ad_for_celltype(args.cell_type)
    vocab = GeneformerVocab.load()

    print(f"[cli] fold={fold['fold_id']} train={fold['train_cohorts']} "
          f"holdout={fold['holdout_cohort']} h5ad={h5ad}", flush=True)

    train_idx, train_ages, train_donors = select_indices(
        h5ad,
        cell_type=args.cell_type,
        cohorts=fold["train_cohorts"],
        max_cells_per_donor=args.max_cells_per_donor,
        rng_seed=args.seed,
    )
    eval_idx, eval_ages, eval_donors = select_indices(
        h5ad,
        cell_type=args.cell_type,
        cohorts=[fold["holdout_cohort"]],
        max_cells_per_donor=args.eval_max_cells_per_donor,
        rng_seed=args.seed,
    )

    if args.smoke:
        # restrict to 4 train donors, 2 eval donors
        td = np.unique(train_donors)[:4]
        keep = np.isin(train_donors, td)
        train_idx, train_ages, train_donors = train_idx[keep], train_ages[keep], train_donors[keep]
        ed = np.unique(eval_donors)[:2]
        keep_e = np.isin(eval_donors, ed)
        eval_idx, eval_ages, eval_donors = eval_idx[keep_e], eval_ages[keep_e], eval_donors[keep_e]
    elif args.gpu_smoke:
        # 10 train donors, 5 eval donors -> ~250 train cells, ~125 eval cells
        td = np.unique(train_donors)[:10]
        keep = np.isin(train_donors, td)
        train_idx, train_ages, train_donors = train_idx[keep], train_ages[keep], train_donors[keep]
        ed = np.unique(eval_donors)[:5]
        keep_e = np.isin(eval_donors, ed)
        eval_idx, eval_ages, eval_donors = eval_idx[keep_e], eval_ages[keep_e], eval_donors[keep_e]

    print(f"[cli] train cells={len(train_idx)} donors={len(np.unique(train_donors))}", flush=True)
    print(f"[cli] eval cells={len(eval_idx)} donors={len(np.unique(eval_donors))}", flush=True)

    # Per-cell mean is the right initialization since per-cell MSE is what we
    # minimize; per-donor median aggregation happens only at eval time.
    train_age_mean = float(np.mean(train_ages)) if len(train_ages) else 0.0
    print(f"[cli] head bias init (= mean train age per-cell): {train_age_mean:.2f}", flush=True)

    train_ds = TokenizedAnnData(
        h5ad, train_idx, vocab, seq_len=args.seq_len,
        ages=train_ages, donors=train_donors,
    )
    eval_ds = TokenizedAnnData(
        h5ad, eval_idx, vocab, seq_len=args.seq_len,
        ages=eval_ages, donors=eval_donors,
    )

    print("[cli] building model + LoRA…", flush=True)
    model = build_geneformer_lora(
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        gradient_checkpointing=args.gradient_checkpointing and device.type == "cuda",
        head_bias_init=train_age_mean,
    )
    model.to(device)
    summary = trainable_param_summary(model)
    print(f"[cli] params: {summary}", flush=True)

    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        head_lr=args.head_lr,
        bf16=args.bf16 and device.type == "cuda",
        eval_bf16=False,
        seq_len=args.seq_len,
        max_train_steps=args.max_train_steps,
        eval_every=args.eval_every,
        log_every=args.log_every,
    )

    out_root = Path(args.output_dir) / args.fm
    run_tag = f"{args.fold}_seed{args.seed}_{_slug(args.cell_type)}"
    if args.smoke:
        run_tag = "smoke_" + run_tag
    elif args.gpu_smoke:
        run_tag = "gpusmoke_" + run_tag
    if args.run_tag_suffix:
        run_tag = run_tag + args.run_tag_suffix
    log_path = Path("logs/phase3") / f"{args.fm}_{run_tag}.jsonl"
    ckpt_path = out_root / "checkpoints" / f"{run_tag}.pt"

    t_start = time.time()
    result = train(
        model=model,
        train_ds=train_ds,
        eval_ds=eval_ds,
        cfg=cfg,
        device=device,
        log_path=log_path,
        ckpt_path=None if (args.smoke or args.gpu_smoke) else ckpt_path,
    )
    wall = time.time() - t_start

    final = result["history"][-1] if result["history"] else {"mae": float("nan"), "pearson_r": float("nan"), "n_donors": 0, "per_donor_predictions": []}
    summary_row = {
        "fm": args.fm,
        "fold": args.fold,
        "seed": args.seed,
        "cell_type": args.cell_type,
        "eval_cohort": fold["holdout_cohort"],
        "n_train_cells": int(len(train_idx)),
        "n_eval_cells": int(len(eval_idx)),
        "n_eval_donors": int(final["n_donors"]),
        "mae_y": float(final["mae"]),
        "pearson_r": float(final["pearson_r"]),
        "wall_s": float(wall),
        "device": str(device),
        "smoke": bool(args.smoke),
        "params": json.dumps(summary),
    }

    summary_csv = out_root / "summary.csv"
    if summary_csv.exists():
        existing = pd.read_csv(summary_csv)
        df = pd.concat([existing, pd.DataFrame([summary_row])], ignore_index=True)
    else:
        df = pd.DataFrame([summary_row])
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(summary_csv, index=False)

    if not args.smoke and final.get("per_donor_predictions"):
        per_donor_dir = out_root / "per_donor"
        per_donor_dir.mkdir(parents=True, exist_ok=True)
        per_donor_path = per_donor_dir / f"{run_tag}.csv"
        pd.DataFrame(final["per_donor_predictions"]).to_csv(per_donor_path, index=False)

    runtime_log = Path("compute/runtime_log.csv")
    runtime_row = {
        "fm": args.fm,
        "fold": args.fold,
        "seed": args.seed,
        "cell_type": args.cell_type,
        "wall_s": float(wall),
        "n_train_cells": int(len(train_idx)),
        "device": str(device),
        "smoke": bool(args.smoke),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if runtime_log.exists():
        rt = pd.read_csv(runtime_log)
        rt = pd.concat([rt, pd.DataFrame([runtime_row])], ignore_index=True)
    else:
        rt = pd.DataFrame([runtime_row])
    runtime_log.parent.mkdir(parents=True, exist_ok=True)
    rt.to_csv(runtime_log, index=False)

    print(f"[cli] done. mae={summary_row['mae_y']:.3f} r={summary_row['pearson_r']:.3f} "
          f"wall={summary_row['wall_s']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
