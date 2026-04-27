"""Score a fine-tuned Geneformer LoRA checkpoint on AIDA CD4+T.

Per-checkpoint inference pass against the AIDA `ancestry_shift_mae` donor half
(307 donors, frozen in `data/aida_split.json`). The other 318-donor half is
reserved for Phase-5 age-axis alignment and is NOT touched here.

Output:
- `results/phase3/aida_per_donor/{run_tag}.csv`
- one row appended to `results/phase3/aida_summary.csv`

Usage:
    uv run python scripts/score_aida.py \\
        --checkpoint results/baselines/fm_finetuned/geneformer/checkpoints/loco_onek1k_seed0_CD4p_T_e5b.pt \\
        --pool mean --run-tag loco_onek1k_seed0_e5b
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.finetune.data_loader import GeneformerVocab, TokenizedAnnData, select_indices
from src.finetune.lora_wrappers.geneformer import build_geneformer_lora
from src.finetune.train_loop import _collate, evaluate_per_donor


CELL_TYPE_TO_FILE = {
    "CD4+ T": "CD4p_T.h5ad",
    "CD8+ T": "CD8p_T.h5ad",
    "B": "B.h5ad",
    "NK": "NK.h5ad",
    "Monocyte": "Monocyte.h5ad",
}


def _load_aida_donors() -> list[str]:
    """Return donor IDs in the harmonized h5ad's namespace (`aida:<bare>`).

    The frozen `aida_split.json` records bare AIDA accession IDs (e.g.
    `IN_NIB_H021`) from the raw metadata, but `data/cohorts/aida_eval/*.h5ad`
    namespaces them as `aida:<bare>` to match the multi-cohort convention used
    by `loco_folds.json`. Translate here so callers can pass-through to
    `select_indices(include_donors=...)`.
    """
    raw = json.loads(Path("data/aida_split.json").read_text())["ancestry_shift_mae_donors"]
    return [d if d.startswith("aida:") else f"aida:{d}" for d in raw]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Path to LoRA + head .pt")
    p.add_argument("--cell-type", default="CD4+ T", choices=list(CELL_TYPE_TO_FILE.keys()))
    p.add_argument("--pool", choices=["cls", "mean"], default="mean",
                   help="must match the training-time pooling")
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--eval-max-cells-per-donor", type=int, default=20)
    p.add_argument("--head-bias-init", type=float, default=48.93,
                   help="ignored — overwritten by the loaded checkpoint")
    p.add_argument("--bf16", action="store_true", default=False,
                   help="off by default to match training-time eval_bf16=False")
    p.add_argument("--device", default=None)
    p.add_argument("--run-tag", required=True,
                   help="identifier for this checkpoint, appears in output filenames")
    p.add_argument("--output-dir", default="results/phase3")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.device in (None, "auto"):
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    h5ad = Path("data/cohorts/aida_eval") / CELL_TYPE_TO_FILE[args.cell_type]
    if not h5ad.exists():
        raise SystemExit(f"AIDA h5ad not found: {h5ad}")
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise SystemExit(f"checkpoint not found: {ckpt_path}")

    aida_donors = _load_aida_donors()
    print(f"[aida] cell_type={args.cell_type} h5ad={h5ad}")
    print(f"[aida] ancestry_shift_mae donors={len(aida_donors)} (from data/aida_split.json)")

    eval_idx, eval_ages, eval_donors = select_indices(
        h5ad,
        cell_type=args.cell_type,
        cohorts=None,                       # AIDA h5ad is single-cohort
        include_donors=aida_donors,
        max_cells_per_donor=args.eval_max_cells_per_donor,
        rng_seed=args.seed,
    )
    n_donors_present = len(np.unique(eval_donors))
    print(f"[aida] eval cells={len(eval_idx)} donors={n_donors_present}")

    if n_donors_present == 0:
        raise SystemExit("zero AIDA donors selected; check donor-id format")

    vocab = GeneformerVocab.load()
    eval_ds = TokenizedAnnData(
        h5ad, eval_idx, vocab, seq_len=args.seq_len,
        ages=eval_ages, donors=eval_donors,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=_collate,
        pin_memory=device.type == "cuda",
    )

    print("[aida] building model + LoRA…")
    model = build_geneformer_lora(
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        gradient_checkpointing=False,        # inference: no need
        head_bias_init=args.head_bias_init,
        pool=args.pool,
    )

    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # Checkpoint contains LoRA + head only; strict=False to tolerate frozen-base keys.
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # Frozen base keys legitimately appear in `missing`; LoRA + head keys must NOT.
    bad_missing = [k for k in missing if "lora_" in k or k.startswith("head.")]
    if bad_missing:
        raise SystemExit(f"checkpoint missing required LoRA/head keys: {bad_missing[:5]}")
    if unexpected:
        print(f"[aida] note: {len(unexpected)} unexpected keys ignored (sample: {unexpected[:3]})")
    model.to(device)
    model.eval()

    t0 = time.time()
    metrics = evaluate_per_donor(model, eval_loader, device, bf16=args.bf16)
    wall = time.time() - t0
    print(f"[aida] mae={metrics['mae']:.3f} r={metrics['pearson_r']:.3f} "
          f"n_donors={metrics['n_donors']} wall={wall:.1f}s")

    out_root = Path(args.output_dir)
    per_donor_dir = out_root / "aida_per_donor"
    per_donor_dir.mkdir(parents=True, exist_ok=True)
    per_donor_csv = per_donor_dir / f"{args.run_tag}.csv"
    pd.DataFrame(metrics["per_donor_predictions"]).to_csv(per_donor_csv, index=False)
    print(f"[aida] wrote per-donor predictions: {per_donor_csv}")

    summary_csv = out_root / "aida_summary.csv"
    row = {
        "run_tag": args.run_tag,
        "checkpoint": str(ckpt_path),
        "cell_type": args.cell_type,
        "pool": args.pool,
        "n_eval_cells": int(len(eval_idx)),
        "n_eval_donors": int(metrics["n_donors"]),
        "mae_y": float(metrics["mae"]),
        "pearson_r": float(metrics["pearson_r"]),
        "wall_s": float(wall),
        "device": str(device),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if summary_csv.exists():
        df = pd.concat([pd.read_csv(summary_csv), pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(summary_csv, index=False)
    print(f"[aida] appended summary row: {summary_csv}")


if __name__ == "__main__":
    main()
