"""Extract per-donor mean Geneformer embeddings for the pseudobulk-parity probe.

For each (checkpoint, cohort, cell type), runs Geneformer inference and
mean-pools the last hidden state across attended tokens per cell, then averages
across cells per donor. Output is one D=768-dim vector per donor.

Two modes:
- `--frozen-base`: load the unmodified Geneformer V2-104M backbone (no LoRA, no
  fine-tuned head). Tells us what the FM's pretrained embeddings encode.
- `--checkpoint PATH`: load LoRA + head from a Phase-3-A fine-tune checkpoint.
  Tells us whether fine-tuning preserved or destroyed age-relevant structure.

Usage:
    # Frozen-base across one (cohort, cell type):
    uv run python scripts/extract_embeddings.py \\
        --frozen-base --cohort onek1k --cell-type "CD4+ T" \\
        --output-tag frozen_base

    # Fine-tuned checkpoint:
    uv run python scripts/extract_embeddings.py \\
        --checkpoint results/baselines/fm_finetuned/geneformer/checkpoints/loco_onek1k_seed0_CD4p_T_e5b.pt \\
        --cohort onek1k --cell-type "CD4+ T" --output-tag loco_onek1k_seed0_e5b
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.finetune.data_loader import GeneformerVocab, TokenizedAnnData, select_indices
from src.finetune.lora_wrappers.geneformer import GeneformerRegressor, build_geneformer_lora
from src.finetune.train_loop import _collate


CELL_TYPE_TO_FILE = {
    "CD4+ T": "CD4p_T.h5ad",
    "CD8+ T": "CD8p_T.h5ad",
    "B": "B.h5ad",
    "NK": "NK.h5ad",
    "Monocyte": "Monocyte.h5ad",
}


def _h5ad_for_cohort(cohort: str, cell_type: str) -> Path:
    if cohort == "aida":
        return Path("data/cohorts/aida_eval") / CELL_TYPE_TO_FILE[cell_type]
    return Path("data/cohorts/integrated") / CELL_TYPE_TO_FILE[cell_type]


def _aida_donors() -> list[str]:
    raw = json.loads(Path("data/aida_split.json").read_text())["ancestry_shift_mae_donors"]
    return [d if d.startswith("aida:") else f"aida:{d}" for d in raw]


def _build_model(checkpoint: Path | None, pool: str, head_bias_init: float, device: torch.device):
    if checkpoint is None:
        # Frozen base: GeneformerRegressor without LoRA, head untrained.
        # We use the regressor wrapper to get the same forward path as fine-tuned, but
        # we won't read the head output — only the pre-head pooled embedding.
        model = GeneformerRegressor(bias_init=head_bias_init, pool=pool)
    else:
        model = build_geneformer_lora(
            gradient_checkpointing=False,
            head_bias_init=head_bias_init,
            pool=pool,
        )
        sd = torch.load(checkpoint, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        bad_missing = [k for k in missing if "lora_" in k or k.startswith("head.")]
        if bad_missing:
            raise SystemExit(f"checkpoint missing required LoRA/head keys: {bad_missing[:5]}")
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def _pool_embedding(model, ids, mask, pool: str, bf16: bool, device: torch.device) -> torch.Tensor:
    """Run backbone forward, return per-cell pooled embedding (B, H)."""
    autocast_dtype = torch.bfloat16 if bf16 else torch.float32
    backbone = getattr(model, "backbone")
    with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=bf16):
        out = backbone(input_ids=ids, attention_mask=mask)
    h = out.last_hidden_state  # (B, T, H)
    if pool == "cls":
        feat = h[:, 0]
    else:  # mean
        m = mask.unsqueeze(-1).to(h.dtype)
        feat = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
    return feat.float()


def main():
    p = argparse.ArgumentParser()
    src_grp = p.add_mutually_exclusive_group(required=True)
    src_grp.add_argument("--frozen-base", action="store_true",
                         help="Use unmodified Geneformer V2-104M; ignore --checkpoint")
    src_grp.add_argument("--checkpoint", type=Path,
                         help="Path to LoRA + head .pt for fine-tuned extraction")
    p.add_argument("--cohort", required=True, choices=["onek1k", "stephenson", "terekhova", "aida"])
    p.add_argument("--cell-type", required=True, choices=list(CELL_TYPE_TO_FILE.keys()))
    p.add_argument("--pool", choices=["cls", "mean"], default="mean",
                   help="must match training-time pool for fine-tuned checkpoints")
    p.add_argument("--max-cells-per-donor", type=int, default=20)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--bf16", action="store_true", default=False,
                   help="off by default to match training-time eval_bf16=False")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-tag", required=True,
                   help="identifier appearing in output filename (e.g. 'frozen_base', 'loco_onek1k_seed0_e5b')")
    p.add_argument("--output-dir", default="results/phase3/embeddings")
    p.add_argument("--head-bias-init", type=float, default=48.93,
                   help="ignored by extraction path; kept for the regressor wrapper API")
    args = p.parse_args()

    if args.device in (None, "auto"):
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    h5ad = _h5ad_for_cohort(args.cohort, args.cell_type)
    if not h5ad.exists():
        raise SystemExit(f"h5ad not found: {h5ad}")
    print(f"[extract] cohort={args.cohort} cell_type={args.cell_type} h5ad={h5ad}")

    if args.cohort == "aida":
        include_donors = _aida_donors()
        cohorts_filter = None
    else:
        include_donors = None
        cohorts_filter = [args.cohort]

    idx, ages, donors = select_indices(
        h5ad,
        cell_type=args.cell_type,
        cohorts=cohorts_filter,
        include_donors=include_donors,
        max_cells_per_donor=args.max_cells_per_donor,
        rng_seed=args.seed,
    )
    n_donors = len(np.unique(donors))
    print(f"[extract] cells={len(idx)} donors={n_donors}")
    if n_donors == 0:
        raise SystemExit("zero donors selected; check filters")

    vocab = GeneformerVocab.load()
    ds = TokenizedAnnData(h5ad, idx, vocab, seq_len=args.seq_len, ages=ages, donors=donors)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, num_workers=0,
        collate_fn=_collate, pin_memory=device.type == "cuda",
    )

    print(f"[extract] mode={'frozen-base' if args.frozen_base else f'fine-tuned ({args.checkpoint.name})'}")
    model = _build_model(
        checkpoint=None if args.frozen_base else args.checkpoint,
        pool=args.pool, head_bias_init=args.head_bias_init, device=device,
    )

    t0 = time.time()
    all_emb: list[np.ndarray] = []
    all_donors: list[str] = []
    all_ages: list[float] = []
    for batch in loader:
        ids = batch["input_ids"].to(device, non_blocking=True)
        mask = batch["attention_mask"].to(device, non_blocking=True)
        feat = _pool_embedding(model, ids, mask, args.pool, args.bf16, device)
        all_emb.append(feat.cpu().numpy())
        all_donors.extend(list(batch["donor"]))
        all_ages.extend(batch["age"].numpy().tolist())
    cell_emb = np.concatenate(all_emb, axis=0)  # (N_cells, H)
    cell_donors = np.asarray(all_donors)
    cell_ages = np.asarray(all_ages)

    # Aggregate per donor: mean embedding, single age (donors are uniform-aged).
    unique_donors = np.unique(cell_donors)
    donor_emb = np.stack([cell_emb[cell_donors == d].mean(axis=0) for d in unique_donors])
    donor_age = np.array([cell_ages[cell_donors == d][0] for d in unique_donors], dtype=np.float32)

    elapsed = time.time() - t0
    print(f"[extract] aggregated to {len(unique_donors)} donors × {donor_emb.shape[1]}-dim in {elapsed:.1f}s")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cell_type_slug = args.cell_type.replace("+", "p").replace(" ", "_")
    out_path = out_dir / f"{args.cohort}_{cell_type_slug}_{args.output_tag}.npz"
    np.savez(
        out_path,
        donor_ids=unique_donors,
        ages=donor_age,
        embeddings=donor_emb.astype(np.float32),
        meta=np.array([
            f"cohort={args.cohort}", f"cell_type={args.cell_type}", f"pool={args.pool}",
            f"max_cells_per_donor={args.max_cells_per_donor}", f"seq_len={args.seq_len}",
            f"checkpoint={args.checkpoint if args.checkpoint else 'frozen-base'}",
            f"timestamp={time.strftime('%Y-%m-%dT%H:%M:%S')}",
        ]),
    )
    print(f"[extract] wrote {out_path} ({donor_emb.shape})")


if __name__ == "__main__":
    main()
