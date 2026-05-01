"""Variant 3: extract mean-pooled hidden states from EVERY layer of frozen Geneformer V2-104M.

For each (cohort, cell type), runs frozen Geneformer inference with
`output_hidden_states=True` and mean-pools each layer's hidden state across
attended tokens per cell, then averages across cells per donor. Output is one
(L, n_donors, 768) tensor per (cohort, cell type) where L = num layers + 1
(layer 0 = embedding output, layers 1..N = transformer layers).

Used by `scripts/donor_ridge_layered.py` to fit ridge per layer and produce
the layer-wise R-vs-depth curve (Variant 3 of the diagnostic ladder).

Usage:
    uv run python scripts/extract_embeddings_layered.py \\
        --cohort onek1k --cell-type "CD4+ T" --output-tag frozen_base_alllayers
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
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


@torch.no_grad()
def _pool_all_layers(model, ids, mask, bf16: bool, device: torch.device) -> torch.Tensor:
    """Returns (L, B, H) per-cell mean-pooled hidden states for every layer."""
    autocast_dtype = torch.bfloat16 if bf16 else torch.float32
    backbone = getattr(model, "backbone")
    with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=bf16):
        out = backbone(input_ids=ids, attention_mask=mask, output_hidden_states=True)
    # tuple of (L+1) tensors of shape (B, T, H), where L is the number of transformer layers.
    hidden_states = out.hidden_states  # length = num_layers + 1 (embedding output is hidden_states[0])
    m = mask.unsqueeze(-1).to(hidden_states[0].dtype)
    pooled_per_layer = []
    denom = m.sum(dim=1).clamp(min=1)
    for h in hidden_states:
        feat = (h * m).sum(dim=1) / denom  # (B, H)
        pooled_per_layer.append(feat.float())
    return torch.stack(pooled_per_layer, dim=0)  # (L, B, H)


def main():
    p = argparse.ArgumentParser()
    src_grp = p.add_mutually_exclusive_group()
    src_grp.add_argument("--frozen-base", action="store_true", default=True,
                         help="Use unmodified Geneformer V2-104M (default)")
    src_grp.add_argument("--checkpoint", type=Path, default=None,
                         help="Path to LoRA + head .pt for fine-tuned extraction")
    p.add_argument("--cohort", required=True, choices=["onek1k", "stephenson", "terekhova", "aida"])
    p.add_argument("--cell-type", required=True, choices=list(CELL_TYPE_TO_FILE.keys()))
    p.add_argument("--max-cells-per-donor", type=int, default=20)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--bf16", action="store_true", default=False)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-tag", default="frozen_base_alllayers")
    p.add_argument("--output-dir", default="results/phase3/embeddings_layered")
    p.add_argument("--lora-rank", type=int, default=16,
                   help="Must match the rank used to train the checkpoint (default 16 = e5b config).")
    args = p.parse_args()
    if args.checkpoint is not None:
        args.frozen_base = False

    if args.device in (None, "auto"):
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    h5ad = _h5ad_for_cohort(args.cohort, args.cell_type)
    if not h5ad.exists():
        raise SystemExit(f"h5ad not found: {h5ad}")
    print(f"[extract-L] cohort={args.cohort} cell_type={args.cell_type} h5ad={h5ad}")

    if args.cohort == "aida":
        include_donors = _aida_donors()
        cohorts_filter = None
    else:
        include_donors = None
        cohorts_filter = [args.cohort]

    idx, ages, donors = select_indices(
        h5ad, cell_type=args.cell_type, cohorts=cohorts_filter,
        include_donors=include_donors, max_cells_per_donor=args.max_cells_per_donor,
        rng_seed=args.seed,
    )
    n_donors = len(np.unique(donors))
    print(f"[extract-L] cells={len(idx)} donors={n_donors}")

    vocab = GeneformerVocab.load()
    ds = TokenizedAnnData(h5ad, idx, vocab, seq_len=args.seq_len, ages=ages, donors=donors)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, num_workers=0,
        collate_fn=_collate, pin_memory=device.type == "cuda",
    )

    if args.frozen_base or args.checkpoint is None:
        print(f"[extract-L] mode=frozen-base")
        model = GeneformerRegressor(bias_init=0.0, pool="mean")
    else:
        print(f"[extract-L] mode=fine-tuned ({args.checkpoint.name})")
        model = build_geneformer_lora(gradient_checkpointing=False, head_bias_init=0.0, pool="mean",
                                       lora_rank=args.lora_rank)
        sd = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        bad_missing = [k for k in missing if "lora_" in k or k.startswith("head.")]
        if bad_missing:
            raise SystemExit(f"checkpoint missing required LoRA/head keys: {bad_missing[:5]}")
    model.to(device).eval()

    t0 = time.time()
    # Streaming per-donor aggregation. We pre-compute the donor index from
    # select_indices' output (donors array) so memory stays O(n_donors × L × H)
    # instead of O(n_cells × L × H). cap=500 onek1k = 455k cells would be 17 GB
    # at float32; on 15 GB RAM that OOMs and the previous (May 1 01:23) crash
    # was on this code path.
    unique_donors = np.unique(donors)
    n_d = len(unique_donors)
    donor_to_idx = {d: i for i, d in enumerate(unique_donors)}
    # donor → first-seen age (donors are constant within a donor)
    first_age_by_donor: dict[str, float] = {}
    for d, a in zip(donors, ages):
        if d not in first_age_by_donor:
            first_age_by_donor[d] = float(a)

    donor_emb_sum: np.ndarray | None = None  # lazy-init on first batch (need L, H)
    donor_count = np.zeros(n_d, dtype=np.int64)
    n_layers = None
    H: int | None = None
    for batch in loader:
        ids = batch["input_ids"].to(device, non_blocking=True)
        mask = batch["attention_mask"].to(device, non_blocking=True)
        feat_lbh = _pool_all_layers(model, ids, mask, args.bf16, device)  # (L, B, H)
        feat_np = feat_lbh.cpu().numpy().astype(np.float32, copy=False)
        if donor_emb_sum is None:
            n_layers = feat_np.shape[0]
            H = feat_np.shape[2]
            donor_emb_sum = np.zeros((n_layers, n_d, H), dtype=np.float32)
        # Map this batch's donors to indices, then scatter-add per layer.
        batch_donor_idx = np.array([donor_to_idx[d] for d in batch["donor"]], dtype=np.int64)
        # np.add.at is unbuffered scatter-add; safe even when same donor_idx
        # appears multiple times in a batch.
        for L in range(n_layers):
            np.add.at(donor_emb_sum[L], batch_donor_idx, feat_np[L])
        np.add.at(donor_count, batch_donor_idx, 1)

    if donor_emb_sum is None:
        raise SystemExit("[extract-L] no batches produced; nothing to aggregate")
    safe_count = np.where(donor_count > 0, donor_count, 1).astype(np.float32)
    donor_emb = donor_emb_sum / safe_count[None, :, None]
    donor_age = np.array([first_age_by_donor[d] for d in unique_donors], dtype=np.float32)

    elapsed = time.time() - t0
    print(f"[extract-L] aggregated {n_d} donors × {n_layers} layers × {H}-dim in {elapsed:.1f}s")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cell_type_slug = args.cell_type.replace("+", "p").replace(" ", "_")
    out_path = out_dir / f"{args.cohort}_{cell_type_slug}_{args.output_tag}.npz"
    np.savez(
        out_path,
        donor_ids=unique_donors,
        ages=donor_age,
        embeddings_per_layer=donor_emb,  # (L, n_donors, H)
        meta=np.array([
            f"cohort={args.cohort}", f"cell_type={args.cell_type}",
            f"max_cells_per_donor={args.max_cells_per_donor}",
            f"seq_len={args.seq_len}", f"n_layers_incl_emb={n_layers}",
            f"timestamp={time.strftime('%Y-%m-%dT%H:%M:%S')}",
        ]),
    )
    print(f"[extract-L] wrote {out_path} (shape {donor_emb.shape})")


if __name__ == "__main__":
    main()
