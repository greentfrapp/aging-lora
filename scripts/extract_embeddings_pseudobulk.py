"""D.18 — Variant 2 pseudobulk-input Geneformer + ridge readout.

Per the step-back review (scratchpad/step_back_review.md): the gene-EN baseline
operates on log1p-mean pseudobulk; an apples-to-apples FM comparison requires
matching that input shape. This script aggregates cells per donor *before*
feeding to Geneformer, then extracts per-donor layered embeddings.

Per donor:
  1. Sum raw counts across selected cells → one pseudo-count vector per donor.
  2. Run Geneformer's standard rank-value tokenization on that vector as if it
     were a single cell.
  3. Forward (frozen base or fine-tuned LoRA) with `output_hidden_states=True`.
  4. Mean-pool across attended positions per layer.

Output is (n_layers, n_donors, 768) per cohort × cell-type — the same shape as
extract_embeddings_layered.py, so donor_ridge_layered.py works unchanged.

Usage:
    uv run python scripts/extract_embeddings_pseudobulk.py \\
        --cohort onek1k --cell-type "CD4+ T" --output-tag pseudobulk_frozen
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp
import torch

from src.finetune.data_loader import GeneformerVocab, build_var_token_arrays, select_indices
from src.finetune.lora_wrappers.geneformer import GeneformerRegressor, build_geneformer_lora


CELL_TYPE_TO_FILE = {
    "CD4+ T": "CD4p_T.h5ad",
    "B": "B.h5ad",
    "NK": "NK.h5ad",
}


def _h5ad(cohort: str, cell_type: str) -> Path:
    base = Path("data/cohorts/aida_eval") if cohort == "aida" else Path("data/cohorts/integrated")
    return base / CELL_TYPE_TO_FILE[cell_type]


def _aida_donors() -> list[str]:
    raw = json.loads(Path("data/aida_split.json").read_text())["ancestry_shift_mae_donors"]
    return [d if d.startswith("aida:") else f"aida:{d}" for d in raw]


def _tokenize_pseudobulk(donor_counts: np.ndarray, token_ids: np.ndarray, medians: np.ndarray,
                          pad_token_id: int, cls_token_id: int, seq_len: int) -> np.ndarray:
    """Apply Geneformer rank-value tokenization to a single donor-aggregated count vector."""
    total = donor_counts.sum()
    if total <= 0:
        ids = np.full(seq_len, pad_token_id, dtype=np.int64)
        ids[0] = cls_token_id
        return ids
    norm = donor_counts * (1e4 / total) / medians
    nz = norm > 0
    if not nz.any():
        ids = np.full(seq_len, pad_token_id, dtype=np.int64)
        ids[0] = cls_token_id
        return ids
    nz_idx = np.where(nz)[0]
    order = nz_idx[np.argsort(-norm[nz_idx], kind="stable")]
    top = order[: seq_len - 1]
    ids = np.full(seq_len, pad_token_id, dtype=np.int64)
    ids[0] = cls_token_id
    ids[1 : 1 + len(top)] = token_ids[top]
    return ids


@torch.no_grad()
def _pool_all_layers(model, ids, mask, bf16: bool, device: torch.device) -> torch.Tensor:
    autocast_dtype = torch.bfloat16 if bf16 else torch.float32
    backbone = getattr(model, "backbone")
    with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=bf16):
        out = backbone(input_ids=ids, attention_mask=mask, output_hidden_states=True)
    hidden_states = out.hidden_states  # length = num_layers + 1
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
    src_grp.add_argument("--frozen-base", action="store_true", default=True)
    src_grp.add_argument("--checkpoint", type=Path, default=None,
                         help="Path to LoRA + head .pt for fine-tuned extraction")
    p.add_argument("--cohort", required=True, choices=["onek1k", "stephenson", "terekhova", "aida"])
    p.add_argument("--cell-type", required=True, choices=list(CELL_TYPE_TO_FILE.keys()))
    p.add_argument("--max-cells-per-donor", type=int, default=500,
                   help="Cap on cells used per donor for the pseudobulk sum.")
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--bf16", action="store_true", default=False)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--output-tag", default="pseudobulk_frozen_alllayers")
    p.add_argument("--output-dir", default="results/phase3/embeddings_pseudobulk")
    args = p.parse_args()
    if args.checkpoint is not None:
        args.frozen_base = False

    if args.device in (None, "auto"):
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    h5ad = _h5ad(args.cohort, args.cell_type)
    if not h5ad.exists():
        raise SystemExit(f"h5ad not found: {h5ad}")
    print(f"[pb-extract] cohort={args.cohort} cell_type={args.cell_type} h5ad={h5ad}", flush=True)

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
    unique_donors = np.unique(donors)
    n_donors = len(unique_donors)
    print(f"[pb-extract] {len(idx)} cells across {n_donors} donors → 1 pseudobulk vector per donor", flush=True)

    vocab = GeneformerVocab.load()
    a = ad.read_h5ad(h5ad, backed="r")
    tids, meds = build_var_token_arrays(a.var, vocab)
    valid = tids >= 0
    valid_token_ids = tids[valid]
    valid_medians = meds[valid]
    valid_cols = np.where(valid)[0]
    print(f"[pb-extract] {len(valid_cols):,} / {len(tids):,} h5ad genes have Geneformer tokens + medians", flush=True)

    # Aggregate per donor: sum raw counts across selected cells → 1 vector per donor
    print(f"[pb-extract] aggregating donor pseudobulk vectors...", flush=True)
    t0 = time.time()
    donor_counts = np.zeros((n_donors, len(valid_cols)), dtype=np.float32)
    donor_age = np.zeros(n_donors, dtype=np.float32)
    for i, d in enumerate(unique_donors):
        m = donors == d
        sub_idx = idx[m]
        rows = a.X[sub_idx]
        if sp.issparse(rows):
            rows_dense = rows.toarray().astype(np.float32)
        else:
            rows_dense = np.asarray(rows, dtype=np.float32)
        donor_counts[i] = rows_dense[:, valid_cols].sum(axis=0)
        donor_age[i] = ages[m][0]
    a.file.close()
    print(f"[pb-extract] aggregated in {time.time()-t0:.1f}s; counts shape {donor_counts.shape}", flush=True)

    # Tokenize per donor as a single pseudo-cell
    donor_input_ids = np.zeros((n_donors, args.seq_len), dtype=np.int64)
    donor_attn = np.zeros((n_donors, args.seq_len), dtype=np.int64)
    for i in range(n_donors):
        ids = _tokenize_pseudobulk(donor_counts[i], valid_token_ids, valid_medians,
                                    vocab.pad_token_id, vocab.cls_token_id, args.seq_len)
        donor_input_ids[i] = ids
        donor_attn[i] = (ids != vocab.pad_token_id).astype(np.int64)
    print(f"[pb-extract] tokenized {n_donors} donor pseudobulks", flush=True)

    # Build model
    if args.frozen_base or args.checkpoint is None:
        print(f"[pb-extract] mode=frozen-base", flush=True)
        model = GeneformerRegressor(bias_init=0.0, pool="mean")
    else:
        print(f"[pb-extract] mode=fine-tuned ({args.checkpoint.name}) lora_rank={args.lora_rank}", flush=True)
        model = build_geneformer_lora(gradient_checkpointing=False, head_bias_init=0.0,
                                       pool="mean", lora_rank=args.lora_rank)
        sd = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(sd, strict=False)
    model.to(device).eval()

    # Forward in batches
    print(f"[pb-extract] forward in batches of {args.batch_size}...", flush=True)
    bs = args.batch_size
    n_layers = None
    donor_emb = None
    t0 = time.time()
    for batch_start in range(0, n_donors, bs):
        ids_b = torch.from_numpy(donor_input_ids[batch_start : batch_start + bs]).to(device)
        mask_b = torch.from_numpy(donor_attn[batch_start : batch_start + bs]).to(device)
        feat_lbh = _pool_all_layers(model, ids_b, mask_b, args.bf16, device)
        feat_np = feat_lbh.cpu().numpy()
        if donor_emb is None:
            n_layers = feat_np.shape[0]
            donor_emb = np.zeros((n_layers, n_donors, feat_np.shape[2]), dtype=np.float32)
        donor_emb[:, batch_start : batch_start + feat_np.shape[1], :] = feat_np

    print(f"[pb-extract] embeddings shape {donor_emb.shape}, elapsed {time.time()-t0:.1f}s", flush=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cell_type_slug = args.cell_type.replace("+", "p").replace(" ", "_")
    out_path = out_dir / f"{args.cohort}_{cell_type_slug}_{args.output_tag}.npz"
    np.savez(
        out_path,
        donor_ids=unique_donors,
        ages=donor_age,
        embeddings_per_layer=donor_emb,
        meta=np.array([
            f"cohort={args.cohort}", f"cell_type={args.cell_type}",
            f"max_cells_per_donor={args.max_cells_per_donor}",
            f"seq_len={args.seq_len}", f"n_layers_incl_emb={n_layers}",
            f"timestamp={time.strftime('%Y-%m-%dT%H:%M:%S')}",
            "input=donor-pseudobulk-summed-counts → rank-value tokenization",
        ]),
    )
    print(f"[pb-extract] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
