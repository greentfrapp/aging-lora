"""scFoundation FM-class diagnostic — extract per-donor mean-pooled cell embeddings.

Mirrors extract_embeddings_layered.py protocol so the per-donor ridge readout
on scFoundation can be compared apples-to-apples to Geneformer §22 / §27 / §28.

Per cohort × cell-type:
  1. select_indices to get up to `max_cells_per_donor` cells per donor;
  2. project per-cell raw-count vectors onto scFoundation's 19264-gene panel
     (zero-pad missing); pre-normalize log1p(CP10k);
  3. forward through frozen scFoundation encoder (`01B-resolution` `cell` key);
  4. pool per-cell as the canonical 3072-d "all-pool" (concat of T/S tokens
     + max-pool + mean-pool of gene tokens) — matches third_party scFoundation
     `model/get_embedding.py`;
  5. mean across cells per donor → 3072-d per donor;
  6. save (n_donors, 3072) .npz alongside `donor_ids`, `ages`.

Usage:
    uv run python scripts/extract_embeddings_scfoundation.py \\
        --cohort onek1k --cell-type "CD4+ T" --output-tag scfoundation_t4_alldonors
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

# Make scFoundation's `model/` folder importable for `from load import ...`.
SCF_DIR = Path("/home/ubuntu/third_party/scFoundation/model")
if str(SCF_DIR) not in sys.path:
    sys.path.insert(0, str(SCF_DIR))

from load import gatherData, load_model_frommmf  # type: ignore

from src.finetune.data_loader import select_indices


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


def _resolve_var_symbols(var: pd.DataFrame) -> np.ndarray:
    """Best-effort gene-symbol per row.

    integrated/*.h5ad: var indexed by ENSG with `gene_symbol` populated for those,
    plus orphan rows where var.index already IS the symbol and gene_symbol is NaN.
    aida_eval/*.h5ad: var indexed by ENSG with `gene_symbol` populated for all rows.
    """
    syms = np.asarray(var.get("gene_symbol", pd.Series(np.nan, index=var.index)).astype(object))
    idx = np.asarray(var.index.astype(str))
    out = np.where(pd.isna(syms) | (syms == "nan"), idx, syms.astype(object).astype(str))
    return out.astype(str)


def _build_gene_projection(var_symbols: np.ndarray, scf_genes: list[str]):
    """Return (col_to_scf, scf_size) — `col_to_scf[i]` is the scFoundation
    panel index for h5ad-column i, or -1 if missing/duplicate."""
    sym_to_scf = {g: i for i, g in enumerate(scf_genes)}
    col_to_scf = np.full(len(var_symbols), -1, dtype=np.int64)
    seen = set()
    for i, s in enumerate(var_symbols):
        j = sym_to_scf.get(s)
        if j is None or j in seen:
            continue
        col_to_scf[i] = j
        seen.add(j)
    return col_to_scf, len(scf_genes)


@torch.no_grad()
def _forward_cell_emb(
    model,
    cfg: dict,
    expr_log_panel: torch.Tensor,  # (B, 19264) log1p(CP10k) on scFoundation gene panel
    log10_total_count: torch.Tensor,  # (B,)
    tgt_t: float,  # target highres value (default 4.0 = 1e4 reads)
    bf16: bool = False,
):
    B = expr_log_panel.shape[0]
    pad_token_id = cfg["pad_token_id"]
    # Append [tgthighres, log10(actual_total_count)] -> shape (B, 19266)
    tgt = torch.full((B, 1), float(tgt_t), device=expr_log_panel.device, dtype=expr_log_panel.dtype)
    tot = log10_total_count.to(device=expr_log_panel.device, dtype=expr_log_panel.dtype).unsqueeze(1)
    pretrain_gene_x = torch.cat([expr_log_panel, tgt, tot], dim=1)
    data_gene_ids = torch.arange(19266, device=pretrain_gene_x.device).repeat(B, 1)
    value_labels = pretrain_gene_x > 0
    x, x_padding = gatherData(pretrain_gene_x, value_labels, pad_token_id)
    position_gene_ids, _ = gatherData(data_gene_ids, value_labels, pad_token_id)
    x = model.token_emb(torch.unsqueeze(x, 2).float(), output_weight=0)
    position_emb = model.pos_emb(position_gene_ids)
    x = x + position_emb
    if bf16:
        with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16):
            geneemb = model.encoder(x, x_padding)
        geneemb = geneemb.float()
    else:
        geneemb = model.encoder(x, x_padding)  # (B, S, 768)

    geneemb1 = geneemb[:, -1, :]
    geneemb2 = geneemb[:, -2, :]
    # mask the padding positions for the mean/max pool over gene tokens
    body = geneemb[:, :-2, :]
    body_pad = x_padding[:, :-2]
    body_masked_for_max = body.masked_fill(body_pad.unsqueeze(-1), float("-inf"))
    geneemb3, _ = torch.max(body_masked_for_max, dim=1)
    body_masked_for_mean = body.masked_fill(body_pad.unsqueeze(-1), 0.0)
    denom = (~body_pad).float().sum(dim=1, keepdim=True).clamp(min=1)
    geneemb4 = body_masked_for_mean.sum(dim=1) / denom
    return torch.cat([geneemb1, geneemb2, geneemb3, geneemb4], dim=1).float()  # (B, 3072)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cohort", required=True, choices=["onek1k", "stephenson", "terekhova", "aida"])
    p.add_argument("--cell-type", required=True, choices=list(CELL_TYPE_TO_FILE.keys()))
    p.add_argument("--max-cells-per-donor", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--tgt-t", type=float, default=4.0,
                   help="target highres value (canonical scFoundation t4 = 4.0).")
    p.add_argument("--ckpt", default="save/scFoundation/models/models.ckpt")
    p.add_argument("--gene-list", default=str(SCF_DIR.parent / "OS_scRNA_gene_index.19264.tsv"))
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bf16", action="store_true", default=False,
                   help="Run encoder forward in bf16 autocast — ~50% memory, near-identical embeddings.")
    p.add_argument("--output-tag", default="scfoundation_t4")
    p.add_argument("--output-dir", default="results/phase3/embeddings_scfoundation")
    args = p.parse_args()

    if args.device in (None, "auto"):
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    h5ad = _h5ad_for_cohort(args.cohort, args.cell_type)
    if not h5ad.exists():
        raise SystemExit(f"h5ad not found: {h5ad}")
    print(f"[scf-extract] cohort={args.cohort} cell_type={args.cell_type} h5ad={h5ad}")

    # Cell selection -- mirrors extract_embeddings_layered.py
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
    n_cells = len(idx)
    n_donors = len(np.unique(donors))
    print(f"[scf-extract] cells={n_cells} donors={n_donors}")

    # scFoundation gene panel
    scf_genes = pd.read_csv(args.gene_list, sep="\t")["gene_name"].astype(str).tolist()
    if len(scf_genes) != 19264:
        raise SystemExit(f"unexpected gene-list size: {len(scf_genes)} (want 19264)")

    # h5ad gene-symbol mapping
    a = ad.read_h5ad(h5ad, backed="r")
    var_syms = _resolve_var_symbols(a.var)
    col_to_scf, scf_size = _build_gene_projection(var_syms, scf_genes)
    n_mapped = int((col_to_scf >= 0).sum())
    print(f"[scf-extract] gene-symbol -> scFoundation panel: {n_mapped:,} / {scf_size:,} mapped (h5ad has {len(var_syms):,} cols)")

    # Load model
    model, cfg = load_model_frommmf(args.ckpt, key="cell")
    model.eval()
    model.to(device)
    print(f"[scf-extract] model loaded: depth={cfg['encoder']['depth']} hidden={cfg['encoder']['hidden_dim']} pad_token_id={cfg['pad_token_id']}")

    # Iterate batches
    t0 = time.time()
    cell_emb_buf = np.zeros((n_cells, 3072), dtype=np.float32)
    cell_donors = np.empty(n_cells, dtype=object)
    cell_ages = np.zeros(n_cells, dtype=np.float32)

    valid_cols = np.where(col_to_scf >= 0)[0]
    valid_scf_idx = col_to_scf[valid_cols]

    bs = args.batch_size
    cursor = 0
    last_log = time.time()

    def _process_batch(batch_idx_arr, batch_donors, batch_ages, bs_cur, dst_cursor):
        rows = a.X[batch_idx_arr]
        if sp.issparse(rows):
            rows_dense = rows.toarray()
        else:
            rows_dense = np.asarray(rows)
        sub = rows_dense[:, valid_cols].astype(np.float32, copy=False)
        panel = np.zeros((sub.shape[0], scf_size), dtype=np.float32)
        panel[:, valid_scf_idx] = sub
        totals = panel.sum(axis=1, keepdims=True)
        safe_totals = np.where(totals > 0, totals, 1.0)
        panel_norm = np.log1p(panel / safe_totals * 1e4)
        log10_total = np.log10(np.clip(totals.squeeze(1), 1.0, None))
        x_t = torch.from_numpy(panel_norm).to(device)
        log10_t = torch.from_numpy(log10_total.astype(np.float32))
        try:
            emb = _forward_cell_emb(model, cfg, x_t, log10_t, args.tgt_t, bf16=args.bf16)
        except torch.cuda.OutOfMemoryError:
            del x_t, log10_t
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            if bs_cur == 1:
                raise
            # Recurse with halved batch
            mid = bs_cur // 2
            _process_batch(batch_idx_arr[:mid], batch_donors[:mid], batch_ages[:mid], mid, dst_cursor)
            _process_batch(batch_idx_arr[mid:], batch_donors[mid:], batch_ages[mid:],
                            bs_cur - mid, dst_cursor + mid)
            return
        n_b = emb.shape[0]
        cell_emb_buf[dst_cursor : dst_cursor + n_b] = emb.detach().cpu().numpy()
        cell_donors[dst_cursor : dst_cursor + n_b] = batch_donors
        cell_ages[dst_cursor : dst_cursor + n_b] = batch_ages
        del emb, x_t, log10_t

    for batch_start in range(0, n_cells, bs):
        batch_end = min(batch_start + bs, n_cells)
        bs_cur = batch_end - batch_start
        _process_batch(
            idx[batch_start:batch_end],
            donors[batch_start:batch_end],
            ages[batch_start:batch_end],
            bs_cur, cursor,
        )
        cursor += bs_cur
        if (batch_start // bs) % 100 == 0:
            torch.cuda.empty_cache()
        now = time.time()
        if now - last_log > 30 or batch_start == 0:
            elapsed = now - t0
            print(f"[scf-extract] {cursor}/{n_cells} cells | {elapsed:.1f}s elapsed | rate {cursor / max(elapsed, 1e-3):.1f} cells/s", flush=True)
            last_log = now

    # Aggregate to per-donor
    unique_donors = np.unique(cell_donors)
    n_d = len(unique_donors)
    H = cell_emb_buf.shape[1]
    donor_emb = np.zeros((n_d, H), dtype=np.float32)
    donor_age = np.zeros(n_d, dtype=np.float32)
    for i, d in enumerate(unique_donors):
        m = cell_donors == d
        donor_emb[i] = cell_emb_buf[m].mean(axis=0)
        donor_age[i] = cell_ages[m][0]

    elapsed = time.time() - t0
    print(f"[scf-extract] aggregated {n_d} donors × {H}-dim in {elapsed:.1f}s ({n_cells / max(elapsed, 1e-3):.1f} cells/s)")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cell_type_slug = args.cell_type.replace("+", "p").replace(" ", "_")
    out_path = out_dir / f"{args.cohort}_{cell_type_slug}_{args.output_tag}.npz"
    np.savez(
        out_path,
        donor_ids=unique_donors,
        ages=donor_age,
        embeddings=donor_emb,  # (n_donors, 3072)
        meta=np.array([
            f"cohort={args.cohort}", f"cell_type={args.cell_type}",
            f"max_cells_per_donor={args.max_cells_per_donor}",
            f"tgt_t={args.tgt_t}", f"n_genes_mapped={n_mapped}",
            f"timestamp={time.strftime('%Y-%m-%dT%H:%M:%S')}",
        ]),
    )
    print(f"[scf-extract] wrote {out_path} (shape {donor_emb.shape})")


if __name__ == "__main__":
    main()
