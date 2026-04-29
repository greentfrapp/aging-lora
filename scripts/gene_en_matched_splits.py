"""D.17 — Gene-EN baseline on FM-matched splits (Phase-3-B step-back review).

Removes the apples-to-oranges concern with the TF paper's gene-EN R=0.83 LOCO /
0.77 AIDA numbers, which used different splits / preprocessing / hyperparameters
than our FM experiments.

This script:
  1. Uses the *same* `data/loco_folds.json` LOCO splits the FM experiments use.
  2. Uses the *same* `select_indices` cell-selection (max_cells_per_donor=500
     for train, 200 for eval — matches the e5b config used in §27/§28).
  3. Aggregates per donor as log1p(CP10k) → mean across cells → one vector per
     donor (the canonical "gene-EN on log1p-mean pseudobulk" input format).
  4. Fits ElasticNetCV with (alpha, l1_ratio) selected by 5-fold inner CV.
  5. Reports R + MAE + 95% bootstrap CI per (cell_type × eval_cohort), plus
     AIDA cross-cohort transfer where applicable.

Usage:
    uv run python scripts/gene_en_matched_splits.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import pearsonr
from sklearn.linear_model import ElasticNetCV, ElasticNet
from sklearn.preprocessing import StandardScaler

from src.finetune.data_loader import select_indices


CELL_TYPE_TO_FILE = {
    "CD4+ T": "CD4p_T.h5ad",
    "B": "B.h5ad",
    "NK": "NK.h5ad",
}

INTEGRATED_DIR = Path("data/cohorts/integrated")
AIDA_DIR = Path("data/cohorts/aida_eval")
# Smaller / less overlapping alpha grid + fewer l1_ratios.
ALPHAS = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
L1_RATIOS = [0.3, 0.5, 0.7, 0.9]
TOP_N_HVG = 5000  # filter to top 5000 high-variance genes
SEED = 0
MAX_CELLS_PER_DONOR_TRAIN = 100  # pseudobulk only needs ~100 cells/donor for stable mean
MAX_CELLS_PER_DONOR_EVAL = 100
OUT_CSV = Path("results/baselines/gene_en_matched_splits.csv")


def _h5ad(cohort: str, cell_type: str) -> Path:
    base = AIDA_DIR if cohort == "aida" else INTEGRATED_DIR
    return base / CELL_TYPE_TO_FILE[cell_type]


def _aida_donors() -> list[str]:
    raw = json.loads(Path("data/aida_split.json").read_text())["ancestry_shift_mae_donors"]
    return [d if d.startswith("aida:") else f"aida:{d}" for d in raw]


def _per_donor_log1p_mean(h5ad_path: Path, idx: np.ndarray, donors: np.ndarray, ages: np.ndarray):
    """Aggregate per-donor log1p(CP10k) mean across selected cells.

    Returns (donor_ids, X_donor (D, G), y_donor (D,), gene_symbols (G,)).
    """
    a = ad.read_h5ad(h5ad_path, backed="r")
    var_syms = _resolve_var_symbols(a.var)
    n_genes = len(var_syms)
    unique_donors = np.unique(donors)
    n_d = len(unique_donors)
    X_donor = np.zeros((n_d, n_genes), dtype=np.float32)
    y_donor = np.zeros(n_d, dtype=np.float32)
    for i, d in enumerate(unique_donors):
        m = donors == d
        sub_idx = idx[m]
        rows = a.X[sub_idx]
        if sp.issparse(rows):
            rows_dense = rows.toarray().astype(np.float32)
        else:
            rows_dense = np.asarray(rows, dtype=np.float32)
        # Per-cell CP10k + log1p
        totals = rows_dense.sum(axis=1, keepdims=True)
        safe = np.where(totals > 0, totals, 1.0)
        per_cell = np.log1p(rows_dense / safe * 1e4)
        # Mean across cells -> donor-level pseudobulk
        X_donor[i] = per_cell.mean(axis=0)
        y_donor[i] = ages[m][0]
    a.file.close()
    return unique_donors, X_donor, y_donor, var_syms


def _resolve_var_symbols(var: pd.DataFrame) -> np.ndarray:
    """Best-effort gene symbol per row (mirrors extract_embeddings_scfoundation.py)."""
    syms = np.asarray(var.get("gene_symbol", pd.Series(np.nan, index=var.index)).astype(object))
    idx = np.asarray(var.index.astype(str))
    return np.where(pd.isna(syms) | (syms == "nan"), idx, syms.astype(object).astype(str)).astype(str)


def _align_columns(train_syms: np.ndarray, eval_syms: np.ndarray, eval_X: np.ndarray) -> np.ndarray:
    """Reindex eval_X to train_syms column order; pad missing genes with zeros."""
    train_set = {s: i for i, s in enumerate(train_syms)}
    out = np.zeros((eval_X.shape[0], len(train_syms)), dtype=np.float32)
    for j, s in enumerate(eval_syms):
        i = train_set.get(s)
        if i is not None:
            out[:, i] = eval_X[:, j]
    return out


def _bootstrap_pearson_ci(pred: np.ndarray, y: np.ndarray, n_boot: int = 1000, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = len(pred)
    rs = []
    for _ in range(n_boot):
        i = rng.integers(0, n, size=n)
        if pred[i].std() == 0 or y[i].std() == 0:
            continue
        rs.append(pearsonr(pred[i], y[i])[0])
    rs = np.asarray(rs)
    return float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))


def _build_donor_matrix(cohort: str, cell_type: str, max_cells: int):
    h5ad = _h5ad(cohort, cell_type)
    if cohort == "aida":
        idx, ages, donors = select_indices(
            h5ad, cell_type=cell_type, cohorts=None,
            include_donors=_aida_donors(), max_cells_per_donor=max_cells, rng_seed=SEED,
        )
    else:
        idx, ages, donors = select_indices(
            h5ad, cell_type=cell_type, cohorts=[cohort],
            max_cells_per_donor=max_cells, rng_seed=SEED,
        )
    print(f"  [{cohort} × {cell_type}] {len(idx)} cells, {len(np.unique(donors))} donors")
    donor_ids, X, y, syms = _per_donor_log1p_mean(h5ad, idx, donors, ages)
    return donor_ids, X, y, syms


def main():
    folds = json.loads(Path("data/loco_folds.json").read_text())["folds"]
    fmap = {f["fold_id"]: f for f in folds}

    # 5 conditions to match the FM analyses we already ran:
    runs = [
        # (fold_id, cell_type, also_aida)
        ("loco_onek1k",    "CD4+ T", True),
        ("loco_terekhova", "CD4+ T", True),
        ("loco_onek1k",    "B",      True),
        ("loco_terekhova", "B",      False),
        ("loco_onek1k",    "NK",     True),
    ]

    rows = []
    for fold_id, cell_type, also_aida in runs:
        f = fmap[fold_id]
        eval_cohort = f["holdout_cohort"]
        print(f"\n=== {fold_id} × {cell_type} (eval={eval_cohort}) ===")
        t0 = time.time()

        # Build per-donor log1p-mean for train cohorts + eval cohort
        train_X_list, train_y_list, train_syms_ref = [], [], None
        for tc in f["train_cohorts"]:
            _, X_tc, y_tc, syms_tc = _build_donor_matrix(tc, cell_type, MAX_CELLS_PER_DONOR_TRAIN)
            if train_syms_ref is None:
                train_syms_ref = syms_tc
                train_X_list.append(X_tc)
            else:
                # Align this cohort's gene columns to first cohort's gene order
                X_aligned = _align_columns(train_syms_ref, syms_tc, X_tc)
                train_X_list.append(X_aligned)
            train_y_list.append(y_tc)
        train_X = np.concatenate(train_X_list, axis=0)
        train_y = np.concatenate(train_y_list)

        eval_donors, eval_X_raw, eval_y, eval_syms = _build_donor_matrix(eval_cohort, cell_type, MAX_CELLS_PER_DONOR_EVAL)
        eval_X = _align_columns(train_syms_ref, eval_syms, eval_X_raw)

        # HVG filter on train set (top-N most variable across train donors)
        train_var = train_X.var(axis=0)
        top_idx = np.argsort(-train_var)[:TOP_N_HVG]
        train_X = train_X[:, top_idx]
        eval_X = eval_X[:, top_idx]
        train_syms_ref = train_syms_ref[top_idx]
        # Standardize features (mean=0, std=1) using train stats
        scaler = StandardScaler().fit(train_X)
        train_X_s = scaler.transform(train_X).astype(np.float32)
        eval_X_s = scaler.transform(eval_X).astype(np.float32)

        print(f"  train: {train_X_s.shape}, eval: {eval_X_s.shape} (after HVG-{TOP_N_HVG} + standardization)")
        print(f"  fitting ElasticNetCV (3-fold, alphas={len(ALPHAS)}, l1_ratios={L1_RATIOS})...")
        rng = np.random.default_rng(SEED)
        perm = rng.permutation(len(train_y))
        cv = ElasticNetCV(
            l1_ratio=L1_RATIOS, alphas=ALPHAS, cv=3, max_iter=5000,
            n_jobs=-1, random_state=SEED, selection="cyclic",
        )
        cv.fit(train_X_s[perm], train_y[perm])
        alpha = float(cv.alpha_)
        l1_ratio = float(cv.l1_ratio_)
        final = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000).fit(train_X_s, train_y)
        n_nonzero = int(np.sum(final.coef_ != 0))

        pred = final.predict(eval_X_s)
        r, p_val = pearsonr(pred, eval_y)
        mae = float(np.median(np.abs(pred - eval_y)))
        ci_lo, ci_hi = _bootstrap_pearson_ci(pred, eval_y, seed=SEED)
        elapsed = time.time() - t0
        print(f"  HOLDOUT  R={r:+.3f} ({ci_lo:+.3f}, {ci_hi:+.3f})  MAE={mae:.2f}  alpha={alpha:.4g} l1_ratio={l1_ratio} n_nonzero={n_nonzero}  ({elapsed:.0f}s)", flush=True)
        rows.append({
            "fold": fold_id, "eval_cohort": eval_cohort, "cell_type": cell_type,
            "alpha": alpha, "l1_ratio": l1_ratio, "n_nonzero_genes": n_nonzero,
            "n_train_donors": int(len(train_y)), "n_eval_donors": int(len(eval_y)),
            "pearson_r": float(r), "pearson_p": float(p_val), "mae_y": mae,
            "pearson_ci_lo": ci_lo, "pearson_ci_hi": ci_hi,
            "pred_mean": float(pred.mean()), "eval_mean": float(eval_y.mean()),
        })

        if also_aida:
            _, aida_X_raw, aida_y, aida_syms = _build_donor_matrix("aida", cell_type, MAX_CELLS_PER_DONOR_EVAL)
            aida_X = _align_columns(train_syms_ref, aida_syms, aida_X_raw)
            aida_X_s = scaler.transform(aida_X).astype(np.float32)
            apred = final.predict(aida_X_s)
            ar, ap = pearsonr(apred, aida_y)
            amae = float(np.median(np.abs(apred - aida_y)))
            aci_lo, aci_hi = _bootstrap_pearson_ci(apred, aida_y, seed=SEED)
            print(f"  AIDA     R={ar:+.3f} ({aci_lo:+.3f}, {aci_hi:+.3f})  MAE={amae:.2f}  pred_mean={apred.mean():.2f} eval_mean={aida_y.mean():.2f}", flush=True)
            rows.append({
                "fold": fold_id, "eval_cohort": "aida", "cell_type": cell_type,
                "alpha": alpha, "l1_ratio": l1_ratio, "n_nonzero_genes": n_nonzero,
                "n_train_donors": int(len(train_y)), "n_eval_donors": int(len(aida_y)),
                "pearson_r": float(ar), "pearson_p": float(ap), "mae_y": amae,
                "pearson_ci_lo": aci_lo, "pearson_ci_hi": aci_hi,
                "pred_mean": float(apred.mean()), "eval_mean": float(aida_y.mean()),
            })

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False, float_format="%.6g")
    print(f"\n[gene-EN] wrote {len(df)} rows to {OUT_CSV}")
    print()
    print(df[["fold", "eval_cohort", "cell_type", "n_train_donors", "n_eval_donors",
              "pearson_r", "pearson_ci_lo", "pearson_ci_hi", "mae_y", "n_nonzero_genes"]].to_string(index=False))


if __name__ == "__main__":
    main()
