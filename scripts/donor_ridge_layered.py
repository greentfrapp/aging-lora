"""Variant 3 ridge fits: one per (fold × cell_type × layer).

Reads layered `.npz` files from `extract_embeddings_layered.py`, fits ridge
(RidgeCV alpha selection) per layer, evaluates Pearson R + MAE on the holdout
cohort. Writes one row per (fold × cell_type × layer × eval_cohort) to
`results/phase3/ridge_summary_layered.csv`.

Usage:
    uv run python scripts/donor_ridge_layered.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from scipy.stats import pearsonr


EMB_DIR = Path("results/phase3/embeddings_layered")
OUT_CSV = Path("results/phase3/ridge_summary_layered.csv")
ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
SEED = 0
TAG = "frozen_base_alllayers"


def _slug(cell_type: str) -> str:
    return cell_type.replace("+", "p").replace(" ", "_")


def _load_npz(cohort: str, cell_type: str):
    path = EMB_DIR / f"{cohort}_{_slug(cell_type)}_{TAG}.npz"
    if not path.exists():
        raise SystemExit(f"missing layered embedding file: {path}")
    z = np.load(path, allow_pickle=True)
    return z["donor_ids"], z["ages"].astype(np.float32), z["embeddings_per_layer"].astype(np.float32)


def _fit_layer(train_X, train_y, eval_X, eval_y):
    cv = RidgeCV(alphas=ALPHAS, cv=3, scoring="neg_mean_absolute_error")
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(train_y))
    cv.fit(train_X[perm], train_y[perm])
    alpha = float(cv.alpha_)
    final = Ridge(alpha=alpha).fit(train_X, train_y)
    pred = final.predict(eval_X)
    r, p = pearsonr(pred, eval_y)
    mae = float(np.median(np.abs(pred - eval_y)))
    return alpha, float(r), float(p), mae, pred


def main():
    folds = json.loads(Path("data/loco_folds.json").read_text())["folds"]
    fold_map = {f["fold_id"]: f for f in folds}

    conditions = [
        ("loco_onek1k", "CD4+ T", True),
        ("loco_terekhova", "CD4+ T", False),
        ("loco_onek1k", "B", True),
        ("loco_terekhova", "B", False),
        ("loco_onek1k", "NK", True),
        ("loco_terekhova", "NK", False),
    ]

    rows = []
    for fold_id, cell_type, also_aida in conditions:
        f = fold_map[fold_id]
        train_cohorts = f["train_cohorts"]
        eval_cohort = f["holdout_cohort"]

        # Load all layered embeddings for this cell type.
        train_X_per_layer, train_y_all = [], []
        for tc in train_cohorts:
            _, ages, emb_LDH = _load_npz(tc, cell_type)
            train_X_per_layer.append(emb_LDH)
            train_y_all.append(ages)
        train_X_layered = np.concatenate(train_X_per_layer, axis=1)  # (L, n_train, H)
        train_y = np.concatenate(train_y_all)

        eval_donors, eval_y, eval_X_layered = _load_npz(eval_cohort, cell_type)
        aida_X_layered = aida_y = None
        if also_aida:
            _, aida_y, aida_X_layered = _load_npz("aida", cell_type)

        n_layers = train_X_layered.shape[0]
        print(f"\n[ridge-L] {fold_id} × {cell_type} | {n_layers} layers, {len(train_y)} train donors, {len(eval_y)} eval donors")
        for layer in range(n_layers):
            t0 = time.time()
            alpha, r, p, mae, pred = _fit_layer(
                train_X_layered[layer], train_y, eval_X_layered[layer], eval_y,
            )
            elapsed = time.time() - t0
            print(f"  layer={layer:>2d} alpha={alpha:>8.2f} R={r:+.3f} p={p:.2e} MAE={mae:.2f} ({elapsed:.1f}s)")
            rows.append({
                "fold": fold_id,
                "eval_cohort": eval_cohort,
                "cell_type": cell_type,
                "layer": layer,
                "alpha": alpha,
                "n_train_donors": int(len(train_y)),
                "n_eval_donors": int(len(eval_y)),
                "pearson_r": r,
                "pearson_p": p,
                "mae_y": mae,
            })
            if also_aida and aida_X_layered is not None:
                # Apply this layer's ridge to AIDA — we need to refit and re-predict, since `final` is local.
                # Re-fitting is cheap; do it once again for transparency.
                final = Ridge(alpha=alpha).fit(train_X_layered[layer], train_y)
                aida_pred = final.predict(aida_X_layered[layer])
                aida_r, aida_p = pearsonr(aida_pred, aida_y)
                aida_mae = float(np.median(np.abs(aida_pred - aida_y)))
                print(f"           AIDA: R={aida_r:+.3f} p={aida_p:.2e} MAE={aida_mae:.2f}")
                rows.append({
                    "fold": fold_id,
                    "eval_cohort": "aida",
                    "cell_type": cell_type,
                    "layer": layer,
                    "alpha": alpha,
                    "n_train_donors": int(len(train_y)),
                    "n_eval_donors": int(len(aida_y)),
                    "pearson_r": float(aida_r),
                    "pearson_p": float(aida_p),
                    "mae_y": aida_mae,
                })

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False, float_format="%.6g")
    print(f"\n[ridge-L] wrote {len(df)} rows to {OUT_CSV}")

    # Pretty summary: best layer per (fold, cell, eval_cohort).
    print("\n=== best layer per (fold × cell × eval_cohort) by R ===")
    best = df.loc[df.groupby(["fold", "cell_type", "eval_cohort"])["pearson_r"].idxmax()]
    print(best[["fold", "eval_cohort", "cell_type", "layer", "pearson_r", "mae_y"]].to_string(index=False))


if __name__ == "__main__":
    main()
