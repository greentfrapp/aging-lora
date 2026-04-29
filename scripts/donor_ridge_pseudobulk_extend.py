"""D.24 — Extend pseudobulk ridge CSV with missing conditions.

The original D.18 frozen-base pseudobulk run covered 9 of 12 standard
conditions but is missing:
  - loco_terekhova × terekhova × NK
  - loco_terekhova × aida × B
  - loco_terekhova × aida × NK

This script reads existing pseudobulk-input embeddings and appends ridge fits
for the missing conditions to `results/phase3/ridge_summary_pseudobulk.csv`.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from scipy.stats import pearsonr


EMB_DIR = Path("results/phase3/embeddings_pseudobulk")
OUT_CSV = Path("results/phase3/ridge_summary_pseudobulk.csv")
ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
SEED = 0
TAG = "pseudobulk_frozen_alllayers"


def _slug(cell_type: str) -> str:
    return cell_type.replace("+", "p").replace(" ", "_")


def _load_npz(cohort: str, cell_type: str):
    path = EMB_DIR / f"{cohort}_{_slug(cell_type)}_{TAG}.npz"
    if not path.exists():
        raise SystemExit(f"missing pseudobulk embedding file: {path}")
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


def _bootstrap_pearson_ci(pred, y, seed=0, n_boot=1000):
    rng = np.random.default_rng(seed)
    n = len(y)
    rs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if np.std(pred[idx]) > 0 and np.std(y[idx]) > 0:
            rs.append(pearsonr(pred[idx], y[idx])[0])
    return float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))


def main():
    folds = json.loads(Path("data/loco_folds.json").read_text())["folds"]
    fold_map = {f["fold_id"]: f for f in folds}

    # Missing conditions:
    conditions = [
        # (fold_id, cell_type, also_aida)
        ("loco_terekhova", "NK", True),  # adds NK × terekhova + NK × aida
        ("loco_terekhova", "B",  True),  # adds B × aida (B × terekhova already exists, but we'll re-emit it)
    ]

    rows = []
    for fold_id, cell_type, also_aida in conditions:
        f = fold_map[fold_id]
        train_cohorts = f["train_cohorts"]
        eval_cohort = f["holdout_cohort"]

        train_X_per_layer, train_y_all = [], []
        for tc in train_cohorts:
            _, ages, emb_LDH = _load_npz(tc, cell_type)
            train_X_per_layer.append(emb_LDH)
            train_y_all.append(ages)
        train_X_layered = np.concatenate(train_X_per_layer, axis=1)
        train_y = np.concatenate(train_y_all)

        eval_donors, eval_y, eval_X_layered = _load_npz(eval_cohort, cell_type)
        aida_X_layered = aida_y = None
        if also_aida:
            _, aida_y, aida_X_layered = _load_npz("aida", cell_type)

        n_layers = train_X_layered.shape[0]
        print(f"\n[ridge-pb] {fold_id} × {cell_type} | {n_layers} layers, {len(train_y)} train donors, {len(eval_y)} eval donors", flush=True)
        for layer in range(n_layers):
            alpha, r, p, mae, pred = _fit_layer(
                train_X_layered[layer], train_y, eval_X_layered[layer], eval_y,
            )
            ci_lo, ci_hi = _bootstrap_pearson_ci(pred, eval_y, seed=SEED)
            print(f"  layer={layer:>2d} alpha={alpha:>8.2f} R={r:+.3f} p={p:.2e} MAE={mae:.2f}", flush=True)
            rows.append({
                "fold": fold_id, "eval_cohort": eval_cohort, "cell_type": cell_type,
                "layer": layer, "alpha": alpha,
                "n_train_donors": int(len(train_y)), "n_eval_donors": int(len(eval_y)),
                "pearson_r": r, "mae_y": mae,
                "ci_lo": ci_lo, "ci_hi": ci_hi,
                "pred_mean": float(pred.mean()), "eval_mean": float(eval_y.mean()),
            })
            if also_aida and aida_X_layered is not None:
                final = Ridge(alpha=alpha).fit(train_X_layered[layer], train_y)
                apred = final.predict(aida_X_layered[layer])
                ar, ap = pearsonr(apred, aida_y)
                amae = float(np.median(np.abs(apred - aida_y)))
                aci_lo, aci_hi = _bootstrap_pearson_ci(apred, aida_y, seed=SEED)
                print(f"           AIDA: R={ar:+.3f} MAE={amae:.2f}", flush=True)
                rows.append({
                    "fold": fold_id, "eval_cohort": "aida", "cell_type": cell_type,
                    "layer": layer, "alpha": alpha,
                    "n_train_donors": int(len(train_y)), "n_eval_donors": int(len(aida_y)),
                    "pearson_r": float(ar), "mae_y": amae,
                    "ci_lo": aci_lo, "ci_hi": aci_hi,
                    "pred_mean": float(apred.mean()), "eval_mean": float(aida_y.mean()),
                })

    df_new = pd.DataFrame(rows)
    # Drop duplicates against existing (the loco_terekhova × terekhova × B condition)
    existing = pd.read_csv(OUT_CSV)
    # Filter to truly new conditions only
    existing_keys = set(zip(existing["fold"], existing["eval_cohort"], existing["cell_type"], existing["layer"]))
    df_new["key"] = list(zip(df_new["fold"], df_new["eval_cohort"], df_new["cell_type"], df_new["layer"]))
    df_truly_new = df_new[~df_new["key"].isin(existing_keys)].drop(columns=["key"])
    print(f"\n[ridge-pb] {len(df_truly_new)} truly new rows (dropped {len(df_new) - len(df_truly_new)} dups)")

    df_combined = pd.concat([existing, df_truly_new], ignore_index=True)
    df_combined.to_csv(OUT_CSV, index=False, float_format="%.6g")
    print(f"[ridge-pb] total {len(df_combined)} rows in {OUT_CSV}")

    # Best-layer summary for new conditions
    print("\n=== best layer per (fold × cell × eval_cohort) by R, ALL pseudobulk conditions ===")
    best = df_combined.loc[df_combined.groupby(["fold", "cell_type", "eval_cohort"])["pearson_r"].idxmax()]
    print(best[["fold", "eval_cohort", "cell_type", "layer", "pearson_r", "mae_y"]].to_string(index=False))


if __name__ == "__main__":
    main()
