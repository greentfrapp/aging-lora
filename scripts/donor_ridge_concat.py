"""Variant 4: concatenated multi-layer probe (cheap follow-up to Variant 3).

For each (fold × cell × eval), concatenate per-donor embeddings from selected
layer subsets and fit ridge. Tests whether multi-resolution probes (e.g.
layer-1 + layer-12) outperform any single-layer probe — the §26.7 candidate
"sweep readouts on Geneformer".

Uses the same `embeddings_per_layer` .npz produced by
`extract_embeddings_layered.py`. No new compute beyond fast ridge fits.

Usage:
    uv run python scripts/donor_ridge_concat.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from scipy.stats import pearsonr


EMB_DIR = Path("results/phase3/embeddings_layered")
OUT_CSV = Path("results/phase3/ridge_summary_concat.csv")
ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
SEED = 0
TAG = "frozen_base_alllayers"

# Layer subsets to try. None means "use ALL layers concatenated".
SUBSETS = {
    "L1": [1],
    "L12": [12],
    "L1+L12": [1, 12],
    "L1+L9+L12": [1, 9, 12],
    "L0..12_all": list(range(13)),
    "early_block_L1+L2+L3": [1, 2, 3],
    "mid_block_L5+L6+L7": [5, 6, 7],
    "late_block_L10+L11+L12": [10, 11, 12],
}


def _slug(cell_type: str) -> str:
    return cell_type.replace("+", "p").replace(" ", "_")


def _load_npz(cohort: str, cell_type: str):
    path = EMB_DIR / f"{cohort}_{_slug(cell_type)}_{TAG}.npz"
    z = np.load(path, allow_pickle=True)
    return z["donor_ids"], z["ages"].astype(np.float32), z["embeddings_per_layer"].astype(np.float32)


def _concat(emb_LDH: np.ndarray, layers: list[int]) -> np.ndarray:
    return np.concatenate([emb_LDH[layer] for layer in layers], axis=1)  # (D, sum_H)


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

        train_X_per_layer, train_y_all = [], []
        for tc in train_cohorts:
            _, ages, emb_LDH = _load_npz(tc, cell_type)
            train_X_per_layer.append(emb_LDH)
            train_y_all.append(ages)
        train_X_layered = np.concatenate(train_X_per_layer, axis=1)  # (L, n_train, H)
        train_y = np.concatenate(train_y_all)
        _, eval_y, eval_X_layered = _load_npz(eval_cohort, cell_type)
        aida_X_layered = aida_y = None
        if also_aida:
            _, aida_y, aida_X_layered = _load_npz("aida", cell_type)

        print(f"\n=== {fold_id} × {cell_type} (eval={eval_cohort}, AIDA={also_aida}) ===")
        for subset_name, layer_idx in SUBSETS.items():
            train_X = _concat(train_X_layered, layer_idx)
            eval_X = _concat(eval_X_layered, layer_idx)
            cv = RidgeCV(alphas=ALPHAS, cv=3, scoring="neg_mean_absolute_error")
            rng = np.random.default_rng(SEED)
            perm = rng.permutation(len(train_y))
            cv.fit(train_X[perm], train_y[perm])
            alpha = float(cv.alpha_)
            final = Ridge(alpha=alpha).fit(train_X, train_y)
            pred = final.predict(eval_X)
            r, p = pearsonr(pred, eval_y)
            mae = float(np.median(np.abs(pred - eval_y)))
            print(f"  {subset_name:>28s}  α={alpha:>7.2f}  R={r:+.3f}  MAE={mae:.2f}  H={train_X.shape[1]}")
            rows.append({
                "fold": fold_id, "eval_cohort": eval_cohort, "cell_type": cell_type,
                "subset": subset_name, "layers": ",".join(str(l) for l in layer_idx),
                "alpha": alpha, "n_train_donors": int(len(train_y)),
                "n_eval_donors": int(len(eval_y)), "feature_dim": int(train_X.shape[1]),
                "pearson_r": float(r), "pearson_p": float(p), "mae_y": mae,
            })
            if also_aida:
                aida_X = _concat(aida_X_layered, layer_idx)
                aida_pred = final.predict(aida_X)
                ar, ap = pearsonr(aida_pred, aida_y)
                amae = float(np.median(np.abs(aida_pred - aida_y)))
                print(f"  {subset_name:>28s}  AIDA: R={ar:+.3f}  MAE={amae:.2f}")
                rows.append({
                    "fold": fold_id, "eval_cohort": "aida", "cell_type": cell_type,
                    "subset": subset_name, "layers": ",".join(str(l) for l in layer_idx),
                    "alpha": alpha, "n_train_donors": int(len(train_y)),
                    "n_eval_donors": int(len(aida_y)), "feature_dim": int(train_X.shape[1]),
                    "pearson_r": float(ar), "pearson_p": float(ap), "mae_y": amae,
                })

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False, float_format="%.6g")
    print(f"\n[concat] wrote {len(df)} rows to {OUT_CSV}")

    # Best subset per (fold × cell × eval) by R.
    print("\n=== best subset per (fold × cell × eval) by R ===")
    best = df.loc[df.groupby(["fold", "cell_type", "eval_cohort"])["pearson_r"].idxmax()]
    print(best[["fold", "eval_cohort", "cell_type", "subset", "pearson_r", "mae_y"]].to_string(index=False))


if __name__ == "__main__":
    main()
