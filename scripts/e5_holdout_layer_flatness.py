"""E.5 — Holdout R-per-layer flatness check.

For each D.37 condition, fit ridge at every layer on full train + evaluate on
actual holdout cohort (and AIDA where available). Output per-layer holdout R
+ flatness metrics:
  - Top-1 R minus R at L0..L12 → ΔR per layer
  - Width of "near-best" band (#layers within 0.01, 0.02 of top)
  - Whether the "oracle" pick is in a flat top region

Output: results/phase3/e5_holdout_layer_flatness.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from scipy.stats import pearsonr


D37_CSV = Path("results/phase3/d37_cv_layer_selection.csv")
EMB_DIR = Path("results/phase3/embeddings_layered")
OUT_CSV = Path("results/phase3/e5_holdout_layer_flatness.csv")
ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
SEED = 0


def _slug(cell_type: str) -> str:
    return cell_type.replace("+", "p").replace(" ", "_")


def _load_npz(cohort: str, cell_type: str, tag: str):
    path = EMB_DIR / f"{cohort}_{_slug(cell_type)}_{tag}.npz"
    if not path.exists():
        return None
    z = np.load(path, allow_pickle=True)
    return z["donor_ids"], z["ages"].astype(np.float32), z["embeddings_per_layer"].astype(np.float32)


def _fit_ridge_eval(X_train, y_train, X_eval, y_eval):
    cv = RidgeCV(alphas=ALPHAS, cv=3, scoring="neg_mean_absolute_error")
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(y_train))
    cv.fit(X_train[perm], y_train[perm])
    final = Ridge(alpha=float(cv.alpha_)).fit(X_train, y_train)
    pred = final.predict(X_eval)
    if np.std(pred) > 0 and np.std(y_eval) > 0 and len(y_eval) > 1:
        r, _ = pearsonr(pred, y_eval)
    else:
        r = 0.0
    mae = float(np.median(np.abs(pred - y_eval)))
    return float(r), mae


# Same configs as D.37
CONFIGS = []
for fold_id, also_aida in [("loco_onek1k", True), ("loco_terekhova", False)]:
    for cell_type in ["CD4+ T", "B", "NK"]:
        CONFIGS.append((fold_id, cell_type, 0, "frozen_base_alllayers", "geneformer_frozen_seed0", also_aida))

for fold_id, also_aida in [("loco_onek1k", True), ("loco_terekhova", False)]:
    for seed, tag in [(1, "frozen_base_seed1_alllayers"), (2, "frozen_base_seed2_alllayers")]:
        CONFIGS.append((fold_id, "NK", seed, tag, f"geneformer_frozen_seed{seed}", also_aida))

rank16_seed_tags = [
    (0, "loco_onek1k_e5b_alllayers"),
    (1, "loco_onek1k_CD4pT_e5b_seed1_alllayers"),
    (2, "loco_onek1k_CD4pT_e5b_seed2_alllayers"),
]
for seed, tag in rank16_seed_tags:
    CONFIGS.append(("loco_onek1k", "CD4+ T", seed, tag, f"geneformer_rank16_seed{seed}", True))

rank32_seed_tags = [
    (0, "loco_onek1k_CD4pT_e5b_r32_alllayers"),
    (1, "loco_onek1k_CD4pT_e5b_r32_seed1_alllayers"),
    (2, "loco_onek1k_CD4pT_e5b_r32_seed2_alllayers"),
]
for seed, tag in rank32_seed_tags:
    CONFIGS.append(("loco_onek1k", "CD4+ T", seed, tag, f"geneformer_rank32_seed{seed}", True))


def main():
    folds = json.loads(Path("data/loco_folds.json").read_text())["folds"]
    fmap = {f["fold_id"]: f for f in folds}

    rows = []
    for fold_id, cell_type, seed, tag, method_label, also_aida in CONFIGS:
        f = fmap[fold_id]
        train_X_per_layer, train_y_all = [], []
        skip = False
        for tc in f["train_cohorts"]:
            ret = _load_npz(tc, cell_type, tag)
            if ret is None:
                skip = True
                break
            _, ages, emb_LDH = ret
            train_X_per_layer.append(emb_LDH)
            train_y_all.append(ages)
        if skip:
            continue
        train_X_layered = np.concatenate(train_X_per_layer, axis=1)
        train_y = np.concatenate(train_y_all)

        eval_ret = _load_npz(f["holdout_cohort"], cell_type, tag)
        if eval_ret is None:
            continue
        _, eval_y, eval_X_layered = eval_ret
        n_layers = eval_X_layered.shape[0]

        aida_ret = _load_npz("aida", cell_type, tag) if also_aida else None

        holdout_R = np.zeros(n_layers)
        holdout_MAE = np.zeros(n_layers)
        aida_R = np.zeros(n_layers) if aida_ret else None
        aida_MAE = np.zeros(n_layers) if aida_ret else None

        for layer in range(n_layers):
            r, mae = _fit_ridge_eval(train_X_layered[layer], train_y, eval_X_layered[layer], eval_y)
            holdout_R[layer] = r
            holdout_MAE[layer] = mae
            if aida_ret:
                _, aida_y, aida_X_layered = aida_ret
                ar, amae = _fit_ridge_eval(train_X_layered[layer], train_y, aida_X_layered[layer], aida_y)
                aida_R[layer] = ar
                aida_MAE[layer] = amae

        L_oracle = int(np.argmax(holdout_R))
        R_top = float(holdout_R[L_oracle])
        within_001 = int(np.sum(holdout_R >= R_top - 0.01))
        within_002 = int(np.sum(holdout_R >= R_top - 0.02))
        within_005 = int(np.sum(holdout_R >= R_top - 0.05))
        layers_within_002 = sorted([int(l) for l in np.where(holdout_R >= R_top - 0.02)[0]])

        # MAE flatness (lower is better)
        L_oracle_MAE = int(np.argmin(holdout_MAE))
        MAE_top = float(holdout_MAE[L_oracle_MAE])
        layers_MAE_within_05y = sorted([int(l) for l in np.where(holdout_MAE <= MAE_top + 0.5)[0]])
        layers_MAE_within_1y = sorted([int(l) for l in np.where(holdout_MAE <= MAE_top + 1.0)[0]])

        rows.append({
            "method": method_label,
            "fold": fold_id,
            "cell_type": cell_type,
            "seed": seed,
            "L_oracle_R": L_oracle,
            "R_top": R_top,
            "R_min": float(holdout_R.min()),
            "R_range": float(holdout_R.max() - holdout_R.min()),
            "n_layers_within_001_R": within_001,
            "n_layers_within_002_R": within_002,
            "n_layers_within_005_R": within_005,
            "layers_within_002_R": str(layers_within_002),
            "L_oracle_MAE": L_oracle_MAE,
            "MAE_top": MAE_top,
            "MAE_max": float(holdout_MAE.max()),
            "MAE_range": float(holdout_MAE.max() - holdout_MAE.min()),
            "n_layers_MAE_within_05y": len(layers_MAE_within_05y),
            "n_layers_MAE_within_1y": len(layers_MAE_within_1y),
            "layers_MAE_within_1y": str(layers_MAE_within_1y),
            "holdout_R_per_layer": [float(x) for x in holdout_R],
            "holdout_MAE_per_layer": [float(x) for x in holdout_MAE],
            "aida_R_per_layer": [float(x) for x in aida_R] if aida_R is not None else None,
        })
        print(f"[E.5] {method_label} {fold_id} {cell_type} seed{seed}: "
              f"R_top={R_top:.3f} R_range={holdout_R.max()-holdout_R.min():.3f} | "
              f"layers within 0.02 of top: {within_002} ({layers_within_002}) | "
              f"oracle L{L_oracle}", flush=True)

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, float_format="%.4f")
    print(f"\n[E.5] wrote {len(df)} rows to {OUT_CSV}")

    # Summary
    print("\n=== E.5 Flatness summary ===")
    print(df[["method", "fold", "cell_type", "seed", "L_oracle_R", "R_top",
              "R_range", "n_layers_within_002_R", "n_layers_within_005_R",
              "L_oracle_MAE", "n_layers_MAE_within_1y"]].to_string(index=False, float_format="%.3f"))

    print("\n=== Per-condition holdout R curve ===")
    for _, r in df.iterrows():
        curve = r["holdout_R_per_layer"]
        bar = " ".join(f"{x:+.2f}" for x in curve)
        print(f"  {r['method']:30s} {r['fold']:14s} {r['cell_type']:6s} s{r['seed']}: {bar}")


if __name__ == "__main__":
    main()
