"""D.32 — Bootstrap CIs on rank-16 LoRA 3-seed L9 AIDA from existing embeddings.

Closes the audit gap on §27/§28 numbers: the 3-seed mean MAE for rank-16 LoRA
on L9 AIDA is computed in §28, but no bootstrap CI on the per-seed numbers
nor on the 3-seed mean. This script:

1. Loads layered embeddings for rank-16 seeds 0, 1, 2.
2. Refits ridge per layer per seed.
3. Bootstrap-resamples donors (n=1000) per seed-layer to compute Pearson R CI
   and MAE CI.
4. Aggregates across seeds: 3-seed mean ± std for L9 AIDA + bootstrap CI on
   the 3-seed mean.

Output: `results/phase3/d32_rank16_3seed_layered_bootstrap_cis.csv`
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from scipy.stats import pearsonr


EMB_DIR = Path("results/phase3/embeddings_layered")
OUT_CSV = Path("results/phase3/d32_rank16_3seed_layered_bootstrap_cis.csv")
ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
SEED = 0
N_BOOT = 1000


def _slug(cell_type: str) -> str:
    return cell_type.replace("+", "p").replace(" ", "_")


def _load_npz(cohort: str, cell_type: str, tag: str):
    path = EMB_DIR / f"{cohort}_{_slug(cell_type)}_{tag}.npz"
    if not path.exists():
        return None
    z = np.load(path, allow_pickle=True)
    return z["donor_ids"], z["ages"].astype(np.float32), z["embeddings_per_layer"].astype(np.float32)


def _bootstrap_R_MAE(pred, y, seed=0, n_boot=N_BOOT):
    rng = np.random.default_rng(seed)
    n = len(y)
    rs, maes = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if np.std(pred[idx]) > 0 and np.std(y[idx]) > 0:
            rs.append(pearsonr(pred[idx], y[idx])[0])
            maes.append(np.median(np.abs(pred[idx] - y[idx])))
    rs = np.array(rs); maes = np.array(maes)
    return (
        float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5)),
        float(np.percentile(maes, 2.5)), float(np.percentile(maes, 97.5)),
    )


def main():
    folds = json.loads(Path("data/loco_folds.json").read_text())["folds"]
    fmap = {f["fold_id"]: f for f in folds}

    seed_tags = [
        (0, "loco_onek1k_e5b_alllayers"),
        (1, "loco_onek1k_CD4pT_e5b_seed1_alllayers"),
        (2, "loco_onek1k_CD4pT_e5b_seed2_alllayers"),
    ]
    fold_id = "loco_onek1k"
    cell_type = "CD4+ T"
    f = fmap[fold_id]

    rows = []
    for seed, tag in seed_tags:
        train_X_per_layer, train_y_all = [], []
        skip = False
        for tc in f["train_cohorts"]:
            ret = _load_npz(tc, cell_type, tag)
            if ret is None:
                print(f"[seed{seed}] missing for {tc}; skip")
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
        aida_ret = _load_npz("aida", cell_type, tag)
        if eval_ret is None or aida_ret is None:
            print(f"[seed{seed}] missing eval/aida; skip")
            continue
        _, eval_y, eval_X_layered = eval_ret
        _, aida_y, aida_X_layered = aida_ret

        n_layers = train_X_layered.shape[0]
        for layer in range(n_layers):
            cv = RidgeCV(alphas=ALPHAS, cv=3, scoring="neg_mean_absolute_error")
            rng = np.random.default_rng(SEED)
            perm = rng.permutation(len(train_y))
            cv.fit(train_X_layered[layer][perm], train_y[perm])
            alpha = float(cv.alpha_)
            final = Ridge(alpha=alpha).fit(train_X_layered[layer], train_y)

            pred = final.predict(eval_X_layered[layer])
            r, _ = pearsonr(pred, eval_y)
            mae = float(np.median(np.abs(pred - eval_y)))
            r_lo, r_hi, mae_lo, mae_hi = _bootstrap_R_MAE(pred, eval_y, seed=0)

            apred = final.predict(aida_X_layered[layer])
            ar, _ = pearsonr(apred, aida_y)
            amae = float(np.median(np.abs(apred - aida_y)))
            ar_lo, ar_hi, amae_lo, amae_hi = _bootstrap_R_MAE(apred, aida_y, seed=0)

            rows.append({
                "seed": seed, "layer": layer, "alpha": alpha,
                "onek1k_r": r, "onek1k_r_ci_lo": r_lo, "onek1k_r_ci_hi": r_hi,
                "onek1k_mae": mae, "onek1k_mae_ci_lo": mae_lo, "onek1k_mae_ci_hi": mae_hi,
                "aida_r": ar, "aida_r_ci_lo": ar_lo, "aida_r_ci_hi": ar_hi,
                "aida_mae": amae, "aida_mae_ci_lo": amae_lo, "aida_mae_ci_hi": amae_hi,
            })

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, float_format="%.6g")
    print(f"\n[d32] wrote {len(df)} rows to {OUT_CSV}")

    # Aggregate L9 AIDA across seeds
    print("\n=== rank-16 LoRA 3-seed L9 AIDA ===")
    l9 = df[df["layer"] == 9]
    print(l9[["seed", "aida_r", "aida_r_ci_lo", "aida_r_ci_hi", "aida_mae", "aida_mae_ci_lo", "aida_mae_ci_hi"]].to_string(index=False, float_format="%.3f"))
    print(f"\n  3-seed mean R = {l9['aida_r'].mean():+.3f} ± {l9['aida_r'].std():.3f}")
    print(f"  3-seed mean MAE = {l9['aida_mae'].mean():.2f}y ± {l9['aida_mae'].std():.2f}y")

    # Layer-by-layer 3-seed agg
    print("\n=== rank-16 LoRA 3-seed layer-by-layer (AIDA) ===")
    agg = df.groupby("layer").agg(
        aida_r_mean=("aida_r", "mean"), aida_r_std=("aida_r", "std"),
        aida_mae_mean=("aida_mae", "mean"), aida_mae_std=("aida_mae", "std"),
    ).reset_index()
    print(agg.to_string(index=False, float_format="%.3f"))


if __name__ == "__main__":
    main()
