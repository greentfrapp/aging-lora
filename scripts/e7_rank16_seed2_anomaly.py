"""E.7 — Verify rank-16 seed-2 anomaly: is the L12 vs L6 deployment gap real
on actual held-out (OneK1K + AIDA), or a CV-R artifact?

Per E.5, rank-16 × CD4+T × loco_onek1k:
  seed 0: L6=0.64, L12=0.63 → 0.01 gap
  seed 1: L6=0.64, L12=0.63 → 0.01 gap
  seed 2: L6=0.62, L12=0.56 → 0.06 gap

Question: is seed-2 L12=0.56 a deployment-real value, or is the L12 vs L6
difference within bootstrap variance on the actual holdout evaluation?

Approach:
  For each rank-16 seed, refit ridge at L6 AND at L12 on full train, predict
  on OneK1K holdout + AIDA. Bootstrap-resample donors (n=1000) within the
  predictions to get CI on R and MAE at each layer. Check whether L6 and
  L12 distributions overlap within seed.

Output: results/phase3/e7_rank16_seed2_anomaly.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from scipy.stats import pearsonr, mannwhitneyu


EMB_DIR = Path("results/phase3/embeddings_layered")
OUT_CSV = Path("results/phase3/e7_rank16_seed2_anomaly.csv")
ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
SEED = 0
N_BOOT = 1000


def _slug(cell_type: str) -> str:
    return cell_type.replace("+", "p").replace(" ", "_")


def _load_npz(cohort: str, cell_type: str, tag: str):
    path = EMB_DIR / f"{cohort}_{_slug(cell_type)}_{tag}.npz"
    z = np.load(path, allow_pickle=True)
    return z["donor_ids"], z["ages"].astype(np.float32), z["embeddings_per_layer"].astype(np.float32)


def _fit_predict(X_train, y_train, X_eval):
    cv = RidgeCV(alphas=ALPHAS, cv=3, scoring="neg_mean_absolute_error")
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(y_train))
    cv.fit(X_train[perm], y_train[perm])
    final = Ridge(alpha=float(cv.alpha_)).fit(X_train, y_train)
    return final.predict(X_eval)


def _bootstrap_R_MAE(pred, y, n_boot=N_BOOT, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    n = len(y)
    R, MAE = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        p, t = pred[idx], y[idx]
        if np.std(p) > 0 and np.std(t) > 0:
            r, _ = pearsonr(p, t)
            R.append(r)
        MAE.append(float(np.median(np.abs(p - t))))
    return np.array(R), np.array(MAE)


def main():
    folds = json.loads(Path("data/loco_folds.json").read_text())["folds"]
    fmap = {f["fold_id"]: f for f in folds}
    f = fmap["loco_onek1k"]

    seed_tags = [
        (0, "loco_onek1k_e5b_alllayers"),
        (1, "loco_onek1k_CD4pT_e5b_seed1_alllayers"),
        (2, "loco_onek1k_CD4pT_e5b_seed2_alllayers"),
    ]

    rows = []
    seed_predictions = {}  # seed → (L6 pred onek1k, L12 pred onek1k, L6 pred aida, L12 pred aida, y_onek1k, y_aida)

    for seed, tag in seed_tags:
        # Build train
        train_X_per_layer, train_y_all = [], []
        for tc in f["train_cohorts"]:
            _, ages, emb_LDH = _load_npz(tc, "CD4+ T", tag)
            train_X_per_layer.append(emb_LDH)
            train_y_all.append(ages)
        train_X_layered = np.concatenate(train_X_per_layer, axis=1)
        train_y = np.concatenate(train_y_all)

        # OneK1K holdout
        _, y_onek1k, X_onek1k = _load_npz("onek1k", "CD4+ T", tag)
        # AIDA cross-ancestry
        _, y_aida, X_aida = _load_npz("aida", "CD4+ T", tag)

        for layer in [6, 12]:
            pred_onek1k = _fit_predict(train_X_layered[layer], train_y, X_onek1k[layer])
            pred_aida = _fit_predict(train_X_layered[layer], train_y, X_aida[layer])

            R_ok, MAE_ok = _bootstrap_R_MAE(pred_onek1k, y_onek1k, rng_seed=seed*100+layer)
            R_ai, MAE_ai = _bootstrap_R_MAE(pred_aida, y_aida, rng_seed=seed*100+layer+1)

            point_r_ok, _ = pearsonr(pred_onek1k, y_onek1k)
            point_mae_ok = float(np.median(np.abs(pred_onek1k - y_onek1k)))
            point_r_ai, _ = pearsonr(pred_aida, y_aida)
            point_mae_ai = float(np.median(np.abs(pred_aida - y_aida)))

            rows.append({
                "seed": seed,
                "layer": layer,
                "onek1k_R_point": float(point_r_ok),
                "onek1k_MAE_point": point_mae_ok,
                "onek1k_R_median_boot": float(np.median(R_ok)),
                "onek1k_R_ci_lo": float(np.percentile(R_ok, 2.5)),
                "onek1k_R_ci_hi": float(np.percentile(R_ok, 97.5)),
                "onek1k_MAE_median_boot": float(np.median(MAE_ok)),
                "onek1k_MAE_ci_lo": float(np.percentile(MAE_ok, 2.5)),
                "onek1k_MAE_ci_hi": float(np.percentile(MAE_ok, 97.5)),
                "aida_R_point": float(point_r_ai),
                "aida_MAE_point": point_mae_ai,
                "aida_R_median_boot": float(np.median(R_ai)),
                "aida_R_ci_lo": float(np.percentile(R_ai, 2.5)),
                "aida_R_ci_hi": float(np.percentile(R_ai, 97.5)),
                "aida_MAE_median_boot": float(np.median(MAE_ai)),
                "aida_MAE_ci_lo": float(np.percentile(MAE_ai, 2.5)),
                "aida_MAE_ci_hi": float(np.percentile(MAE_ai, 97.5)),
                "_R_ok_boot": R_ok,
                "_R_ai_boot": R_ai,
                "_MAE_ok_boot": MAE_ok,
                "_MAE_ai_boot": MAE_ai,
            })
            print(f"  seed{seed} L{layer}: OneK1K R={point_r_ok:+.3f} CI[{np.percentile(R_ok,2.5):+.3f}, {np.percentile(R_ok,97.5):+.3f}], MAE={point_mae_ok:.2f} CI[{np.percentile(MAE_ok,2.5):.2f}, {np.percentile(MAE_ok,97.5):.2f}] | AIDA R={point_r_ai:+.3f} CI[{np.percentile(R_ai,2.5):+.3f}, {np.percentile(R_ai,97.5):+.3f}]", flush=True)

    # Cross-layer comparison per seed
    print("\n=== L6 vs L12 within-seed comparison ===")
    for seed in [0, 1, 2]:
        L6_row = next(r for r in rows if r["seed"] == seed and r["layer"] == 6)
        L12_row = next(r for r in rows if r["seed"] == seed and r["layer"] == 12)

        # Mann-Whitney on bootstrap distributions
        u_ok_R, p_ok_R = mannwhitneyu(L6_row["_R_ok_boot"], L12_row["_R_ok_boot"], alternative="greater")
        u_ai_R, p_ai_R = mannwhitneyu(L6_row["_R_ai_boot"], L12_row["_R_ai_boot"], alternative="greater")

        # CI overlap?
        L6_ok_lo, L6_ok_hi = L6_row["onek1k_R_ci_lo"], L6_row["onek1k_R_ci_hi"]
        L12_ok_lo, L12_ok_hi = L12_row["onek1k_R_ci_lo"], L12_row["onek1k_R_ci_hi"]
        ok_overlap = max(L6_ok_lo, L12_ok_lo) <= min(L6_ok_hi, L12_ok_hi)
        L6_ai_lo, L6_ai_hi = L6_row["aida_R_ci_lo"], L6_row["aida_R_ci_hi"]
        L12_ai_lo, L12_ai_hi = L12_row["aida_R_ci_lo"], L12_row["aida_R_ci_hi"]
        ai_overlap = max(L6_ai_lo, L12_ai_lo) <= min(L6_ai_hi, L12_ai_hi)

        print(f"\n  seed {seed}:")
        print(f"    OneK1K: L6 R = {L6_row['onek1k_R_point']:+.3f} CI[{L6_ok_lo:+.3f}, {L6_ok_hi:+.3f}], L12 R = {L12_row['onek1k_R_point']:+.3f} CI[{L12_ok_lo:+.3f}, {L12_ok_hi:+.3f}]")
        print(f"      CI overlap: {'YES' if ok_overlap else 'NO'} | Mann-Whitney L6 > L12 p={p_ok_R:.3e}")
        print(f"    AIDA:    L6 R = {L6_row['aida_R_point']:+.3f} CI[{L6_ai_lo:+.3f}, {L6_ai_hi:+.3f}], L12 R = {L12_row['aida_R_point']:+.3f} CI[{L12_ai_lo:+.3f}, {L12_ai_hi:+.3f}]")
        print(f"      CI overlap: {'YES' if ai_overlap else 'NO'} | Mann-Whitney L6 > L12 p={p_ai_R:.3e}")

    # 3-seed pooled comparison
    print("\n=== 3-seed pooled L6 vs L12 ===")
    for which, key_R, key_MAE in [("OneK1K", "_R_ok_boot", "_MAE_ok_boot"),
                                    ("AIDA", "_R_ai_boot", "_MAE_ai_boot")]:
        L6_R_pooled = np.concatenate([r[key_R] for r in rows if r["layer"] == 6])
        L12_R_pooled = np.concatenate([r[key_R] for r in rows if r["layer"] == 12])
        L6_MAE_pooled = np.concatenate([r[key_MAE] for r in rows if r["layer"] == 6])
        L12_MAE_pooled = np.concatenate([r[key_MAE] for r in rows if r["layer"] == 12])
        u, p = mannwhitneyu(L6_R_pooled, L12_R_pooled, alternative="greater")
        print(f"  {which}: L6 median R = {np.median(L6_R_pooled):+.3f} [CI {np.percentile(L6_R_pooled, 2.5):+.3f}, {np.percentile(L6_R_pooled, 97.5):+.3f}]")
        print(f"           L12 median R = {np.median(L12_R_pooled):+.3f} [CI {np.percentile(L12_R_pooled, 2.5):+.3f}, {np.percentile(L12_R_pooled, 97.5):+.3f}]")
        print(f"           Mann-Whitney L6 > L12: p = {p:.3e}")
        print(f"           L6  median MAE = {np.median(L6_MAE_pooled):.2f} [CI {np.percentile(L6_MAE_pooled, 2.5):.2f}, {np.percentile(L6_MAE_pooled, 97.5):.2f}]")
        print(f"           L12 median MAE = {np.median(L12_MAE_pooled):.2f} [CI {np.percentile(L12_MAE_pooled, 2.5):.2f}, {np.percentile(L12_MAE_pooled, 97.5):.2f}]")

    # Drop bootstrap arrays for CSV
    out_rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out_rows).to_csv(OUT_CSV, index=False, float_format="%.4f")
    print(f"\n[E.7] wrote {len(out_rows)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
