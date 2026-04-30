"""F.3 ridge readout — layer-wise frozen CD4+T at cap=5, cap=20 (existing),
cap=100 across loco_onek1k + loco_terekhova + AIDA cross-ancestry.

Tests whether cell-type-conditional layer asymmetry (CD4+T at L9.7) is robust
to per-donor cell count or driven by SNR.

Output: results/phase3/f3_cell_count_layered_ridge.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from scipy.stats import pearsonr


EMB_DIR = Path("results/phase3/embeddings_layered")
OUT_CSV = Path("results/phase3/f3_cell_count_layered_ridge.csv")
ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
SEED = 0


def _slug(cell_type: str) -> str:
    return cell_type.replace("+", "p").replace(" ", "_")


def _load(cohort: str, cell_type: str, tag: str):
    p = EMB_DIR / f"{cohort}_{_slug(cell_type)}_{tag}.npz"
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=True)
    return z["donor_ids"], z["ages"].astype(np.float32), z["embeddings_per_layer"].astype(np.float32)


def _fit(X_train, y_train, X_eval, y_eval):
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


def main():
    folds = json.loads(Path("data/loco_folds.json").read_text())["folds"]
    fmap = {f["fold_id"]: f for f in folds}

    rows = []
    for cap, tag in [(5, "frozen_base_cap5_alllayers"),
                     (20, "frozen_base_alllayers"),
                     (100, "frozen_base_cap100_alllayers")]:
        for fold_id in ["loco_onek1k", "loco_terekhova"]:
            f = fmap[fold_id]
            train_X_per_layer, train_y_all = [], []
            skip = False
            for tc in f["train_cohorts"]:
                ret = _load(tc, "CD4+ T", tag)
                if ret is None:
                    print(f"  [SKIP cap={cap} {fold_id}] missing {tc}_CD4p_T_{tag}")
                    skip = True
                    break
                _, ages, emb_LDH = ret
                train_X_per_layer.append(emb_LDH)
                train_y_all.append(ages)
            if skip:
                continue
            train_X_layered = np.concatenate(train_X_per_layer, axis=1)
            train_y = np.concatenate(train_y_all)

            eval_ret = _load(f["holdout_cohort"], "CD4+ T", tag)
            if eval_ret is None:
                continue
            _, eval_y, eval_X_layered = eval_ret
            n_layers = eval_X_layered.shape[0]

            aida_ret = _load("aida", "CD4+ T", tag) if fold_id == "loco_onek1k" else None

            for layer in range(n_layers):
                r, mae = _fit(train_X_layered[layer], train_y, eval_X_layered[layer], eval_y)
                row = {"cap": cap, "fold": fold_id, "layer": layer, "holdout_R": r, "holdout_MAE": mae}
                if aida_ret:
                    _, aida_y, aida_X = aida_ret
                    ar, amae = _fit(train_X_layered[layer], train_y, aida_X[layer], aida_y)
                    row["aida_R"] = ar
                    row["aida_MAE"] = amae
                rows.append(row)

            # Per-fold summary
            holdout_R_vec = np.array([r["holdout_R"] for r in rows[-n_layers:]])
            L_best = int(np.argmax(holdout_R_vec))
            print(f"[F.3] cap={cap} {fold_id} CD4+T: L_best={L_best} R={holdout_R_vec[L_best]:.3f} | "
                  f"R curve: {[f'{x:.2f}' for x in holdout_R_vec]}", flush=True)

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, float_format="%.4f")
    print(f"\n[F.3] wrote {len(df)} rows to {OUT_CSV}")

    # Summary table
    print("\n=== F.3 Best-layer per (cap × fold) ===")
    for cap in sorted(df["cap"].unique()):
        for fold in sorted(df["fold"].unique()):
            sub = df[(df["cap"] == cap) & (df["fold"] == fold)]
            if len(sub) == 0:
                continue
            L_best_holdout = int(sub.loc[sub["holdout_R"].idxmax(), "layer"])
            R_best_holdout = sub["holdout_R"].max()
            line = f"  cap={cap:3d} {fold:14s} CD4+T: L_best_holdout=L{L_best_holdout} (R={R_best_holdout:.3f})"
            if "aida_R" in sub.columns and sub["aida_R"].notna().any():
                L_best_aida = int(sub.loc[sub["aida_R"].idxmax(), "layer"])
                R_best_aida = sub["aida_R"].max()
                line += f" | L_best_AIDA=L{L_best_aida} (R={R_best_aida:.3f})"
            print(line)

    print("\n=== Decision rule ===")
    rows_loco = df[df["fold"] == "loco_onek1k"].copy()
    L_best_per_cap = {}
    for cap in [5, 20, 100]:
        sub = rows_loco[rows_loco["cap"] == cap]
        if len(sub) > 0:
            L_best_per_cap[cap] = int(sub.loc[sub["holdout_R"].idxmax(), "layer"])
    if 5 in L_best_per_cap and 20 in L_best_per_cap:
        L_5 = L_best_per_cap[5]
        L_20 = L_best_per_cap[20]
        shift = L_20 - L_5
        if L_5 >= 9:
            print(f"  cap=5 still picks L{L_5} (>=L9): cell-type-conditional asymmetry is robust to per-donor cell count → BIOLOGY")
        elif L_5 <= 5:
            print(f"  cap=5 picks L{L_5} (<=L5, near NK regime L3.3): asymmetry is SNR-driven → DATA QUALITY")
        else:
            print(f"  cap=5 picks L{L_5} (mid-range): partial shift, mixed interpretation")


if __name__ == "__main__":
    main()
