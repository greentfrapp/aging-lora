"""I.2 — Ridge readout on cap=100 NK + B frozen embeddings.

Tests whether the F.3 cap=100 layer-shift (CD4+T cap=20 L12 → cap=100 L2)
generalizes to NK and B cell types.

Output: results/phase3/i2_nk_b_cap100_layered_ridge.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from scipy.stats import pearsonr


EMB_DIR = Path("results/phase3/embeddings_layered")
OUT_CSV = Path("results/phase3/i2_nk_b_cap100_layered_ridge.csv")
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
    for cell_type in ["NK", "B"]:
        for cap, tag in [(20, "frozen_base_alllayers"),
                          (100, "frozen_base_cap100_alllayers")]:
            for fold_id in ["loco_onek1k", "loco_terekhova"]:
                f = fmap[fold_id]
                train_X_per_layer, train_y_all = [], []
                skip = False
                for tc in f["train_cohorts"]:
                    ret = _load(tc, cell_type, tag)
                    if ret is None:
                        skip = True
                        break
                    _, ages, emb = ret
                    train_X_per_layer.append(emb)
                    train_y_all.append(ages)
                if skip:
                    continue
                train_X_layered = np.concatenate(train_X_per_layer, axis=1)
                train_y = np.concatenate(train_y_all)

                eval_ret = _load(f["holdout_cohort"], cell_type, tag)
                if eval_ret is None:
                    continue
                _, eval_y, eval_X_layered = eval_ret
                aida_ret = _load("aida", cell_type, tag) if fold_id == "loco_onek1k" else None
                n_layers = eval_X_layered.shape[0]

                for layer in range(n_layers):
                    r, mae = _fit(train_X_layered[layer], train_y, eval_X_layered[layer], eval_y)
                    row = {"cell_type": cell_type, "cap": cap, "fold": fold_id, "layer": layer,
                           "holdout_R": r, "holdout_MAE": mae}
                    if aida_ret:
                        _, aida_y, aida_X = aida_ret
                        ar, amae = _fit(train_X_layered[layer], train_y, aida_X[layer], aida_y)
                        row["aida_R"] = ar
                        row["aida_MAE"] = amae
                    rows.append(row)

                # Print best layer per condition
                holdout_R = np.array([r["holdout_R"] for r in rows[-n_layers:]])
                L_best = int(np.argmax(holdout_R))
                line = f"[I.2] {cell_type} cap={cap:3d} {fold_id:14s}: L_best={L_best} R={holdout_R[L_best]:+.3f}"
                if aida_ret:
                    aida_R = np.array([r.get("aida_R", 0.0) for r in rows[-n_layers:]])
                    L_best_aida = int(np.argmax(aida_R))
                    line += f" | AIDA L_best={L_best_aida} R={aida_R[L_best_aida]:+.3f}"
                print(line, flush=True)

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, float_format="%.4f")
    print(f"\n[I.2] wrote {len(df)} rows to {OUT_CSV}")

    print("\n=== I.2 Best-layer table ===")
    for cell in ["NK", "B"]:
        for fold in ["loco_onek1k", "loco_terekhova"]:
            for cap in [20, 100]:
                sub = df[(df["cell_type"] == cell) & (df["fold"] == fold) & (df["cap"] == cap)]
                if len(sub) == 0:
                    continue
                L_best = int(sub.loc[sub["holdout_R"].idxmax(), "layer"])
                R_best = sub["holdout_R"].max()
                line = f"  {cell:3s} cap={cap:3d} {fold:14s}: L_best=L{L_best} R={R_best:+.3f}"
                if "aida_R" in sub.columns and sub["aida_R"].notna().any():
                    L_best_aida = int(sub.loc[sub["aida_R"].idxmax(), "layer"])
                    R_best_aida = sub["aida_R"].max()
                    line += f" | AIDA L{L_best_aida} R={R_best_aida:+.3f}"
                print(line)


if __name__ == "__main__":
    main()
