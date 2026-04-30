"""I.4 — 3-seed ridge readout for cap=100 CD4+T frozen.

Combines F.3 (seed=0) cap=100 with I.4 (seed=1, seed=2) extractions. Tests
§28-lesson stability of F.3's headline R=0.706 AIDA result.

Output: results/phase3/i4_cap100_3seed_layered_ridge.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from scipy.stats import pearsonr


EMB_DIR = Path("results/phase3/embeddings_layered")
OUT_CSV = Path("results/phase3/i4_cap100_3seed_layered_ridge.csv")
ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
SEED = 0


def _load(cohort: str, tag: str):
    p = EMB_DIR / f"{cohort}_CD4p_T_{tag}.npz"
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

    seed_tags = [
        (0, "frozen_base_cap100_alllayers"),
        (1, "frozen_base_cap100_seed1_alllayers"),
        (2, "frozen_base_cap100_seed2_alllayers"),
    ]

    rows = []
    for seed, tag in seed_tags:
        for fold_id in ["loco_onek1k", "loco_terekhova"]:
            f = fmap[fold_id]
            train_X_per_layer, train_y_all = [], []
            skip = False
            for tc in f["train_cohorts"]:
                ret = _load(tc, tag)
                if ret is None:
                    print(f"  [SKIP seed{seed} {fold_id}] missing {tc}_{tag}", flush=True)
                    skip = True
                    break
                _, ages, emb = ret
                train_X_per_layer.append(emb)
                train_y_all.append(ages)
            if skip:
                continue
            train_X_layered = np.concatenate(train_X_per_layer, axis=1)
            train_y = np.concatenate(train_y_all)

            eval_ret = _load(f["holdout_cohort"], tag)
            if eval_ret is None:
                continue
            _, eval_y, eval_X_layered = eval_ret
            aida_ret = _load("aida", tag) if fold_id == "loco_onek1k" else None
            n_layers = eval_X_layered.shape[0]

            for layer in range(n_layers):
                r, mae = _fit(train_X_layered[layer], train_y, eval_X_layered[layer], eval_y)
                row = {"seed": seed, "fold": fold_id, "layer": layer,
                       "holdout_R": r, "holdout_MAE": mae}
                if aida_ret:
                    _, aida_y, aida_X = aida_ret
                    ar, amae = _fit(train_X_layered[layer], train_y, aida_X[layer], aida_y)
                    row["aida_R"] = ar
                    row["aida_MAE"] = amae
                rows.append(row)

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, float_format="%.4f")
    print(f"\n[I.4] wrote {len(df)} rows to {OUT_CSV}")

    # 3-seed mean per layer per fold
    print("\n=== I.4 3-seed mean ± SD per layer ===")
    for fold in ["loco_onek1k", "loco_terekhova"]:
        sub = df[df["fold"] == fold]
        if len(sub) == 0:
            continue
        print(f"\nFold: {fold}")
        layer_R = sub.groupby("layer")["holdout_R"].agg(["mean", "std", "count"])
        for layer, row in layer_R.iterrows():
            line = f"  L{layer:2d}: holdout R = {row['mean']:+.3f} ± {row['std']:.3f}  (n={row['count']})"
            if "aida_R" in sub.columns and sub[sub["layer"] == layer]["aida_R"].notna().any():
                aida_R_vals = sub[sub["layer"] == layer]["aida_R"].dropna()
                line += f" | AIDA R = {aida_R_vals.mean():+.3f} ± {aida_R_vals.std():.3f}"
            print(line)

    # Decision rule
    print("\n=== I.4 Decision rule ===")
    aida_at_L2 = df[(df["fold"] == "loco_onek1k") & (df["layer"] == 2) & (df["aida_R"].notna())]
    if len(aida_at_L2) > 0:
        mean_aida_L2 = aida_at_L2["aida_R"].mean()
        std_aida_L2 = aida_at_L2["aida_R"].std()
        print(f"  3-seed mean AIDA R at L2 (cap=100) = {mean_aida_L2:+.3f} ± {std_aida_L2:.3f}")
        if mean_aida_L2 >= 0.65:
            print("  → cap=100 effect ROBUST; F.3 headline holds.")
        elif mean_aida_L2 >= 0.55:
            print("  → cap effect real but smaller than single-seed; report 3-seed mean as headline.")
        else:
            print("  → single-seed F.3 was a fluke; cap effect much smaller.")


if __name__ == "__main__":
    main()
