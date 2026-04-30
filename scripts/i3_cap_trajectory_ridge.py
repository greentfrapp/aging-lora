"""I.3 — Cap-trajectory plateau ridge readout for CD4+T frozen.

Combines existing F.3 caps (5/20/100) with new I.3 caps (50/200/500).
Tests whether cap=20 → cap=100 R gain plateaus at cap=200 or keeps climbing.

Output: results/phase3/i3_cap_trajectory.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from scipy.stats import pearsonr


EMB_DIR = Path("results/phase3/embeddings_layered")
OUT_CSV = Path("results/phase3/i3_cap_trajectory.csv")
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

    # F.3 cap=5/20/100 are in different file naming; only cap=100 has the
    # consistent _alllayers tag. Use I.3 caps 50/200/500 + cap=100 (F.3) for
    # the trajectory.
    cap_tags = [
        (50, "frozen_base_cap50_alllayers"),
        (100, "frozen_base_cap100_alllayers"),
        (200, "frozen_base_cap200_alllayers"),
        (500, "frozen_base_cap500_alllayers"),
    ]

    rows = []
    for cap, tag in cap_tags:
        for fold_id in ["loco_onek1k", "loco_terekhova"]:
            f = fmap[fold_id]
            train_X_per_layer, train_y_all = [], []
            skip = False
            for tc in f["train_cohorts"]:
                ret = _load(tc, tag)
                if ret is None:
                    print(f"  [SKIP cap={cap} {fold_id}] missing {tc}_{tag}", flush=True)
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
                row = {"cap": cap, "fold": fold_id, "layer": layer,
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
    print(f"\n[I.3] wrote {len(df)} rows to {OUT_CSV}")

    # Best-layer trajectory
    print("\n=== I.3 Cap trajectory (best-layer R per cap × fold) ===")
    for fold in ["loco_onek1k", "loco_terekhova"]:
        sub = df[df["fold"] == fold]
        if len(sub) == 0:
            continue
        print(f"\nFold: {fold}")
        for cap in sorted(sub["cap"].unique()):
            sub2 = sub[sub["cap"] == cap]
            L_best = int(sub2.loc[sub2["holdout_R"].idxmax(), "layer"])
            R_best = sub2["holdout_R"].max()
            line = f"  cap={cap:3d}: L_best=L{L_best:2d} R={R_best:+.3f}"
            if "aida_R" in sub2.columns and sub2["aida_R"].notna().any():
                L_best_aida = int(sub2.loc[sub2["aida_R"].idxmax(), "layer"])
                R_best_aida = sub2["aida_R"].max()
                line += f" | AIDA L{L_best_aida:2d} R={R_best_aida:+.3f}"
            print(line)

    # Decision rule
    print("\n=== I.3 Decision rule ===")
    for fold in ["loco_onek1k"]:
        sub = df[df["fold"] == fold]
        if "aida_R" not in sub.columns:
            continue
        cap200 = sub[sub["cap"] == 200]["aida_R"].max() if 200 in sub["cap"].values else None
        cap100 = sub[sub["cap"] == 100]["aida_R"].max() if 100 in sub["cap"].values else None
        cap500 = sub[sub["cap"] == 500]["aida_R"].max() if 500 in sub["cap"].values else None
        if cap100 is not None and cap200 is not None:
            print(f"  AIDA best-layer R: cap=100={cap100:+.3f}, cap=200={cap200:+.3f}", end="")
            if cap500 is not None:
                print(f", cap=500={cap500:+.3f}")
            else:
                print()
            delta = cap200 - cap100
            if abs(delta) <= 0.01:
                print("  → plateau reached at cap=100; cap=100 is the recipe.")
            elif delta < -0.01:
                print(f"  → cap=200 dropped {delta:+.3f} below cap=100; investigate.")
            elif cap500 is not None and cap500 - cap200 > 0.04:
                print("  → trajectory keeps climbing; recipe is 'as many cells as possible'.")
            else:
                print(f"  → moderate gain at cap=200 ({delta:+.3f}); cap=200-500 sweet spot.")


if __name__ == "__main__":
    main()
