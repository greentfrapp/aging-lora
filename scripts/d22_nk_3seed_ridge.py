"""D.22 — NK frozen-base 3-seed verification ridge analysis.

Reads layered embeddings for NK × seeds 0, 1, 2 across train + holdout + AIDA
cohorts. Fits ridge per layer per seed, aggregates 3-seed mean ± std per
layer. Applies the pre-committed decision rule from
`notes/decision_rules_phase3.md` §D.22 (NK ΔR(best vs L12) > +0.05 across
all 3 cohorts at 3-seed mean → anchor-ready).

Output: `results/phase3/d22_nk_3seed_layered_ridge.csv`
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from scipy.stats import pearsonr


EMB_DIR = Path("results/phase3/embeddings_layered")
OUT_CSV = Path("results/phase3/d22_nk_3seed_layered_ridge.csv")
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


def main():
    folds = json.loads(Path("data/loco_folds.json").read_text())["folds"]
    fmap = {f["fold_id"]: f for f in folds}

    seed_tags = [
        (0, "frozen_base_alllayers"),
        (1, "frozen_base_seed1_alllayers"),
        (2, "frozen_base_seed2_alllayers"),
    ]
    cell_type = "NK"
    rows = []

    # 6 conditions: 2 folds × {holdout cohort, AIDA cross-ancestry} (loco_terekhova doesn't have AIDA in original ridge_summary_layered)
    # Match the existing ridge_summary_layered.csv structure:
    fold_setups = [
        ("loco_onek1k", True),       # train: stephenson + terekhova; holdout: onek1k; AIDA cross
        ("loco_terekhova", False),   # train: stephenson + onek1k; holdout: terekhova; no AIDA in §22 conditions
    ]

    for fold_id, also_aida in fold_setups:
        f = fmap[fold_id]
        for seed, tag in seed_tags:
            train_X_per_layer, train_y_all = [], []
            skip = False
            for tc in f["train_cohorts"]:
                ret = _load_npz(tc, cell_type, tag)
                if ret is None:
                    print(f"[seed{seed} {fold_id}] missing {tc}; skip")
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
            aida_ret = _load_npz("aida", cell_type, tag) if also_aida else None
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
                rows.append({
                    "fold": fold_id, "eval_cohort": f["holdout_cohort"], "cell_type": cell_type,
                    "seed": seed, "layer": layer, "alpha": alpha,
                    "pearson_r": float(r), "mae_y": mae,
                    "n_train": int(len(train_y)), "n_eval": int(len(eval_y)),
                })
                if also_aida and aida_ret is not None:
                    _, aida_y, aida_X_layered = aida_ret
                    apred = final.predict(aida_X_layered[layer])
                    ar, _ = pearsonr(apred, aida_y)
                    amae = float(np.median(np.abs(apred - aida_y)))
                    rows.append({
                        "fold": fold_id, "eval_cohort": "aida", "cell_type": cell_type,
                        "seed": seed, "layer": layer, "alpha": alpha,
                        "pearson_r": float(ar), "mae_y": amae,
                        "n_train": int(len(train_y)), "n_eval": int(len(aida_y)),
                    })

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, float_format="%.6g")
    print(f"\n[d22] wrote {len(df)} rows to {OUT_CSV}")

    # 3-seed aggregation per (fold, eval_cohort, layer)
    agg = df.groupby(["fold", "eval_cohort", "layer"]).agg(
        R_mean=("pearson_r", "mean"), R_std=("pearson_r", "std"),
        MAE_mean=("mae_y", "mean"), MAE_std=("mae_y", "std"),
        n_seeds=("seed", "count"),
    ).reset_index()
    agg_csv = OUT_CSV.parent / "d22_nk_3seed_aggregated.csv"
    agg.to_csv(agg_csv, index=False, float_format="%.6g")
    print(f"\n[d22] wrote 3-seed aggregated to {agg_csv}")

    # Apply decision rule: ΔR(best vs L12) per (fold, eval_cohort) at 3-seed mean
    print("\n=== D.22 decision rule check (NK best-layer vs L12 at 3-seed mean) ===\n")
    print("Per `notes/decision_rules_phase3.md` §D.22:")
    print("  ΔR > +0.05 across all 3 cohorts at 3-seed mean → cell-type-conditional finding ANCHOR-READY")
    print("  2/3 cohorts → partial support")
    print("  ≤1/3 cohorts → finding is single-seed artifact, demote to supplementary")
    print()
    cohort_results = []
    for (fold, eval_cohort), g in agg.groupby(["fold", "eval_cohort"]):
        L_best = int(g.loc[g["R_mean"].idxmax(), "layer"])
        R_best = float(g.loc[g["layer"] == L_best, "R_mean"].iloc[0])
        R_12 = float(g.loc[g["layer"] == 12, "R_mean"].iloc[0])
        delta = R_best - R_12
        passes = delta > 0.05
        cohort_results.append((fold, eval_cohort, L_best, R_best, R_12, delta, passes))
        print(f"  {fold} × {eval_cohort}: L_best={L_best}, R={R_best:+.3f}, L12 R={R_12:+.3f}, ΔR={delta:+.3f} {'PASS' if passes else 'FAIL'} (>+0.05)")

    n_pass = sum(1 for r in cohort_results if r[6])
    n_total = len(cohort_results)
    print(f"\n  Result: {n_pass}/{n_total} cohorts pass ΔR > +0.05 threshold")
    if n_pass == n_total:
        print("  Decision: NK cell-type-conditional finding is ANCHOR-READY for headline.")
    elif n_pass >= n_total * 2 // 3:
        print("  Decision: PARTIAL support. Finding survives with cohort-specific caveat.")
    else:
        print("  Decision: NK finding is single-seed artifact; demote to supplementary.")

    # Layer-by-layer detail
    print("\n=== NK 3-seed layer profile (mean ± std) per condition ===\n")
    for (fold, eval_cohort), g in agg.groupby(["fold", "eval_cohort"]):
        print(f"\n  {fold} × {eval_cohort}:")
        for _, r in g.iterrows():
            print(f"    L{int(r['layer']):>2d}  R={r['R_mean']:+.3f}±{r['R_std']:.3f}  MAE={r['MAE_mean']:.2f}±{r['MAE_std']:.2f}")


if __name__ == "__main__":
    main()
