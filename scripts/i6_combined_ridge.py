"""I.6 — Combined FM ridge readout across cap-matrix.

For each (cap, seed) condition, runs per-layer ridge readout and produces
the 3-seed mean ± SD per layer per fold. cap=1000 is single-seed.

Output: results/phase3/i6_fm_ridge_caps.csv
Combined comparison table: results/phase3/i6_summary.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from scipy.stats import pearsonr


EMB_DIR = Path("results/phase3/embeddings_layered")
OUT_CSV = Path("results/phase3/i6_fm_ridge_caps.csv")
SUMMARY_CSV = Path("results/phase3/i6_summary.csv")
ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]


def _tag(cap: int, seed: int) -> str:
    if seed == 0:
        return f"frozen_base_cap{cap}_alllayers"
    return f"frozen_base_cap{cap}_seed{seed}_alllayers"


def _load(cohort: str, cap: int, seed: int):
    p = EMB_DIR / f"{cohort}_CD4p_T_{_tag(cap, seed)}.npz"
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=True)
    return z["donor_ids"], z["ages"].astype(np.float32), z["embeddings_per_layer"].astype(np.float32)


def _fit(X_train, y_train, X_eval, y_eval, seed=0):
    cv = RidgeCV(alphas=ALPHAS, cv=3, scoring="neg_mean_absolute_error")
    rng = np.random.default_rng(seed)
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

    cap_seed_pairs = [
        (50, 0), (50, 1), (50, 2),
        (100, 0), (100, 1), (100, 2),
        (500, 0), (500, 1), (500, 2),
        (1000, 0),
    ]

    rows = []
    for cap, seed in cap_seed_pairs:
        for fold_id in ["loco_onek1k", "loco_terekhova"]:
            f = fmap[fold_id]
            train_X_per_layer, train_y_all = [], []
            skip = False
            for tc in f["train_cohorts"]:
                ret = _load(tc, cap, seed)
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

            eval_ret = _load(f["holdout_cohort"], cap, seed)
            # AIDA eval for both folds (matched comparison with gene-EN, which
            # also evaluated AIDA for both fold directions).
            aida_ret = _load("aida", cap, seed)
            # Skip only if BOTH holdout and AIDA are missing (no eval data at all).
            # When onek1k cap=500/1000 is intentionally skipped, holdout is None
            # but AIDA-only eval should still proceed for loco_onek1k fold.
            if eval_ret is None and aida_ret is None:
                continue
            n_layers = (eval_ret[2] if eval_ret is not None else aida_ret[2]).shape[0]

            for layer in range(n_layers):
                row = {"cap": cap, "seed": seed, "fold": fold_id, "layer": layer}
                if eval_ret is not None:
                    _, eval_y, eval_X_layered = eval_ret
                    r, mae = _fit(train_X_layered[layer], train_y, eval_X_layered[layer], eval_y)
                    row["holdout_R"] = r
                    row["holdout_MAE"] = mae
                if aida_ret:
                    _, aida_y, aida_X = aida_ret
                    ar, amae = _fit(train_X_layered[layer], train_y, aida_X[layer], aida_y)
                    row["aida_R"] = ar
                    row["aida_MAE"] = amae
                rows.append(row)

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, float_format="%.4f")
    print(f"[I.6] wrote {len(df)} FM-ridge rows to {OUT_CSV}")

    # PER-LAYER 3-seed mean (the honest readout, per I.4 lesson):
    # for each layer, compute mean ± SD across seeds; then pick the layer
    # that wins on per-layer-3-seed-mean. NOT per-seed-argmax (that's
    # post-hoc layer selection on eval and is what F.3 did wrong).
    print("\n=== I.6 FM per-layer 3-seed mean per (cap × fold) — best by 3-seed mean ===")
    summary_rows = []
    for fold in ["loco_onek1k", "loco_terekhova"]:
        sub = df[df["fold"] == fold]
        if len(sub) == 0:
            continue
        print(f"\nFold: {fold}")
        for cap in sorted(sub["cap"].unique()):
            sub2 = sub[sub["cap"] == cap]
            n_seeds = sub2["seed"].nunique()
            row = {"fold": fold, "cap": cap, "method": "FM", "n_seeds": n_seeds}
            line = f"  cap={cap:5d}:"
            # Holdout: only if any holdout_R is non-NaN (skipped when onek1k
            # extractions are absent and onek1k is the holdout cohort).
            has_holdout = "holdout_R" in sub2.columns and sub2["holdout_R"].notna().any()
            if has_holdout:
                per_layer = sub2.dropna(subset=["holdout_R"]).groupby("layer")["holdout_R"].agg(["mean", "std"]).reset_index()
                best_layer_holdout = int(per_layer.loc[per_layer["mean"].idxmax(), "layer"])
                best_R_holdout = float(per_layer["mean"].max())
                best_SD_holdout = float(per_layer.loc[per_layer["mean"].idxmax(), "std"])
                competitive = per_layer[per_layer["mean"] >= best_R_holdout - 0.02]
                stable_layer_holdout = int(competitive.loc[competitive["std"].idxmin(), "layer"])
                stable_R_holdout = float(competitive.loc[competitive["std"].idxmin(), "mean"])
                stable_SD_holdout = float(competitive["std"].min())
                line += (f" holdout best-by-mean L{best_layer_holdout:2d} = {best_R_holdout:+.3f} ± {best_SD_holdout:.3f}"
                         f"  | stable L{stable_layer_holdout:2d} = {stable_R_holdout:+.3f} ± {stable_SD_holdout:.3f}  (n_seeds={n_seeds})")
                row.update({"best_layer_holdout": best_layer_holdout,
                            "holdout_R_mean": best_R_holdout, "holdout_R_std": best_SD_holdout,
                            "stable_layer_holdout": stable_layer_holdout,
                            "stable_holdout_R_mean": stable_R_holdout,
                            "stable_holdout_R_std": stable_SD_holdout})
            else:
                line += f" holdout SKIPPED (no holdout NPZs)  (n_seeds={n_seeds})"
            if "aida_R" in sub2.columns and sub2["aida_R"].notna().any():
                per_layer_aida = sub2.dropna(subset=["aida_R"]).groupby("layer")["aida_R"].agg(["mean", "std"]).reset_index()
                best_layer_aida = int(per_layer_aida.loc[per_layer_aida["mean"].idxmax(), "layer"])
                best_R_aida = float(per_layer_aida["mean"].max())
                best_SD_aida = float(per_layer_aida.loc[per_layer_aida["mean"].idxmax(), "std"])
                competitive_a = per_layer_aida[per_layer_aida["mean"] >= best_R_aida - 0.02]
                stable_layer_aida = int(competitive_a.loc[competitive_a["std"].idxmin(), "layer"])
                stable_R_aida = float(competitive_a.loc[competitive_a["std"].idxmin(), "mean"])
                stable_SD_aida = float(competitive_a["std"].min())
                line += (f"\n         AIDA    best-by-mean L{best_layer_aida:2d} = {best_R_aida:+.3f} ± {best_SD_aida:.3f}"
                         f"  | stable L{stable_layer_aida:2d} = {stable_R_aida:+.3f} ± {stable_SD_aida:.3f}")
                row["best_layer_aida"] = best_layer_aida
                row["aida_R_mean"] = best_R_aida
                row["aida_R_std"] = best_SD_aida
                row["stable_layer_aida"] = stable_layer_aida
                row["stable_aida_R_mean"] = stable_R_aida
                row["stable_aida_R_std"] = stable_SD_aida
            print(line)
            summary_rows.append(row)

    # Cross-method (FM vs gene-EN) summary if gene-EN CSV exists.
    gene_csv = Path("results/phase3/i6_gene_en_3seed_caps.csv")
    if gene_csv.exists():
        gdf = pd.read_csv(gene_csv)
        print("\n=== I.6 gene-EN 3-seed mean per (cap × fold) ===")
        for fold in ["loco_onek1k", "loco_terekhova"]:
            sub = gdf[gdf["fold"] == fold]
            if len(sub) == 0:
                continue
            print(f"\nFold: {fold}")
            for cap in sorted(sub["cap"].unique()):
                holdout_vals = sub[(sub["cap"] == cap) & (sub["eval_cohort"] != "aida")]["R"]
                aida_vals = sub[(sub["cap"] == cap) & (sub["eval_cohort"] == "aida")]["R"]
                line = f"  cap={cap:5d}: holdout R = {holdout_vals.mean():+.3f} ± {holdout_vals.std():.3f}"
                row = {"fold": fold, "cap": cap, "method": "gene-EN",
                       "holdout_R_mean": holdout_vals.mean(), "holdout_R_std": holdout_vals.std(),
                       "n_seeds": len(holdout_vals)}
                if len(aida_vals) > 0:
                    line += f" | AIDA = {aida_vals.mean():+.3f} ± {aida_vals.std():.3f}"
                    row["aida_R_mean"] = aida_vals.mean()
                    row["aida_R_std"] = aida_vals.std()
                print(line)
                summary_rows.append(row)

        # Matched-cap matched-method comparison (both folds)
        print("\n=== I.6 Matched-cap FM-vs-gene-EN gap (3-seed mean AIDA R) ===")
        for fold in ["loco_onek1k", "loco_terekhova"]:
            print(f"\nFold: {fold}")
            for cap in sorted({50, 100, 500, 1000}):
                fm_row = next((r for r in summary_rows if r["fold"] == fold and r["cap"] == cap and r["method"] == "FM"), None)
                gen_row = next((r for r in summary_rows if r["fold"] == fold and r["cap"] == cap and r["method"] == "gene-EN"), None)
                if fm_row and gen_row and "aida_R_mean" in fm_row and "aida_R_mean" in gen_row:
                    gap = fm_row["aida_R_mean"] - gen_row["aida_R_mean"]
                    print(f"  cap={cap:5d}: FM AIDA R = {fm_row['aida_R_mean']:+.3f} ± {fm_row['aida_R_std']:.3f}  "
                          f"vs gene-EN = {gen_row['aida_R_mean']:+.3f} ± {gen_row['aida_R_std']:.3f}  "
                          f"→ gap = {gap:+.3f}")

    sdf = pd.DataFrame(summary_rows)
    sdf.to_csv(SUMMARY_CSV, index=False, float_format="%.4f")
    print(f"\n[I.6] wrote {len(sdf)} summary rows to {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
