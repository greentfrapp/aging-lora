"""I.9 — combined FM ridge readout for NK and B cell types.

For each cell type, runs the same per-layer 3-seed-mean readout as
i6_combined_ridge.py and merges with i9_gene_en_full.csv to produce
matched-cap gap tables per cell type.

Output:
  results/phase3/i9_fm_ridge_caps.csv (per cell_type × cap × seed × fold × layer)
  results/phase3/i9_summary.csv (per cell_type × cap × fold × method aggregates)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from scipy.stats import pearsonr


EMB_DIR = Path("results/phase3/embeddings_layered")
OUT_CSV = Path("results/phase3/i9_fm_ridge_caps.csv")
SUMMARY_CSV = Path("results/phase3/i9_summary.csv")
ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]

CELL_TYPES = ["NK", "B"]
CAP_SEED_PAIRS = [
    (1, 0), (1, 1), (1, 2),
    (5, 0), (5, 1), (5, 2),
    (10, 0), (10, 1), (10, 2),
    (20, 0), (20, 1), (20, 2),
    (50, 0), (50, 1), (50, 2),
    (100, 0), (100, 1), (100, 2),
    (500, 0), (500, 1), (500, 2),
    (1000, 0), (1000, 1), (1000, 2),
]


def _slug(cell_type: str) -> str:
    return cell_type.replace("+", "p").replace(" ", "_")


def _tag(cap: int, seed: int) -> str:
    if seed == 0:
        return f"frozen_base_cap{cap}_alllayers"
    return f"frozen_base_cap{cap}_seed{seed}_alllayers"


def _load(cohort: str, cell_type: str, cap: int, seed: int):
    p = EMB_DIR / f"{cohort}_{_slug(cell_type)}_{_tag(cap, seed)}.npz"
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
    if np.std(pred) > 1e-3 and np.std(y_eval) > 0 and len(y_eval) > 1:
        r, _ = pearsonr(pred, y_eval)
        if not np.isfinite(r):
            r = 0.0
    else:
        r = 0.0
    mae = float(np.median(np.abs(pred - y_eval)))
    return float(r), mae


def main():
    folds = json.loads(Path("data/loco_folds.json").read_text())["folds"]
    fmap = {f["fold_id"]: f for f in folds}

    rows = []
    for cell_type in CELL_TYPES:
        for cap, seed in CAP_SEED_PAIRS:
            for fold_id in ["loco_onek1k", "loco_terekhova"]:
                f = fmap[fold_id]
                train_X_per_layer, train_y_all = [], []
                skip = False
                for tc in f["train_cohorts"]:
                    ret = _load(tc, cell_type, cap, seed)
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

                eval_ret = _load(f["holdout_cohort"], cell_type, cap, seed)
                aida_ret = _load("aida", cell_type, cap, seed)
                if eval_ret is None and aida_ret is None:
                    continue
                n_layers = (eval_ret[2] if eval_ret is not None else aida_ret[2]).shape[0]

                for layer in range(n_layers):
                    row = {"cell_type": cell_type, "cap": cap, "seed": seed,
                           "fold": fold_id, "layer": layer}
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
    print(f"[I.9] wrote {len(df)} FM-ridge rows to {OUT_CSV}")

    summary_rows = []
    for cell_type in CELL_TYPES:
        for fold in ["loco_onek1k", "loco_terekhova"]:
            sub = df[(df["cell_type"] == cell_type) & (df["fold"] == fold)]
            if len(sub) == 0:
                continue
            print(f"\n{cell_type} fold={fold}: per-layer 3-seed mean")
            for cap in sorted(sub["cap"].unique()):
                sub2 = sub[sub["cap"] == cap]
                n_seeds = sub2["seed"].nunique()
                row = {"cell_type": cell_type, "fold": fold, "cap": cap,
                       "method": "FM", "n_seeds": n_seeds}
                line = f"  cap={cap:5d}:"

                has_holdout = "holdout_R" in sub2.columns and sub2["holdout_R"].notna().any()
                if has_holdout:
                    per_layer = sub2.dropna(subset=["holdout_R"]).groupby("layer")["holdout_R"].agg(["mean", "std"]).reset_index()
                    bL = int(per_layer.loc[per_layer["mean"].idxmax(), "layer"])
                    bR = float(per_layer["mean"].max())
                    bSD_raw = per_layer.loc[per_layer["mean"].idxmax(), "std"]
                    bSD = float(bSD_raw) if pd.notna(bSD_raw) else float("nan")
                    line += f" holdout best L{bL:2d} = {bR:+.3f} ± {bSD:.3f} (n_seeds={n_seeds})"
                    row["best_layer_holdout"] = bL
                    row["holdout_R_mean"] = bR
                    row["holdout_R_std"] = bSD

                if "aida_R" in sub2.columns and sub2["aida_R"].notna().any():
                    per_layer_a = sub2.dropna(subset=["aida_R"]).groupby("layer")["aida_R"].agg(["mean", "std"]).reset_index()
                    bL = int(per_layer_a.loc[per_layer_a["mean"].idxmax(), "layer"])
                    bR = float(per_layer_a["mean"].max())
                    bSD_raw = per_layer_a.loc[per_layer_a["mean"].idxmax(), "std"]
                    bSD = float(bSD_raw) if pd.notna(bSD_raw) else float("nan")
                    line += f"\n         AIDA    best L{bL:2d} = {bR:+.3f} ± {bSD:.3f}"
                    row["best_layer_aida"] = bL
                    row["aida_R_mean"] = bR
                    row["aida_R_std"] = bSD
                print(line)
                summary_rows.append(row)

    # Add gene-EN rows from i9_gene_en_full.csv if present
    gene_csv = Path("results/phase3/i9_gene_en_full.csv")
    if gene_csv.exists():
        gdf = pd.read_csv(gene_csv)
        print("\n=== I.9 gene-EN 3-seed mean per (cell_type × cap × fold) ===")
        for cell_type in CELL_TYPES:
            for fold in ["loco_onek1k", "loco_terekhova"]:
                sub = gdf[(gdf["cell_type"] == cell_type) & (gdf["fold"] == fold)]
                if len(sub) == 0:
                    continue
                print(f"\n{cell_type} fold={fold}:")
                for cap in sorted(sub["cap"].unique()):
                    holdout_vals = sub[(sub["cap"] == cap) & (sub["eval_cohort"] != "aida")]["R"]
                    aida_vals = sub[(sub["cap"] == cap) & (sub["eval_cohort"] == "aida")]["R"]
                    line = f"  cap={cap:5d}: holdout = {holdout_vals.mean():+.3f} ± {holdout_vals.std():.3f}"
                    row = {"cell_type": cell_type, "fold": fold, "cap": cap,
                           "method": "gene-EN", "n_seeds": len(holdout_vals),
                           "holdout_R_mean": holdout_vals.mean(), "holdout_R_std": holdout_vals.std()}
                    if len(aida_vals) > 0:
                        line += f" | AIDA = {aida_vals.mean():+.3f} ± {aida_vals.std():.3f}"
                        row["aida_R_mean"] = aida_vals.mean()
                        row["aida_R_std"] = aida_vals.std()
                    print(line)
                    summary_rows.append(row)

        print("\n=== I.9 Matched-cap FM-vs-gene-EN gap (3-seed mean AIDA R) ===")
        for cell_type in CELL_TYPES:
            for fold in ["loco_onek1k", "loco_terekhova"]:
                print(f"\n{cell_type} fold={fold}:")
                for cap in sorted({1, 5, 10, 20, 50, 100, 500, 1000}):
                    fm = next((r for r in summary_rows
                               if r.get("cell_type") == cell_type and r["fold"] == fold
                               and r["cap"] == cap and r["method"] == "FM"), None)
                    gen = next((r for r in summary_rows
                                if r.get("cell_type") == cell_type and r["fold"] == fold
                                and r["cap"] == cap and r["method"] == "gene-EN"), None)
                    if fm and gen and "aida_R_mean" in fm and "aida_R_mean" in gen:
                        gap = fm["aida_R_mean"] - gen["aida_R_mean"]
                        print(f"  cap={cap:5d}: FM {fm['aida_R_mean']:+.3f} ± {fm['aida_R_std']:.3f}  "
                              f"vs gene-EN {gen['aida_R_mean']:+.3f} ± {gen['aida_R_std']:.3f}  "
                              f"→ gap = {gap:+.3f}")

    sdf = pd.DataFrame(summary_rows)
    sdf.to_csv(SUMMARY_CSV, index=False, float_format="%.4f")
    print(f"\n[I.9] wrote {len(sdf)} summary rows to {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
