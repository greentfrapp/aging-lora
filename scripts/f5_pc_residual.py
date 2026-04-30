"""F.5 — Per-layer PC-residual age recovery.

For each multi-seed condition × layer × k ∈ {5, 10, 25, 50}:
  Fit PCA on training embeddings, project out top-k PCs.
  Refit ridge on residual subspace.
  Compare R/MAE to full-embedding ridge.

Tests additional_concerns.md #3: is age signal a low-variance residual axis
competing with stronger cell-type/batch axes?

Decision rule (pre-commit):
  Improves substantially (ΔR ≥ 0.05) on >50% of conditions → age is residual.
  Mixed (ΔR ∈ [-0.02, +0.05]) → no clean reframe.
  Degrades (ΔR ≤ -0.05) on >50% conditions → age in high-variance subspace.

Output: results/phase3/f5_pc_residual.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr


EMB_DIR = Path("results/phase3/embeddings_layered")
OUT_CSV = Path("results/phase3/f5_pc_residual.csv")
ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
PC_KS = [5, 10, 25, 50]
SEED = 0


def _slug(cell_type: str) -> str:
    return cell_type.replace("+", "p").replace(" ", "_")


def _load(cohort: str, cell_type: str, tag: str):
    p = EMB_DIR / f"{cohort}_{_slug(cell_type)}_{tag}.npz"
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=True)
    return z["donor_ids"], z["ages"].astype(np.float32), z["embeddings_per_layer"].astype(np.float32)


def _r_mae(pred, y):
    if np.std(pred) > 0 and np.std(y) > 0 and len(y) > 1:
        r, _ = pearsonr(pred, y)
    else:
        r = 0.0
    return float(r), float(np.median(np.abs(pred - y)))


def _ridge(X_tr, y_tr, X_ev, y_ev):
    cv = RidgeCV(alphas=ALPHAS, cv=3, scoring="neg_mean_absolute_error")
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(y_tr))
    cv.fit(X_tr[perm], y_tr[perm])
    final = Ridge(alpha=float(cv.alpha_)).fit(X_tr, y_tr)
    pred = final.predict(X_ev)
    return _r_mae(pred, y_ev)


def _residualize(X_tr, X_ev, k):
    """Project out top-k PCs of X_tr from both X_tr and X_ev. Returns residuals."""
    scaler = StandardScaler().fit(X_tr)
    Xs_tr = scaler.transform(X_tr)
    Xs_ev = scaler.transform(X_ev)
    pca = PCA(n_components=k).fit(Xs_tr)
    Xs_tr_resid = Xs_tr - pca.inverse_transform(pca.transform(Xs_tr))
    Xs_ev_resid = Xs_ev - pca.inverse_transform(pca.transform(Xs_ev))
    return Xs_tr_resid, Xs_ev_resid


# Same multi-seed conditions as F.4
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
            ret = _load(tc, cell_type, tag)
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

        eval_ret = _load(f["holdout_cohort"], cell_type, tag)
        if eval_ret is None:
            continue
        _, eval_y, eval_X_layered = eval_ret
        aida_ret = _load("aida", cell_type, tag) if also_aida else None
        n_layers = eval_X_layered.shape[0]

        print(f"\n=== {method_label} | {fold_id} × {cell_type} × seed{seed} ===", flush=True)
        for layer in range(n_layers):
            X_tr = train_X_layered[layer]
            X_ev = eval_X_layered[layer]

            # Full-embed baseline
            r_full, mae_full = _ridge(X_tr, train_y, X_ev, eval_y)
            r_full_aida, mae_full_aida = (None, None)
            if aida_ret is not None:
                _, aida_y, aida_X = aida_ret
                r_full_aida, mae_full_aida = _ridge(X_tr, train_y, aida_X[layer], aida_y)

            for k in PC_KS:
                if k >= min(X_tr.shape):
                    continue
                X_tr_res, X_ev_res = _residualize(X_tr, X_ev, k)
                r_res, mae_res = _ridge(X_tr_res, train_y, X_ev_res, eval_y)
                r_res_aida, mae_res_aida = (None, None)
                if aida_ret is not None:
                    _, aida_y, aida_X = aida_ret
                    _, X_aida_res = _residualize(X_tr, aida_X[layer], k)
                    r_res_aida, mae_res_aida = _ridge(X_tr_res, train_y, X_aida_res, aida_y)

                rows.append({
                    "method": method_label, "fold": fold_id, "cell_type": cell_type,
                    "seed": seed, "layer": layer, "k_pc": k,
                    "R_full": r_full, "MAE_full": mae_full,
                    "R_residual": r_res, "MAE_residual": mae_res,
                    "deltaR_holdout": r_res - r_full,
                    "R_aida_full": r_full_aida, "R_aida_residual": r_res_aida,
                    "deltaR_aida": (r_res_aida - r_full_aida) if r_full_aida is not None else None,
                })

            # Print best k per layer
            sub = [r for r in rows if r["method"] == method_label and r["fold"] == fold_id
                   and r["cell_type"] == cell_type and r["seed"] == seed and r["layer"] == layer]
            if sub:
                best_k_holdout = max(sub, key=lambda r: r["R_residual"])
                print(f"  L{layer}: full R={r_full:+.3f}, best resid (k={best_k_holdout['k_pc']}) R={best_k_holdout['R_residual']:+.3f}, ΔR={best_k_holdout['deltaR_holdout']:+.3f}")

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, float_format="%.4f")
    print(f"\n[F.5] wrote {len(df)} rows to {OUT_CSV}")

    # Decision: count conditions where best PC-residual ΔR >= 0.05
    n_improve, n_no_change, n_degrade, n_total = 0, 0, 0, 0
    for (m, fold, ct, seed), grp in df.groupby(["method", "fold", "cell_type", "seed"]):
        # At each layer, find max ΔR across k
        layer_max_dR = grp.groupby("layer")["deltaR_holdout"].max()
        max_layer_dR = layer_max_dR.max()
        if max_layer_dR >= 0.05:
            n_improve += 1
        elif max_layer_dR <= -0.05:
            n_degrade += 1
        else:
            n_no_change += 1
        n_total += 1
    print(f"\n=== Decision ===")
    print(f"  Improve (max ΔR >= +0.05): {n_improve}/{n_total}")
    print(f"  No change:                 {n_no_change}/{n_total}")
    print(f"  Degrade (max ΔR <= -0.05): {n_degrade}/{n_total}")
    if n_improve > 0.5 * n_total:
        print("  → DECISION: age is residual axis; reframe.")
    elif n_degrade > 0.5 * n_total:
        print("  → DECISION: age in high-variance subspace; no reframe.")
    else:
        print("  → DECISION: mixed; no clean reframe.")


if __name__ == "__main__":
    main()
