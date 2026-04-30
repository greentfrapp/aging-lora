"""H.1 multi-seed verification of G.2 — 3-seed PC-residualized FM probe on B × Terekhova.

Replicates G.2's CV-honest pipeline (inner 3-fold CV by donor on train, pick (layer × k_pc)
by mean CV R, evaluate at picked recipe on Terekhova holdout + AIDA) for seeds 0, 1, 2.
Aggregates to 3-seed mean ± std.

Decision rule (pre-commit, 3-seed mean R on Terekhova holdout, gene-EN R = 0.321):
  3-seed mean ≥ 0.27 AND σ(R) ≤ 0.05 → MATCHES verdict survives multi-seed.
  3-seed mean ≥ 0.27 AND σ(R) > 0.05 → high seed variance; deployment needs ≥3 seeds.
  3-seed mean ∈ [0.17, 0.27) → NARROWS GAP downgrade.
  3-seed mean < 0.17 → single-seed luck; B-cell parity claim dropped.

Output: results/phase3/h1_b_cell_multi_seed.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from scipy.stats import pearsonr


EMB_DIR = Path("results/phase3/embeddings_layered")
OUT_CSV = Path("results/phase3/h1_b_cell_multi_seed.csv")
OUT_GRID_CSV = Path("results/phase3/h1_b_cell_multi_seed_cv_grid.csv")
ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
PC_KS = [5, 10, 25, 50]
SEED = 0
GENE_EN_R_TEREKHOVA = 0.3210  # gene-EN B × loco_terekhova → terekhova holdout
GENE_EN_R_ONEK1K = 0.1358  # gene-EN B × loco_onek1k → onek1k holdout
FOLD_IDS = ["loco_terekhova", "loco_onek1k"]
CELL_TYPE_SLUG = "B"

SEED_TAGS = [
    (0, "frozen_base_alllayers"),
    (1, "frozen_base_seed1_alllayers"),
    (2, "frozen_base_seed2_alllayers"),
]


def _load(cohort, tag):
    p = EMB_DIR / f"{cohort}_{CELL_TYPE_SLUG}_{tag}.npz"
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=True)
    donor_ids = np.asarray([str(d) for d in z["donor_ids"]])
    return donor_ids, z["ages"].astype(np.float32), z["embeddings_per_layer"].astype(np.float32)


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
    return _r_mae(final.predict(X_ev), y_ev)


def _residualize(X_tr, X_ev_list, k):
    scaler = StandardScaler().fit(X_tr)
    Xs_tr = scaler.transform(X_tr)
    pca = PCA(n_components=k).fit(Xs_tr)
    Xs_tr_resid = Xs_tr - pca.inverse_transform(pca.transform(Xs_tr))
    out_evs = []
    for X_ev in X_ev_list:
        Xs_ev = scaler.transform(X_ev)
        out_evs.append(Xs_ev - pca.inverse_transform(pca.transform(Xs_ev)))
    return Xs_tr_resid, out_evs


def _inner_cv_pick(X_tr_layered, y_tr):
    n_layers = X_tr_layered.shape[0]
    n = len(y_tr)
    kf = KFold(n_splits=3, shuffle=True, random_state=SEED)
    rows = []
    for layer in range(n_layers):
        Xl = X_tr_layered[layer]
        for k in PC_KS:
            if k >= min(Xl.shape):
                continue
            cv_rs = []
            for tr_idx, va_idx in kf.split(np.arange(n)):
                X_tr_l = Xl[tr_idx]
                X_va_l = Xl[va_idx]
                X_tr_res, [X_va_res] = _residualize(X_tr_l, [X_va_l], k)
                r, _ = _ridge(X_tr_res, y_tr[tr_idx], X_va_res, y_tr[va_idx])
                cv_rs.append(r)
            rows.append({"layer": layer, "k_pc": k, "cv_R": float(np.mean(cv_rs))})
    grid = pd.DataFrame(rows)
    best = grid.loc[grid["cv_R"].idxmax()]
    return int(best["layer"]), int(best["k_pc"]), grid


def _process_fold(fold_id, gene_en_r, all_folds):
    f = next(x for x in all_folds if x["fold_id"] == fold_id)
    eval_cohort = f["holdout_cohort"]

    rows = []
    grids = []
    for seed, tag in SEED_TAGS:
        train_X_per_layer, train_y_all = [], []
        skip = False
        for tc in f["train_cohorts"]:
            ret = _load(tc, tag)
            if ret is None:
                print(f"[H.1] SKIP fold={fold_id} seed={seed}: missing {tc} {tag}", flush=True)
                skip = True
                break
            _, ages, emb_LDH = ret
            train_X_per_layer.append(emb_LDH)
            train_y_all.append(ages)
        if skip:
            continue
        train_X = np.concatenate(train_X_per_layer, axis=1)
        train_y = np.concatenate(train_y_all)

        eval_ret = _load(eval_cohort, tag)
        aida_ret = _load("aida", tag)
        if eval_ret is None or aida_ret is None:
            print(f"[H.1] SKIP fold={fold_id} seed={seed}: missing eval/aida", flush=True)
            continue
        _, eval_y, eval_X = eval_ret
        _, aida_y, aida_X = aida_ret
        n_layers = train_X.shape[0]

        print(f"\n=== H.1 fold={fold_id} seed={seed} | n_train={len(train_y)} ===", flush=True)
        cv_layer, cv_k, grid = _inner_cv_pick(train_X, train_y)
        grid["seed"] = seed
        grid["fold"] = fold_id
        grids.append(grid)
        cv_R_train = float(grid.loc[grid["cv_R"].idxmax(), "cv_R"])
        print(f"  CV-picked: L{cv_layer}, k={cv_k}, train CV R={cv_R_train:.4f}", flush=True)

        X_tr_l = train_X[cv_layer]
        X_ev_l = eval_X[cv_layer]
        X_aida_l = aida_X[cv_layer]
        X_tr_res, [X_ev_res, X_aida_res] = _residualize(X_tr_l, [X_ev_l, X_aida_l], cv_k)

        r_holdout, mae_holdout = _ridge(X_tr_res, train_y, X_ev_res, eval_y)
        r_aida, mae_aida = _ridge(X_tr_res, train_y, X_aida_res, aida_y)

        scaler = StandardScaler().fit(X_tr_l)
        r_full_holdout, mae_full_holdout = _ridge(scaler.transform(X_tr_l), train_y,
                                                   scaler.transform(X_ev_l), eval_y)
        r_full_aida, mae_full_aida = _ridge(scaler.transform(X_tr_l), train_y,
                                             scaler.transform(X_aida_l), aida_y)

        full_R_per_layer_holdout = []
        for L in range(n_layers):
            sc = StandardScaler().fit(train_X[L])
            r, _ = _ridge(sc.transform(train_X[L]), train_y, sc.transform(eval_X[L]), eval_y)
            full_R_per_layer_holdout.append(r)
        full_best_L = int(np.argmax(full_R_per_layer_holdout))
        full_best_R = float(np.max(full_R_per_layer_holdout))

        rows.append({
            "fold": fold_id, "seed": seed, "cv_layer": cv_layer, "cv_k_pc": cv_k,
            "train_cv_R": cv_R_train,
            "R_holdout_resid": r_holdout, "MAE_holdout_resid": mae_holdout,
            "R_holdout_full_at_cv_layer": r_full_holdout, "MAE_holdout_full_at_cv_layer": mae_full_holdout,
            "R_aida_resid": r_aida, "MAE_aida_resid": mae_aida,
            "R_aida_full_at_cv_layer": r_full_aida, "MAE_aida_full_at_cv_layer": mae_full_aida,
            "full_best_L_holdout": full_best_L, "full_best_R_holdout": full_best_R,
            "deltaR_vs_full_holdout": r_holdout - r_full_holdout,
            "deltaR_vs_geneEN_holdout": r_holdout - gene_en_r,
            "geneEN_R": gene_en_r,
        })
        print(f"  Eval @(L{cv_layer}, k={cv_k}): holdout R_resid={r_holdout:+.4f} (R_full={r_full_holdout:+.4f}, ΔR={r_holdout-r_full_holdout:+.4f}); MAE={mae_holdout:.2f}", flush=True)
        print(f"                            AIDA R_resid={r_aida:+.4f} (R_full={r_full_aida:+.4f}, ΔR={r_aida-r_full_aida:+.4f}); MAE={mae_aida:.2f}", flush=True)
        print(f"  Full-embed best (post-hoc): L{full_best_L} R={full_best_R:+.4f}", flush=True)
    return rows, grids


def main():
    folds = json.loads(Path("data/loco_folds.json").read_text())["folds"]

    all_rows = []
    all_grids = []
    fold_gene_en = {"loco_terekhova": GENE_EN_R_TEREKHOVA, "loco_onek1k": GENE_EN_R_ONEK1K}
    for fold_id in FOLD_IDS:
        rows, grids = _process_fold(fold_id, fold_gene_en[fold_id], folds)
        all_rows.extend(rows)
        all_grids.extend(grids)

    df = pd.DataFrame(all_rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, float_format="%.4f")
    if all_grids:
        pd.concat(all_grids, ignore_index=True).to_csv(OUT_GRID_CSV, index=False, float_format="%.4f")
    print(f"\n[H.1] wrote {len(df)} rows to {OUT_CSV}\n")
    print(df.to_string(index=False, float_format="%.3f"))

    # === Aggregation + decision rule (per fold) ===
    print("\n=== H.1 Multi-seed aggregation per fold ===")
    for fold_id in FOLD_IDS:
        sub = df[df["fold"] == fold_id]
        if len(sub) < 2:
            continue
        gene_en_r = fold_gene_en[fold_id]
        mean_R = sub["R_holdout_resid"].mean()
        std_R = sub["R_holdout_resid"].std()
        mean_R_aida = sub["R_aida_resid"].mean()
        std_R_aida = sub["R_aida_resid"].std()
        n_seeds = len(sub)
        print(f"\n  --- {fold_id} (n={n_seeds}) ---")
        print(f"  Holdout: mean R = {mean_R:+.4f} ± {std_R:.4f}")
        print(f"  AIDA:    mean R = {mean_R_aida:+.4f} ± {std_R_aida:.4f}")
        print(f"  vs gene-EN (R = {gene_en_r:.3f}): mean ΔR = {mean_R - gene_en_r:+.4f}")

        # Decision rule (only Terekhova is the load-bearing fold per G.2 spec)
        if fold_id == "loco_terekhova":
            print("  Decision (load-bearing fold):")
            if mean_R >= 0.27:
                if std_R <= 0.05:
                    print(f"    → MATCHES gene-EN (mean ≥ 0.27, σ ≤ 0.05). Multi-seed CONFIRMS cell-type-conditional extension.")
                else:
                    print(f"    → MATCHES on average but high seed variance (σ = {std_R:.3f}).")
            elif mean_R >= 0.17:
                print(f"    → NARROWS GAP (mean ∈ [0.17, 0.27)).")
            else:
                print(f"    → SINGLE-SEED LUCK (mean < 0.17). B-cell parity claim dropped.")


if __name__ == "__main__":
    main()
