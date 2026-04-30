"""G.2 — PC-residualized FM probe on B × loco_terekhova.

Tests F.5 reviewer's hypothesis: does PC-residualized ridge close the gap
between FM and gene-EN on B-cells, where gene-EN got R=0.321 (D.23) and
FM-frozen ridge missed it?

Two selection regimes for (layer × k_pc):
  (A) Inner-CV on train (3-fold by donor): the honest, reportable headline.
  (B) Holdout-peek (F.5-best on Terekhova holdout R_residual): post-hoc upper bound.

Comparators:
  - Full-embedding ridge at the F.4/F.5 R_full-best layer (FM-frozen baseline)
  - gene-EN × B × loco_terekhova → Terekhova R = 0.321 (from gene_en_matched_splits.csv)

Decision rule (pre-commit, applied to honest CV-picked R on Terekhova holdout):
  R ≥ 0.27 (gene-EN R − 0.05) → matches gene-EN; methodology contribution extends to multi-cell-type.
  0.17 ≤ R < 0.27 → narrows gap; "PC-residual narrows the gap" not "matches."
  R < 0.17 → still a gene-EN win; gene-EN-on-B remains a methodology limitation.

Output: results/phase3/g2_pc_residual_b_cell.csv
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
OUT_CSV = Path("results/phase3/g2_pc_residual_b_cell.csv")
ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
PC_KS = [5, 10, 25, 50]
SEED = 0
GENE_EN_R = 0.3210  # D.23 / gene_en_matched_splits.csv: B × loco_terekhova → Terekhova
CELL_TYPE = "B"
CELL_TYPE_SLUG = "B"
FOLD_ID = "loco_terekhova"
TAG = "frozen_base_alllayers"


def _load(cohort, tag):
    p = EMB_DIR / f"{cohort}_{CELL_TYPE_SLUG}_{tag}.npz"
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
    pred = final.predict(X_ev)
    return _r_mae(pred, y_ev)


def _residualize_ev(X_tr, X_ev_list, k):
    """Fit StandardScaler + PCA on X_tr, project out top-k PCs from X_tr and each X_ev."""
    scaler = StandardScaler().fit(X_tr)
    Xs_tr = scaler.transform(X_tr)
    pca = PCA(n_components=k).fit(Xs_tr)
    Xs_tr_resid = Xs_tr - pca.inverse_transform(pca.transform(Xs_tr))
    out_evs = []
    for X_ev in X_ev_list:
        Xs_ev = scaler.transform(X_ev)
        Xs_ev_resid = Xs_ev - pca.inverse_transform(pca.transform(Xs_ev))
        out_evs.append(Xs_ev_resid)
    return Xs_tr_resid, out_evs


def _inner_cv_pick_best(X_tr_layered, y_tr, k_pcs):
    """3-fold CV by donor on train. For each (layer, k_pc), compute mean CV R.
    Returns (best_layer, best_k_pc, cv_R_grid as DataFrame).
    """
    n_layers = X_tr_layered.shape[0]
    n = len(y_tr)
    kf = KFold(n_splits=3, shuffle=True, random_state=SEED)
    rows = []
    for layer in range(n_layers):
        Xl = X_tr_layered[layer]
        for k in k_pcs:
            if k >= min(Xl.shape):
                continue
            cv_rs = []
            for tr_idx, va_idx in kf.split(np.arange(n)):
                X_tr_l = Xl[tr_idx]
                X_va_l = Xl[va_idx]
                # PC-residualize using only inner-train
                X_tr_res, [X_va_res] = _residualize_ev(X_tr_l, [X_va_l], k)
                r, _ = _ridge(X_tr_res, y_tr[tr_idx], X_va_res, y_tr[va_idx])
                cv_rs.append(r)
            cv_R = float(np.mean(cv_rs))
            rows.append({"layer": layer, "k_pc": k, "cv_R": cv_R})
    grid = pd.DataFrame(rows)
    best = grid.loc[grid["cv_R"].idxmax()]
    return int(best["layer"]), int(best["k_pc"]), grid


def main():
    folds = json.loads(Path("data/loco_folds.json").read_text())["folds"]
    f = next(x for x in folds if x["fold_id"] == FOLD_ID)
    eval_cohort = f["holdout_cohort"]

    # Load train + eval + AIDA layered embeddings (frozen base, single seed)
    train_X_per_layer, train_y_all, train_donors_all = [], [], []
    for tc in f["train_cohorts"]:
        d_ids, ages, emb_LDH = _load(tc, TAG)
        train_X_per_layer.append(emb_LDH)
        train_y_all.append(ages)
        train_donors_all.append(d_ids)
    train_X = np.concatenate(train_X_per_layer, axis=1)  # shape (L, D_total, H)
    train_y = np.concatenate(train_y_all)
    train_donors = np.concatenate(train_donors_all)
    print(f"[G.2] train: {train_X.shape} ({len(train_y)} donors), folds {f['train_cohorts']}", flush=True)

    _, eval_y, eval_X = _load(eval_cohort, TAG)
    print(f"[G.2] eval ({eval_cohort}): {eval_X.shape} ({len(eval_y)} donors)", flush=True)
    _, aida_y, aida_X = _load("aida", TAG)
    print(f"[G.2] aida: {aida_X.shape} ({len(aida_y)} donors)", flush=True)

    n_layers = train_X.shape[0]

    # ===== A. Honest inner-CV pick =====
    print("\n[G.2-A] Inner-CV picking best (layer × k_pc) on train...", flush=True)
    cv_layer, cv_k, cv_grid = _inner_cv_pick_best(train_X, train_y, PC_KS)
    print(f"  CV-picked: layer={cv_layer}, k_pc={cv_k}, train CV R={float(cv_grid.loc[cv_grid['cv_R'].idxmax(),'cv_R']):.4f}", flush=True)

    # ===== B. F.5-holdout-peek best (post-hoc upper bound) =====
    f5 = pd.read_csv("results/phase3/f5_pc_residual.csv")
    f5_b = f5[(f5["cell_type"] == "B") & (f5["fold"] == FOLD_ID)
              & (f5["method"] == "geneformer_frozen_seed0") & (f5["seed"] == 0)]
    f5_best = f5_b.loc[f5_b["R_residual"].idxmax()]
    holdout_layer = int(f5_best["layer"])
    holdout_k = int(f5_best["k_pc"])
    print(f"  F.5-holdout-best: layer={holdout_layer}, k_pc={holdout_k}, R_residual={float(f5_best['R_residual']):.4f} (post-hoc, leaky)", flush=True)

    # ===== Full-embed comparators =====
    full_R_per_layer_holdout = []
    full_R_per_layer_aida = []
    for layer in range(n_layers):
        X_tr_l = train_X[layer]
        scaler = StandardScaler().fit(X_tr_l)
        Xs_tr = scaler.transform(X_tr_l)
        Xs_ev = scaler.transform(eval_X[layer])
        Xs_aida = scaler.transform(aida_X[layer])
        r_h, _ = _ridge(Xs_tr, train_y, Xs_ev, eval_y)
        r_a, _ = _ridge(Xs_tr, train_y, Xs_aida, aida_y)
        full_R_per_layer_holdout.append(r_h)
        full_R_per_layer_aida.append(r_a)
    full_best_layer_holdout = int(np.argmax(full_R_per_layer_holdout))
    full_best_R_holdout = float(np.max(full_R_per_layer_holdout))
    print(f"  full-embed best layer (holdout, post-hoc): L{full_best_layer_holdout}, R={full_best_R_holdout:.4f}", flush=True)

    # ===== Final eval at each picked (layer × k_pc) =====
    rows = []
    for label, layer, k in [("CV-picked", cv_layer, cv_k),
                             ("F.5-holdout-best", holdout_layer, holdout_k)]:
        X_tr_l = train_X[layer]
        X_ev_l = eval_X[layer]
        X_aida_l = aida_X[layer]

        # PC-residual ridge
        X_tr_res, [X_ev_res, X_aida_res] = _residualize_ev(X_tr_l, [X_ev_l, X_aida_l], k)
        r_holdout, mae_holdout = _ridge(X_tr_res, train_y, X_ev_res, eval_y)
        r_aida, mae_aida = _ridge(X_tr_res, train_y, X_aida_res, aida_y)

        # Full-embed ridge at same layer (no PC-resid)
        scaler = StandardScaler().fit(X_tr_l)
        r_full_holdout, mae_full_holdout = _ridge(scaler.transform(X_tr_l), train_y,
                                                   scaler.transform(X_ev_l), eval_y)
        r_full_aida, mae_full_aida = _ridge(scaler.transform(X_tr_l), train_y,
                                             scaler.transform(X_aida_l), aida_y)

        rows.append({
            "regime": label, "layer": layer, "k_pc": k,
            "R_holdout_resid": r_holdout, "MAE_holdout_resid": mae_holdout,
            "R_holdout_full": r_full_holdout, "MAE_holdout_full": mae_full_holdout,
            "R_aida_resid": r_aida, "MAE_aida_resid": mae_aida,
            "R_aida_full": r_full_aida, "MAE_aida_full": mae_full_aida,
            "deltaR_vs_full_holdout": r_holdout - r_full_holdout,
            "deltaR_vs_geneEN_holdout": r_holdout - GENE_EN_R,
        })
        print(f"\n  [{label}] L{layer} × k={k}:", flush=True)
        print(f"    Holdout: R_resid={r_holdout:+.4f} (R_full={r_full_holdout:+.4f}, ΔR={r_holdout - r_full_holdout:+.4f}); MAE_resid={mae_holdout:.2f}", flush=True)
        print(f"    AIDA:    R_resid={r_aida:+.4f} (R_full={r_full_aida:+.4f}, ΔR={r_aida - r_full_aida:+.4f}); MAE_resid={mae_aida:.2f}", flush=True)

    # Also include "full-embed best layer" baseline as a comparator row
    layer = full_best_layer_holdout
    X_tr_l = train_X[layer]
    scaler = StandardScaler().fit(X_tr_l)
    r_full_holdout, mae_full_holdout = _ridge(scaler.transform(X_tr_l), train_y,
                                               scaler.transform(eval_X[layer]), eval_y)
    r_full_aida, mae_full_aida = _ridge(scaler.transform(X_tr_l), train_y,
                                         scaler.transform(aida_X[layer]), aida_y)
    rows.append({
        "regime": "full-embed-best (post-hoc)", "layer": layer, "k_pc": -1,
        "R_holdout_resid": np.nan, "MAE_holdout_resid": np.nan,
        "R_holdout_full": r_full_holdout, "MAE_holdout_full": mae_full_holdout,
        "R_aida_resid": np.nan, "MAE_aida_resid": np.nan,
        "R_aida_full": r_full_aida, "MAE_aida_full": mae_full_aida,
        "deltaR_vs_full_holdout": np.nan, "deltaR_vs_geneEN_holdout": r_full_holdout - GENE_EN_R,
    })

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, float_format="%.4f")
    cv_grid.to_csv(OUT_CSV.with_name("g2_pc_residual_b_cell_cv_grid.csv"), index=False, float_format="%.4f")
    print(f"\n[G.2] wrote {len(df)} rows to {OUT_CSV}\n")
    print(df.to_string(index=False, float_format="%.3f"))

    # Decision rule
    print("\n=== G.2 Decision rule (pre-commit, on CV-picked B × Terekhova holdout R) ===")
    cv_row = df[df["regime"] == "CV-picked"].iloc[0]
    R_cv = float(cv_row["R_holdout_resid"])
    print(f"  CV-picked R = {R_cv:+.4f} vs gene-EN R = {GENE_EN_R:.4f}")
    if R_cv >= GENE_EN_R - 0.05:
        print(f"  → DECISION: MATCHES gene-EN (R ≥ {GENE_EN_R - 0.05:.2f}). Methodology contribution extends to B-cells.")
    elif R_cv >= GENE_EN_R - 0.15:
        print(f"  → DECISION: NARROWS gap ({GENE_EN_R - 0.15:.2f} ≤ R < {GENE_EN_R - 0.05:.2f}). PC-residual narrows but not matches.")
    else:
        print(f"  → DECISION: STILL gene-EN win (R < {GENE_EN_R - 0.15:.2f}). B-cell remains a gene-EN-only success.")
    print(f"\n  Honest-CV note: F.5-holdout-best R = {df[df['regime']=='F.5-holdout-best'].iloc[0]['R_holdout_resid']:+.4f} (upper bound; leaky).")


if __name__ == "__main__":
    main()
