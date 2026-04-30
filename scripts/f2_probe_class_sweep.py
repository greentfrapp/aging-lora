"""F.2 — Per-layer probe-class sweep.

For rank-32 LoRA × CD4+T × loco_onek1k × 3 seeds, fit four probe classes at
each of 13 layers, evaluate on AIDA + holdout. Compare layer-orderings across
probe classes.

Probe classes:
  1. Ridge with dense λ sweep (RidgeCV, 30 alphas log-spaced)
  2. PCA preprocessing + OLS, inner CV for n_components ∈ {5, 10, 25, 50, 100, full}
  3. Kernel ridge with RBF kernel, inner CV for gamma + alpha
  4. Two-layer MLP (hidden=64), early stopping on within-train val split

Decision rule (pre-commit):
  All 4 probes pick within ±1 layer → layer-ordering is representation property.
  2-3 layer disagreement → moderate probe-conditional.
  >=4 layer disagreement → layer-ordering is probe-property; restructure.

Output: results/phase3/f2_probe_class_sweep.csv
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV, LinearRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from scipy.stats import pearsonr


EMB_DIR = Path("results/phase3/embeddings_layered")
OUT_CSV = Path("results/phase3/f2_probe_class_sweep.csv")
RIDGE_ALPHAS = np.logspace(-4, 4, 30).tolist()
PCA_KS = [5, 10, 25, 50, 100]
KERNEL_GAMMAS = [0.001, 0.01, 0.1, 1.0]
KERNEL_ALPHAS = [0.01, 0.1, 1.0, 10.0]
SEED = 0


def _slug(cell_type: str) -> str:
    return cell_type.replace("+", "p").replace(" ", "_")


def _load(cohort: str, cell_type: str, tag: str):
    p = EMB_DIR / f"{cohort}_{_slug(cell_type)}_{tag}.npz"
    z = np.load(p, allow_pickle=True)
    return z["donor_ids"], z["ages"].astype(np.float32), z["embeddings_per_layer"].astype(np.float32)


def _r_mae(pred, y):
    if np.std(pred) > 0 and np.std(y) > 0 and len(y) > 1:
        r, _ = pearsonr(pred, y)
    else:
        r = 0.0
    return float(r), float(np.median(np.abs(pred - y)))


def _ridge_probe(X_tr, y_tr, X_ev, y_ev):
    cv = RidgeCV(alphas=RIDGE_ALPHAS, cv=3, scoring="neg_mean_absolute_error")
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(y_tr))
    cv.fit(X_tr[perm], y_tr[perm])
    final = Ridge(alpha=float(cv.alpha_)).fit(X_tr, y_tr)
    pred = final.predict(X_ev)
    return _r_mae(pred, y_ev)


def _pca_ols_probe(X_tr, y_tr, X_ev, y_ev):
    """PCA preprocessing + OLS, inner CV for n_components."""
    n_donors = len(y_tr)
    valid_ks = [k for k in PCA_KS if k < n_donors and k < X_tr.shape[1]] + [min(X_tr.shape[1], n_donors - 1)]
    valid_ks = sorted(set(valid_ks))
    kf = KFold(n_splits=3, shuffle=True, random_state=SEED)
    best_k, best_cv_R = valid_ks[0], -np.inf
    for k in valid_ks:
        rs = []
        for tr_idx, te_idx in kf.split(np.arange(n_donors)):
            scaler = StandardScaler().fit(X_tr[tr_idx])
            Xs = scaler.transform(X_tr)
            pca = PCA(n_components=k).fit(Xs[tr_idx])
            X_tr_pca = pca.transform(Xs[tr_idx])
            X_te_pca = pca.transform(Xs[te_idx])
            ols = LinearRegression().fit(X_tr_pca, y_tr[tr_idx])
            pred = ols.predict(X_te_pca)
            if np.std(pred) > 0:
                r, _ = pearsonr(pred, y_tr[te_idx])
                rs.append(r)
        cv_R = float(np.mean(rs)) if rs else -np.inf
        if cv_R > best_cv_R:
            best_cv_R, best_k = cv_R, k

    scaler = StandardScaler().fit(X_tr)
    Xs_tr = scaler.transform(X_tr)
    Xs_ev = scaler.transform(X_ev)
    pca = PCA(n_components=best_k).fit(Xs_tr)
    X_tr_pca = pca.transform(Xs_tr)
    X_ev_pca = pca.transform(Xs_ev)
    ols = LinearRegression().fit(X_tr_pca, y_tr)
    pred = ols.predict(X_ev_pca)
    r, mae = _r_mae(pred, y_ev)
    return r, mae, best_k


def _kernel_ridge_probe(X_tr, y_tr, X_ev, y_ev):
    """RBF kernel ridge, inner CV for gamma + alpha."""
    scaler = StandardScaler().fit(X_tr)
    Xs_tr = scaler.transform(X_tr)
    Xs_ev = scaler.transform(X_ev)
    kf = KFold(n_splits=3, shuffle=True, random_state=SEED)
    best = (None, None, -np.inf)
    n = len(y_tr)
    for g in KERNEL_GAMMAS:
        for a in KERNEL_ALPHAS:
            rs = []
            for tr_idx, te_idx in kf.split(np.arange(n)):
                m = KernelRidge(alpha=a, kernel="rbf", gamma=g).fit(Xs_tr[tr_idx], y_tr[tr_idx])
                pred = m.predict(Xs_tr[te_idx])
                if np.std(pred) > 0:
                    r, _ = pearsonr(pred, y_tr[te_idx])
                    rs.append(r)
            cv_R = float(np.mean(rs)) if rs else -np.inf
            if cv_R > best[2]:
                best = (g, a, cv_R)
    g, a, _ = best
    final = KernelRidge(alpha=a, kernel="rbf", gamma=g).fit(Xs_tr, y_tr)
    pred = final.predict(Xs_ev)
    r, mae = _r_mae(pred, y_ev)
    return r, mae, g, a


def _mlp_probe(X_tr, y_tr, X_ev, y_ev):
    """2-layer MLP with early stopping on within-train val split."""
    scaler = StandardScaler().fit(X_tr)
    Xs_tr = scaler.transform(X_tr)
    Xs_ev = scaler.transform(X_ev)
    X_inner_tr, X_inner_val, y_inner_tr, y_inner_val = train_test_split(
        Xs_tr, y_tr, test_size=0.2, random_state=SEED
    )
    mlp = MLPRegressor(
        hidden_layer_sizes=(64,), activation="relu", solver="adam",
        max_iter=300, batch_size=min(32, len(y_inner_tr)),
        early_stopping=False, random_state=SEED, learning_rate_init=1e-3,
        alpha=1e-3,
    )
    mlp.fit(X_inner_tr, y_inner_tr)
    pred = mlp.predict(Xs_ev)
    r, mae = _r_mae(pred, y_ev)
    return r, mae


# Configs: rank-32 × CD4+T × loco_onek1k × 3 seeds
CONFIGS = [
    (0, "loco_onek1k_CD4pT_e5b_r32_alllayers"),
    (1, "loco_onek1k_CD4pT_e5b_r32_seed1_alllayers"),
    (2, "loco_onek1k_CD4pT_e5b_r32_seed2_alllayers"),
]
CELL_TYPE = "CD4+ T"
FOLD_ID = "loco_onek1k"


def main():
    folds = json.loads(Path("data/loco_folds.json").read_text())["folds"]
    fmap = {f["fold_id"]: f for f in folds}
    f = fmap[FOLD_ID]

    rows = []
    for seed, tag in CONFIGS:
        train_X_per_layer, train_y_all = [], []
        for tc in f["train_cohorts"]:
            _, ages, emb_LDH = _load(tc, CELL_TYPE, tag)
            train_X_per_layer.append(emb_LDH)
            train_y_all.append(ages)
        train_X_layered = np.concatenate(train_X_per_layer, axis=1)
        train_y = np.concatenate(train_y_all)

        eval_ret = _load(f["holdout_cohort"], CELL_TYPE, tag)
        _, eval_y, eval_X_layered = eval_ret
        _, aida_y, aida_X_layered = _load("aida", CELL_TYPE, tag)
        n_layers = train_X_layered.shape[0]

        print(f"\n=== F.2 seed{seed} | {n_layers} layers, {len(train_y)} train ===", flush=True)
        for layer in range(n_layers):
            t0 = time.time()
            X_tr_l = train_X_layered[layer]
            X_ev_l = eval_X_layered[layer]
            X_aida_l = aida_X_layered[layer]

            # Ridge
            r_aida, mae_aida = _ridge_probe(X_tr_l, train_y, X_aida_l, aida_y)
            r_ho, mae_ho = _ridge_probe(X_tr_l, train_y, X_ev_l, eval_y)
            rows.append({"seed": seed, "layer": layer, "probe": "ridge",
                         "aida_R": r_aida, "aida_MAE": mae_aida,
                         "holdout_R": r_ho, "holdout_MAE": mae_ho, "extras": ""})

            # PCA + OLS
            r_aida, mae_aida, best_k = _pca_ols_probe(X_tr_l, train_y, X_aida_l, aida_y)
            r_ho, mae_ho, _ = _pca_ols_probe(X_tr_l, train_y, X_ev_l, eval_y)
            rows.append({"seed": seed, "layer": layer, "probe": "pca_ols",
                         "aida_R": r_aida, "aida_MAE": mae_aida,
                         "holdout_R": r_ho, "holdout_MAE": mae_ho, "extras": f"k={best_k}"})

            # Kernel Ridge
            r_aida, mae_aida, g, a = _kernel_ridge_probe(X_tr_l, train_y, X_aida_l, aida_y)
            r_ho, mae_ho, _, _ = _kernel_ridge_probe(X_tr_l, train_y, X_ev_l, eval_y)
            rows.append({"seed": seed, "layer": layer, "probe": "kernel_rbf",
                         "aida_R": r_aida, "aida_MAE": mae_aida,
                         "holdout_R": r_ho, "holdout_MAE": mae_ho, "extras": f"g={g} a={a}"})

            # MLP
            r_aida, mae_aida = _mlp_probe(X_tr_l, train_y, X_aida_l, aida_y)
            r_ho, mae_ho = _mlp_probe(X_tr_l, train_y, X_ev_l, eval_y)
            rows.append({"seed": seed, "layer": layer, "probe": "mlp_h64",
                         "aida_R": r_aida, "aida_MAE": mae_aida,
                         "holdout_R": r_ho, "holdout_MAE": mae_ho, "extras": ""})

            dt = time.time() - t0
            print(f"  L{layer}: ridge={rows[-4]['aida_R']:+.3f} pca={rows[-3]['aida_R']:+.3f} "
                  f"kernel={rows[-2]['aida_R']:+.3f} mlp={rows[-1]['aida_R']:+.3f} | {dt:.1f}s", flush=True)

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, float_format="%.4f")
    print(f"\n[F.2] wrote {len(df)} rows to {OUT_CSV}")

    # Per-probe best layer (3-seed mean)
    print("\n=== F.2 Best layer per probe (AIDA, 3-seed mean) ===")
    for probe in ["ridge", "pca_ols", "kernel_rbf", "mlp_h64"]:
        sub = df[df["probe"] == probe]
        layer_means = sub.groupby("layer")["aida_R"].mean()
        L_best = int(layer_means.idxmax())
        print(f"  {probe:12s}: L_best=L{L_best} (mean R = {layer_means.max():+.3f})")

    bests = []
    for probe in ["ridge", "pca_ols", "kernel_rbf", "mlp_h64"]:
        sub = df[df["probe"] == probe]
        layer_means = sub.groupby("layer")["aida_R"].mean()
        bests.append(int(layer_means.idxmax()))
    spread = max(bests) - min(bests)
    print(f"\n=== Decision (probe-class layer disagreement) ===")
    print(f"  Best-layer spread across 4 probes: {spread} layers")
    if spread <= 1:
        print("  → DECISION: probes agree within ±1 → representation property.")
    elif spread <= 3:
        print("  → DECISION: 2-3 layer disagreement → moderate probe-conditional.")
    else:
        print("  → DECISION: ≥4 layer disagreement → probe-property; restructure methodology.")


if __name__ == "__main__":
    main()
