"""F.4 — CCA upper bound on per-layer linear age info.

For each multi-seed condition × layer:
  CCA-best-direction R: closed form for 1-D target.
    With X (n × p) and y (n × 1), the maximum train correlation between Xw
    and y over unit-norm w ∈ R^p is the multiple correlation coefficient.
    For p > n it overfits to 1.0; for p < n it's the OLS R.
  OLS unregularized R on holdout: when n > p, well-defined.
  Ridge-CV R (existing protocol).

Compare CCA-best-layer + OLS-best-layer + Ridge-best-layer per condition.

Decision rule (pre-commit):
  CCA-best-layer matches Ridge-best in >=75% conditions → ridge near-maximal.
  25-50% disagreement → moderate regularization shaping.
  >50% disagreement → ridge regularization shapes layer ordering.

Output: results/phase3/f4_cca_upper_bound.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV, LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr


EMB_DIR = Path("results/phase3/embeddings_layered")
OUT_CSV = Path("results/phase3/f4_cca_upper_bound.csv")
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


def _r(pred, y):
    if np.std(pred) > 0 and np.std(y) > 0 and len(y) > 1:
        r, _ = pearsonr(pred, y)
        return float(r)
    return 0.0


def _cca_train_R(X_train, y_train):
    """Maximum train correlation between X_train w and y_train, closed-form for 1-D target.

    Returns the multiple correlation coefficient. When p > n, this is 1.0 (overfit);
    when p < n, it's the OLS-on-train R.
    """
    n, p = X_train.shape
    if p >= n:
        # Use Moore-Penrose pseudoinverse OLS — train R will be 1.0 by construction
        # for p >= n. Report 1.0 as the upper bound.
        return 1.0
    Xc = X_train - X_train.mean(0, keepdims=True)
    yc = y_train - y_train.mean()
    beta = np.linalg.lstsq(Xc, yc, rcond=None)[0]
    pred = Xc @ beta
    return _r(pred, yc)


def _ols_holdout_R(X_train, y_train, X_eval, y_eval):
    """OLS unregularized on train, predict on eval. Returns holdout R or None if p>=n."""
    n, p = X_train.shape
    if p >= n:
        return None  # not well-defined
    final = LinearRegression().fit(X_train, y_train)
    return _r(final.predict(X_eval), y_eval)


def _ridge_holdout_R(X_train, y_train, X_eval, y_eval):
    cv = RidgeCV(alphas=ALPHAS, cv=3, scoring="neg_mean_absolute_error")
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(y_train))
    cv.fit(X_train[perm], y_train[perm])
    final = Ridge(alpha=float(cv.alpha_)).fit(X_train, y_train)
    return _r(final.predict(X_eval), y_eval)


# Multi-seed conditions
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

        print(f"\n=== {method_label} | {fold_id} × {cell_type} × seed{seed} (n_train={len(train_y)}) ===", flush=True)

        # Standardize per layer (fit on train)
        for layer in range(n_layers):
            scaler = StandardScaler().fit(train_X_layered[layer])
            X_tr = scaler.transform(train_X_layered[layer])
            X_ev = scaler.transform(eval_X_layered[layer])

            cca_R_train = _cca_train_R(X_tr, train_y)
            ols_R_holdout = _ols_holdout_R(X_tr, train_y, X_ev, eval_y)
            ridge_R_holdout = _ridge_holdout_R(X_tr, train_y, X_ev, eval_y)

            row = {
                "method": method_label, "fold": fold_id, "cell_type": cell_type,
                "seed": seed, "layer": layer,
                "cca_train_R": cca_R_train,
                "ols_holdout_R": ols_R_holdout,
                "ridge_holdout_R": ridge_R_holdout,
            }
            if aida_ret is not None:
                _, aida_y, aida_X = aida_ret
                aida_X_l = scaler.transform(aida_X[layer])
                row["ols_aida_R"] = _ols_holdout_R(X_tr, train_y, aida_X_l, aida_y)
                row["ridge_aida_R"] = _ridge_holdout_R(X_tr, train_y, aida_X_l, aida_y)
            rows.append(row)

        # Print per-layer
        sub_rows = [r for r in rows if r["method"] == method_label and r["fold"] == fold_id and r["cell_type"] == cell_type and r["seed"] == seed]
        for r in sub_rows:
            ols_str = f"{r['ols_holdout_R']:+.3f}" if r["ols_holdout_R"] is not None else "  n/a"
            print(f"  L{r['layer']:2d}: CCA-train={r['cca_train_R']:.3f}  OLS-holdout={ols_str}  Ridge-holdout={r['ridge_holdout_R']:+.3f}")

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, float_format="%.4f")
    print(f"\n[F.4] wrote {len(df)} rows to {OUT_CSV}")

    # Decision: how often does CCA-best-layer agree with Ridge-best-layer?
    n_agree = 0
    n_total = 0
    for (m, fold, ct, seed), grp in df.groupby(["method", "fold", "cell_type", "seed"]):
        ridge_best = int(grp.loc[grp["ridge_holdout_R"].idxmax(), "layer"])
        # For CCA, when n > p the cca_train_R is OLS train R; pick by argmax
        cca_best = int(grp.loc[grp["cca_train_R"].idxmax(), "layer"])
        if cca_best == ridge_best:
            n_agree += 1
        n_total += 1
        print(f"  {m:25s} {fold:14s} {ct:6s} s{seed}: Ridge L{ridge_best}, CCA L{cca_best}, match={cca_best == ridge_best}")

    pct = n_agree / n_total if n_total else 0
    print(f"\n=== Decision (CCA vs Ridge layer agreement) ===")
    print(f"  Agreement: {n_agree}/{n_total} = {100*pct:.1f}%")
    if pct >= 0.75:
        print("  → DECISION: ridge near-maximal; methodology robust to regularization.")
    elif pct >= 0.50:
        print("  → DECISION: moderate regularization shaping; report both.")
    else:
        print("  → DECISION: ridge regularization substantially shapes layer ordering.")


if __name__ == "__main__":
    main()
