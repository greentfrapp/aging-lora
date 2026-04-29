"""D.23 — Extend gene-EN matched-splits CSV with missing NK/B conditions.

The original D.17 run covered most conditions but is missing:
  - NK × loco_terekhova → terekhova
  - NK × loco_terekhova → aida
  - B × loco_terekhova → aida

This script reuses the same preprocessing + hyperparameter grid as
gene_en_matched_splits.py and appends new rows to the existing CSV
(`results/baselines/gene_en_matched_splits.csv`).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import ElasticNetCV, ElasticNet
from sklearn.preprocessing import StandardScaler

# Reuse helpers from the main script
import sys
sys.path.insert(0, str(Path(__file__).parent))
from gene_en_matched_splits import (
    _build_donor_matrix, _align_columns, _bootstrap_pearson_ci,
    ALPHAS, L1_RATIOS, TOP_N_HVG, SEED,
    MAX_CELLS_PER_DONOR_TRAIN, MAX_CELLS_PER_DONOR_EVAL, OUT_CSV,
)


def main():
    folds = json.loads(Path("data/loco_folds.json").read_text())["folds"]
    fmap = {f["fold_id"]: f for f in folds}

    # Missing conditions:
    runs = [
        # (fold_id, cell_type, also_aida)
        ("loco_terekhova", "NK", True),  # adds NK × terekhova + NK × aida
        ("loco_terekhova", "B",  True),  # adds B × aida (B × terekhova already exists)
    ]

    new_rows = []
    for fold_id, cell_type, also_aida in runs:
        f = fmap[fold_id]
        eval_cohort = f["holdout_cohort"]
        print(f"\n=== {fold_id} × {cell_type} (eval={eval_cohort}) ===", flush=True)
        t0 = time.time()

        train_X_list, train_y_list, train_syms_ref = [], [], None
        for tc in f["train_cohorts"]:
            _, X_tc, y_tc, syms_tc = _build_donor_matrix(tc, cell_type, MAX_CELLS_PER_DONOR_TRAIN)
            if train_syms_ref is None:
                train_syms_ref = syms_tc
                train_X_list.append(X_tc)
            else:
                X_aligned = _align_columns(train_syms_ref, syms_tc, X_tc)
                train_X_list.append(X_aligned)
            train_y_list.append(y_tc)
        train_X = np.concatenate(train_X_list, axis=0)
        train_y = np.concatenate(train_y_list)

        eval_donors, eval_X_raw, eval_y, eval_syms = _build_donor_matrix(eval_cohort, cell_type, MAX_CELLS_PER_DONOR_EVAL)
        eval_X = _align_columns(train_syms_ref, eval_syms, eval_X_raw)

        train_var = train_X.var(axis=0)
        top_idx = np.argsort(-train_var)[:TOP_N_HVG]
        train_X = train_X[:, top_idx]
        eval_X = eval_X[:, top_idx]
        train_syms_ref = train_syms_ref[top_idx]
        scaler = StandardScaler().fit(train_X)
        train_X_s = scaler.transform(train_X).astype(np.float32)
        eval_X_s = scaler.transform(eval_X).astype(np.float32)

        print(f"  train: {train_X_s.shape}, eval: {eval_X_s.shape}")
        rng = np.random.default_rng(SEED)
        perm = rng.permutation(len(train_y))
        cv = ElasticNetCV(
            l1_ratio=L1_RATIOS, alphas=ALPHAS, cv=3, max_iter=5000,
            n_jobs=-1, random_state=SEED, selection="cyclic",
        )
        cv.fit(train_X_s[perm], train_y[perm])
        alpha = float(cv.alpha_)
        l1_ratio = float(cv.l1_ratio_)
        final = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000).fit(train_X_s, train_y)
        n_nonzero = int(np.sum(final.coef_ != 0))

        pred = final.predict(eval_X_s)
        r, p_val = pearsonr(pred, eval_y)
        mae = float(np.median(np.abs(pred - eval_y)))
        ci_lo, ci_hi = _bootstrap_pearson_ci(pred, eval_y, seed=SEED)
        elapsed = time.time() - t0
        print(f"  HOLDOUT  R={r:+.3f} ({ci_lo:+.3f}, {ci_hi:+.3f})  MAE={mae:.2f}  alpha={alpha:.4g} l1_ratio={l1_ratio} n_nonzero={n_nonzero}  ({elapsed:.0f}s)", flush=True)
        new_rows.append({
            "fold": fold_id, "eval_cohort": eval_cohort, "cell_type": cell_type,
            "alpha": alpha, "l1_ratio": l1_ratio, "n_nonzero_genes": n_nonzero,
            "n_train_donors": int(len(train_y)), "n_eval_donors": int(len(eval_y)),
            "pearson_r": float(r), "pearson_p": float(p_val), "mae_y": mae,
            "pearson_ci_lo": ci_lo, "pearson_ci_hi": ci_hi,
            "pred_mean": float(pred.mean()), "eval_mean": float(eval_y.mean()),
        })

        if also_aida:
            _, aida_X_raw, aida_y, aida_syms = _build_donor_matrix("aida", cell_type, MAX_CELLS_PER_DONOR_EVAL)
            aida_X = _align_columns(train_syms_ref, aida_syms, aida_X_raw)
            aida_X_s = scaler.transform(aida_X).astype(np.float32)
            apred = final.predict(aida_X_s)
            ar, ap = pearsonr(apred, aida_y)
            amae = float(np.median(np.abs(apred - aida_y)))
            aci_lo, aci_hi = _bootstrap_pearson_ci(apred, aida_y, seed=SEED)
            print(f"  AIDA     R={ar:+.3f} ({aci_lo:+.3f}, {aci_hi:+.3f})  MAE={amae:.2f}", flush=True)
            new_rows.append({
                "fold": fold_id, "eval_cohort": "aida", "cell_type": cell_type,
                "alpha": alpha, "l1_ratio": l1_ratio, "n_nonzero_genes": n_nonzero,
                "n_train_donors": int(len(train_y)), "n_eval_donors": int(len(aida_y)),
                "pearson_r": float(ar), "pearson_p": float(ap), "mae_y": amae,
                "pearson_ci_lo": aci_lo, "pearson_ci_hi": aci_hi,
                "pred_mean": float(apred.mean()), "eval_mean": float(aida_y.mean()),
            })

    # Append to existing CSV
    existing = pd.read_csv(OUT_CSV)
    df_new = pd.DataFrame(new_rows)
    df = pd.concat([existing, df_new], ignore_index=True)
    df.to_csv(OUT_CSV, index=False, float_format="%.6g")
    print(f"\n[gene-EN-extra] appended {len(df_new)} rows; total {len(df)} rows in {OUT_CSV}")
    print()
    print(df_new[["fold", "eval_cohort", "cell_type", "n_train_donors", "n_eval_donors",
                  "pearson_r", "pearson_ci_lo", "pearson_ci_hi", "mae_y", "n_nonzero_genes"]].to_string(index=False))


if __name__ == "__main__":
    main()
