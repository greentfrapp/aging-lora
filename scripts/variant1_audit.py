"""D.9 + D.10 audit of Variant 1 frozen-base ridge fits.

Reads `.npz` per-donor embeddings, re-fits ridge with the same RidgeCV alpha
selection used in `donor_ridge.py`, and computes bias-variance diagnostics
flagged in `scratchpad/variant_1_review.md`:

- Exact R, 95% bootstrap CI (n=1000 paired-resamples), Pearson p
- pred_sd vs eval_sd (compression test)
- pred range vs eval range
- OLS slope of pred ~ true (slope < 1 = mean-compression)
- AIDA-specific: pred mean vs train age mean (bias-toward-training-mean)

Writes `results/phase3/variant1_audit.csv` with one row per condition + a
`results/phase3/variant1_predictions/` directory of per-condition .npz files
holding the raw (true, pred, donor) triples for downstream plotting.

Usage:
    uv run python scripts/variant1_audit.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from scipy.stats import pearsonr


EMB_DIR = Path("results/phase3/embeddings")
OUT_CSV = Path("results/phase3/variant1_audit.csv")
PRED_DIR = Path("results/phase3/variant1_predictions")
ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
SEED = 0
N_BOOTSTRAP = 1000


def _slug(cell_type: str) -> str:
    return cell_type.replace("+", "p").replace(" ", "_")


def _load_npz(cohort: str, cell_type: str, tag: str = "frozen_base"):
    path = EMB_DIR / f"{cohort}_{_slug(cell_type)}_{tag}.npz"
    z = np.load(path, allow_pickle=True)
    return z["donor_ids"], z["ages"].astype(np.float32), z["embeddings"].astype(np.float32)


def _bootstrap_pearson_ci(y_true, y_pred, n_boot=N_BOOTSTRAP, seed=SEED, alpha=0.05):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    rs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_pred[idx]
        if yt.std() < 1e-9 or yp.std() < 1e-9:
            rs[i] = np.nan
            continue
        rs[i] = np.corrcoef(yt, yp)[0, 1]
    rs = rs[~np.isnan(rs)]
    lo = float(np.quantile(rs, alpha / 2))
    hi = float(np.quantile(rs, 1 - alpha / 2))
    return lo, hi


def _ols_slope(y_true, y_pred):
    """Regress pred on true (independent var = true age). slope < 1 = compression of preds."""
    X = np.column_stack([np.ones_like(y_true), y_true.astype(np.float64)])
    beta, *_ = np.linalg.lstsq(X, y_pred.astype(np.float64), rcond=None)
    intercept, slope = float(beta[0]), float(beta[1])
    return slope, intercept


def _audit_one(fold_id, train_cohorts, eval_cohort, cell_type, train_age_mean_override=None, also_aida=False):
    train_X, train_y = [], []
    for tc in train_cohorts:
        _, ages, emb = _load_npz(tc, cell_type)
        train_X.append(emb)
        train_y.append(ages)
    train_X = np.concatenate(train_X, axis=0)
    train_y = np.concatenate(train_y)
    train_mean_age = float(train_y.mean())

    cv = RidgeCV(alphas=ALPHAS, cv=3, scoring="neg_mean_absolute_error")
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(train_y))
    cv.fit(train_X[perm], train_y[perm])
    alpha = float(cv.alpha_)
    final = Ridge(alpha=alpha).fit(train_X, train_y)

    rows = []
    eval_pairs = [(eval_cohort, eval_cohort)]
    if also_aida and eval_cohort != "aida":
        eval_pairs.append(("aida", "aida"))

    for ec_label, ec in eval_pairs:
        donor_ids, eval_y, eval_X = _load_npz(ec, cell_type)
        pred = final.predict(eval_X)
        r, p = pearsonr(pred, eval_y)
        r_lo, r_hi = _bootstrap_pearson_ci(eval_y, pred)
        slope, intercept = _ols_slope(eval_y, pred)
        pred_sd, eval_sd = float(pred.std(ddof=1)), float(eval_y.std(ddof=1))
        pred_min, pred_max = float(pred.min()), float(pred.max())
        eval_min, eval_max = float(eval_y.min()), float(eval_y.max())
        residuals = pred - eval_y
        rows.append({
            "fold": fold_id,
            "eval_cohort": ec_label,
            "cell_type": cell_type,
            "alpha": alpha,
            "n_train_donors": int(len(train_y)),
            "n_eval_donors": int(len(eval_y)),
            "train_age_mean": train_mean_age,
            "eval_age_mean": float(eval_y.mean()),
            "pred_mean": float(pred.mean()),
            "pred_sd": pred_sd,
            "eval_sd": eval_sd,
            "compression_ratio_sd": pred_sd / max(eval_sd, 1e-9),
            "pred_min": pred_min,
            "pred_max": pred_max,
            "eval_min": eval_min,
            "eval_max": eval_max,
            "compression_ratio_range": (pred_max - pred_min) / max(eval_max - eval_min, 1e-9),
            "ols_slope_pred_on_true": slope,
            "ols_intercept_pred_on_true": intercept,
            "pearson_r": float(r),
            "pearson_r_ci_lo": r_lo,
            "pearson_r_ci_hi": r_hi,
            "pearson_p": float(p),
            "mae": float(np.median(np.abs(residuals))),
            "mean_signed_residual": float(np.mean(residuals)),
        })

        # Persist raw predictions for downstream plotting.
        PRED_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(
            PRED_DIR / f"{fold_id}_{ec_label}_{_slug(cell_type)}.npz",
            donor_ids=donor_ids,
            true_age=eval_y,
            pred_age=pred.astype(np.float32),
            train_age_mean=np.float32(train_mean_age),
            alpha=np.float32(alpha),
        )
    return rows


def main():
    folds = json.loads(Path("data/loco_folds.json").read_text())["folds"]
    fold_map = {f["fold_id"]: f for f in folds}

    conditions = [
        ("loco_onek1k", "CD4+ T", True),
        ("loco_terekhova", "CD4+ T", False),
        ("loco_onek1k", "B", True),
        ("loco_terekhova", "B", False),
        ("loco_onek1k", "NK", True),
        ("loco_terekhova", "NK", False),
    ]

    all_rows = []
    for fold_id, cell_type, also_aida in conditions:
        f = fold_map[fold_id]
        print(f"[audit] {fold_id} × {cell_type}")
        rows = _audit_one(
            fold_id=fold_id,
            train_cohorts=f["train_cohorts"],
            eval_cohort=f["holdout_cohort"],
            cell_type=cell_type,
            also_aida=also_aida,
        )
        all_rows.extend(rows)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_CSV, index=False, float_format="%.6g")
    print(f"\n[audit] wrote {len(df)} rows to {OUT_CSV}")

    # Print human-readable summary tables.
    print("\n=== D.9 STAT TABLE (R + 95% CI + p, all conditions) ===")
    cols = ["fold", "eval_cohort", "cell_type", "n_eval_donors", "pearson_r",
            "pearson_r_ci_lo", "pearson_r_ci_hi", "pearson_p"]
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    print("\n=== D.10 BIAS-VARIANCE TABLE ===")
    cols = ["fold", "eval_cohort", "cell_type",
            "pred_sd", "eval_sd", "compression_ratio_sd",
            "ols_slope_pred_on_true",
            "train_age_mean", "eval_age_mean", "pred_mean"]
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.4g}"))


if __name__ == "__main__":
    main()
