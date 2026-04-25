"""Phase 2 Task 2.7: retrain LASSO on our 3 cohorts (training-matched comparator).

For each LOCO fold (e.g., OneK1K-out: train on Stephenson + Terekhova, evaluate
on OneK1K), train one LASSO per cell type using the same pseudocell-aggregation
+ marker-gene panel + log1p(CP10k) preprocessing as the upstream pretrained
sc-ImmuAging LASSO. This isolates the "training-cohort-set" effect from the
"architecture" effect when comparing FM fine-tunes to LASSO.

Training procedure (matches `data/scImmuAging/codes/Lasso_training.R` semantics):
  - per training donor: 100 pseudocells × 15 cells, sample → mean
  - log1p(CP10k) per pseudocell
  - align to the 1100 marker genes for that cell type (from all_model.RDS)
  - sklearn LassoCV (10-fold internal CV, default α grid; α=1.0 = pure L1)

Evaluation: same pseudocell scheme on the held-out cohort, score per donor as
median of pseudocell predictions. Output rows append to loco_baseline_table.csv
with `baseline=LASSO-retrained-3cohort, training_cohorts=our-three-cohort`.
"""
from __future__ import annotations
import argparse
import logging
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import pearsonr
from sklearn.linear_model import LassoCV

from src.baselines.score_pretrained_lasso import (
    extract_lasso_coefs,  # to get the marker-gene panel per cell type
    _align_to_marker_genes,
    _log1p_cp10k_rows,
)

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
log = logging.getLogger("retrain")

PROJ_ROOT = Path(__file__).resolve().parents[2]
INTEGRATED_DIR = PROJ_ROOT / "data" / "cohorts" / "integrated"
RESULTS_DIR = PROJ_ROOT / "results" / "baselines"

CT_CODES = ["CD4T", "CD8T", "MONO", "NK", "B"]
CODE_TO_CANONICAL = {"CD4T": "CD4+ T", "CD8T": "CD8+ T", "MONO": "Monocyte", "NK": "NK", "B": "B"}
CANONICAL_TO_FILENAME = {"CD4+ T": "CD4p_T", "CD8+ T": "CD8p_T", "Monocyte": "Monocyte", "NK": "NK", "B": "B"}
COHORTS = ["onek1k", "stephenson", "terekhova"]


def build_pseudocell_matrix(
    adata_backed: ad.AnnData,
    cell_mask: np.ndarray,
    marker_ids: list[str],
    *,
    pseudocell_n: int = 100,
    pseudocell_size: int = 15,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Per-donor: sample pseudocells, log1p(CP10k), align to markers.

    Operates on a BACKED AnnData + boolean cell-mask, materializing one donor's
    cells at a time. This bounds peak memory to the largest single donor's
    cells × genes matrix (typically a few thousand cells × ~49K genes ≈ a few
    hundred MB), which is critical for the largest LOCO training folds (1.5M+
    cells in CD4T Stephenson-out).

    Returns
    -------
    X_pseudo : (n_donors * pseudocell_n, len(marker_ids)) float32
    y_pseudo : (n_donors * pseudocell_n,) float32 — repeated donor age
    donor_ids : list[str] — repeated per pseudocell, so result rows can be joined back
    """
    rng = np.random.default_rng(seed)
    obs_full = adata_backed.obs
    donor_series = obs_full["donor_id"].astype(str)
    age_series = obs_full["age"].astype(float)
    var_ids = np.asarray(adata_backed.var_names, dtype=object)
    id_to_var_idx = {g: i for i, g in enumerate(var_ids)}
    # Pre-compute marker→source-column mapping
    marker_src_idx = np.array([id_to_var_idx.get(g, -1) for g in marker_ids], dtype=np.int64)
    present_target = np.nonzero(marker_src_idx >= 0)[0]
    present_source = marker_src_idx[marker_src_idx >= 0]
    n_marker = len(marker_ids)

    # Only iterate donors that have cells in the mask
    cell_idx_global = np.nonzero(cell_mask)[0]
    donors_in_mask = donor_series.values[cell_idx_global]
    unique_donors = pd.unique(donors_in_mask)

    X_blocks = []
    y_blocks = []
    donor_blocks: list[str] = []
    for donor in unique_donors:
        donor_mask = (donor_series.values == donor) & cell_mask
        donor_idx = np.nonzero(donor_mask)[0]
        n_cells = len(donor_idx)
        if n_cells < 1:
            continue
        # Materialize only this donor's slice (cells × all genes)
        donor_sub = adata_backed[donor_idx].to_memory()
        donor_X_full = donor_sub.X
        if not sparse.issparse(donor_X_full):
            donor_X_full = sparse.csr_matrix(donor_X_full)
        donor_X_full = _log1p_cp10k_rows(donor_X_full)
        # Align to markers (cells × n_marker)
        sub = donor_X_full.tocsc()[:, present_source].toarray().astype(np.float32)
        donor_X_aligned = np.zeros((n_cells, n_marker), dtype=np.float32)
        donor_X_aligned[:, present_target] = sub

        replace = n_cells <= pseudocell_size
        if replace:
            sample_idx = rng.choice(n_cells, size=(pseudocell_n, pseudocell_size), replace=True)
        else:
            sample_idx = np.empty((pseudocell_n, pseudocell_size), dtype=np.int64)
            for k in range(pseudocell_n):
                sample_idx[k] = rng.choice(n_cells, size=pseudocell_size, replace=False)
        pseudocells = donor_X_aligned[sample_idx].mean(axis=1)  # (pseudocell_n, n_markers)
        donor_age = float(age_series.values[donor_mask][0])
        X_blocks.append(pseudocells)
        y_blocks.append(np.full(pseudocell_n, donor_age, dtype=np.float32))
        donor_blocks.extend([donor] * pseudocell_n)
        del donor_sub, donor_X_full, donor_X_aligned, sub

    if not X_blocks:
        raise RuntimeError("no pseudocells built — check input adata + mask")
    X_pseudo = np.vstack(X_blocks)
    y_pseudo = np.concatenate(y_blocks)
    return X_pseudo, y_pseudo, donor_blocks


def per_donor_predictions(
    X_pseudo: np.ndarray, donor_ids: list[str], y_true: np.ndarray, model: LassoCV
) -> pd.DataFrame:
    """Predict per pseudocell, aggregate per-donor median."""
    preds = model.predict(X_pseudo)
    df = pd.DataFrame({"donor_id": donor_ids, "true_age": y_true, "pred": preds})
    grouped = df.groupby("donor_id").agg(
        true_age=("true_age", "first"),
        predicted_age=("pred", "median"),
    ).reset_index()
    return grouped


def loco_one_fold(
    holdout_cohort: str, ct_code: str, *, pseudocell_n: int = 100, pseudocell_size: int = 15, seed: int = 0,
) -> dict:
    canonical = CODE_TO_CANONICAL[ct_code]
    fname = CANONICAL_TO_FILENAME[canonical]
    h5_path = INTEGRATED_DIR / f"{fname}.h5ad"
    log.info(f"=== holdout={holdout_cohort} cell_type={ct_code} ({canonical}) ===")

    # Use backed mode + per-donor materialization to bound peak memory
    a = ad.read_h5ad(h5_path, backed="r")
    train_mask = (a.obs["cohort_id"] != holdout_cohort).values
    test_mask = (a.obs["cohort_id"] == holdout_cohort).values
    n_train_donors = a.obs.loc[train_mask, "donor_id"].nunique()
    n_test_donors = a.obs.loc[test_mask, "donor_id"].nunique()
    log.info(f"  train: {int(train_mask.sum()):,} cells from cohorts "
             f"{sorted(a.obs.loc[train_mask, 'cohort_id'].unique().tolist())}, "
             f"{n_train_donors} donors")
    log.info(f"  test:  {int(test_mask.sum()):,} cells from {holdout_cohort}, "
             f"{n_test_donors} donors")

    # Marker gene panel (reuse the upstream sc-ImmuAging panel for direct comparability)
    _, _, marker_ids = extract_lasso_coefs(cell_type=ct_code)
    log.info(f"  using {len(marker_ids)} marker genes from upstream sc-ImmuAging panel")

    # Build training pseudocells (streaming per-donor materialization)
    t0 = time.time()
    X_train, y_train, _ = build_pseudocell_matrix(
        a, train_mask, marker_ids,
        pseudocell_n=pseudocell_n, pseudocell_size=pseudocell_size, seed=seed,
    )
    log.info(f"  X_train shape: {X_train.shape}; built in {time.time()-t0:.1f}s")

    # Fit LassoCV (10-fold CV; default α grid; pure L1)
    t0 = time.time()
    model = LassoCV(cv=10, n_alphas=50, max_iter=5_000, n_jobs=-1, random_state=seed)
    model.fit(X_train, y_train)
    log.info(f"  fit done in {time.time()-t0:.1f}s; alpha_={model.alpha_:.4g}; "
             f"non-zero coefs={int((model.coef_ != 0).sum())} / {len(model.coef_)}")

    # Build test pseudocells with the same marker panel
    X_test, y_test, donor_test = build_pseudocell_matrix(
        a, test_mask, marker_ids,
        pseudocell_n=pseudocell_n, pseudocell_size=pseudocell_size, seed=seed,
    )
    per_donor = per_donor_predictions(X_test, donor_test, y_test, model)

    # Metrics
    err = per_donor["predicted_age"] - per_donor["true_age"]
    abs_err = err.abs()
    if len(per_donor) >= 3 and per_donor["predicted_age"].std() > 0 and per_donor["true_age"].std() > 0:
        r, p = pearsonr(per_donor["predicted_age"], per_donor["true_age"])
    else:
        r, p = float("nan"), float("nan")

    out = {
        "baseline": "LASSO-retrained-3cohort",
        "training_cohorts": "our-three-cohort",
        "eval_cohort": holdout_cohort,
        "cell_type": ct_code,
        "n_donors": int(len(per_donor)),
        "median_abs_err_yr": float(np.median(abs_err)),
        "mean_abs_err_yr": float(np.mean(abs_err)),
        "pearson_r": float(r),
        "pearson_p": float(p),
        "mean_bias_yr": float(np.mean(err)),
        "alpha_lambda_min": float(model.alpha_),
        "n_nonzero_coefs": int((model.coef_ != 0).sum()),
        "pseudocell_n": pseudocell_n,
        "pseudocell_size": pseudocell_size,
    }
    log.info(f"  -> n={out['n_donors']} MAE={out['median_abs_err_yr']:.2f}y "
             f"R={out['pearson_r']:.3f} p={out['pearson_p']:.2e} bias={out['mean_bias_yr']:+.2f}y")

    # Write per-donor predictions
    per_donor_dir = RESULTS_DIR / "lasso_retrained_3cohort" / "per_donor"
    per_donor_dir.mkdir(parents=True, exist_ok=True)
    per_donor.to_csv(per_donor_dir / f"{holdout_cohort}_{ct_code}.csv", index=False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohorts", nargs="+", default=COHORTS)
    ap.add_argument("--cell-types", nargs="+", default=CT_CODES)
    ap.add_argument("--out-csv", default=str(RESULTS_DIR / "lasso_retrained_3cohort" / "summary.csv"))
    ap.add_argument("--pseudocell-n", type=int, default=100)
    ap.add_argument("--pseudocell-size", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = []
    for cohort in args.cohorts:
        for ct in args.cell_types:
            res = loco_one_fold(
                cohort, ct,
                pseudocell_n=args.pseudocell_n,
                pseudocell_size=args.pseudocell_size,
                seed=args.seed,
            )
            rows.append(res)

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)
    log.info(f"wrote {args.out_csv} ({len(df)} rows)")


if __name__ == "__main__":
    main()
