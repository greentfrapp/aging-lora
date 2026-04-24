"""Score the pre-trained sc-ImmuAging LASSO clocks in pure Python.

Implements the same pipeline as the scImmuAging R package:
    PreProcess -> AgingClockCalculator -> Age_Donor

by reading the serialized R cv.glmnet objects via the `rdata` library,
extracting coefficients at lambda.min, and re-implementing the
log-normalize -> pseudocell -> linear-predict logic in numpy.

This bypasses the need to install R or Seurat on the target machine.
Tested equivalence against the R package is deferred to Task 1e output
review (the reported MAE must fall within 2x of the paper's internal
validation value; if not, fall back to the R pipeline).

Key reference: data/scImmuAging/R/main.R in the vendored repo.

Design
------
* `extract_lasso_coefs(rds_path, features_rds_path, cell_type)` returns
  (coef_vec, intercept, gene_ids) at lambda.min. The coef_vec is aligned
  to `gene_ids`, which come from all_model_inputfeatures.RDS (1000 genes
  per cell type in Ensembl form).
* `score_pretrained_lasso(adata, ...)` takes a per-cell-type AnnData
  with raw counts in .X, `obs['donor_id']`, `obs['age']`, and Ensembl
  gene ids as `.var_names` (or in `.var['ensembl_id']`). Returns a
  per-donor DataFrame with predicted age + true age.

The scoring reproduces `PreProcess()`:
  1. Log1p(CP10k) normalize each cell (Seurat NormalizeData default).
  2. For each donor, sample 15 cells with replacement-dynamic, average
     (colMeans) to form a pseudocell. Repeat 100 times. Seed is fixed
     for reproducibility.
  3. Align each pseudocell to the 1000 marker genes; pad missing with 0.
  4. predict = pseudocell @ coefs + intercept.
  5. Per-donor prediction = mean across pseudocells.

Usage
-----
    uv run python -m src.baselines.score_pretrained_lasso --cell-type CD8T

Adds `results/baselines/pretrained_sanity_check.csv` with per-donor
predictions and summary metrics.
"""

from __future__ import annotations

import argparse
import logging
import time
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import rdata
from scipy import sparse
from scipy.stats import pearsonr

warnings.filterwarnings("ignore", category=UserWarning, module="rdata")
warnings.filterwarnings("ignore", category=UserWarning, module="anndata")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


MODEL_RDS = Path("data/scImmuAging/data/all_model.RDS")
FEATURES_RDS = Path("data/scImmuAging/data/all_model_inputfeatures.RDS")
ONEK1K_H5AD = Path("data/cohorts/raw/onek1k_cellxgene/a3f5651f-cd1a-4d26-8165-74964b79b4f2.h5ad")
RESULTS_DIR = Path("results/baselines")

# Map (scImmuAging cell-type code) -> (CellxGene cell_type_ontology_term_id list)
# CD4+ T, CD8+ T, Monocyte, NK, B all have many subtype ontology IDs; we capture
# all descendants of the five parent classes.
CELLTYPE_CODE_TO_ONTOLOGY = {
    "CD4T": {
        "CL:0000624",  # CD4-positive, alpha-beta T cell
        "CL:0000895",  # naive thymus-derived CD4-positive, alpha-beta T cell
        "CL:0000904",  # central memory CD4-positive, alpha-beta T cell
        "CL:0000905",  # effector memory CD4-positive, alpha-beta T cell
        "CL:0000815",  # regulatory T cell           (OneK1K uses this, not CL:0000792)
        "CL:0002038",  # T follicular helper cell
        "CL:0002678",  # memory CD4+ T               (supplementary; rare)
        "CL:0000897",  # CD4-positive, alpha-beta memory T cell
        "CL:0000899",  # T-helper 22 cell
    },
    "CD8T": {
        "CL:0000625",  # CD8-positive, alpha-beta T cell
        "CL:0000900",  # naive thymus-derived CD8-positive, alpha-beta T cell
        "CL:0000907",  # central memory CD8-positive, alpha-beta T cell
        "CL:0000913",  # effector memory CD8-positive, alpha-beta T cell
        "CL:0000909",  # CD8-positive, alpha-beta memory T cell
        # NOTE: CL:0000934 is "CD4-positive, alpha-beta cytotoxic T cell" (CD4!) — do NOT include here
    },
    "MONO": {
        # OneK1K-actual monocyte ontology IDs (verified 2026-04-24 from h5ad obs)
        "CL:0001054",  # CD14-positive monocyte
        "CL:0002396",  # CD14-low, CD16-positive monocyte
        # Also include ancestor classes in case future cohorts use them
        "CL:0000576",  # monocyte
        "CL:0000860",  # classical monocyte
        "CL:0000875",  # non-classical monocyte
    },
    "NK": {
        "CL:0000623",  # natural killer cell
        "CL:0000938",  # CD16-negative, CD56-bright natural killer cell, human
        "CL:0000939",  # CD16-positive, CD56-dim natural killer cell, human
    },
    "B": {
        "CL:0000236",  # B cell
        "CL:0000788",  # naive B cell
        "CL:0000787",  # memory B cell
        "CL:0000818",  # transitional stage B cell
        "CL:0000972",  # class switched memory B cell
        "CL:0000970",  # unswitched memory B cell
        "CL:0000980",  # plasmablast (borderline; matches our harmonize_cohorts CELL_TYPE_MAP)
    },
}


# ---------------------------------------------------------------------------
# LASSO coefficient extraction from all_model.RDS
# ---------------------------------------------------------------------------
def _dgcmatrix_column(beta: object, col_idx: int, n_rows: int) -> np.ndarray:
    """Extract a single CSC column from an rdata-parsed dgCMatrix.

    dgCMatrix storage uses:
      beta.i = row indices of non-zero entries   (int32, nnz,)
      beta.p = column pointers                   (int32, ncol+1,)
      beta.x = non-zero values                   (float64, nnz,)
    """
    i = np.asarray(beta.i)
    p = np.asarray(beta.p)
    x = np.asarray(beta.x)
    start, stop = int(p[col_idx]), int(p[col_idx + 1])
    col = np.zeros(n_rows, dtype=np.float64)
    col[i[start:stop]] = x[start:stop]
    return col


def extract_lasso_coefs(
    model_rds: Path = MODEL_RDS,
    features_rds: Path = FEATURES_RDS,
    cell_type: str = "CD8T",
) -> tuple[np.ndarray, float, list[str]]:
    """Return (coef_vec, intercept, gene_ids) for one cell-type clock at lambda.min."""
    log.info(f"loading LASSO model {model_rds} (cell_type={cell_type})")
    model_parsed = rdata.parser.parse_file(str(model_rds))
    models = rdata.conversion.convert(model_parsed)
    if cell_type not in models:
        raise KeyError(f"cell_type {cell_type!r} not in model keys {list(models.keys())}")
    mdl = models[cell_type]

    lambdas = np.asarray(mdl["glmnet.fit"]["lambda"])
    lambda_min = float(mdl["lambda.min"][0])
    idx_min = int(np.argmin(np.abs(lambdas - lambda_min)))
    log.info(f"  lambda.min = {lambda_min:.6g}  (column {idx_min}/{len(lambdas)})")

    beta = mdl["glmnet.fit"]["beta"]
    n_genes = int(beta.Dim[0])
    coef_vec = _dgcmatrix_column(beta, idx_min, n_genes)
    gene_ids = [str(g) for g in beta.Dimnames[0]]
    if len(gene_ids) != n_genes:
        raise RuntimeError(
            f"gene_id dim ({len(gene_ids)}) != beta.Dim[0] ({n_genes})"
        )

    # Intercept at lambda.min: glmnet.fit$a0[idx_min] is a DataArray
    a0 = mdl["glmnet.fit"]["a0"]
    a0_vals = a0.values if hasattr(a0, "values") else np.asarray(a0)
    intercept = float(a0_vals[idx_min])
    log.info(f"  intercept = {intercept:.4f}  nonzero coefs = {(coef_vec != 0).sum()} / {n_genes}")

    # Verify the feature list in all_model_inputfeatures.RDS matches beta rows
    feat_parsed = rdata.parser.parse_file(str(features_rds))
    feats = rdata.conversion.convert(feat_parsed)
    expected = [str(g) for g in feats[cell_type]]
    if expected != gene_ids:
        raise RuntimeError(
            f"gene order mismatch between all_model_inputfeatures and beta rows "
            f"({len(set(expected) & set(gene_ids))} / {len(gene_ids)} shared)"
        )

    return coef_vec, intercept, gene_ids


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def _align_to_marker_genes(X: sparse.csr_matrix, var_ids: np.ndarray, marker_ids: list[str]) -> sparse.csr_matrix:
    """Reorder X's columns to match marker_ids; missing genes become zero columns.

    Vectorized: convert to CSC for fast column slicing, take the present-genes
    submatrix, then scatter into a (n_cells, len(marker_ids)) output with zeros
    for missing marker ids.
    """
    id_to_idx = {g: i for i, g in enumerate(var_ids)}
    col_idx = np.array([id_to_idx.get(g, -1) for g in marker_ids], dtype=np.int64)
    present_mask = col_idx >= 0
    present_target = np.nonzero(present_mask)[0]
    present_source = col_idx[present_mask]

    X_csc = X.tocsc()
    # Pick only the columns of X we actually need, in their source order.
    sub = X_csc[:, present_source]  # (n_cells, n_present)

    n_cells = X.shape[0]
    n_marker = len(marker_ids)
    # Scatter into output at target positions.
    out = sparse.lil_matrix((n_cells, n_marker), dtype=X.dtype)
    out[:, present_target] = sub
    return out.tocsr()


def _log1p_cp10k_rows(X: sparse.csr_matrix) -> sparse.csr_matrix:
    """Per-cell counts-per-10,000 + log1p, matching Seurat NormalizeData default."""
    X = X.astype(np.float32).tocsr()
    row_sums = np.asarray(X.sum(axis=1)).ravel()
    # Safe division
    safe_sums = np.where(row_sums > 0, row_sums, 1.0).astype(np.float32)
    scale = (1e4 / safe_sums).astype(np.float32)
    # Multiply each row by its scale factor.
    D = sparse.diags(scale, format="csr")
    X = D @ X
    # log1p preserves sparsity.
    X.data = np.log1p(X.data)
    return X


def score_pretrained_lasso(
    adata: ad.AnnData,
    cell_type: str,
    *,
    pseudocell_size: int = 15,
    pseudocell_n: int = 100,
    seed: int = 0,
) -> pd.DataFrame:
    """Score the pre-trained sc-ImmuAging LASSO on a per-cell-type AnnData.

    Parameters
    ----------
    adata : one cell type only; .X raw counts (int or float), .var_names = Ensembl IDs,
            .obs must have 'donor_id' and 'age' columns.
    cell_type : one of {"CD4T","CD8T","MONO","NK","B"} — the clock code.

    Returns
    -------
    DataFrame with columns [donor_id, true_age, predicted_age, n_cells, pseudocell_n]
    """
    coef_vec, intercept, marker_ids = extract_lasso_coefs(cell_type=cell_type)
    required = {"donor_id", "age"}
    missing = required - set(adata.obs.columns)
    if missing:
        raise KeyError(f"adata.obs missing columns {missing}")

    t0 = time.time()
    X = adata.X
    if not sparse.issparse(X):
        X = sparse.csr_matrix(X)
    X = X.tocsr()

    log.info(f"[{cell_type}] log1p(CP10k) normalizing {X.shape[0]:,} cells x {X.shape[1]:,} genes")
    X = _log1p_cp10k_rows(X)

    log.info(f"[{cell_type}] aligning to {len(marker_ids)} marker genes")
    var_ids = np.asarray(adata.var_names, dtype=object)
    X_aligned = _align_to_marker_genes(X, var_ids, marker_ids)
    # also report how many marker genes are actually present
    present = sum(1 for g in marker_ids if g in set(var_ids))
    log.info(f"[{cell_type}] {present}/{len(marker_ids)} marker genes present in adata.var_names")

    # Pseudocell sampling + prediction per donor
    rng = np.random.default_rng(seed)
    donor_series = adata.obs["donor_id"].astype(str)
    age_series = adata.obs["age"].astype(float)
    unique_donors = donor_series.unique()

    log.info(f"[{cell_type}] predicting for {len(unique_donors)} donors "
             f"({pseudocell_n} pseudocells x {pseudocell_size} cells each)")

    rows = []
    for donor in unique_donors:
        mask = (donor_series.values == donor)
        n_cells = int(mask.sum())
        if n_cells < 1:
            continue
        # Select this donor's aligned matrix (dense slice, typically <few 1000 cells x 1000 genes)
        donor_X = X_aligned[mask].toarray().astype(np.float32)  # (n_cells, 1000)
        replace = n_cells <= pseudocell_size
        # R's pseudocell samples 15 cells without replacement PER pseudocell (not
        # globally distinct across all 100 pseudocells). We match that semantic:
        # loop when replace=False, vectorize when replace=True.
        if replace:
            sample_idx = rng.choice(n_cells, size=(pseudocell_n, pseudocell_size), replace=True)
        else:
            sample_idx = np.empty((pseudocell_n, pseudocell_size), dtype=np.int64)
            for k in range(pseudocell_n):
                sample_idx[k] = rng.choice(n_cells, size=pseudocell_size, replace=False)
        pseudocells = donor_X[sample_idx].mean(axis=1)  # (pseudocell_n, 1000)
        preds = pseudocells @ coef_vec + intercept
        donor_true_age = float(age_series.values[mask][0])
        rows.append({
            "donor_id": donor,
            "true_age": donor_true_age,
            "predicted_age": float(preds.mean()),
            "predicted_age_sd": float(preds.std()),
            "n_cells": n_cells,
            "pseudocell_n": pseudocell_n,
        })

    df = pd.DataFrame(rows)
    log.info(f"[{cell_type}] scoring complete in {time.time() - t0:.1f}s")
    return df


def _onek1k_subset_to_cell_type(h5ad_path: Path, cell_type_code: str) -> ad.AnnData:
    """Stream-load OneK1K and subset to one cell type.

    OneK1K is 1.25M cells x 35k genes; loading into memory is feasible (~6 GB)
    but slow. We use backed mode to slice by the cell_type_ontology_term_id.
    """
    log.info(f"opening {h5ad_path} in backed mode")
    a = ad.read_h5ad(h5ad_path, backed="r")
    ontology_ids = CELLTYPE_CODE_TO_ONTOLOGY[cell_type_code]
    ct_col = "cell_type_ontology_term_id"
    keep_mask = a.obs[ct_col].astype(str).isin(ontology_ids).values
    n_keep = int(keep_mask.sum())
    log.info(f"[{cell_type_code}] {n_keep:,} cells match ontology IDs "
             f"{sorted(ontology_ids)}")
    # Materialize the subset into memory. Backed slicing keeps this cheap.
    sub = a[keep_mask].to_memory()
    return sub


# Map scImmuAging cell-type code -> canonical cell_type label (as used in harmonized h5ads).
CELLTYPE_CODE_TO_CANONICAL = {
    "CD4T": "CD4+ T",
    "CD8T": "CD8+ T",
    "MONO": "Monocyte",
    "NK":   "NK",
    "B":    "B",
}


def _harmonized_subset(
    integrated_dir: Path,
    cell_type_code: str,
    cohort_id: str | None = None,
) -> ad.AnnData:
    """Load a per-cell-type h5ad from data/cohorts/integrated/, optionally filter by cohort_id."""
    canonical = CELLTYPE_CODE_TO_CANONICAL[cell_type_code]
    # Accept both "CD4+_T" (my filename pattern "CD4p_T") and "CD4+ T" variants.
    stem = canonical.replace("+", "p").replace(" ", "_")
    h5ad_path = integrated_dir / f"{stem}.h5ad"
    if not h5ad_path.exists():
        # Fallback: try plain `canonical.h5ad`
        h5ad_path = integrated_dir / f"{canonical}.h5ad"
    if not h5ad_path.exists():
        raise FileNotFoundError(
            f"Can't find integrated h5ad for {canonical} under {integrated_dir}; "
            f"tried {stem}.h5ad and {canonical}.h5ad."
        )
    log.info(f"opening {h5ad_path}")
    a = ad.read_h5ad(h5ad_path)
    if cohort_id is not None:
        mask = (a.obs["cohort_id"].astype(str) == cohort_id).values
        n_keep = int(mask.sum())
        log.info(f"[{cell_type_code}] filtered to cohort_id={cohort_id!r}: {n_keep:,} cells")
        a = a[mask].copy()
    # harmonize_cohorts writes Ensembl IDs as var_names; verify.
    if a.var_names[0].startswith("ENSG"):
        pass
    elif "ensembl_id" in a.var.columns:
        a.var_names = a.var["ensembl_id"].astype(str).values
    return a


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cell-type", default="CD8T",
                    choices=list(CELLTYPE_CODE_TO_ONTOLOGY.keys()))
    ap.add_argument("--source", default="onek1k-raw",
                    choices=("onek1k-raw", "harmonized"),
                    help="onek1k-raw = load directly from the OneK1K CellxGene h5ad (default, "
                         "for the Task 1e sanity check). "
                         "harmonized = load from data/cohorts/integrated/ and optionally "
                         "filter by --cohort-id (for Task 1f / Phase 2 LOCO scoring).")
    ap.add_argument("--h5ad", default=str(ONEK1K_H5AD),
                    help="source h5ad for --source onek1k-raw")
    ap.add_argument("--integrated-dir", default="data/cohorts/integrated",
                    help="directory of harmonized per-cell-type h5ads for --source harmonized")
    ap.add_argument("--cohort-id", default=None,
                    help="filter --source harmonized by this cohort_id (e.g. 'terekhova')")
    ap.add_argument("--out-csv", default=None,
                    help=f"per-donor output CSV; default: {RESULTS_DIR}/pretrained_sanity_{{cell_type}}.csv")
    ap.add_argument("--summary-csv", default=None,
                    help=f"per-cell-type summary CSV the script appends to; "
                         f"default: {RESULTS_DIR}/pretrained_sanity_summary.csv. "
                         f"Use a different file for non-OneK1K runs e.g. Task 1f on Terekhova "
                         f"so the OneK1K sanity summary is not overwritten.")
    ap.add_argument("--pseudocell-n", type=int, default=100)
    ap.add_argument("--pseudocell-size", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cell_type_code = args.cell_type
    if args.source == "onek1k-raw":
        adata = _onek1k_subset_to_cell_type(Path(args.h5ad), cell_type_code)
    elif args.source == "harmonized":
        adata = _harmonized_subset(
            Path(args.integrated_dir),
            cell_type_code,
            cohort_id=args.cohort_id,
        )
    else:
        raise ValueError(f"unknown --source {args.source!r}")

    # Ensure obs has canonical columns
    if "age" not in adata.obs.columns:
        # OneK1K has both 'age' (numeric) and 'development_stage' — we want numeric.
        raise RuntimeError("OneK1K obs missing 'age' column; adjust schema parsing")
    adata.obs["donor_id"] = adata.obs["donor_id"].astype(str)
    adata.obs["age"] = pd.to_numeric(adata.obs["age"], errors="coerce")

    # adata.var_names are Ensembl IDs already
    df = score_pretrained_lasso(
        adata,
        cell_type=cell_type_code,
        pseudocell_n=args.pseudocell_n,
        pseudocell_size=args.pseudocell_size,
        seed=args.seed,
    )

    if df.empty:
        log.error(
            f"[{cell_type_code}] zero donors scored — check that "
            f"CELLTYPE_CODE_TO_ONTOLOGY[{cell_type_code!r}] matches the h5ad's "
            f"cell_type_ontology_term_id values."
        )
        raise SystemExit(2)

    # Summary
    mae = float(np.median(np.abs(df["predicted_age"] - df["true_age"])))
    mean_abs_err = float(np.mean(np.abs(df["predicted_age"] - df["true_age"])))
    r, pval = pearsonr(df["predicted_age"], df["true_age"])
    bias = float(np.mean(df["predicted_age"] - df["true_age"]))
    log.info(
        f"[{cell_type_code}] n_donors={len(df)} "
        f"median|err|={mae:.2f}y  mean|err|={mean_abs_err:.2f}y  "
        f"Pearson R={r:.3f} (p={pval:.2g})  mean_bias={bias:+.2f}y"
    )

    out_csv = Path(args.out_csv) if args.out_csv else RESULTS_DIR / f"pretrained_sanity_{cell_type_code}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    log.info(f"wrote {out_csv}")

    # Also append a single-row summary
    summary_csv = Path(args.summary_csv) if args.summary_csv else RESULTS_DIR / "pretrained_sanity_summary.csv"
    summary_row = {
        "cell_type": cell_type_code,
        "n_donors": len(df),
        "median_abs_err_yr": round(mae, 3),
        "mean_abs_err_yr": round(mean_abs_err, 3),
        "pearson_r": round(r, 4),
        "pearson_p": pval,
        "mean_bias_yr": round(bias, 3),
        "pseudocell_n": args.pseudocell_n,
        "pseudocell_size": args.pseudocell_size,
    }
    if summary_csv.exists():
        existing = pd.read_csv(summary_csv)
        existing = existing[existing["cell_type"] != cell_type_code]
        out = pd.concat([existing, pd.DataFrame([summary_row])], ignore_index=True)
    else:
        out = pd.DataFrame([summary_row])
    out.to_csv(summary_csv, index=False)
    log.info(f"appended summary to {summary_csv}")


if __name__ == "__main__":
    main()
