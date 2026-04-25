"""Harmonize AIDA into per-cell-type h5ads under data/cohorts/aida_eval/.

AIDA's CellxGene h5ad is 1.27M cells × 35K genes WITH .raw counts. The high-
level `load_cellxgene_cohort` from harmonize_cohorts.py runs out of memory
when subsetting .raw on this scale (15+ GB indices array). Mirror the
load_terekhova streaming pattern: iterate canonical cell types, materialize
each subset separately, write per-cell-type files.

Output:
  data/cohorts/aida_eval/{B,CD4p_T,CD8p_T,Monocyte,NK}.h5ad

Held back from the FM training corpus (data/cohorts/integrated/). Both
directories share the same per-cell-type file layout, so scoring scripts
work on either with `--integrated-dir`.
"""
from __future__ import annotations
import gc
import logging
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from src.data.harmonize_cohorts import (
    canonicalize_cell_type,
    parse_age,
    write_cohort_summary,
    CANONICAL_CELL_TYPES,
)

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
log = logging.getLogger("aida")

PROJ_ROOT = Path(__file__).resolve().parents[2]
AIDA_H5AD = PROJ_ROOT / "data" / "cohorts" / "raw" / "aida" / "9deda9ad-6a71-401e-b909-5263919d85f9.h5ad"
OUT_DIR = PROJ_ROOT / "data" / "cohorts" / "aida_eval"
COHORT_ID = "aida"
CANONICAL_TO_FILENAME = {"CD4+ T": "CD4p_T", "CD8+ T": "CD8p_T", "Monocyte": "Monocyte", "NK": "NK", "B": "B"}


def main():
    if not AIDA_H5AD.exists():
        raise FileNotFoundError(f"AIDA h5ad not found at {AIDA_H5AD}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info(f"reading {AIDA_H5AD} (backed)")
    adata = ad.read_h5ad(AIDA_H5AD, backed="r")
    log.info(f"loaded: {adata.n_obs:,} cells x {adata.n_vars:,} genes (raw is not None: {adata.raw is not None})")

    # All-cell filters computed on obs (cheap)
    obs = adata.obs
    keep = np.ones(len(obs), dtype=bool)

    # Disease == normal (AIDA is all healthy by design, but keep the filter for robustness)
    if "disease" in obs.columns:
        is_healthy = obs["disease"].astype(str).str.lower().isin({"normal", "healthy"}).values
        n_drop = int((~is_healthy & keep).sum())
        keep &= is_healthy
        if n_drop:
            log.info(f"disease filter drops {n_drop:,} non-healthy cells")

    age, precision = parse_age(obs)
    age_ok = age.notna().values
    keep &= age_ok
    keep &= (age.values >= 18) & age_ok
    log.info(f"age + adult filters keep {int(keep.sum()):,} / {len(obs):,}")

    # Cell-type canonicalization
    ct_col = next((c for c in ("cell_type", "cell_type_ontology_term_id", "celltype")
                   if c in obs.columns), None)
    if ct_col is None:
        raise KeyError(f"no cell_type column in obs; have {list(obs.columns)}")
    canonical = canonicalize_cell_type(obs[ct_col])
    canon_ok = canonical.isin(CANONICAL_CELL_TYPES).values
    keep &= canon_ok
    log.info(f"after cell-type filter: {int(keep.sum()):,} cells across canonical 5")

    # Var schema (Ensembl as index, gene_symbol from feature_name)
    full_var = adata.var.copy()
    full_var.index = full_var.index.astype(str)
    if "feature_name" in full_var.columns:
        full_var["gene_symbol"] = full_var["feature_name"].astype(str)
    elif "gene_symbol" not in full_var.columns:
        full_var["gene_symbol"] = full_var.index
    full_var["ensembl_id"] = full_var.index.astype(str)
    new_var = full_var[["gene_symbol", "ensembl_id"]].copy()

    # Donor / sex passthrough
    donor_col = next((c for c in ("donor_id", "donor", "sample_id") if c in obs.columns), None)
    if donor_col is None:
        raise KeyError(f"no donor-id column in obs; have {list(obs.columns)}")

    # Per-cell-type streaming materialization
    canonical_values = canonical.values
    for ct in CANONICAL_CELL_TYPES:
        ct_mask = (canonical_values == ct) & keep
        n_ct = int(ct_mask.sum())
        if n_ct == 0:
            log.warning(f"  {ct}: no cells after filters; skipping")
            continue
        log.info(f"  {ct}: materializing {n_ct:,} cells")
        sub = adata[ct_mask].to_memory()
        # Use .raw.X for raw counts (CellxGene convention; AIDA verified to have integer
        # counts in .raw.X and log-normalized in .X)
        if sub.raw is not None:
            X = sub.raw.X
        else:
            log.warning(f"  {ct}: no .raw on subset; falling back to .X (may be normalized)")
            X = sub.X
        if not sparse.issparse(X):
            X = sparse.csr_matrix(X)
        X = X.astype(np.float32).tocsr()

        sex_col = sub.obs["sex"].astype(str) if "sex" in sub.obs.columns else pd.Series("unknown", index=sub.obs.index)
        assay_col = sub.obs["assay"].astype(str) if "assay" in sub.obs.columns else pd.Series("unknown", index=sub.obs.index)
        ages_for_ct = age[ct_mask]
        precision_for_ct = precision[ct_mask]

        out = ad.AnnData(
            X=X,
            obs=pd.DataFrame(
                {
                    "cohort_id": COHORT_ID,
                    "donor_id": (COHORT_ID + ":" + sub.obs[donor_col].astype(str)).values,
                    "age": ages_for_ct.astype(float).values,
                    "age_precision": precision_for_ct.values,
                    "sex": sex_col.values,
                    "assay": assay_col.values,
                    "cell_type": ct,
                    "self_reported_ethnicity": (
                        sub.obs["self_reported_ethnicity"].astype(str).values
                        if "self_reported_ethnicity" in sub.obs.columns else "unknown"
                    ),
                },
                index=sub.obs.index,
            ),
            var=new_var,
        )
        out_path = OUT_DIR / f"{CANONICAL_TO_FILENAME[ct]}.h5ad"
        out.write_h5ad(out_path, compression="gzip")
        log.info(f"  {ct}: wrote {out_path} ({out.n_obs:,} cells x {out.n_vars:,} genes, "
                 f"{out.obs['donor_id'].nunique()} donors)")
        del sub, X, out
        gc.collect()

    # Per-cohort summary using the just-written files
    log.info("writing AIDA cohort summary")
    rows = []
    for ct in CANONICAL_CELL_TYPES:
        p = OUT_DIR / f"{CANONICAL_TO_FILENAME[ct]}.h5ad"
        if not p.exists():
            continue
        a = ad.read_h5ad(p, backed="r")
        n_cells = a.n_obs
        n_donors = a.obs["donor_id"].nunique()
        rows.append({
            "cohort_id": COHORT_ID,
            "cell_type": ct,
            "n_cells": int(n_cells),
            "n_donors": int(n_donors),
            "age_min": float(a.obs["age"].min()),
            "age_max": float(a.obs["age"].max()),
        })
        del a
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "aida_summary.csv", index=False)
    log.info(f"summary:\n{summary.to_string(index=False)}")
    log.info("Done.")


if __name__ == "__main__":
    main()
