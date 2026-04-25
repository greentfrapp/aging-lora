"""Pseudobulk our harmonized h5ads per (donor × cell-type) and write tab-separated
matrices suitable for the Pasta R pipeline.

Output (one file per cohort × cell_type):
  results/baselines/pasta_pseudobulk/{cohort}_{cell_type_code}.tsv
    rows  = Ensembl gene IDs (var.index)
    cols  = donor_id values
    cells = summed raw counts (integer; cast to int when saving)

Plus a single companion file with donor metadata:
  results/baselines/pasta_pseudobulk/{cohort}_{cell_type_code}_meta.csv
    cols: donor_id, age, sex, n_cells_aggregated
"""
from __future__ import annotations
import argparse
import logging
from pathlib import Path
import gc

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
log = logging.getLogger("pseudobulk")

PROJ_ROOT = Path(__file__).resolve().parents[2]
INTEGRATED_DIR = PROJ_ROOT / "data" / "cohorts" / "integrated"
OUT_DIR = PROJ_ROOT / "results" / "baselines" / "pasta_pseudobulk"

CODE_TO_CANONICAL = {"CD4T": "CD4+ T", "CD8T": "CD8+ T", "MONO": "Monocyte", "NK": "NK", "B": "B"}
CANONICAL_TO_FILENAME = {"CD4+ T": "CD4p_T", "CD8+ T": "CD8p_T", "Monocyte": "Monocyte", "NK": "NK", "B": "B"}


def pseudobulk_one(cohort: str, ct_code: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    canonical = CODE_TO_CANONICAL[ct_code]
    fname = CANONICAL_TO_FILENAME[canonical]
    a = ad.read_h5ad(INTEGRATED_DIR / f"{fname}.h5ad")
    keep = (a.obs["cohort_id"] == cohort).values
    sub = a[keep].copy()
    log.info(f"{cohort} × {ct_code}: {sub.n_obs:,} cells, {sub.obs['donor_id'].nunique()} donors")

    # Strip Ensembl version suffix only from genuine ENSG IDs to align with
    # Pasta's v_genes_model (unversioned ENSG). Other identifiers like clone
    # accessions (AC000065.1 vs AC000065.2) are distinct genomic features and
    # MUST keep their version suffix.
    raw = sub.var.index.astype(str)
    var_ens = raw.where(~raw.str.startswith("ENSG"),
                        raw.str.split(".").str[0])

    X = sub.X if sparse.issparse(sub.X) else sparse.csr_matrix(sub.X)
    X = X.tocsr()

    donors = sub.obs["donor_id"].astype(str).values
    unique_donors = pd.Index(sorted(set(donors)))
    donor_to_idx = {d: i for i, d in enumerate(unique_donors)}
    rows = np.array([donor_to_idx[d] for d in donors], dtype=np.int32)

    # Build a (n_donors × n_cells) sparse indicator and multiply: P = D @ X
    n_donors = len(unique_donors)
    n_cells = X.shape[0]
    cols = np.arange(n_cells, dtype=np.int32)
    data = np.ones(n_cells, dtype=np.float32)
    D = sparse.csr_matrix((data, (rows, cols)), shape=(n_donors, n_cells))
    P = (D @ X).toarray().astype(np.float32)  # n_donors × n_genes
    log.info(f"  pseudobulk shape: {P.shape}; total counts: {P.sum():,.0f}")

    # Build the gene × donor matrix with Ensembl IDs as rownames
    pseudobulk_df = pd.DataFrame(P.T.astype(np.int32), index=var_ens, columns=unique_donors)

    # Donor metadata: take first-occurrence age/sex per donor
    meta = (
        sub.obs[["donor_id", "age", "sex"]]
        .astype({"donor_id": str})
        .drop_duplicates(subset=["donor_id"])
        .set_index("donor_id")
        .reindex(unique_donors)
    )
    n_cells_per = sub.obs.groupby("donor_id", observed=True).size().reindex(unique_donors)
    meta = meta.assign(n_cells_aggregated=n_cells_per.values).reset_index().rename(columns={"index": "donor_id"})

    return pseudobulk_df, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohorts", nargs="+", default=["onek1k", "stephenson", "terekhova"])
    ap.add_argument("--cell-types", nargs="+", default=["CD4T", "CD8T", "MONO", "NK", "B"])
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for cohort in args.cohorts:
        for ct in args.cell_types:
            log.info(f"=== {cohort} × {ct} ===")
            pb_df, meta_df = pseudobulk_one(cohort, ct)
            pb_path = out_dir / f"{cohort}_{ct}.tsv"
            meta_path = out_dir / f"{cohort}_{ct}_meta.csv"
            pb_df.to_csv(pb_path, sep="\t")
            meta_df.to_csv(meta_path, index=False)
            log.info(f"  wrote {pb_path}; {meta_path}")
            del pb_df, meta_df
            gc.collect()


if __name__ == "__main__":
    main()
