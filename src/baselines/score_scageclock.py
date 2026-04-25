"""Score scAgeClock (Xie 2026) on harmonized LOCO holdouts.

Loads the pretrained GMA checkpoint, formats each (cohort, cell_type) slice to
scAgeClock's input schema (4 categorical features + 19,234 protein-coding gene
features), and runs chunked inference on CPU. Aggregates per-cell predictions
to per-donor median, then computes MAE/R per slice.

Output: appends rows to results/baselines/loco_baseline_table.csv with
        baseline=scAgeClock, training_cohorts=CELLxGENE-Census-2024-07-01.

Memory: chunks at 50K cells. Largest cell-type slice (Terekhova CD4T = 901K
cells × 19,238 dense float32 ≈ 67 GB if loaded at once) is broken into ~18
chunks of ≤50K cells each (~3.7 GB peak per chunk).
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
from scipy.sparse import csr_matrix, hstack
import scanpy as sc
import torch
from scipy.stats import pearsonr

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
log = logging.getLogger("scageclock")

PROJ_ROOT = Path(__file__).resolve().parents[2]
SCAGECLOCK_REPO = PROJ_ROOT / "data" / "scAgeClock" / "scageclock"
MODEL_PATH = SCAGECLOCK_REPO / "data" / "trained_models" / "scAgeClock_GMA_model_state_dict.pth"
META_DIR = SCAGECLOCK_REPO / "data" / "metadata"
CAT_DIR = META_DIR / "categorical_features_index"
INTEGRATED_DIR = PROJ_ROOT / "data" / "cohorts" / "integrated"
RESULTS_DIR = PROJ_ROOT / "results" / "baselines"

# Cohort -> assay categorical_value (must match obs['assay'] in our integrated h5ads)
# Verified against integrated/B.h5ad obs 2026-04-25.
COHORT_TO_ASSAY = {
    "onek1k":     "10x 3' v2",
    "stephenson": "10x 3' transcription profiling",
    "terekhova":  "10x 5' v2",
    "aida":       "10x 5' v2",
}

# Map our canonical 5 cell types -> scAgeClock cell_type categorical_value
CT_TO_SCAGECLOCK = {
    "CD4+ T":   "CD4-positive, alpha-beta T cell",     # numeric_index 19
    "CD8+ T":   "CD8-positive, alpha-beta T cell",     # 26
    "Monocyte": "monocyte",                            # 441
    "NK":       "natural killer cell",                 # 472
    "B":        "B cell",                              # 0
}

# Our cell-type-code (used in score_pretrained_lasso) -> our canonical
CODE_TO_CANONICAL = {"CD4T": "CD4+ T", "CD8T": "CD8+ T", "MONO": "Monocyte", "NK": "NK", "B": "B"}

# File naming under data/cohorts/integrated/{B,CD4p_T,CD8p_T,Monocyte,NK}.h5ad
CANONICAL_TO_FILENAME = {"CD4+ T": "CD4p_T", "CD8+ T": "CD8p_T", "Monocyte": "Monocyte", "NK": "NK", "B": "B"}


def load_categorical_indices() -> dict[str, dict[str, int]]:
    """Load scAgeClock's 4 categorical -> integer-code dicts."""
    out: dict[str, dict[str, int]] = {}
    for cat in ["assay", "cell_type", "tissue_general", "sex"]:
        df = pd.read_csv(CAT_DIR / f"{cat}_numeric_index.tsv", sep="\t")
        out[cat] = dict(zip(df["categorical_value"], df["numeric_index"].astype(int)))
    return out


def load_model_genes() -> list[str]:
    """Load scAgeClock's 19,234 numeric-feature gene names in model order.

    Source: h5ad_var.tsv rows 4-19237 (skipping the 4 categorical entries).
    These are mostly gene symbols with some ENSG IDs for unresolved cases.
    """
    df = pd.read_csv(META_DIR / "h5ad_var.tsv", sep="\t")
    # First 4 rows are categorical features; remainder are model genes in order
    gene_rows = df.iloc[4:].copy()
    return gene_rows["h5ad_var"].astype(str).tolist()


def load_protein_coding_table() -> pd.DataFrame:
    """Helper: gene_id (Ensembl) <-> gene_name (symbol) for the 19,234 protein
    coding genes scAgeClock uses. Used to translate our Ensembl-indexed var
    to scAgeClock's symbol-or-Ensembl model_genes vocabulary.
    """
    df = pd.read_csv(META_DIR / "CZCELLxGENE_ProteinCodingGenes_Selected.tsv", sep="\t")
    return df  # cols: gene_id, gene_name, number_of_cells_expressed


def slice_cohort(canonical_ct: str, cohort_id: str, integrated_dir: Path = INTEGRATED_DIR) -> ad.AnnData:
    """Read the harmonized per-cell-type h5ad and filter to one cohort."""
    fname = CANONICAL_TO_FILENAME[canonical_ct]
    path = integrated_dir / f"{fname}.h5ad"
    log.info(f"loading {path}")
    a = ad.read_h5ad(path)
    keep = (a.obs["cohort_id"] == cohort_id).values
    sub = a[keep].copy()
    log.info(f"  {cohort_id} {canonical_ct}: {sub.n_obs:,} cells")
    return sub


def reshape_to_scageclock_vocab(adata: ad.AnnData, model_genes: list[str], pc_table: pd.DataFrame) -> ad.AnnData:
    """Reindex adata's var to scAgeClock's 19,234 protein-coding gene vocabulary.

    Strategy: match by Ensembl ID (our var.index) first, fall back to gene_symbol.
    Genes in scAgeClock vocab not present in our data become zero columns;
    genes in our data not in scAgeClock vocab are dropped.
    """
    # Build {scAgeClock-vocab-token -> column index in adata}
    # 1) build {ensembl_id_no_suffix -> our-col-idx} and {symbol -> our-col-idx}
    our_ensembl = adata.var.index.astype(str)
    our_ens_norm = pd.Index([e.split(".")[0] for e in our_ensembl])  # strip version suffix
    ens_to_col = {e: i for i, e in enumerate(our_ens_norm) if not e.startswith("nan")}
    sym_to_col: dict[str, int] = {}
    if "gene_symbol" in adata.var.columns:
        for i, s in enumerate(adata.var["gene_symbol"].astype(str)):
            sym_to_col.setdefault(s, i)  # keep first symbol if duplicate

    # 2) for each scAgeClock model gene token, translate via pc_table to (ENSG, symbol),
    #    look up in ens_to_col, fall back to sym_to_col.
    pc_by_id = dict(zip(pc_table["gene_id"].astype(str), pc_table["gene_name"].astype(str)))
    pc_by_name = {row["gene_name"]: row["gene_id"] for _, row in pc_table.iterrows()}

    n_target = len(model_genes)
    target_col_lookup: list[int] = [-1] * n_target  # -1 = not found, fill with zeros
    for j, tok in enumerate(model_genes):
        # tok is mostly a symbol; sometimes an Ensembl-like string
        ens = None
        sym = None
        if tok.startswith("ENSG"):
            ens = tok.split(".")[0]
            sym = pc_by_id.get(ens)
        else:
            sym = tok
            ens = pc_by_name.get(sym)
        # match by ensembl first
        if ens is not None and ens in ens_to_col:
            target_col_lookup[j] = ens_to_col[ens]
        elif sym is not None and sym in sym_to_col:
            target_col_lookup[j] = sym_to_col[sym]
        # else stays -1

    n_matched = sum(1 for c in target_col_lookup if c >= 0)
    log.info(f"  vocab match: {n_matched:,}/{n_target:,} scAgeClock genes mapped from our data")

    # 3) Build the reindexed sparse matrix
    src = adata.X if sparse.issparse(adata.X) else sparse.csr_matrix(adata.X)
    src = src.tocsc()
    n_cells = src.shape[0]
    cols = []
    for tgt_idx, src_col in enumerate(target_col_lookup):
        if src_col >= 0:
            cols.append(src.getcol(src_col))
        else:
            cols.append(csr_matrix((n_cells, 1), dtype=src.dtype))
    new_X = sparse.hstack(cols, format="csr").astype(np.float32)
    new_var = pd.DataFrame({"feature_name": model_genes}, index=pd.Index(model_genes))
    out = ad.AnnData(X=new_X, obs=adata.obs.copy(), var=new_var)
    return out


def normalize_cp10k_log1p(adata: ad.AnnData) -> None:
    """In-place CP10k + log1p (scAgeClock training preprocessing)."""
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)


def assemble_scageclock_input(adata: ad.AnnData, cat_indices: dict, cohort_id: str, canonical_ct: str) -> tuple[ad.AnnData, pd.DataFrame]:
    """Prepend the 4 categorical-feature columns (assay, cell_type, tissue_general, sex)
    to the gene-expression matrix, returning (formatted_adata, donor_age_df).

    Sex is per-cell from obs['sex']; defaults to 'unknown' for any value not in
    {female, male}. Assay and cell_type are constant per slice.
    """
    n = adata.n_obs

    assay_val = COHORT_TO_ASSAY[cohort_id]
    if assay_val not in cat_indices["assay"]:
        log.warning(f"assay '{assay_val}' not in scAgeClock vocab; falling back to '10x 3' transcription profiling'")
        assay_val = "10x 3' transcription profiling"
    assay_idx = cat_indices["assay"][assay_val]

    ct_val = CT_TO_SCAGECLOCK[canonical_ct]
    ct_idx = cat_indices["cell_type"][ct_val]

    tissue_idx = cat_indices["tissue_general"]["blood"]

    # Per-cell sex
    sex_col = adata.obs["sex"].astype(str).str.lower().fillna("unknown")
    sex_col = sex_col.map(lambda s: s if s in ("female", "male") else "unknown")
    sex_idx = sex_col.map(cat_indices["sex"]).astype(int).values

    cat_df = pd.DataFrame({
        "assay": np.full(n, assay_idx, dtype=np.int32),
        "cell_type": np.full(n, ct_idx, dtype=np.int32),
        "tissue_general": np.full(n, tissue_idx, dtype=np.int32),
        "sex": sex_idx.astype(np.int32),
    })

    X_merged = hstack([csr_matrix(cat_df.values.astype(np.float32)), adata.X], format="csr")

    formatted = ad.AnnData(X=X_merged, obs=adata.obs.copy())

    return formatted, adata.obs[["donor_id", "age"]].copy()


def run_chunked_inference(model: torch.nn.Module, X: sparse.csr_matrix, chunk_size: int = 50_000, device: str = "cpu") -> np.ndarray:
    """Run model.forward on X in chunks, returning concatenated 1D age predictions."""
    n = X.shape[0]
    out = np.empty(n, dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0, n, chunk_size):
            stop = min(start + chunk_size, n)
            x_dense = X[start:stop].toarray().astype(np.float32)
            x_t = torch.from_numpy(x_dense).to(device)
            y = model(x_t)
            out[start:stop] = y.cpu().numpy().flatten()
            del x_dense, x_t, y
    return out


def aggregate_per_donor(donor_age_df: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
    """Group predictions by donor_id; report median predicted age and true age."""
    df = donor_age_df.copy()
    df["predicted_age"] = predictions
    grouped = df.groupby("donor_id").agg(
        true_age=("age", "first"),
        predicted_age=("predicted_age", "median"),
        n_cells=("predicted_age", "count"),
    ).reset_index()
    return grouped


def score(predictions_df: pd.DataFrame) -> dict:
    err = predictions_df["predicted_age"] - predictions_df["true_age"]
    abs_err = np.abs(err)
    r, p = pearsonr(predictions_df["predicted_age"], predictions_df["true_age"])
    return {
        "n_donors": len(predictions_df),
        "median_abs_err_yr": float(np.median(abs_err)),
        "mean_abs_err_yr": float(np.mean(abs_err)),
        "pearson_r": float(r),
        "pearson_p": float(p),
        "mean_bias_yr": float(np.mean(err)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohorts", nargs="+", default=["onek1k", "stephenson", "terekhova"])
    ap.add_argument("--cell-types", nargs="+", default=["CD4T", "CD8T", "MONO", "NK", "B"])
    ap.add_argument("--chunk-size", type=int, default=50_000)
    ap.add_argument("--integrated-dir", default=str(INTEGRATED_DIR),
                    help="Source dir of harmonized per-cell-type h5ads (default: data/cohorts/integrated). "
                         "Use data/cohorts/aida_eval to score AIDA.")
    ap.add_argument("--out-csv", default=str(RESULTS_DIR / "scageclock_loco_summary.csv"))
    ap.add_argument("--per-donor-dir", default=str(RESULTS_DIR / "scageclock_per_donor"))
    args = ap.parse_args()
    integrated_dir = Path(args.integrated_dir)

    Path(args.per_donor_dir).mkdir(parents=True, exist_ok=True)

    # Setup
    import sys
    sys.path.insert(0, str(SCAGECLOCK_REPO))
    from scageclock.scAgeClock import load_GMA_model

    log.info(f"loading model from {MODEL_PATH}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"device: {device}")
    model = load_GMA_model(model_file=str(MODEL_PATH))
    model.to(device)
    model.eval()

    cat_indices = load_categorical_indices()
    model_genes = load_model_genes()
    pc_table = load_protein_coding_table()
    log.info(f"loaded {len(model_genes):,} model genes; {len(cat_indices)} categorical features")

    summaries: list[dict] = []
    for cohort in args.cohorts:
        for ct_code in args.cell_types:
            canonical = CODE_TO_CANONICAL[ct_code]
            log.info(f"=== {cohort} × {canonical} ({ct_code}) ===")
            # 1) Load + filter cohort
            adata_raw = slice_cohort(canonical, cohort, integrated_dir=integrated_dir)
            if adata_raw.n_obs == 0:
                log.warning(f"  empty slice; skipping")
                continue
            # 2) Normalize on the FULL gene set (matches scAgeClock's official
            #    formatter which calls sc.pp.normalize_total before filtering to
            #    model_genes — this keeps row sums consistent with scAgeClock's
            #    training-time CP10k).
            log.info(f"  normalizing CP10k + log1p (on full {adata_raw.n_vars}-gene matrix)")
            normalize_cp10k_log1p(adata_raw)
            # 3) Reindex to scAgeClock vocab (drops non-model genes; pads missing
            #    model genes with zeros).
            adata_v = reshape_to_scageclock_vocab(adata_raw, model_genes, pc_table)
            del adata_raw
            gc.collect()
            # 4) Prepend categorical columns
            adata_f, donor_age = assemble_scageclock_input(adata_v, cat_indices, cohort, canonical)
            del adata_v
            gc.collect()
            log.info(f"  formatted shape: {adata_f.shape}")
            # 5) Chunked inference
            X = adata_f.X
            preds = run_chunked_inference(model, X, chunk_size=args.chunk_size, device=device)
            # 6) Aggregate per donor
            per_donor = aggregate_per_donor(donor_age, preds)
            per_donor_path = Path(args.per_donor_dir) / f"{cohort}_{ct_code}.csv"
            per_donor.to_csv(per_donor_path, index=False)
            # 7) Summarize
            metrics = score(per_donor)
            summaries.append({
                "baseline": "scAgeClock",
                "training_cohorts": "CELLxGENE-Census-2024-07-01",
                "eval_cohort": cohort,
                "cell_type": ct_code,
                **metrics,
                "chunk_size": args.chunk_size,
            })
            log.info(f"  n={metrics['n_donors']} median|err|={metrics['median_abs_err_yr']:.2f}y "
                     f"R={metrics['pearson_r']:.3f} p={metrics['pearson_p']:.2e} bias={metrics['mean_bias_yr']:+.2f}y")
            del adata_f, X, preds, donor_age, per_donor
            gc.collect()

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(args.out_csv, index=False)
    log.info(f"wrote summary to {args.out_csv}")


if __name__ == "__main__":
    main()
