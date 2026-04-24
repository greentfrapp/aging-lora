"""
Harmonize the three training cohorts into per-cell-type AnnData files.

Sources (Case 1, revised 2026-04-24 — Barreiro dropped, Terekhova promoted)
-------
    data/cohorts/raw/onek1k_cellxgene/a3f5651f-*.h5ad         (~4.2 GB; Yazar 2022, 981 donors, 1.25M cells, 10x 3' v2)
    data/cohorts/raw/stephenson_covid_portal/c17079d3-*.h5ad  (~6.6 GB; Stephenson 2021 COVID PBMC, 29 healthy donors)
    data/cohorts/raw/terekhova/<h5ad or rds>                  (Synapse syn49637038; 166 donors, 25-85 yr, 10x 5' v2;
                                                                blocked on human Synapse download, see HUMAN_TASKS.md #2)

Outputs
-------
    data/cohorts/integrated/{CD4+ T,CD8+ T,Monocyte,NK,B}.h5ad
        .X                     raw counts (CSR, int) — primary consumer is the foundation models
        .obs['cohort_id']      one of {onek1k, stephenson, terekhova}
        .obs['donor_id']       cohort-prefixed unique donor identifier
        .obs['age']            float years
        .obs['age_precision']  {'exact', 'decade', 'none'} — Stephenson mixes exact-year and decade-bin labels
        .obs['sex']            optional, passthrough where available
        .obs['cell_type']      canonical (CD4+ T | CD8+ T | Monocyte | NK | B)
        .obs['assay']          e.g. "10x 3' v2", "10x 5' v2" — downstream chemistry-aware batch correction keys on this
        .var['gene_symbol']    HGNC symbol
        .var['ensembl_id']     ENSG...
    data/cohort_summary.csv    per-cohort x per-cell-type n_cells, n_donors, age_min, age_max

Design notes
------------
* Raw counts only in .X. Log-normalization is generated on-demand when exporting to R for
  the pre-trained sc-ImmuAging LASSO baseline (which expects Seurat `assays$RNA@data`).
* Healthy filter: Stephenson drops all donors whose `disease` is not "normal"; OneK1K is
  healthy-by-design; Terekhova is healthy-only per the Immunity 2023 paper.
* Cell-type labels: all three sources ship canonical labels; remapped via `CELL_TYPE_MAP`
  (extended below with CellxGene ontology names). No CellTypist re-annotation needed.
* Gene identifiers: preserved in both Ensembl and symbol form. **Gene-set intersection is
  deliberately deferred to model-specific preprocessing** — each foundation model has its
  own vocabulary, so we keep the widest per-cohort gene list here.
* Chemistry heterogeneity: OneK1K+Stephenson are 10x 3', Terekhova is 10x 5' v2. Batch
  correction (Harmony/scran) is applied at model-training time keyed on obs['assay'],
  not here — we keep the harmonized counts chemistry-native.

Usage
-----
    uv run python src/data/harmonize_cohorts.py --out-dir data/cohorts/integrated
    uv run python src/data/harmonize_cohorts.py --skip-terekhova   # while Synapse download is pending

Run time is dominated by loading OneK1K + Terekhova (~2M cells total); expect 30-60 min
on a 32 GB box.
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Optional

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from src.data.download_cohorts import CANONICAL_CELL_TYPES, CELL_TYPE_MAP

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths (relative to project root; overridable via CLI)
# ---------------------------------------------------------------------------
ONEK1K_H5AD = Path("data/cohorts/raw/onek1k_cellxgene/a3f5651f-cd1a-4d26-8165-74964b79b4f2.h5ad")
STEPHENSON_H5AD = Path("data/cohorts/raw/stephenson_covid_portal/c17079d3-204f-487e-bc54-d63bb947a5a2.h5ad")
# Terekhova path is discovered at run-time (Synapse filename not yet known); see
# _discover_terekhova_h5ad() below. TEREKHOVA_DIR is defined alongside the loader.

OUT_DIR_DEFAULT = Path("data/cohorts/integrated")
SUMMARY_CSV = Path("data/cohort_summary.csv")

# ---------------------------------------------------------------------------
# Extended cell-type label map. Extends CELL_TYPE_MAP from download_cohorts.py
# with CellxGene-specific labels observed in OneK1K and Stephenson.
# Left unmapped labels fall through to "Other" and are dropped.
# ---------------------------------------------------------------------------
EXTRA_CELL_TYPE_MAP = {
    # CellxGene / OBO cell ontology labels
    "CD4-positive, alpha-beta T cell": "CD4+ T",
    "CD4-positive, alpha-beta memory T cell": "CD4+ T",
    "naive thymus-derived CD4-positive, alpha-beta T cell": "CD4+ T",
    "central memory CD4-positive, alpha-beta T cell": "CD4+ T",
    "effector memory CD4-positive, alpha-beta T cell": "CD4+ T",
    "regulatory T cell": "CD4+ T",
    "T follicular helper cell": "CD4+ T",
    "T helper cell": "CD4+ T",
    "CD4-positive, alpha-beta cytotoxic T cell": "CD4+ T",

    "CD8-positive, alpha-beta T cell": "CD8+ T",
    "CD8-positive, alpha-beta memory T cell": "CD8+ T",
    "naive thymus-derived CD8-positive, alpha-beta T cell": "CD8+ T",
    "central memory CD8-positive, alpha-beta T cell": "CD8+ T",
    "effector memory CD8-positive, alpha-beta T cell": "CD8+ T",
    "effector CD8-positive, alpha-beta T cell": "CD8+ T",

    "monocyte": "Monocyte",
    "classical monocyte": "Monocyte",
    "non-classical monocyte": "Monocyte",
    "intermediate monocyte": "Monocyte",
    "CD14-positive monocyte": "Monocyte",
    "CD14-low, CD16-positive monocyte": "Monocyte",

    "natural killer cell": "NK",
    "CD16-positive, CD56-dim natural killer cell, human": "NK",
    "CD16-negative, CD56-bright natural killer cell, human": "NK",

    "B cell": "B",
    "naive B cell": "B",
    "memory B cell": "B",
    "class switched memory B cell": "B",
    "unswitched memory B cell": "B",
    "transitional stage B cell": "B",
    "plasmablast": "B",           # borderline; include per sc-ImmuAging convention
    "plasma cell": "B",
}
_FULL_CELL_TYPE_MAP: dict[str, str] = {**CELL_TYPE_MAP, **EXTRA_CELL_TYPE_MAP}


def canonicalize_cell_type(labels: pd.Series) -> pd.Series:
    """Map arbitrary cell-type strings to canonical five or 'Other'."""
    return labels.astype(str).str.strip().map(_FULL_CELL_TYPE_MAP).fillna("Other")


# ---------------------------------------------------------------------------
# Age parsing. CellxGene datasets encode age in several ways:
#   * obs['age']                          numeric years (e.g. 42)
#   * obs['development_stage']            "42-year-old stage"  OR  "sixth decade stage"
#   * obs['development_stage_ontology_term_id']  "HsapDv:0000163"
#
# Stephenson's healthy subset mixes exact-age labels (11/29 donors) and
# decade-bin labels (18/29 donors). We parse exact years when available and
# fall back to decade midpoints (±5 yr precision). `age_precision` records
# which path each cell took so downstream code can weight or exclude
# decade-coded donors.
# ---------------------------------------------------------------------------
_AGE_EXACT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*-?\s*year", flags=re.IGNORECASE)
_DECADE_WORD_TO_MIDPOINT = {
    "first": 5.0, "second": 15.0, "third": 25.0, "fourth": 35.0,
    "fifth": 45.0, "sixth": 55.0, "seventh": 65.0, "eighth": 75.0,
    "ninth": 85.0, "tenth": 95.0,
}
_DECADE_RE = re.compile(r"(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\s+decade",
                        flags=re.IGNORECASE)


def parse_age(obs: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Parse ages into (age_years: float, precision: {'exact','decade','none'}).

    Returns a float Series of ages and a string Series of precision labels
    aligned on the same index.
    """
    n = len(obs)
    age = pd.Series(np.full(n, np.nan, dtype=float), index=obs.index)
    precision = pd.Series(np.full(n, "none", dtype=object), index=obs.index)

    # 1) numeric `age` column wins
    if "age" in obs.columns:
        numeric = pd.to_numeric(obs["age"], errors="coerce")
        filled = numeric.notna() & age.isna()
        age.loc[filled] = numeric.loc[filled].astype(float)
        precision.loc[filled] = "exact"

    # 2) exact-year regex over any stage/age string column
    for col in ("development_stage", "age_group", "Age", "age_years"):
        if col not in obs.columns:
            continue
        s = obs[col].astype(str)
        extracted = s.str.extract(_AGE_EXACT_RE, expand=False)
        numeric = pd.to_numeric(extracted, errors="coerce")
        filled = numeric.notna() & age.isna()
        age.loc[filled] = numeric.loc[filled].astype(float)
        precision.loc[filled] = "exact"

    # 3) decade bins → midpoint with 'decade' precision flag
    if "development_stage" in obs.columns:
        s = obs["development_stage"].astype(str)
        decade_word = s.str.extract(_DECADE_RE, expand=False).str.lower()
        midpoint = decade_word.map(_DECADE_WORD_TO_MIDPOINT)
        filled = midpoint.notna() & age.isna()
        age.loc[filled] = midpoint.loc[filled].astype(float)
        precision.loc[filled] = "decade"

    return age, precision


# ---------------------------------------------------------------------------
# CellxGene cohort loader
# ---------------------------------------------------------------------------
def load_cellxgene_cohort(
    h5ad_path: Path,
    cohort_id: str,
    *,
    healthy_only: bool = True,
    adult_only: bool = True,
    backed: Optional[str] = None,
) -> ad.AnnData:
    """Load a CellxGene h5ad and coerce it to the project's canonical schema.

    Parameters
    ----------
    h5ad_path : path to the CellxGene-curated h5ad
    cohort_id : short cohort tag, used as a prefix on donor_id and as obs['cohort_id']
    healthy_only : drop cells whose donor disease != 'normal'
    adult_only : drop cells whose age < 18
    backed : pass to anndata.read_h5ad (e.g. 'r' for low-memory loading)
    """
    log.info(f"[{cohort_id}] reading {h5ad_path}  backed={backed!r}")
    adata = ad.read_h5ad(h5ad_path, backed=backed)
    log.info(f"[{cohort_id}] loaded: {adata.n_obs:,} cells x {adata.n_vars:,} genes")

    # -- Compute all filters on obs (cheap, obs is already in memory even under backed='r')
    obs = adata.obs
    keep = np.ones(len(obs), dtype=bool)

    # disease
    if healthy_only and "disease" in obs.columns:
        is_healthy = obs["disease"].astype(str).str.lower().isin({"normal", "healthy"}).values
        n_drop = int((~is_healthy & keep).sum())
        keep &= is_healthy
        if n_drop:
            log.info(f"[{cohort_id}] disease filter drops {n_drop:,} non-healthy cells")

    # age parsing (feeds age + precision columns)
    age, precision = parse_age(obs)
    age_ok = age.notna().values
    n_drop = int((~age_ok & keep).sum())
    keep &= age_ok
    if n_drop:
        log.warning(f"[{cohort_id}] age filter drops {n_drop:,} cells with unparseable age")

    if adult_only:
        adult_mask = (age.values >= 18) & age_ok  # NaN ages get filtered by age_ok above
        n_drop = int((~adult_mask & keep).sum())
        keep &= adult_mask
        if n_drop:
            log.info(f"[{cohort_id}] adult filter drops {n_drop:,} cells from donors under 18")

    # cell-type canonicalization (applied to full obs; the 'Other' rows drop below)
    ct_col = next((c for c in ("cell_type", "cell_type_ontology_term_id", "celltype")
                   if c in obs.columns), None)
    if ct_col is None:
        raise KeyError(f"[{cohort_id}] no cell_type column in obs; have {list(obs.columns)}")
    canonical = canonicalize_cell_type(obs[ct_col])
    canon_keep = canonical.isin(CANONICAL_CELL_TYPES).values
    n_drop = int((~canon_keep & keep).sum())
    keep &= canon_keep
    if n_drop:
        log.info(f"[{cohort_id}] cell-type filter drops {n_drop:,} cells not in canonical five")

    n_kept = int(keep.sum())
    log.info(f"[{cohort_id}] combined filters keep {n_kept:,} / {adata.n_obs:,} cells "
             f"({100 * n_kept / max(adata.n_obs, 1):.1f}%)")

    # Single materialization: only read the kept rows' X from disk.
    # .to_memory() handles backed -> in-memory correctly; .copy() works on both.
    if backed is not None:
        adata = adata[keep].to_memory()
    else:
        adata = adata[keep].copy()

    # Re-bind post-filter views
    obs = adata.obs
    age = age[keep]
    precision = precision[keep]
    canonical = canonical[keep]

    n_decade = int((precision == "decade").sum())
    if n_decade:
        log.info(f"[{cohort_id}] {n_decade:,} cells carry decade-precision age (midpoint); "
                 f"flagged via obs['age_precision']")

    # -- donor id: prefer donor_id, fall back to donor, then sample_id
    donor_col = next((c for c in ("donor_id", "donor", "sample_id") if c in obs.columns), None)
    if donor_col is None:
        raise KeyError(f"[{cohort_id}] no donor-id column in obs; have {list(obs.columns)}")
    donor = obs[donor_col].astype(str)

    # -- sex / assay passthrough
    sex = obs["sex"].astype(str) if "sex" in obs.columns else pd.Series("unknown", index=obs.index)
    assay = obs["assay"].astype(str) if "assay" in obs.columns else pd.Series("unknown", index=obs.index)

    # -- var: gene symbol + ensembl id
    var = adata.var.copy()
    ensembl_id = var.index.astype(str)
    gene_symbol = var["feature_name"].astype(str) if "feature_name" in var.columns else ensembl_id
    new_var = pd.DataFrame(
        {"gene_symbol": gene_symbol.values, "ensembl_id": ensembl_id.values},
        index=ensembl_id.values,
    )

    # -- counts: CellxGene puts raw counts in adata.raw.X, normalized in adata.X.
    #    We need raw counts in .X for foundation models.
    if adata.raw is not None:
        log.info(f"[{cohort_id}] using adata.raw.X as counts (CellxGene convention)")
        X = adata.raw.X
        # raw.X may have a different var; assume same ordering (CellxGene standard)
        if adata.raw.shape[1] != adata.n_vars:
            log.warning(f"[{cohort_id}] raw.X has {adata.raw.shape[1]} genes vs X {adata.n_vars}; "
                        f"using raw.var alignment")
            raw_var = adata.raw.var
            raw_ensembl = raw_var.index.astype(str)
            raw_symbol = raw_var["feature_name"].astype(str) if "feature_name" in raw_var.columns else raw_ensembl
            new_var = pd.DataFrame(
                {"gene_symbol": raw_symbol.values, "ensembl_id": raw_ensembl.values},
                index=raw_ensembl.values,
            )
    else:
        log.warning(f"[{cohort_id}] no adata.raw; using adata.X (may already be normalized!)")
        X = adata.X
    if not sparse.issparse(X):
        X = sparse.csr_matrix(X)

    harmonized = ad.AnnData(
        X=X.astype(np.float32),
        obs=pd.DataFrame(
            {
                "cohort_id": cohort_id,
                "donor_id": (cohort_id + ":" + donor).values,
                "age": age.astype(float).values,
                "age_precision": precision.values,
                "sex": sex.values,
                "assay": assay.values,
                "cell_type": canonical.values,
            },
            index=obs.index,
        ),
        var=new_var,
    )
    # Cell-type + healthy/adult/age filters already applied above; harmonized
    # is fully filtered. Log the final shape.
    log.info(
        f"[{cohort_id}] harmonized: {harmonized.n_obs:,} cells | "
        f"{harmonized.obs['donor_id'].nunique()} donors | "
        f"ages {harmonized.obs['age'].min():.0f}-{harmonized.obs['age'].max():.0f}"
    )
    return harmonized


# ---------------------------------------------------------------------------
# Terekhova (Synapse syn49637038) loader
# ---------------------------------------------------------------------------
#
# Schema discovered 2026-04-24 after extraction of all_pbmcs.tar.gz (syn51197006,
# the fallback for the corrupt raw_counts_h5ad.tar.gz — see HUMAN_TASKS.md #5):
#
#   all_pbmcs/all_pbmcs_rna.h5ad       1.916M cells x 36,601 genes, 27 GB.
#                                       obs is empty; var is empty; var.index are
#                                       HGNC gene symbols (e.g. MIR1302-2HG); X is
#                                       raw integer counts.
#   all_pbmcs/all_pbmcs_metadata.csv   406 MB. One row per cell barcode. Columns
#                                       include Donor_id, Age (int years), Sex,
#                                       Batch, Cluster_names (top-level cell type).
#
# load_terekhova joins the metadata CSV onto the empty obs, applies the canonical
# cell-type remap, and produces the same {cohort_id, donor_id, age, age_precision,
# sex, assay, cell_type} schema as load_cellxgene_cohort.
#
# Gene identifier alignment: var.index is HGNC symbols; OneK1K and Stephenson var
# uses Ensembl IDs (with symbols in a `feature_name`/`gene_symbol` column). To
# keep all three cohorts compatible, we map Terekhova's symbols back to Ensembl
# using a supplied symbol_to_ensembl dictionary (typically built from OneK1K's
# var during the harmonization run). Unmapped symbols retain only the symbol
# and carry ensembl_id = NaN.
# ---------------------------------------------------------------------------
TEREKHOVA_DIR = Path("data/cohorts/raw/terekhova")

# Terekhova's Cluster_names top-level labels, mapped to canonical five.
# "Myeloid cells" is coarser than scImmuAging's "Monocyte" (~90% monocytes, ~10%
# dendritic in PBMC). Acceptable trade-off for a PBMC aging clock; the DC
# contamination is orders-of-magnitude lower than the donor-age signal.
_TEREKHOVA_CELLTYPE_MAP = {
    "CD4+ T cells": "CD4+ T",
    "TRAV1-2- CD8+ T cells": "CD8+ T",
    "NK cells": "NK",
    "B cells": "B",
    "Myeloid cells": "Monocyte",
    # Explicit drops (not in canonical 5):
    "gd T cells": "Other",
    "MAIT cells": "Other",
    "Progenitor cells": "Other",
    "DN T cells": "Other",
}


def load_terekhova(
    raw_dir: Path = TEREKHOVA_DIR,
    cohort_id: str = "terekhova",
    *,
    symbol_to_ensembl: dict[str, str] | None = None,
) -> ad.AnnData:
    """Load Terekhova 2023 PBMC atlas from all_pbmcs/{h5ad,metadata.csv}.

    Parameters
    ----------
    raw_dir          directory containing the extracted `all_pbmcs/` folder
    cohort_id        cohort tag for obs['cohort_id'] and donor_id prefix
    symbol_to_ensembl  optional {symbol -> ensembl_id} map used to fill the
                       var.ensembl_id column. If None, ensembl_id is left as
                       the gene symbol (downstream code that keys on
                       `ensembl_id` should intersect-on-symbol in that case).
    """
    extracted_dir = raw_dir / "all_pbmcs"
    h5ad_path = extracted_dir / "all_pbmcs_rna.h5ad"
    meta_path = extracted_dir / "all_pbmcs_metadata.csv"
    if not h5ad_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"[terekhova] expected all_pbmcs/all_pbmcs_rna.h5ad + "
            f"all_pbmcs_metadata.csv under {raw_dir}. "
            f"Extract via: tar --force-local -xzvf all_pbmcs.tar.gz -C {raw_dir}/"
        )

    log.info(f"[{cohort_id}] reading metadata {meta_path}")
    meta = pd.read_csv(meta_path, low_memory=False)
    meta = meta.rename(columns={"Unnamed: 0": "barcode"}).set_index("barcode")
    log.info(f"[{cohort_id}] metadata: {len(meta):,} rows, {meta['Donor_id'].nunique()} donors, "
             f"ages {int(meta['Age'].min())}-{int(meta['Age'].max())}")

    log.info(f"[{cohort_id}] opening {h5ad_path} (backed)")
    adata = ad.read_h5ad(h5ad_path, backed="r")
    log.info(f"[{cohort_id}] loaded: {adata.n_obs:,} cells x {adata.n_vars:,} genes")

    # Align metadata to adata.obs_names
    missing = set(adata.obs_names) - set(meta.index)
    if missing:
        raise RuntimeError(
            f"[{cohort_id}] {len(missing)} barcodes in h5ad but not in metadata; "
            f"sample missing: {list(missing)[:3]}"
        )
    meta_aligned = meta.reindex(adata.obs_names)

    # Compute base filters (age + adult). Cell-type filter applied per-iteration
    # below so the backed subset materialization never needs to hold all 1.83M
    # rows' CSR indices at once — that blew past 17 GiB and OOM'd earlier.
    age = meta_aligned["Age"].astype(float)
    age_ok = age.notna().values
    base_keep = age_ok & (age.values >= 18)
    canonical_series = meta_aligned["Cluster_names"].astype(str).map(_TEREKHOVA_CELLTYPE_MAP).fillna("Other")
    log.info(f"[{cohort_id}] base filters (age + adult) keep "
             f"{int(base_keep.sum()):,} / {len(meta_aligned):,} cells")

    # Build var once (same for every cell type).
    # Critical: var.index must match OneK1K/Stephenson's Ensembl-indexed var so
    # that `ad.concat(join="outer")` aligns genes across cohorts. For Terekhova
    # symbols that do map, we set the index to the Ensembl ID. Unmapped symbols
    # keep the symbol as the index (will be a Terekhova-only gene in the union).
    gene_symbols = np.asarray(adata.var_names, dtype=object)
    if symbol_to_ensembl:
        ensembl_ids = np.array(
            [symbol_to_ensembl.get(s, s) for s in gene_symbols],
            dtype=object,
        )
        n_mapped = int(sum(1 for s in gene_symbols if s in symbol_to_ensembl))
        log.info(f"[{cohort_id}] mapped {n_mapped:,} / {len(gene_symbols):,} "
                 f"gene symbols to Ensembl via supplied table — "
                 f"using Ensembl as var.index for mapped rows")
    else:
        ensembl_ids = gene_symbols.copy()
        log.warning(f"[{cohort_id}] no symbol->ensembl map supplied; var.index stays as symbols "
                    f"(cross-cohort gene alignment WILL BE BROKEN)")
    # Index on Ensembl ID where available, symbol otherwise. This aligns with
    # OneK1K/Stephenson whose var_names are Ensembl IDs.
    new_var = pd.DataFrame(
        {"gene_symbol": gene_symbols, "ensembl_id": ensembl_ids},
        index=pd.Index(ensembl_ids, name=None),
    )
    # Drop any duplicate rows in var that collide after remapping (if two
    # Terekhova symbols map to the same Ensembl ID). Keep the first.
    if new_var.index.has_duplicates:
        dupe_mask = new_var.index.duplicated(keep="first")
        n_dupes = int(dupe_mask.sum())
        log.warning(f"[{cohort_id}] {n_dupes} duplicate Ensembl IDs after symbol remap; "
                    f"keeping first occurrence per Ensembl ID")
        new_var = new_var.loc[~dupe_mask].copy()
        # We also need to deduplicate X columns. Track the kept indices.
        kept_cols = np.nonzero(~dupe_mask)[0]
    else:
        kept_cols = None

    # Per-cell-type streaming materialization — bound peak memory to the largest
    # cell type (CD4+ T at ~901K cells, ~3-8 GB int64 indices, manageable on 32 GB).
    import gc
    canonical_values = canonical_series.values
    parts: list[ad.AnnData] = []
    for ct in CANONICAL_CELL_TYPES:
        ct_mask = (canonical_values == ct) & base_keep
        n_ct = int(ct_mask.sum())
        if n_ct == 0:
            log.warning(f"[{cohort_id}] no cells mapped to {ct}; skipping")
            continue
        log.info(f"[{cohort_id}]   {ct}: materializing {n_ct:,} cells")
        sub = adata[ct_mask].to_memory()
        X_sub = sub.X
        if not sparse.issparse(X_sub):
            X_sub = sparse.csr_matrix(X_sub)
        X_sub = X_sub.astype(np.float32).tocsr()
        # If symbol->ensembl remap produced duplicate Ensembl IDs, keep the
        # first column per duplicate so X's columns match new_var's rows.
        if kept_cols is not None:
            X_sub = X_sub[:, kept_cols]

        meta_sub = meta.reindex(sub.obs_names)

        # Reverse log1p(CP10k) -> raw integer counts.
        # Verified 2026-04-24 that all_pbmcs_rna.h5ad stores log1p(CP10k):
        # row_sum(expm1(X)) == 10000.0 exactly, and expm1(X) * nCount_RNA / 10000
        # recovers integer counts to 100% tolerance <0.01. Without this step,
        # Terekhova's X would be log-normalized while OneK1K/Stephenson are raw
        # counts, breaking both FM fine-tuning and LASSO CP10k preprocessing.
        n_counts = meta_sub["nCount_RNA"].astype(np.float64).values
        if np.isnan(n_counts).any():
            raise RuntimeError(
                f"[{cohort_id}] {ct}: {int(np.isnan(n_counts).sum())} cells missing "
                f"nCount_RNA in metadata; cannot reverse-normalize"
            )
        # In-place per-row scaling: for CSR, row i's data is data[indptr[i]:indptr[i+1]].
        # This avoids building a diag sparse matrix and keeps peak memory to 1x X_sub.
        X_sub.data = np.expm1(X_sub.data)
        scale = (n_counts / 10000.0).astype(np.float32)
        indptr = X_sub.indptr
        for i in range(X_sub.shape[0]):
            X_sub.data[indptr[i]:indptr[i+1]] *= scale[i]
        X_sub.data = np.rint(X_sub.data).astype(np.float32)
        # Report integer-likeness as a sanity check
        n_sample = min(X_sub.data.size, 100_000)
        if n_sample:
            residual = np.abs(X_sub.data[:n_sample] - np.round(X_sub.data[:n_sample])).max()
            log.info(f"[{cohort_id}]   {ct}: reverse-normalized to raw counts "
                     f"(max int-residual in first {n_sample:,} nnz: {residual:.3g})")
        part = ad.AnnData(
            X=X_sub,
            obs=pd.DataFrame(
                {
                    "cohort_id": cohort_id,
                    "donor_id": (cohort_id + ":" + meta_sub["Donor_id"].astype(str)).values,
                    "age": meta_sub["Age"].astype(float).values,
                    "age_precision": "exact",
                    "sex": meta_sub["Sex"].astype(str).values,
                    "assay": "10x 5' v2",
                    "cell_type": ct,
                    "batch": meta_sub["Batch"].astype(str).values,
                },
                index=sub.obs_names,
            ),
            var=new_var,
        )
        parts.append(part)
        del sub, X_sub, meta_sub
        gc.collect()
        log.info(f"[{cohort_id}]   {ct}: done ({part.n_obs:,} cells)")

    if not parts:
        raise RuntimeError(f"[{cohort_id}] no per-cell-type shards produced")

    harmonized = ad.concat(parts, axis=0, join="outer", merge="first")
    log.info(
        f"[{cohort_id}] harmonized: {harmonized.n_obs:,} cells | "
        f"{harmonized.obs['donor_id'].nunique()} donors | "
        f"ages {harmonized.obs['age'].min():.0f}-{harmonized.obs['age'].max():.0f}"
    )
    # Free the per-type parts now that they're stacked in `harmonized`.
    parts.clear()
    gc.collect()
    return harmonized


# ---------------------------------------------------------------------------
# Splitting and summary
# ---------------------------------------------------------------------------
def split_and_write(
    adatas: list[ad.AnnData],
    out_dir: Path,
) -> dict[str, ad.AnnData]:
    """Split each cohort by cell type, accumulate per-cell-type shards, concat+write.

    Memory-streaming design (important for the three-cohort run): we never hold
    all cohorts concatenated into one big AnnData at once. Instead we do

      for each cohort: for each cell type: slice + keep in a per-type list
      for each cell type: concat the (<=3) shards and write the h5ad

    Peak memory is roughly (largest-cohort) + (largest-cell-type-shard), which
    is ~1/3 of the naive ad.concat-then-split approach for a 3-cohort corpus.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    shards: dict[str, list[ad.AnnData]] = {ct: [] for ct in CANONICAL_CELL_TYPES}

    for a in adatas:
        cid = a.obs["cohort_id"].iloc[0] if a.n_obs else "<empty>"
        for ct in CANONICAL_CELL_TYPES:
            mask = (a.obs["cell_type"].astype(str) == ct).values
            n = int(mask.sum())
            if n == 0:
                continue
            sub = a[mask].copy()
            # Attach a stable source_cohort label so downstream can deduplicate if needed.
            sub.obs["source_cohort"] = cid
            shards[ct].append(sub)
            log.info(f"[shard] {cid} {ct}: {n:,} cells")

    per_type: dict[str, ad.AnnData] = {}
    for ct in CANONICAL_CELL_TYPES:
        if not shards[ct]:
            log.warning(f"[write] {ct}: no cells across any cohort — skipping file")
            continue
        combined = ad.concat(
            shards[ct], axis=0, join="outer", merge="first", index_unique="|",
        )
        out_path = out_dir / f"{ct.replace('+', 'p').replace(' ', '_')}.h5ad"
        combined.write_h5ad(out_path, compression="gzip")
        log.info(
            f"[write] {out_path}  {combined.n_obs:,} cells x {combined.n_vars:,} genes "
            f"across {combined.obs['cohort_id'].nunique()} cohorts"
        )
        per_type[ct] = combined
        # Drop the reference promptly so the next cell type starts from a clean slate.
        del combined
    return per_type


def write_cohort_summary(per_type: dict[str, ad.AnnData], out_path: Path = SUMMARY_CSV) -> pd.DataFrame:
    rows = []
    for ct, sub in per_type.items():
        for cohort_id, grp in sub.obs.groupby("cohort_id", observed=True):
            rows.append({
                "cohort_id": cohort_id,
                "cell_type": ct,
                "n_cells": int(len(grp)),
                "n_donors": int(grp["donor_id"].nunique()),
                "age_min": float(grp["age"].min()),
                "age_max": float(grp["age"].max()),
            })
    df = pd.DataFrame(rows).sort_values(["cohort_id", "cell_type"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    log.info(f"[summary] wrote {out_path}")
    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--out-dir", default=str(OUT_DIR_DEFAULT))
    parser.add_argument("--skip-onek1k", action="store_true")
    parser.add_argument("--skip-stephenson", action="store_true")
    parser.add_argument("--skip-terekhova", action="store_true",
                        help="Skip Terekhova (useful while the Synapse download is still pending).")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    adatas: list[ad.AnnData] = []
    # symbol->ensembl map is built from the first CellxGene cohort we load (OneK1K
    # or Stephenson — both carry gene_symbol + ensembl_id pairs in their var) and
    # passed to load_terekhova to translate Terekhova's symbol-indexed var into
    # Ensembl for downstream gene-vocab alignment.
    symbol_to_ensembl: dict[str, str] | None = None

    def _harvest_symbol_map(a: ad.AnnData) -> dict[str, str]:
        if "gene_symbol" not in a.var.columns or "ensembl_id" not in a.var.columns:
            return {}
        # Build a dict{symbol: ensembl}. If multiple Ensembl IDs share a symbol
        # (happens for ~1% of readthrough / paralog genes), keep the first.
        out: dict[str, str] = {}
        for sym, ens in zip(a.var["gene_symbol"], a.var["ensembl_id"]):
            sym_s = str(sym)
            if sym_s and sym_s not in out:
                out[sym_s] = str(ens)
        return out

    if not args.skip_onek1k:
        if not ONEK1K_H5AD.exists():
            raise FileNotFoundError(f"OneK1K h5ad not found at {ONEK1K_H5AD}; run Task 1d first")
        onek1k = load_cellxgene_cohort(ONEK1K_H5AD, cohort_id="onek1k")
        adatas.append(onek1k)
        symbol_to_ensembl = _harvest_symbol_map(onek1k)
        log.info(f"[main] harvested {len(symbol_to_ensembl):,} symbol->ensembl mappings from OneK1K")

    if not args.skip_stephenson:
        if not STEPHENSON_H5AD.exists():
            raise FileNotFoundError(f"Stephenson h5ad not found at {STEPHENSON_H5AD}")
        stephenson = load_cellxgene_cohort(STEPHENSON_H5AD, cohort_id="stephenson")
        adatas.append(stephenson)
        if symbol_to_ensembl is None or len(symbol_to_ensembl) == 0:
            symbol_to_ensembl = _harvest_symbol_map(stephenson)
            log.info(f"[main] harvested {len(symbol_to_ensembl):,} symbol->ensembl mappings from Stephenson")

    if not args.skip_terekhova:
        adatas.append(load_terekhova(TEREKHOVA_DIR, cohort_id="terekhova",
                                      symbol_to_ensembl=symbol_to_ensembl))

    if not adatas:
        log.error("No cohorts loaded (all --skip flags set)")
        return

    per_type = split_and_write(adatas, out_dir)
    write_cohort_summary(per_type)
    log.info("Harmonization complete.")


if __name__ == "__main__":
    main()
