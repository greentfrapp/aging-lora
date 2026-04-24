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
    log.info(f"[{cohort_id}] reading {h5ad_path}")
    adata = ad.read_h5ad(h5ad_path, backed=backed)
    log.info(f"[{cohort_id}] loaded: {adata.n_obs:,} cells x {adata.n_vars:,} genes")

    obs = adata.obs.copy()

    # -- disease filter
    if healthy_only and "disease" in obs.columns:
        is_healthy = obs["disease"].astype(str).str.lower().isin({"normal", "healthy"})
        n_drop = (~is_healthy).sum()
        if n_drop:
            log.info(f"[{cohort_id}] dropping {n_drop:,} cells with non-healthy disease label")
        adata = adata[is_healthy.values].copy()
        obs = adata.obs.copy()

    # -- age parsing
    age, precision = parse_age(obs)
    n_missing_age = age.isna().sum()
    if n_missing_age:
        log.warning(f"[{cohort_id}] dropping {n_missing_age:,} cells with unparseable age")
        keep = age.notna().values
        adata = adata[keep].copy()
        obs = adata.obs.copy()
        age = age[keep]
        precision = precision[keep]
    n_decade = (precision == "decade").sum()
    if n_decade:
        log.info(f"[{cohort_id}] {n_decade:,} cells carry decade-precision age (midpoint); "
                 f"flagged via obs['age_precision']")

    if adult_only:
        keep = age.values >= 18
        n_drop = (~keep).sum()
        if n_drop:
            log.info(f"[{cohort_id}] dropping {n_drop:,} cells from donors under 18")
        adata = adata[keep].copy()
        obs = adata.obs.copy()
        age = age[keep]
        precision = precision[keep]

    # -- donor id: prefer donor_id, fall back to donor, then sample_id
    donor_col = next((c for c in ("donor_id", "donor", "sample_id") if c in obs.columns), None)
    if donor_col is None:
        raise KeyError(f"[{cohort_id}] no donor-id column in obs; have {list(obs.columns)}")
    donor = obs[donor_col].astype(str)

    # -- cell type
    ct_col = next((c for c in ("cell_type", "cell_type_ontology_term_id", "celltype")
                   if c in obs.columns), None)
    if ct_col is None:
        raise KeyError(f"[{cohort_id}] no cell_type column in obs; have {list(obs.columns)}")
    canonical = canonicalize_cell_type(obs[ct_col])

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
    # drop 'Other' cells
    keep = harmonized.obs["cell_type"].isin(CANONICAL_CELL_TYPES).values
    n_drop = (~keep).sum()
    if n_drop:
        log.info(f"[{cohort_id}] dropping {n_drop:,} cells not in canonical five cell types")
    harmonized = harmonized[keep].copy()

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
# Terekhova et al. 2023 Immunity. 166 healthy donors aged 25-85, 10x 5' v2.
# Download requires a free Synapse account + DUC acceptance — see HUMAN_TASKS.md #2.
#
# The Synapse-hosted format is expected to be an h5ad (single or per-cell-type),
# but the exact schema cannot be verified until download. This loader is written
# against the typical CellxGene/scanpy schema (obs: donor_id/age/cell_type/disease)
# and will be tightened after first inspection of the downloaded file. If the
# release is an RDS / Seurat object instead, a one-off R-to-h5ad conversion
# script lives at src/data/convert_terekhova_rds.R (not written until needed).
# ---------------------------------------------------------------------------
TEREKHOVA_DIR = Path("data/cohorts/raw/terekhova")


def _discover_terekhova_h5ad(raw_dir: Path) -> Path:
    """Return the single expected h5ad under raw_dir; fail loudly otherwise."""
    candidates = sorted(raw_dir.glob("*.h5ad"))
    if len(candidates) == 0:
        raise FileNotFoundError(
            f"[terekhova] no h5ad found under {raw_dir}. Download from Synapse "
            f"syn49637038 — see HUMAN_TASKS.md #2."
        )
    if len(candidates) > 1:
        log.warning(f"[terekhova] multiple h5ads under {raw_dir}; using the largest: "
                    + ", ".join(p.name for p in candidates))
        candidates.sort(key=lambda p: p.stat().st_size, reverse=True)
    return candidates[0]


def load_terekhova(
    raw_dir: Path = TEREKHOVA_DIR,
    cohort_id: str = "terekhova",
) -> ad.AnnData:
    """Load the Terekhova 2023 PBMC atlas.

    If the Synapse release is a standard CellxGene-shaped h5ad, this delegates
    to `load_cellxgene_cohort`. If the schema diverges (custom column names,
    RDS format, per-cell-type sharding), inspect the file once downloaded and
    tighten this function — fail loudly until then.
    """
    h5ad_path = _discover_terekhova_h5ad(raw_dir)
    log.info(f"[terekhova] using {h5ad_path}")
    # Terekhova is healthy-only by paper design, but the schema may not carry a
    # 'disease' column — pass healthy_only=False and adult_only=True (age range
    # is 25-85). The cohort-side adult filter in load_cellxgene_cohort handles
    # this correctly.
    return load_cellxgene_cohort(
        h5ad_path=h5ad_path,
        cohort_id=cohort_id,
        healthy_only=False,
        adult_only=True,
    )


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

    if not args.skip_onek1k:
        if not ONEK1K_H5AD.exists():
            raise FileNotFoundError(f"OneK1K h5ad not found at {ONEK1K_H5AD}; run Task 1d first")
        adatas.append(load_cellxgene_cohort(ONEK1K_H5AD, cohort_id="onek1k"))

    if not args.skip_stephenson:
        if not STEPHENSON_H5AD.exists():
            raise FileNotFoundError(f"Stephenson h5ad not found at {STEPHENSON_H5AD}")
        adatas.append(load_cellxgene_cohort(STEPHENSON_H5AD, cohort_id="stephenson"))

    if not args.skip_terekhova:
        adatas.append(load_terekhova(TEREKHOVA_DIR, cohort_id="terekhova"))

    if not adatas:
        log.error("No cohorts loaded (all --skip flags set)")
        return

    per_type = split_and_write(adatas, out_dir)
    write_cohort_summary(per_type)
    log.info("Harmonization complete.")


if __name__ == "__main__":
    main()
