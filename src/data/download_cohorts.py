"""
Download and quality-control the three public training cohorts for the immune aging clock.

Cohorts (Case 1 baseline, revised 2026-04-24 — Barreiro/Randolph dropped, Terekhova
promoted from Case 3):
  OneK1K (CellxGene dde06e0f / dataset 3faad104) - Yazar et al. 2022 Science     - 981 donors, 10x 3' v2
  Stephenson (CellxGene ddfad306)                - Stephenson et al. 2021 COVID   -  29 healthy donors, 10x 3'
  Terekhova (Synapse syn49637038)                - Terekhova et al. 2023 Immunity - 166 donors, 10x 5' v2

Barreiro/Randolph 2021 (GSE162632) was dropped because: (a) GEO has no donor ages, and
(b) the release is genotype-multiplexed with no demux files in the archive. Tracked in
FUTURE_WORK.md; revisit if donor ages and genotype VCFs become accessible.

Chemistry heterogeneity: OneK1K+Stephenson are 10x 3', Terekhova is 10x 5' v2. The
pre-trained sc-ImmuAging LASSO baseline was trained on 3' only — Terekhova LOCO is
reported both naive (measures real-world generalization) and chemistry-corrected via
Harmony/scran (isolates aging-specific signal).

OneK1K source decision (2026-04-24): CellxGene-curated h5ad used instead of GEO RAW.
The GEO release is 75 multiplexed pools requiring genotype demultiplexing; the CellxGene
version is pre-demultiplexed with donor_id, age, and cell-type annotations.

Usage:
    uv run python src/data/download_cohorts.py --out-dir data/cohorts
"""

import argparse
import hashlib
import logging
import subprocess
from pathlib import Path

import GEOparse
import pandas as pd
import requests
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Registry of all five cohorts
# -------------------------------------------------------------------
COHORTS = [
    {
        "id": "onek1k_cellxgene",
        "name": "OneK1K",
        "source": "CellPortal",
        "url": "https://cellxgene.cziscience.com/collections/dde06e0f-ab3b-46be-96a2-a8082383c4a1",
        "download_url": "https://datasets.cellxgene.cziscience.com/a3f5651f-cd1a-4d26-8165-74964b79b4f2.h5ad",
        "dataset_id": "3faad104-2ab8-4434-816d-474d8d2641db",
        "n_donors_expected": 981,
        "n_cells_expected": 1248980,
        "cell_types": ["CD4+ T", "CD8+ T", "Monocyte", "NK", "B"],
        "notes": "Yazar et al. 2022 Science. OneK1K eQTL dataset. 981 healthy donors, 1.25M cells. CellxGene-curated (pre-demultiplexed); superseded GEO GSE196830 RAW path.",
    },
    {
        "id": "stephenson_covid_portal",
        "name": "Stephenson-COVID-CellPortal",
        "source": "CellPortal",
        "url": "https://cellxgene.cziscience.com/collections/ddfad306-714d-4cc0-9985-d9072820c530",
        "download_url": "https://datasets.cellxgene.cziscience.com/c17079d3-204f-487e-bc54-d63bb947a5a2.h5ad",
        "n_donors_expected": 130,
        "cell_types": ["CD4+ T", "CD8+ T", "Monocyte", "NK", "B"],
        "notes": "Stephenson et al. 2021 Nature Medicine COVID PBMC (doi:10.1038/s41591-021-01329-2). 647k cells, 130 donors. ArrayExpress: E-MTAB-10026.",
    },
    {
        "id": "terekhova",
        "name": "Terekhova-2023",
        "source": "Synapse",
        "url": "https://www.synapse.org/Synapse:syn49637038",
        "synapse_id": "syn49637038",
        "n_donors_expected": 166,
        "cell_types": ["CD4+ T", "CD8+ T", "Monocyte", "NK", "B"],
        "notes": "Terekhova et al. 2023 Immunity. Healthy PBMC across 25-85 yr. 10x 5' v2. Requires free Synapse account + DUC acceptance; see HUMAN_TASKS.md #2.",
    },
]

# Cell-type label harmonisation map: source label → canonical label
CELL_TYPE_MAP = {
    # CD4+ T variants
    "CD4+ T": "CD4+ T",
    "CD4 T": "CD4+ T",
    "CD4+T": "CD4+ T",
    "T CD4+": "CD4+ T",
    "Naive CD4 T": "CD4+ T",
    "Memory CD4 T": "CD4+ T",
    "CD4_T": "CD4+ T",
    # CD8+ T variants
    "CD8+ T": "CD8+ T",
    "CD8 T": "CD8+ T",
    "CD8+T": "CD8+ T",
    "T CD8+": "CD8+ T",
    "Cytotoxic T": "CD8+ T",
    "CD8_T": "CD8+ T",
    # Monocyte variants
    "Monocyte": "Monocyte",
    "Monocytes": "Monocyte",
    "CD14+ Mono": "Monocyte",
    "CD16+ Mono": "Monocyte",
    "Classical monocyte": "Monocyte",
    "Non-classical monocyte": "Monocyte",
    "Mono": "Monocyte",
    # NK variants
    "NK": "NK",
    "NK cell": "NK",
    "Natural killer cell": "NK",
    "NK_cell": "NK",
    # B cell variants
    "B": "B",
    "B cell": "B",
    "B cells": "B",
    "Naive B": "B",
    "Memory B": "B",
    "B_cell": "B",
}

CANONICAL_CELL_TYPES = ["CD4+ T", "CD8+ T", "Monocyte", "NK", "B"]


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as fh, tqdm(total=total, unit="B", unit_scale=True,
                                       desc=dest.name) as bar:
        for chunk in resp.iter_content(1 << 16):
            fh.write(chunk)
            bar.update(len(chunk))


def download_geo_soft(accession: str, out_dir: Path) -> Path:
    """Download GEO SOFT metadata file for an accession."""
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Fetching GEO metadata for {accession}")
    gse = GEOparse.get_GEO(geo=accession, destdir=str(out_dir), silent=True)
    return gse


def clone_scimmuaging(out_dir: Path) -> Path:
    """Clone the official sc-ImmuAging integration pipeline."""
    repo_url = "https://github.com/CiiM-Bioinformatics-group/scImmuAging"
    dest = out_dir / "scImmuAging"
    if dest.exists():
        log.info(f"scImmuAging repo already exists at {dest}, pulling latest")
        subprocess.run(["git", "-C", str(dest), "pull"], check=True)
    else:
        log.info(f"Cloning {repo_url} → {dest}")
        subprocess.run(["git", "clone", repo_url, str(dest)], check=True)
    return dest


def validate_cohort_counts(adata, cohort_id: str) -> dict:
    """
    Validate that an AnnData object contains expected metadata columns
    and return per-cell-type donor counts.
    """
    required_cols = {"cell_type", "donor_id", "age"}
    missing = required_cols - set(adata.obs.columns)
    if missing:
        raise ValueError(f"{cohort_id}: obs missing columns {missing}")

    # Harmonize cell type labels
    adata.obs["cell_type_canonical"] = (
        adata.obs["cell_type"].map(CELL_TYPE_MAP).fillna("Other")
    )

    summary = {}
    for ct in CANONICAL_CELL_TYPES:
        mask = adata.obs["cell_type_canonical"] == ct
        n_cells = mask.sum()
        n_donors = adata.obs.loc[mask, "donor_id"].nunique()
        summary[ct] = {"n_cells": int(n_cells), "n_donors": int(n_donors)}

    return summary


def build_cohort_summary(summaries: dict, out_path: Path) -> pd.DataFrame:
    """Write per-cohort per-cell-type donor and cell counts to CSV."""
    rows = []
    for cohort_id, cell_summaries in summaries.items():
        for ct, counts in cell_summaries.items():
            rows.append({
                "cohort_id": cohort_id,
                "cell_type": ct,
                "n_cells": counts["n_cells"],
                "n_donors": counts["n_donors"],
            })
    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    log.info(f"Cohort summary written to {out_path}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Download and QC sc-ImmuAging cohorts")
    parser.add_argument("--out-dir", default="data/cohorts", help="Output directory root")
    parser.add_argument("--repo-dir", default="data/scImmuAging", help="scImmuAging repo path")
    parser.add_argument("--skip-clone", action="store_true",
                        help="Skip cloning scImmuAging (repo already present)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip GEO downloads (already done)")
    parser.add_argument("--summary-only", action="store_true",
                        help="Only rebuild cohort_summary.csv from existing h5ad files")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    repo_dir = Path(args.repo_dir)

    # Step 1: Clone pipeline
    if not args.skip_clone:
        clone_scimmuaging(repo_dir.parent)

    # Step 2: Download GEO cohorts
    if not args.skip_download:
        for cohort in COHORTS:
            cid = cohort["id"]
            if cohort["source"] == "GEO":
                log.info(f"Downloading GEO cohort {cid}")
                download_geo_soft(cid, out_dir / "geo_soft" / cid)
            elif cohort["source"] == "CellPortal":
                log.warning(
                    f"Cohort {cid} is from CellxGene portal. "
                    f"Download manually from: {cohort.get('url', 'see LANDSCAPE.md')}"
                )

    # Step 3: Harmonization pipeline — see src/data/harmonize_cohorts.py (written
    # separately). The scImmuAging R repo is an inference-only package and does
    # not provide an integration script; harmonization is implemented in scanpy
    # from scratch against the three confirmed cohort sources.

    # Step 4: Build cohort summary from integrated h5ads
    import anndata as ad

    summaries = {}
    integrated_dir = out_dir / "integrated"
    if not integrated_dir.exists():
        log.error(
            "Integrated data directory not found. Run full pipeline first."
        )
        return

    for h5ad_path in sorted(integrated_dir.glob("*.h5ad")):
        cohort_id = h5ad_path.stem
        log.info(f"Validating {cohort_id}")
        adata = ad.read_h5ad(h5ad_path)
        try:
            summary = validate_cohort_counts(adata, cohort_id)
            summaries[cohort_id] = summary
        except ValueError as e:
            log.error(f"Validation failed for {cohort_id}: {e}")

    build_cohort_summary(summaries, Path("data/cohort_summary.csv"))
    log.info("Done.")


if __name__ == "__main__":
    main()
