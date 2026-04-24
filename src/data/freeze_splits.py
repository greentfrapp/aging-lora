"""Freeze LOCO fold assignments and the AIDA 50/50 stratified split.

Inputs (read at run time)
-------------------------
    data/cohort_summary.csv                    - produced by harmonize_cohorts.py
    data/cohorts/integrated/{cell_type}.h5ad   - per-cell-type harmonized AnnData
    data/cohorts/raw/aida/*.h5ad               - raw AIDA CellxGene h5ad

Outputs (immutable once written)
--------------------------------
    data/loco_folds.json
    data/aida_split.json

LOCO fold rules (mirrors success criteria in roadmap/phase-1.md):
  - One fold per training cohort (leave-one-cohort-out).
  - An additional leave-one-chemistry-out fold: train on 10x 3' cohorts,
    test on 10x 5' cohorts. With the current three-cohort setup this is
    identical to the Terekhova LOCO fold, so we mark it as an alias.
  - Folds with < 80 held-out donors are flagged `primary: false`.
  - OneK1K's fold is annotated "training-set recapitulation" because
    OneK1K was in the sc-ImmuAging authors' training set — scoring the
    pre-trained LASSO against it is NOT a clean generalization test.

AIDA split rules:
  - 50/50 stratified by (age_decile x self_reported_ethnicity) bins.
  - Half labelled `ancestry_shift_mae` — used as a true holdout evaluation set.
  - Other half labelled `age_axis_alignment` — used for Phase 5 age-axis
    cosine-similarity analysis.

Usage
-----
    uv run python -m src.data.freeze_splits

The script refuses to overwrite existing output files (the assumption is that
once frozen, these files are immutable and any re-run is a mistake). Delete
the output and re-run if intentional.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

COHORT_SUMMARY = Path("data/cohort_summary.csv")
INTEGRATED_DIR = Path("data/cohorts/integrated")
LOCO_OUT = Path("data/loco_folds.json")

AIDA_DIR = Path("data/cohorts/raw/aida")
AIDA_SPLIT_OUT = Path("data/aida_split.json")
DETECT_FLOOR = Path("data/detectability_floor.json")

DONOR_THRESHOLD = 80     # minimum for *any* primary claim, regardless of power
DETECT_RHO_ASSUMED = 0.8  # paired-error correlation we plan around; sensitivity in detectability_floor.json

# Known cohort -> chemistry mapping (populated during harmonization; we replicate here).
COHORT_CHEMISTRY = {
    "onek1k":     "10x 3' v2",
    "stephenson": "10x 3' transcription profiling",
    "terekhova":  "10x 5' v2",
}

# Notes that stick with each cohort's fold — training-set asymmetry and caveats.
COHORT_NOTES = {
    "onek1k": (
        "OneK1K (GSE196830) was in the sc-ImmuAging authors' training set. "
        "Pre-trained LASSO scoring against the OneK1K LOCO fold reflects "
        "training-set recapitulation, not generalization. Phase 2 reports this "
        "fold with a `training_set_asymmetry` flag; treat with care."
    ),
    "stephenson": (
        "Stephenson 2021 COVID-19 Cell Atlas, healthy controls only (29 donors). "
        "Below the 80-donor primary threshold. 18/29 donors carry decade-precision "
        "age labels (midpoint + age_precision='decade' obs column)."
    ),
    "terekhova": (
        "Terekhova 2023 Immunity (166 donors). True holdout — NOT in sc-ImmuAging "
        "training. 10x 5' v2 chemistry vs. training 3'; report naive and "
        "chemistry-corrected MAE per Task 1f."
    ),
}


# ---------------------------------------------------------------------------
# LOCO folds
# ---------------------------------------------------------------------------
def _load_donor_inventory(integrated_dir: Path) -> pd.DataFrame:
    """Walk per-cell-type integrated h5ads and return an (cohort, donor, cell_type, n_cells) table."""
    rows = []
    for p in sorted(integrated_dir.glob("*.h5ad")):
        log.info(f"reading {p}")
        a = ad.read_h5ad(p, backed="r")
        obs = a.obs
        grp = obs.groupby(["cohort_id", "donor_id"], observed=True).size().reset_index(name="n_cells")
        grp["cell_type"] = p.stem.replace("_", " ").replace("p", "+")
        rows.append(grp)
    inventory = pd.concat(rows, ignore_index=True)
    return inventory


def _load_detectability_floor(path: Path = DETECT_FLOOR) -> dict:
    """Return {cell_type: n_required} at the planned pairing_rho, if available."""
    if not path.exists():
        log.warning(f"{path} not found — run src.data.detectability_floor first. "
                    "Fold 'primary_per_cell_type' flags will fall back to the 80-donor heuristic.")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    key = f"rho={DETECT_RHO_ASSUMED:.1f}"
    if key in d.get("sensitivity_pairing_rho", {}):
        return d["sensitivity_pairing_rho"][key]
    # fall back to the default rho in per_cell_type
    return {ct: v["n_required_wilcoxon"] for ct, v in d.get("per_cell_type", {}).items()}


def build_loco_folds(integrated_dir: Path = INTEGRATED_DIR) -> dict:
    """Enumerate LOCO folds across all training cohorts and the chemistry fold."""
    if not integrated_dir.exists():
        raise FileNotFoundError(
            f"{integrated_dir} not found. Run src.data.harmonize_cohorts first."
        )

    inventory = _load_donor_inventory(integrated_dir)
    # Per-cohort donor counts (unique donors across all cell types)
    donors_per_cohort = (
        inventory.groupby("cohort_id")["donor_id"].nunique().to_dict()
    )
    # Per-cohort per-cell-type donor counts, used for per-cell-type primary flags.
    per_cohort_ct_donors = (
        inventory.groupby(["cohort_id", "cell_type"])["donor_id"]
                 .nunique().unstack(fill_value=0).to_dict(orient="index")
    )

    cohorts = sorted(donors_per_cohort.keys())
    log.info(f"cohorts found: {cohorts} with donor counts {donors_per_cohort}")

    # Pull the detectability floor (may be empty if not yet computed)
    floor = _load_detectability_floor()
    # Map canonical names to scImmuAging-style codes used by detectability_floor.
    ct_to_code = {"CD4+ T": "CD4T", "CD8+ T": "CD8T", "Monocyte": "MONO", "NK": "NK", "B": "B"}

    folds = []
    for holdout in cohorts:
        train = [c for c in cohorts if c != holdout]
        n_holdout = int(donors_per_cohort[holdout])
        is_primary = n_holdout >= DONOR_THRESHOLD
        # Per-cell-type primary flag: the held-out cell-type-level donor count
        # must meet both the 80-donor heuristic AND the statistical power floor.
        primary_per_ct = {}
        for canonical_ct, code in ct_to_code.items():
            n_holdout_ct = int(per_cohort_ct_donors.get(holdout, {}).get(canonical_ct, 0))
            required = floor.get(code, DONOR_THRESHOLD)
            primary_per_ct[canonical_ct] = {
                "n_holdout_donors": n_holdout_ct,
                "n_required": int(required),
                "primary": bool(n_holdout_ct >= DONOR_THRESHOLD and n_holdout_ct >= required),
            }
        fold = {
            "fold_id": f"loco_{holdout}",
            "kind": "leave-one-cohort-out",
            "holdout_cohort": holdout,
            "holdout_chemistry": COHORT_CHEMISTRY.get(holdout, "unknown"),
            "train_cohorts": train,
            "n_holdout_donors": n_holdout,
            "primary": bool(is_primary),
            "donor_threshold": DONOR_THRESHOLD,
            "primary_per_cell_type": primary_per_ct,
            "notes": COHORT_NOTES.get(holdout, ""),
        }
        folds.append(fold)

    # Chemistry fold: hold out all 10x 5' cohorts, train on 10x 3'.
    chemistry_5prime = [c for c in cohorts if "5'" in COHORT_CHEMISTRY.get(c, "")]
    chemistry_3prime = [c for c in cohorts if "5'" not in COHORT_CHEMISTRY.get(c, "")]
    n_holdout_chem = int(sum(donors_per_cohort[c] for c in chemistry_5prime))
    folds.append({
        "fold_id": "chemistry_5prime",
        "kind": "leave-one-chemistry-out",
        "holdout_cohorts": chemistry_5prime,
        "train_cohorts": chemistry_3prime,
        "n_holdout_donors": n_holdout_chem,
        "primary": bool(n_holdout_chem >= DONOR_THRESHOLD),
        "donor_threshold": DONOR_THRESHOLD,
        "notes": (
            "Leave-one-chemistry-out: train on 3', test on 5'. With the current "
            "three-cohort setup this is identical to loco_terekhova; kept as a "
            "separate fold so that future cohort additions don't silently change "
            "meaning. Compare directly against loco_terekhova MAE — they must match."
        ),
    })

    out = {
        "version": 1,
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "cohort_donor_counts": donors_per_cohort,
        "cohort_chemistry": {c: COHORT_CHEMISTRY.get(c, "unknown") for c in cohorts},
        "donor_threshold": DONOR_THRESHOLD,
        "folds": folds,
        "per_cell_type_donor_inventory": (
            inventory.groupby(["cohort_id", "cell_type"])["donor_id"].nunique()
                     .reset_index(name="n_donors").to_dict(orient="records")
        ),
    }
    return out


# ---------------------------------------------------------------------------
# AIDA split
# ---------------------------------------------------------------------------
def _discover_aida_h5ad(aida_dir: Path) -> Path:
    candidates = sorted(aida_dir.glob("*.h5ad"))
    if not candidates:
        raise FileNotFoundError(
            f"{aida_dir} has no h5ad — run the AIDA download first."
        )
    if len(candidates) > 1:
        candidates.sort(key=lambda p: p.stat().st_size, reverse=True)
        log.warning(f"multiple AIDA h5ads; using largest: {candidates[0].name}")
    return candidates[0]


def build_aida_split(
    aida_dir: Path = AIDA_DIR,
    seed: int = 0,
) -> dict:
    """Split AIDA donors 50/50, stratified by (age decile x self_reported_ethnicity)."""
    h5ad_path = _discover_aida_h5ad(aida_dir)
    log.info(f"reading {h5ad_path} (backed)")
    a = ad.read_h5ad(h5ad_path, backed="r")
    obs = a.obs

    # Age — AIDA uses CellxGene's development_stage encoding; we parse year form.
    age_col = None
    for c in ("age", "age_years", "Age"):
        if c in obs.columns:
            age_col = c
            break
    if age_col is None:
        # Parse from development_stage "42-year-old stage"
        import re
        _AGE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*-?\s*year", flags=re.IGNORECASE)
        ds = obs["development_stage"].astype(str)
        ages = pd.to_numeric(ds.str.extract(_AGE_RE, expand=False), errors="coerce")
        log.info(f"parsed AIDA age from development_stage; {ages.notna().sum():,}/{len(ages)} cells have ages")
    else:
        ages = pd.to_numeric(obs[age_col], errors="coerce")

    eth_col = None
    for c in ("self_reported_ethnicity", "ethnicity", "ancestry"):
        if c in obs.columns:
            eth_col = c
            break
    if eth_col is None:
        raise RuntimeError(f"AIDA obs missing ethnicity column; have {list(obs.columns)}")

    donor_col = next(c for c in ("donor_id", "donor") if c in obs.columns)

    donor_df = (
        pd.DataFrame({
            "donor_id": obs[donor_col].astype(str).values,
            "age": ages.astype(float).values,
            "ethnicity": obs[eth_col].astype(str).values,
        })
        .dropna(subset=["age", "ethnicity"])
        .drop_duplicates(subset="donor_id")
        .reset_index(drop=True)
    )
    log.info(f"AIDA unique donors with age + ethnicity: {len(donor_df):,}")

    # Stratify: (age_decile x ethnicity)
    donor_df["age_decile"] = (donor_df["age"] // 10 * 10).astype(int)
    donor_df["stratum"] = donor_df["age_decile"].astype(str) + "|" + donor_df["ethnicity"]

    rng = np.random.default_rng(seed)
    donor_df["half"] = ""
    for stratum, grp in donor_df.groupby("stratum"):
        perm = grp.sample(frac=1.0, random_state=rng.integers(0, 2**31)).reset_index(drop=True)
        half_point = len(perm) // 2
        if len(perm) == 1:
            # Single-donor stratum — random-assign with a coin flip.
            donor_df.loc[perm.index[0], "half"] = (
                "ancestry_shift_mae" if rng.integers(0, 2) else "age_axis_alignment"
            )
            continue
        halves = ["ancestry_shift_mae"] * half_point + ["age_axis_alignment"] * (len(perm) - half_point)
        rng.shuffle(halves)
        for i, h in zip(perm.index, halves):
            donor_df.loc[i, "half"] = h

    counts = donor_df["half"].value_counts().to_dict()
    log.info(f"AIDA split counts: {counts}")

    ancestry_mae_donors = sorted(donor_df.loc[donor_df["half"] == "ancestry_shift_mae", "donor_id"].tolist())
    age_axis_donors = sorted(donor_df.loc[donor_df["half"] == "age_axis_alignment", "donor_id"].tolist())

    # Per-stratum balance summary
    balance = (
        donor_df.groupby(["stratum", "half"]).size().unstack(fill_value=0).reset_index()
    ).to_dict(orient="records")

    out = {
        "version": 1,
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "source_h5ad": str(h5ad_path),
        "stratification": "age_decile x self_reported_ethnicity",
        "seed": seed,
        "n_donors_total": int(len(donor_df)),
        "n_donors_ancestry_shift_mae": len(ancestry_mae_donors),
        "n_donors_age_axis_alignment": len(age_axis_donors),
        "ancestry_shift_mae_donors": ancestry_mae_donors,
        "age_axis_alignment_donors": age_axis_donors,
        "stratum_balance": balance,
    }
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def _write_immutable(path: Path, obj: dict, force: bool) -> None:
    if path.exists() and not force:
        raise SystemExit(
            f"{path} already exists; split files are treated as immutable. "
            f"Pass --force to overwrite (only if you are sure)."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
    log.info(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--skip-loco", action="store_true")
    ap.add_argument("--skip-aida", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing split files (dangerous — they're meant to be immutable).")
    ap.add_argument("--aida-seed", type=int, default=0)
    args = ap.parse_args()

    if not args.skip_loco:
        folds = build_loco_folds()
        _write_immutable(LOCO_OUT, folds, args.force)

    if not args.skip_aida:
        split = build_aida_split(seed=args.aida_seed)
        _write_immutable(AIDA_SPLIT_OUT, split, args.force)

    log.info("done.")


if __name__ == "__main__":
    main()
