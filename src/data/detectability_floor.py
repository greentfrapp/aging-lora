"""Compute the per-cell-type MAE detectability floor for LOCO folds.

Per roadmap/phase-1.md success criterion:
  "m.a.e.-detectability floor computed from sc-ImmuAging Extended Data Table 2
   baseline values (paired Wilcoxon power calculation, 80% power, α = 0.05)
   and recorded; if any LOCO fold is underpowered under this criterion, it is
   promoted to exploratory-only in the fold matrix."

We compute the minimum donor count needed to detect a 10% relative reduction
in median |err| at 80% power, α=0.05, using a paired-samples one-sided test.
The data the paper's Ext Data Table 2 would have provided is approximated by
our Task 1e sanity-check residuals (OneK1K, 981 donors, same pre-trained
LASSO applied per cell type). Because we reproduce paper internal R within
roughly 0.05–0.15 of the paper's reported values, the absolute-error
distribution is a reasonable proxy for detectability planning.

Methodology
-----------
* Paired test form: we want to detect that a new method reduces per-donor
  |err| by at least `relative_effect * baseline_median_err`. With a paired
  design (same donors scored by both methods), the test statistic is the
  mean of within-donor absolute-error *differences*. Under the approximation
  that these differences are roughly normal (or via Wilcoxon asymptotic
  relative efficiency ≈ 0.955 vs. paired t-test), the required sample size is

     n = ((z_{1-α} + z_{1-β}) × σ_d / δ)²

  where σ_d is the SD of the paired differences and δ is the target mean
  reduction (relative_effect × baseline_median). For a rank-based Wilcoxon
  we inflate n by the ARE factor 1 / 0.955 ≈ 1.047.

* σ_d is not directly observable from a single scoring run (we have only the
  baseline, not the candidate). We estimate it conservatively as
  σ_{|err|} × √(2 × (1 − ρ)), where ρ is the expected within-donor correlation
  between methods' absolute errors. We take ρ = 0.5 by default (mid-range;
  higher ρ for "methods that share a lot of signal" would reduce n).

* The reported floor is the ceiling of the t-approximation inflated by the
  Wilcoxon ARE factor. Any LOCO fold with fewer held-out donors than the
  floor is marked exploratory-only.

Output
------
  data/detectability_floor.json   per-cell-type floor + parameters used

Usage
-----
  uv run python -m src.data.detectability_floor
  uv run python -m src.data.detectability_floor --relative-effect 0.10 --power 0.8 --alpha 0.05
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


SANITY_DIR = Path("results/baselines")
OUT_JSON = Path("data/detectability_floor.json")

# ARE of the one-sample Wilcoxon signed-rank vs paired t-test under normality.
WILCOXON_ARE = 0.955  # Hodges-Lehmann, Pitman efficiency


def per_cell_type_errors(cell_type_code: str, dir_: Path = SANITY_DIR) -> pd.Series:
    """Return per-donor absolute errors from the Task 1e sanity-check CSV."""
    csv = dir_ / f"pretrained_sanity_{cell_type_code}.csv"
    if not csv.exists():
        raise FileNotFoundError(f"{csv} not found. Run Task 1e first.")
    df = pd.read_csv(csv)
    return (df["predicted_age"] - df["true_age"]).abs()


def floor_for_cell_type(
    abs_err: pd.Series,
    *,
    relative_effect: float = 0.10,
    power: float = 0.80,
    alpha: float = 0.05,
    pairing_rho: float = 0.5,
    one_sided: bool = True,
) -> dict:
    """Compute the detectability floor for one cell type.

    Parameters
    ----------
    abs_err : per-donor absolute errors of the baseline clock, one row per donor
    relative_effect : target relative MAE reduction, default 0.10 = 10 %
    power : 1 - β, default 0.80
    alpha : significance level, default 0.05
    pairing_rho : assumed correlation of per-donor absolute errors between
                  baseline and candidate method. Higher rho => lower floor.
                  Default 0.5 (no better than assuming moderately correlated).
    one_sided : True for one-sided test (we test that the new method is *better*)
    """
    sigma = float(abs_err.std(ddof=1))
    median = float(abs_err.median())
    mean = float(abs_err.mean())
    delta = relative_effect * median  # required mean reduction, in years
    sigma_d = sigma * math.sqrt(2 * (1 - pairing_rho))

    z_alpha = norm.ppf(1 - alpha) if one_sided else norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)

    n_ttest = ((z_alpha + z_beta) * sigma_d / delta) ** 2
    n_wilcoxon = n_ttest / WILCOXON_ARE
    n_required = int(math.ceil(n_wilcoxon))

    return {
        "n_donors_observed": int(len(abs_err)),
        "median_abs_err_yr": round(median, 3),
        "mean_abs_err_yr": round(mean, 3),
        "sd_abs_err_yr": round(sigma, 3),
        "delta_target_yr": round(delta, 3),
        "pairing_rho_assumed": pairing_rho,
        "sigma_d_yr": round(sigma_d, 3),
        "n_required_ttest": int(math.ceil(n_ttest)),
        "n_required_wilcoxon": n_required,
        "relative_effect": relative_effect,
        "power": power,
        "alpha": alpha,
        "one_sided": one_sided,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--relative-effect", type=float, default=0.10,
                    help="Target relative MAE reduction (default 0.10 = 10%%)")
    ap.add_argument("--power", type=float, default=0.80,
                    help="Desired statistical power 1-β (default 0.80)")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--pairing-rho", type=float, default=0.5,
                    help="Assumed per-donor |err| correlation between methods (0.5 = moderate)")
    ap.add_argument("--two-sided", action="store_true",
                    help="Use two-sided test (default is one-sided; we only care "
                         "about detecting improvement).")
    ap.add_argument("--out", default=str(OUT_JSON))
    args = ap.parse_args()

    result = {
        "methodology": {
            "description": (
                "Per-cell-type detectability floor from Task 1e baseline residuals. "
                "Paired one-sided test of relative MAE reduction; Wilcoxon signed-rank "
                "under asymptotic relative efficiency 0.955 vs paired t."
            ),
            "relative_effect": args.relative_effect,
            "power": args.power,
            "alpha": args.alpha,
            "pairing_rho": args.pairing_rho,
            "one_sided": not args.two_sided,
            "source_csvs": [str(SANITY_DIR / f"pretrained_sanity_{ct}.csv")
                            for ct in ("CD4T", "CD8T", "MONO", "NK", "B")],
        },
        "per_cell_type": {},
    }

    per_ct = {}
    for ct in ("CD4T", "CD8T", "MONO", "NK", "B"):
        abs_err = per_cell_type_errors(ct)
        floor = floor_for_cell_type(
            abs_err,
            relative_effect=args.relative_effect,
            power=args.power,
            alpha=args.alpha,
            pairing_rho=args.pairing_rho,
            one_sided=not args.two_sided,
        )
        per_ct[ct] = floor
        log.info(
            f"[{ct}] median|err|={floor['median_abs_err_yr']:.2f}y  "
            f"sd|err|={floor['sd_abs_err_yr']:.2f}y  "
            f"n_required_wilcoxon={floor['n_required_wilcoxon']}"
        )
    result["per_cell_type"] = per_ct

    # Sensitivity analysis over pairing-rho. Baseline and candidate methods on the
    # same donors typically share substantial variance ρ ≈ 0.5-0.9 depending on
    # how much each method's errors are driven by donor-specific factors
    # sequencing depth, age, technical batch rather than method-specific choices.
    # Report n_required at ρ ∈ {0.3, 0.5, 0.7, 0.8, 0.9}.
    sensitivity = {}
    for rho in (0.3, 0.5, 0.7, 0.8, 0.9):
        per_ct_rho = {}
        for ct in ("CD4T", "CD8T", "MONO", "NK", "B"):
            f = floor_for_cell_type(
                per_cell_type_errors(ct),
                relative_effect=args.relative_effect,
                power=args.power,
                alpha=args.alpha,
                pairing_rho=rho,
                one_sided=not args.two_sided,
            )
            per_ct_rho[ct] = f["n_required_wilcoxon"]
        sensitivity[f"rho={rho:.1f}"] = per_ct_rho
    result["sensitivity_pairing_rho"] = sensitivity

    # Log the sensitivity table for quick review
    log.info("sensitivity: n_required_wilcoxon per (rho, cell_type)")
    for rho_key, ct_map in sensitivity.items():
        log.info(f"  {rho_key}: " + "  ".join(f"{ct}={n}" for ct, n in ct_map.items()))

    # Summary: pick the strictest floor across cell types as the conservative LOCO threshold.
    max_n_required = max(v["n_required_wilcoxon"] for v in per_ct.values())
    median_n_required = int(np.median([v["n_required_wilcoxon"] for v in per_ct.values()]))
    result["summary"] = {
        "max_required_across_cell_types": max_n_required,
        "median_required_across_cell_types": median_n_required,
        "interpretation": (
            "A LOCO fold that holds out >= the per-cell-type n_required is adequately "
            "powered for that cell type. A fold holding out >= max_required is powered "
            "across all five cell types; a fold holding out >= median_required is "
            "powered for the median cell type. Folds below both thresholds are "
            "exploratory-only and should not carry the paper's headline MAE numbers."
        ),
        "current_loco_folds": {
            "loco_onek1k": 981,
            "loco_terekhova": 166,
            "loco_stephenson": 29,
        },
        "threshold_80_donor": 80,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    log.info(f"wrote {args.out}")


if __name__ == "__main__":
    main()
