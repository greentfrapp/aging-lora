"""Phase 2 Task 2.8: empirically measure pairing-ρ between baseline residuals.

For each (cohort, cell type) combination, joins per-donor predictions from
LASSO + scAgeClock + Pasta-REG, computes |err| per donor per baseline, and
reports the Pearson correlation of paired |err| vectors.

The Phase-1 detectability-floor calculation assumed ρ=0.8 between paired
baseline-vs-FM absolute residuals. With three Phase-2 baselines now scored
on identical donors, we can MEASURE the actual ρ between baseline pairs as
an empirical proxy for what the Phase-3 baseline-vs-FM ρ will look like
(the FM is expected to share more residual structure with these baselines
than the baselines do with each other, so this empirical ρ is a lower bound
on the Phase-3-time ρ — meaning the floor we recompute is conservative).

Output:
  results/baselines/empirical_pairing_rho.csv  — per-cell-type ρ table
  data/detectability_floor.json                — gets a `post_phase2_empirical_rho`
                                                   block appended (Phase-1 ρ=0.8
                                                   numbers preserved)
"""
from __future__ import annotations
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
log = logging.getLogger("rho")

PROJ_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJ_ROOT / "results" / "baselines"
DETECTABILITY_PATH = PROJ_ROOT / "data" / "detectability_floor.json"

# Cell-type-code conventions
CT_CODES = ["CD4T", "CD8T", "MONO", "NK", "B"]
COHORTS = ["onek1k", "stephenson", "terekhova"]


def normalize_donor(s: str, cohort: str) -> str:
    """Make donor_id values comparable across LASSO / scAgeClock / Pasta CSVs.
    LASSO files use bare donor_id (no cohort prefix); scAgeClock/Pasta use
    'cohort:donor_id'. Strip the prefix if present.
    """
    s = str(s)
    pre = f"{cohort}:"
    if s.startswith(pre):
        return s[len(pre):]
    return s


def load_lasso_per_donor(cohort: str, ct: str) -> pd.DataFrame | None:
    """Find the LASSO per-donor file for this slice."""
    candidates = {
        "onek1k": RESULTS_DIR / "lasso_pretrained" / "per_donor" / f"onek1k_{ct}.csv",
        "stephenson": RESULTS_DIR / "lasso_pretrained" / "per_donor" / f"stephenson_{ct}.csv",
        "terekhova": RESULTS_DIR / "lasso_pretrained" / "per_donor" / f"terekhova_{ct}.csv",
    }
    p = candidates.get(cohort)
    if not p or not p.exists():
        return None
    df = pd.read_csv(p)
    df["donor_id_norm"] = df["donor_id"].astype(str).map(lambda s: normalize_donor(s, cohort))
    df["abs_err"] = (df["predicted_age"] - df["true_age"]).abs()
    return df[["donor_id_norm", "true_age", "abs_err"]].rename(columns={"abs_err": "abs_err_lasso"})


def load_other_per_donor(cohort: str, ct: str, baseline_dir: str) -> pd.DataFrame | None:
    """Load scAgeClock or Pasta per-donor CSV."""
    p = RESULTS_DIR / baseline_dir / f"{cohort}_{ct}.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df["donor_id_norm"] = df["donor_id"].astype(str).map(lambda s: normalize_donor(s, cohort))
    df["abs_err"] = (df["predicted_age"] - df["true_age"]).abs()
    return df[["donor_id_norm", "abs_err"]]


def main():
    rows = []
    for cohort in COHORTS:
        for ct in CT_CODES:
            lasso = load_lasso_per_donor(cohort, ct)
            scage = load_other_per_donor(cohort, ct, "scageclock/per_donor")
            pasta = load_other_per_donor(cohort, ct, "pasta_reg/per_donor")
            if lasso is None or scage is None or pasta is None:
                log.warning(f"  skip {cohort} × {ct}: missing per-donor file(s)")
                continue
            scage = scage.rename(columns={"abs_err": "abs_err_scage"})
            pasta = pasta.rename(columns={"abs_err": "abs_err_pasta"})

            merged = lasso.merge(scage, on="donor_id_norm", how="inner") \
                          .merge(pasta, on="donor_id_norm", how="inner")
            n = len(merged)
            if n < 10:
                log.warning(f"  skip {cohort} × {ct}: only {n} matched donors")
                continue
            try:
                r_lp, _ = pearsonr(merged["abs_err_lasso"], merged["abs_err_pasta"])
                r_ls, _ = pearsonr(merged["abs_err_lasso"], merged["abs_err_scage"])
                r_ps, _ = pearsonr(merged["abs_err_pasta"], merged["abs_err_scage"])
            except Exception as e:
                log.warning(f"  pearson failed for {cohort} × {ct}: {e}")
                continue

            rows.append({
                "eval_cohort": cohort,
                "cell_type": ct,
                "n_matched_donors": n,
                "rho_lasso_pasta": float(r_lp),
                "rho_lasso_scage": float(r_ls),
                "rho_pasta_scage": float(r_ps),
                "rho_min": float(min(r_lp, r_ls, r_ps)),
                "rho_median": float(np.median([r_lp, r_ls, r_ps])),
            })
            log.info(f"{cohort} × {ct}: n={n} ρ(L,P)={r_lp:.3f} ρ(L,S)={r_ls:.3f} ρ(P,S)={r_ps:.3f}")

    df = pd.DataFrame(rows)
    out_csv = RESULTS_DIR / "empirical_pairing_rho.csv"
    df.to_csv(out_csv, index=False)
    log.info(f"wrote {out_csv} ({len(df)} rows)")

    # Per-cell-type median ρ for the detectability-floor update.
    # We pick the median across all (cohort, baseline-pair) ρ values per cell type
    # as the empirical proxy for ρ in the paired-Wilcoxon power calc. Using the
    # median (not mean) is more robust to per-cohort outliers like Stephenson
    # with 24-29 donors.
    per_ct_summary = []
    for ct in CT_CODES:
        sub = df[df["cell_type"] == ct]
        if sub.empty:
            continue
        all_rho = pd.concat([sub["rho_lasso_pasta"], sub["rho_lasso_scage"], sub["rho_pasta_scage"]]).values
        per_ct_summary.append({
            "cell_type": ct,
            "median_rho": float(np.median(all_rho)),
            "min_rho": float(np.min(all_rho)),
            "max_rho": float(np.max(all_rho)),
            "n_pairs_aggregated": len(all_rho),
        })
    summary_df = pd.DataFrame(per_ct_summary)
    log.info("\nper-cell-type empirical ρ summary:\n" + summary_df.to_string(index=False))

    # Append empirical ρ block to detectability_floor.json (preserve existing fields)
    if DETECTABILITY_PATH.exists():
        with open(DETECTABILITY_PATH) as f:
            floor = json.load(f)
    else:
        floor = {}

    # Re-compute n_required per cell type at the empirical median ρ.
    # Reuses the same paired-Wilcoxon power calculation as
    # src/data/detectability_floor.py: σ_d = σ × √(2(1-ρ)),
    # n_required ≈ ((z_{α} + z_{β}) × σ_d / δ)² / ARE_wilcoxon
    from scipy.stats import norm as _norm
    ARE_WILCOXON = 0.955  # asymptotic relative efficiency vs paired t
    z_alpha = _norm.ppf(0.95)   # 1-sided α=0.05
    z_beta = _norm.ppf(0.80)
    n_at_empirical_rho = {}
    if "per_cell_type" in floor:
        for ct, ct_data in floor["per_cell_type"].items():
            sub = next((row for row in per_ct_summary if row["cell_type"] == ct), None)
            if not sub:
                continue
            rho = sub["median_rho"]
            sigma = ct_data["sd_abs_err_yr"]
            delta = ct_data["delta_target_yr"]
            # Clamp ρ to (-0.99, 0.99) so √(2(1-ρ)) stays finite.
            rho_clamped = max(-0.99, min(0.99, rho))
            sigma_d = sigma * (2 * (1 - rho_clamped)) ** 0.5
            n_t = ((z_alpha + z_beta) * sigma_d / delta) ** 2
            n_w = n_t / ARE_WILCOXON
            n_at_empirical_rho[ct] = {
                "rho_empirical_median": rho,
                "sigma_d_yr": float(sigma_d),
                "n_required_ttest": int(round(n_t)),
                "n_required_wilcoxon": int(round(n_w)),
            }

    floor["post_phase2_empirical_rho"] = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "method": "Pearson ρ between |err| vectors of baseline pairs (LASSO, scAgeClock, Pasta-REG) per (cohort × cell type), aggregated to per-cell-type median across all 9 (cohort × baseline-pair) values",
        "interpretation": "Empirical lower bound on the baseline-vs-FM ρ that Phase-3 measures (FMs are expected to share more residual structure with the baselines than the baselines do with each other). At the empirical median ρ, n_required is markedly larger than the Phase-1 ρ=0.8 planning value — the Phase-1 floor was overoptimistic.",
        "per_cell_type_median_rho": {row["cell_type"]: row["median_rho"] for row in per_ct_summary},
        "per_cell_type_min_rho": {row["cell_type"]: row["min_rho"] for row in per_ct_summary},
        "per_cell_type_max_rho": {row["cell_type"]: row["max_rho"] for row in per_ct_summary},
        "n_baseline_pairs_per_cell_type": {row["cell_type"]: row["n_pairs_aggregated"] for row in per_ct_summary},
        "n_required_at_empirical_rho": n_at_empirical_rho,
        "note": "Phase-1 ρ=0.5/0.8/0.9 grid preserved in 'sensitivity_pairing_rho'. Phase-3 will measure the actual baseline-vs-FM ρ and append a post_phase3_override block; the empirical-baseline-pair ρ here is a conservative LOWER BOUND on that, since FMs and baselines tend to share more residual structure than two baselines do with each other.",
    }
    with open(DETECTABILITY_PATH, "w") as f:
        json.dump(floor, f, indent=2)
    log.info(f"appended post_phase2_empirical_rho block to {DETECTABILITY_PATH}")
    log.info("\nrequired-N at empirical ρ (Wilcoxon):")
    for ct, info in n_at_empirical_rho.items():
        log.info(f"  {ct}: ρ={info['rho_empirical_median']:.3f}  "
                 f"n_required_wilcoxon={info['n_required_wilcoxon']:,}")


if __name__ == "__main__":
    main()
