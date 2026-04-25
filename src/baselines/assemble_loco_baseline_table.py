"""Assemble results/baselines/loco_baseline_table.csv from the three per-cohort
per-cell-type pre-trained LASSO summaries (Task 1e OneK1K, Task 1f Terekhova,
Phase 2 Task 1 Stephenson).

Adds the three Phase-4 stratification columns so downstream aggregation can
filter on (leakage_status, chemistry_match_to_baseline_training, detectability_flag)
without touching the individual score files. See roadmap/phase-4.md for the
stratification policy.
"""
from pathlib import Path
import pandas as pd

RESULTS_DIR = Path("results/baselines")

SOURCES = {
    "onek1k": RESULTS_DIR / "pretrained_sanity_summary.csv",
    "stephenson": RESULTS_DIR / "stephenson_loco_summary.csv",
    "terekhova": RESULTS_DIR / "terekhova_chemistry_shift_naive.csv",
}

SCAGECLOCK_SUMMARY = RESULTS_DIR / "scageclock_loco_summary.csv"
PASTA_SUMMARY = RESULTS_DIR / "pasta_loco_summary.csv"

# Cohort -> chemistry (10x 3' vs 5'). LASSO was trained on 3', so 3' = match.
CHEMISTRY_BY_COHORT = {
    "onek1k":     ("10x 3' v2",  "match"),
    "stephenson": ("10x 3'",     "match"),
    "terekhova":  ("10x 5' v2",  "shifted"),
}

# LOCO detectability flags at ρ=0.8 from data/detectability_floor.json +
# data/loco_folds.json (Phase 1 primary/exploratory flags).
# loco_onek1k: 981 donors -> all 5 powered.
# loco_stephenson: 29 donors -> all exploratory regardless.
# loco_terekhova: 166 donors -> CD4T (132 floor) powered; NK (156) powered;
#                 B (155) powered; CD8T (180) under; MONO (229) under.
DETECTABILITY = {
    "onek1k":     {"CD4T": "powered", "CD8T": "powered", "MONO": "powered", "NK": "powered", "B": "powered"},
    "stephenson": {"CD4T": "underpowered", "CD8T": "underpowered", "MONO": "underpowered", "NK": "underpowered", "B": "underpowered"},
    "terekhova":  {"CD4T": "powered", "CD8T": "underpowered", "MONO": "underpowered", "NK": "powered", "B": "powered"},
}

# pre-trained LASSO is cohort-external (trained on the paper's 5 original cohorts,
# not on ours), so `leakage_status` is always `clean` for this baseline row.
# The leakage audit in data/leakage_audit.csv covers foundation models only.


# scAgeClock leakage from data/leakage_audit.csv (recorded 2026-04-25)
SCAGECLOCK_LEAKAGE = {"onek1k": "overlapping", "stephenson": "overlapping", "terekhova": "clean"}

# Pasta is trained on bulk transcriptomics (microarray/bulk RNA-seq from public GEO
# datasets per Salignon 2025); does not ingest any single-cell PBMC cohort, so all
# our single-cell cohorts are leakage-`clean` for Pasta. Chemistry-match doesn't
# apply directly (different modality entirely), but we record it as `bulk-vs-sc`
# so the stratification column is non-empty for downstream filtering.
PASTA_LEAKAGE = {"onek1k": "clean", "stephenson": "clean", "terekhova": "clean"}


def main():
    rows = []
    # ---- LASSO (sc-ImmuAging pretrained) ----
    for cohort, src in SOURCES.items():
        df = pd.read_csv(src)
        chemistry, chem_match = CHEMISTRY_BY_COHORT[cohort]
        for _, row in df.iterrows():
            ct = row["cell_type"]
            rows.append({
                "baseline": "scImmuAging-LASSO",
                "training_cohorts": "original-five",
                "eval_cohort": cohort,
                "eval_chemistry": chemistry,
                "cell_type": ct,
                "n_donors": int(row["n_donors"]),
                "median_abs_err_yr": float(row["median_abs_err_yr"]),
                "mean_abs_err_yr": float(row["mean_abs_err_yr"]),
                "pearson_r": float(row["pearson_r"]),
                "pearson_p": float(row["pearson_p"]),
                "mean_bias_yr": float(row["mean_bias_yr"]),
                "leakage_status": "clean",  # LASSO trained on EGA cohorts disjoint from ours
                "chemistry_match_to_baseline_training": chem_match,
                "detectability_flag": DETECTABILITY[cohort][ct],
            })

    # ---- scAgeClock (Xie 2026, CELLxGENE Census) ----
    if SCAGECLOCK_SUMMARY.exists():
        for _, row in pd.read_csv(SCAGECLOCK_SUMMARY).iterrows():
            cohort = row["eval_cohort"]
            ct = row["cell_type"]
            chemistry, _ = CHEMISTRY_BY_COHORT[cohort]
            rows.append({
                "baseline": "scAgeClock",
                "training_cohorts": "CELLxGENE-Census-2024-07-01",
                "eval_cohort": cohort,
                "eval_chemistry": chemistry,
                "cell_type": ct,
                "n_donors": int(row["n_donors"]),
                "median_abs_err_yr": float(row["median_abs_err_yr"]),
                "mean_abs_err_yr": float(row["mean_abs_err_yr"]),
                "pearson_r": float(row["pearson_r"]),
                "pearson_p": float(row["pearson_p"]),
                "mean_bias_yr": float(row["mean_bias_yr"]),
                "leakage_status": SCAGECLOCK_LEAKAGE[cohort],
                # scAgeClock saw heterogeneous CELLxGENE chemistries (3', 5', SS2, etc.)
                # so chemistry_match is "match" by default for any 10x assay.
                "chemistry_match_to_baseline_training": "match",
                "detectability_flag": DETECTABILITY[cohort][ct],
            })

    # ---- Pasta REG (bulk-transcriptomic, Salignon 2025) ----
    if PASTA_SUMMARY.exists():
        for _, row in pd.read_csv(PASTA_SUMMARY).iterrows():
            cohort = row["eval_cohort"]
            ct = row["cell_type"]
            chemistry, _ = CHEMISTRY_BY_COHORT[cohort]
            rows.append({
                "baseline": "Pasta-REG",
                "training_cohorts": "Pasta-pretraining-bulk",
                "eval_cohort": cohort,
                "eval_chemistry": chemistry,
                "cell_type": ct,
                "n_donors": int(row["n_donors"]),
                "median_abs_err_yr": float(row["median_abs_err_yr"]),
                "mean_abs_err_yr": float(row["mean_abs_err_yr"]),
                "pearson_r": float(row["pearson_r"]),
                "pearson_p": float(row["pearson_p"]),
                "mean_bias_yr": float(row["mean_bias_yr"]),
                "leakage_status": PASTA_LEAKAGE[cohort],
                # Pasta is bulk-trained — chemistry-match doesn't apply directly;
                # we tag it `bulk-vs-sc` so stratification queries can exclude it
                # from same-modality FM-vs-baseline contrasts.
                "chemistry_match_to_baseline_training": "bulk-vs-sc",
                "detectability_flag": DETECTABILITY[cohort][ct],
            })

    out = pd.DataFrame(rows)
    # Sort: cohort, cell_type in canonical order, then baseline.
    ct_order = {"CD4T": 0, "CD8T": 1, "MONO": 2, "NK": 3, "B": 4}
    baseline_order = {"scImmuAging-LASSO": 0, "scAgeClock": 1, "Pasta-REG": 2}
    out["_ct_ord"] = out["cell_type"].map(ct_order)
    out["_b_ord"] = out["baseline"].map(baseline_order)
    out = out.sort_values(["eval_cohort", "_ct_ord", "_b_ord"]).drop(columns=["_ct_ord", "_b_ord"]).reset_index(drop=True)

    out_path = RESULTS_DIR / "loco_baseline_table.csv"
    out.to_csv(out_path, index=False)
    print(f"wrote {out_path}: {len(out)} rows")
    # Headline summary: per-cohort × per-cell-type pivot of best-baseline MAE
    for cohort in ("onek1k", "stephenson", "terekhova"):
        sub = out[out["eval_cohort"] == cohort]
        if sub.empty:
            continue
        print(f"\n=== {cohort} ({sub.iloc[0]['eval_chemistry']}) ===")
        pivot = sub.pivot(index="cell_type", columns="baseline",
                          values="median_abs_err_yr").round(2)
        print("median_abs_err_yr by baseline:")
        print(pivot)
        pivot_r = sub.pivot(index="cell_type", columns="baseline",
                            values="pearson_r").round(3)
        print("\npearson_r by baseline:")
        print(pivot_r)


if __name__ == "__main__":
    main()
