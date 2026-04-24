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


def main():
    rows = []
    for cohort, src in SOURCES.items():
        df = pd.read_csv(src)
        chemistry, chem_match = CHEMISTRY_BY_COHORT[cohort]
        for _, row in df.iterrows():
            ct = row["cell_type"]
            rows.append({
                "baseline": "scImmuAging-pretrained",
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
                "leakage_status": "clean",
                "chemistry_match_to_baseline_training": chem_match,
                "detectability_flag": DETECTABILITY[cohort][ct],
                "pseudocell_n": int(row["pseudocell_n"]),
                "pseudocell_size": int(row["pseudocell_size"]),
            })

    out = pd.DataFrame(rows)
    # Sort: cohort, cell_type in canonical order.
    ct_order = {"CD4T": 0, "CD8T": 1, "MONO": 2, "NK": 3, "B": 4}
    out["_ct_ord"] = out["cell_type"].map(ct_order)
    out = out.sort_values(["eval_cohort", "_ct_ord"]).drop(columns=["_ct_ord"]).reset_index(drop=True)

    out_path = RESULTS_DIR / "loco_baseline_table.csv"
    out.to_csv(out_path, index=False)
    print(f"wrote {out_path}: {len(out)} rows")
    # Headline summary
    for cohort in ("onek1k", "stephenson", "terekhova"):
        sub = out[out["eval_cohort"] == cohort]
        print(f"\n=== {cohort} ({sub.iloc[0]['eval_chemistry']}) ===")
        print(sub[["cell_type", "n_donors", "median_abs_err_yr", "pearson_r",
                   "leakage_status", "chemistry_match_to_baseline_training", "detectability_flag"]].to_string(index=False))


if __name__ == "__main__":
    main()
