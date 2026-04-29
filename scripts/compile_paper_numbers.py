"""D.35 — Compile all paper-relevant numbers into a single unified CSV.

Reads from:
  - results/baselines/gene_en_matched_splits.csv (D.17)
  - results/phase3/ridge_summary_layered.csv (frozen base layered)
  - results/phase3/ridge_summary_layered_finetune.csv (rank-16 LoRA seed 0 layered)
  - results/phase3/ridge_summary_post_finetune.csv (rank-16 LoRA seed 1+2 layered)
  - results/phase3/ridge_summary_pseudobulk.csv (D.18)
  - results/phase3/ridge_summary_r32_smoke.csv (rank-32 seed 0 layered)
  - results/phase3/ridge_summary_scfoundation.csv (D.7/§29)
  - results/phase3/d32_rank16_3seed_layered_bootstrap_cis.csv (3-seed agg)
  - results/phase3/layer_asymmetry_cis.csv (D.26)

Outputs:
  - results/phase3/paper_numbers_unified.csv (one row per method × cell × eval condition)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def main():
    rows = []

    # 1. Gene-EN matched
    df = pd.read_csv("results/baselines/gene_en_matched_splits.csv")
    for _, r in df.iterrows():
        rows.append({
            "method": "gene_en_matched",
            "cell_type": r["cell_type"], "eval_cohort": r["eval_cohort"],
            "fold": r["fold"], "layer": pd.NA, "seed": 0,
            "R": r["pearson_r"], "MAE": r["mae_y"],
            "ci_lo": r["pearson_ci_lo"], "ci_hi": r["pearson_ci_hi"],
            "n_train": r["n_train_donors"], "n_eval": r["n_eval_donors"],
            "notes": "ElasticNetCV top-5000 HVG StandardScaler",
        })

    # 2. Frozen Geneformer layered (best layer per condition)
    df = pd.read_csv("results/phase3/ridge_summary_layered.csv")
    best = df.loc[df.groupby(["fold", "cell_type", "eval_cohort"])["pearson_r"].idxmax()]
    for _, r in best.iterrows():
        rows.append({
            "method": "geneformer_frozen_per_cell_ridge_best_layer",
            "cell_type": r["cell_type"], "eval_cohort": r["eval_cohort"],
            "fold": r["fold"], "layer": int(r["layer"]), "seed": 0,
            "R": r["pearson_r"], "MAE": r["mae_y"],
            "ci_lo": pd.NA, "ci_hi": pd.NA,
            "n_train": r["n_train_donors"], "n_eval": r["n_eval_donors"],
            "notes": f"frozen base, best-R layer; alpha={r['alpha']}",
        })

    # 3. Pseudobulk-input Geneformer (best layer per condition)
    df = pd.read_csv("results/phase3/ridge_summary_pseudobulk.csv")
    best = df.loc[df.groupby(["fold", "cell_type", "eval_cohort"])["pearson_r"].idxmax()]
    for _, r in best.iterrows():
        rows.append({
            "method": "geneformer_pseudobulk_input_ridge_best_layer",
            "cell_type": r["cell_type"], "eval_cohort": r["eval_cohort"],
            "fold": r["fold"], "layer": int(r["layer"]), "seed": 0,
            "R": r["pearson_r"], "MAE": r["mae_y"],
            "ci_lo": r.get("ci_lo", pd.NA), "ci_hi": r.get("ci_hi", pd.NA),
            "n_train": r["n_train_donors"], "n_eval": r["n_eval_donors"],
            "notes": "pseudobulk-input, frozen base, best-R layer",
        })

    # 4. scFoundation frozen (uses pool='all', no layered)
    df = pd.read_csv("results/phase3/ridge_summary_scfoundation.csv")
    for _, r in df.iterrows():
        rows.append({
            "method": "scfoundation_frozen_pool_all",
            "cell_type": r["cell_type"], "eval_cohort": r["eval_cohort"],
            "fold": r["fold"], "layer": pd.NA, "seed": 0,
            "R": r["pearson_r"], "MAE": r["mae_y"],
            "ci_lo": r.get("pearson_ci_lo", pd.NA), "ci_hi": r.get("pearson_ci_hi", pd.NA),
            "n_train": r["n_train_donors"], "n_eval": r["n_eval_donors"],
            "notes": "scFoundation 3B, pool=concat(T+S+max+mean)",
        })

    # 5. Rank-16 LoRA 3-seed (post-finetune) — aggregate per layer
    df = pd.read_csv("results/phase3/ridge_summary_post_finetune.csv")
    # Add seed 0 from layered_finetune
    seed0 = pd.read_csv("results/phase3/ridge_summary_layered_finetune.csv")
    seed0 = seed0[seed0["tag"] == "loco_onek1k_e5b_alllayers"].copy()
    seed0["label"] = "CD4pT_loco_onek1k_s0"
    seed0["seed"] = 0
    df["seed"] = df["label"].str.extract(r"s(\d)").astype(int)
    seed0 = seed0[df.columns.intersection(seed0.columns)]
    df_combined = pd.concat([seed0, df[df["seed"].isin([1, 2])]], ignore_index=True)

    df_combined = df_combined[df_combined["cell_type"] == "CD4+ T"]
    if len(df_combined) > 0 and "layer" in df_combined.columns:
        # Aggregate: mean ± std across seeds per layer per eval cohort
        agg = df_combined.groupby(["fold", "cell_type", "eval_cohort", "layer"]).agg(
            R_mean=("pearson_r", "mean"), R_std=("pearson_r", "std"),
            MAE_mean=("mae_y", "mean"), MAE_std=("mae_y", "std"),
            n_seeds=("seed", "count"),
        ).reset_index()
        for _, r in agg.iterrows():
            rows.append({
                "method": f"geneformer_rank16_lora_{int(r['n_seeds'])}seed_layer{int(r['layer'])}",
                "cell_type": r["cell_type"], "eval_cohort": r["eval_cohort"],
                "fold": r["fold"], "layer": int(r["layer"]), "seed": pd.NA,
                "R": r["R_mean"], "MAE": r["MAE_mean"],
                "ci_lo": pd.NA, "ci_hi": pd.NA,
                "n_train": pd.NA, "n_eval": pd.NA,
                "notes": f"rank-16 LoRA {int(r['n_seeds'])}-seed mean (R_std={r['R_std']:.3f}, MAE_std={r['MAE_std']:.2f})",
            })

    # 6. Rank-32 single-seed
    df = pd.read_csv("results/phase3/ridge_summary_r32_smoke.csv")
    for _, r in df.iterrows():
        rows.append({
            "method": "geneformer_rank32_lora_seed0_layer" + str(int(r["layer"])),
            "cell_type": "CD4+ T", "eval_cohort": "onek1k",
            "fold": "loco_onek1k", "layer": int(r["layer"]), "seed": 0,
            "R": r["pearson_r"], "MAE": r["mae"],
            "ci_lo": r.get("ci_lo", pd.NA), "ci_hi": r.get("ci_hi", pd.NA),
            "n_train": pd.NA, "n_eval": pd.NA,
            "notes": "rank-32 single-seed",
        })
        rows.append({
            "method": "geneformer_rank32_lora_seed0_layer" + str(int(r["layer"])),
            "cell_type": "CD4+ T", "eval_cohort": "aida",
            "fold": "loco_onek1k", "layer": int(r["layer"]), "seed": 0,
            "R": r["aida_r"], "MAE": r["aida_mae"],
            "ci_lo": pd.NA, "ci_hi": pd.NA,
            "n_train": pd.NA, "n_eval": pd.NA,
            "notes": "rank-32 single-seed AIDA cross-ancestry",
        })

    df_out = pd.DataFrame(rows)
    out = Path("results/phase3/paper_numbers_unified.csv")
    df_out.to_csv(out, index=False, float_format="%.4f")
    print(f"[compile] wrote {len(df_out)} rows to {out}")

    # AIDA cross-ancestry summary
    print("\n=== AIDA cross-ancestry summary ===")
    aida = df_out[df_out["eval_cohort"] == "aida"].copy()
    aida = aida.sort_values(by=["cell_type", "method"])
    print(aida[["method", "cell_type", "fold", "layer", "R", "MAE", "notes"]].to_string(index=False, float_format="%.3f"))


if __name__ == "__main__":
    main()
