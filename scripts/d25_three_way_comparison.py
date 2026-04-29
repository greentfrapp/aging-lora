"""D.25 — Three-way matched-splits comparison: gene-EN | Geneformer | scFoundation.

Reads the three CSVs and emits a single table comparing R + MAE for each
(cell × eval cohort) condition. Tests whether scFoundation also achieves
matched-splits parity (and thus closes scFoundation-LoRA from the queue) or
whether the matched-splits parity finding is Geneformer-specific.

Output: `results/phase3/d25_three_way_matched_splits.csv` and stdout table.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def main():
    # 1. Gene-EN matched
    gene_en = pd.read_csv("results/baselines/gene_en_matched_splits.csv")
    # 2. scFoundation frozen + ridge
    scf = pd.read_csv("results/phase3/ridge_summary_scfoundation.csv")
    # 3. Geneformer per-cell ridge readout (per-cell mean-pool, best layer)
    gf_layered = pd.read_csv("results/phase3/ridge_summary_layered.csv")
    # 4. Geneformer pseudobulk-input (best layer)
    gf_pb = pd.read_csv("results/phase3/ridge_summary_pseudobulk.csv")

    # Reduce gf_layered to best-R per (fold × cell × eval_cohort)
    gf_layered_best = gf_layered.loc[
        gf_layered.groupby(["fold", "cell_type", "eval_cohort"])["pearson_r"].idxmax()
    ][["fold", "cell_type", "eval_cohort", "layer", "pearson_r", "mae_y"]].rename(
        columns={"layer": "gf_pc_best_layer", "pearson_r": "gf_pc_R", "mae_y": "gf_pc_MAE"}
    )
    gf_pb_best = gf_pb.loc[
        gf_pb.groupby(["fold", "cell_type", "eval_cohort"])["pearson_r"].idxmax()
    ][["fold", "cell_type", "eval_cohort", "layer", "pearson_r", "mae_y"]].rename(
        columns={"layer": "gf_pb_best_layer", "pearson_r": "gf_pb_R", "mae_y": "gf_pb_MAE"}
    )
    gene_en_red = gene_en[["fold", "cell_type", "eval_cohort", "pearson_r", "mae_y"]].rename(
        columns={"pearson_r": "gene_en_R", "mae_y": "gene_en_MAE"}
    )
    scf_red = scf[["fold", "cell_type", "eval_cohort", "pearson_r", "mae_y"]].rename(
        columns={"pearson_r": "scf_R", "mae_y": "scf_MAE"}
    )

    keys = ["fold", "cell_type", "eval_cohort"]
    df = gene_en_red.merge(gf_layered_best, on=keys, how="outer")
    df = df.merge(gf_pb_best, on=keys, how="outer")
    df = df.merge(scf_red, on=keys, how="outer")

    # Order rows: CD4+T first, then NK, then B; loco_onek1k first
    cell_order = {"CD4+ T": 0, "NK": 1, "B": 2}
    fold_order = {"loco_onek1k": 0, "loco_terekhova": 1}
    eval_order = {"onek1k": 0, "terekhova": 0, "aida": 1}
    df["_co"] = df["cell_type"].map(cell_order)
    df["_fo"] = df["fold"].map(fold_order)
    df["_eo"] = df["eval_cohort"].map(eval_order)
    df = df.sort_values(["_co", "_fo", "_eo"]).drop(columns=["_co", "_fo", "_eo"]).reset_index(drop=True)

    out = Path("results/phase3/d25_three_way_matched_splits.csv")
    df.to_csv(out, index=False, float_format="%.4f")
    print(f"[d25] wrote {len(df)} rows to {out}\n")

    # Pretty print
    print("=== matched-splits three-way: gene-EN vs Geneformer per-cell vs Geneformer pseudobulk vs scFoundation ===\n")
    cols = ["fold", "cell_type", "eval_cohort", "gene_en_R", "gene_en_MAE",
            "gf_pc_R", "gf_pc_best_layer", "gf_pc_MAE",
            "gf_pb_R", "gf_pb_best_layer", "gf_pb_MAE",
            "scf_R", "scf_MAE"]
    print(df[cols].to_string(index=False, float_format="%.3f"))

    print("\n=== summary: scFoundation vs gene-EN (R deltas) ===")
    df_cd4 = df[df["cell_type"] == "CD4+ T"]
    print("\nCD4+T conditions:")
    for _, row in df_cd4.iterrows():
        gene_en_R = row.gene_en_R
        scf_R = row.scf_R
        gf_pc_R = row.gf_pc_R
        if pd.notna(gene_en_R) and pd.notna(scf_R):
            delta = scf_R - gene_en_R
            print(f"  {row.fold} × {row.eval_cohort}: gene-EN R={gene_en_R:.3f} | scF R={scf_R:.3f} (Δ={delta:+.3f}) | gf-pc R={gf_pc_R:.3f} (Δ={gf_pc_R-gene_en_R:+.3f})")

    print("\n=== INTERPRETATION ===")
    print("If scF Δ ≈ gf_pc Δ across CD4+T conditions → matched-splits parity is FM-class")
    print("If scF Δ ≪ gf_pc Δ (scF lags) → matched-splits parity is Geneformer-specific, scFoundation does worse")
    print("If scF Δ > gf_pc Δ → scFoundation is competitive too; matched-splits parity universal")


if __name__ == "__main__":
    main()
