"""G.3 — Cell-type-conditional probing recipe table.

For each (method × fold × cell_type × seed) condition in F.5:
  - Apply cell-type-conditional recipe: (layer_c, k_c) chosen per cell type.
    * CD4+T: full-embed at the per-condition R_full-best layer (k=0).
    * B: PC-residual at (per-condition R_residual-best layer × k_pc).
    * NK: PC-residual at (per-condition R_residual-best layer × k_pc).
  - Apply fixed-recipe baseline: full-embed ridge at the per-condition R_full-best layer (k=0).
  - Compute ΔR_holdout, ΔR_aida.

Aggregation: mean ΔR across (cell × fold × seed) conditions.

Decision rule (pre-commit, applied to mean ΔR_holdout):
  ≥ +0.05 → unified cell-type-conditional recipe is a methodology contribution; primary headline.
  ∈ [0, +0.05] → small but consistent gains; report in supplement.
  < 0 → recipe doesn't generalize; cell-type-conditional finding stays biological.

NOTE: this is a packaging/aggregation task on existing F.5 results. Recipe selection is
"holdout-peek" (per-condition best-of-(layer, k_pc) by R_residual) rather than CV-selected;
this is a post-hoc upper bound, flagged as such in the writeup.

Output: results/phase3/g3_cell_type_conditional_recipe.csv
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


F5_CSV = Path("results/phase3/f5_pc_residual.csv")
OUT_CSV = Path("results/phase3/g3_cell_type_conditional_recipe.csv")
GENE_EN_CSV = Path("results/baselines/gene_en_matched_splits.csv")


def _aida_eligible(method, fold):
    """Replicate F.5 CONFIGS' also_aida flag."""
    return fold == "loco_onek1k"


def main():
    df = pd.read_csv(F5_CSV)
    gene_en = pd.read_csv(GENE_EN_CSV)
    rows = []
    conditions = df.groupby(["method", "fold", "cell_type", "seed"])
    for (method, fold, cell_type, seed), grp in conditions:
        # === Cell-type-conditional recipe selection ===
        if cell_type == "CD4+ T":
            # Use full-embed: pick layer where R_full is highest (k_pc choice irrelevant for R_full)
            best_layer = int(grp.loc[grp["R_full"].idxmax(), "layer"])
            cond_recipe = ("full-embed", best_layer, 0)
            R_holdout_cond = float(grp[(grp["layer"] == best_layer)].iloc[0]["R_full"])
            MAE_holdout_cond = float(grp[(grp["layer"] == best_layer)].iloc[0]["MAE_full"])
            R_aida_cond = float(grp[(grp["layer"] == best_layer)].iloc[0].get("R_aida_full", np.nan))
        else:
            # Use PC-residual: pick (layer × k_pc) where R_residual is highest
            idx = grp["R_residual"].idxmax()
            best_layer = int(grp.loc[idx, "layer"])
            best_k = int(grp.loc[idx, "k_pc"])
            cond_recipe = ("pc-residual", best_layer, best_k)
            R_holdout_cond = float(grp.loc[idx, "R_residual"])
            MAE_holdout_cond = float(grp.loc[idx, "MAE_residual"])
            R_aida_cond = float(grp.loc[idx].get("R_aida_residual", np.nan))

        # === Fixed-recipe baseline (full-embed ridge at per-condition best R_full layer) ===
        fixed_layer = int(grp.loc[grp["R_full"].idxmax(), "layer"])
        # R_full is same for all k_pc rows at a given layer; pick first
        fixed_grp = grp[grp["layer"] == fixed_layer].iloc[0]
        R_holdout_fixed = float(fixed_grp["R_full"])
        MAE_holdout_fixed = float(fixed_grp["MAE_full"])
        R_aida_fixed = float(fixed_grp.get("R_aida_full", np.nan))

        # === Gene-EN reference (for benchmarking) ===
        ge_holdout = gene_en[
            (gene_en["fold"] == fold) & (gene_en["cell_type"] == cell_type)
            & (gene_en["eval_cohort"] != "aida")
        ]
        ge_aida = gene_en[
            (gene_en["fold"] == fold) & (gene_en["cell_type"] == cell_type)
            & (gene_en["eval_cohort"] == "aida")
        ]
        R_geneEN_holdout = float(ge_holdout["pearson_r"].iloc[0]) if len(ge_holdout) else np.nan
        R_geneEN_aida = float(ge_aida["pearson_r"].iloc[0]) if len(ge_aida) else np.nan

        rows.append({
            "method": method, "fold": fold, "cell_type": cell_type, "seed": seed,
            "cond_recipe": cond_recipe[0], "cond_layer": cond_recipe[1], "cond_k_pc": cond_recipe[2],
            "fixed_layer": fixed_layer,
            "R_holdout_cond": R_holdout_cond, "MAE_holdout_cond": MAE_holdout_cond,
            "R_holdout_fixed": R_holdout_fixed, "MAE_holdout_fixed": MAE_holdout_fixed,
            "deltaR_holdout": R_holdout_cond - R_holdout_fixed,
            "R_aida_cond": R_aida_cond, "R_aida_fixed": R_aida_fixed,
            "deltaR_aida": (R_aida_cond - R_aida_fixed) if not (np.isnan(R_aida_cond) or np.isnan(R_aida_fixed)) else np.nan,
            "R_geneEN_holdout": R_geneEN_holdout, "R_geneEN_aida": R_geneEN_aida,
            "R_holdout_cond_vs_geneEN": R_holdout_cond - R_geneEN_holdout if not np.isnan(R_geneEN_holdout) else np.nan,
        })

    out = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False, float_format="%.4f")
    print(f"[G.3] wrote {len(out)} rows to {OUT_CSV}\n")
    print(out[[
        "method", "fold", "cell_type", "seed", "cond_recipe", "cond_layer", "cond_k_pc",
        "R_holdout_cond", "R_holdout_fixed", "deltaR_holdout",
        "R_geneEN_holdout", "R_holdout_cond_vs_geneEN",
    ]].to_string(index=False, float_format="%.3f"))

    # === Aggregation ===
    print("\n=== G.3 Aggregation: mean ΔR by cell type and overall ===")
    for ct in ["CD4+ T", "B", "NK"]:
        sub = out[out["cell_type"] == ct]
        dR_holdout = sub["deltaR_holdout"].mean()
        dR_aida = sub["deltaR_aida"].dropna().mean() if sub["deltaR_aida"].notna().any() else np.nan
        print(f"  {ct:8s} | n={len(sub):2d} | mean ΔR_holdout = {dR_holdout:+.4f} | mean ΔR_aida = {dR_aida:+.4f}")
    print(f"\n  Overall ΔR_holdout (all conditions): {out['deltaR_holdout'].mean():+.4f}")

    overall_dR = float(out["deltaR_holdout"].mean())
    print("\n=== G.3 Decision rule (mean ΔR_holdout across all 16 conditions) ===")
    if overall_dR >= 0.05:
        print(f"  Mean ΔR = {overall_dR:+.4f} ≥ +0.05 → CONTRIBUTION: cell-type-conditional recipe is a primary methodology contribution.")
    elif overall_dR >= 0:
        print(f"  Mean ΔR = {overall_dR:+.4f} ∈ [0, +0.05) → REFINEMENT: small but consistent; supplement only.")
    else:
        print(f"  Mean ΔR = {overall_dR:+.4f} < 0 → BIOLOGICAL ONLY: recipe doesn't generalize; cell-type-conditional finding stays biological.")


if __name__ == "__main__":
    main()
