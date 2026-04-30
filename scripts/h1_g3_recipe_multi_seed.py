"""H.1-augmented G.3 — Cell-type-conditional probing recipe table with multi-seed B-cell.

Updates G.3's recipe table by replacing F.5's single-seed B-cell rows with H.1's CV-honest
multi-seed B-cell rows. Non-B-cell rows still come from F.5 (seed 0 + NK seeds 1, 2).

Recipe selection:
  - CD4+T: full-embed at per-condition R_full-best layer (k=0). Tautologically equal to
    fixed-recipe baseline. Comes from F.5.
  - B: CV-honest PC-residual at (cv_layer, cv_k_pc) per (fold × seed). Comes from H.1.
  - NK: PC-residual at per-condition R_residual-best layer × k_pc. Comes from F.5
    (note: this is post-hoc holdout-best; consistent with G.3 v1 framing).

Decision rule (pre-commit, mean ΔR_holdout across all conditions):
  ≥ +0.05 → primary contribution.
  ∈ [0, +0.05] → refinement.
  < 0 → biological-only.

Output: results/phase3/h1_g3_recipe_multi_seed.csv
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


F5_CSV = Path("results/phase3/f5_pc_residual.csv")
H1_CSV = Path("results/phase3/h1_b_cell_multi_seed.csv")
GENE_EN_CSV = Path("results/baselines/gene_en_matched_splits.csv")
OUT_CSV = Path("results/phase3/h1_g3_recipe_multi_seed.csv")


def main():
    f5 = pd.read_csv(F5_CSV)
    h1 = pd.read_csv(H1_CSV)
    gene_en = pd.read_csv(GENE_EN_CSV)

    rows = []

    # === Non-B rows from F.5 ===
    f5_nonB = f5[f5["cell_type"] != "B"]
    for (method, fold, cell_type, seed), grp in f5_nonB.groupby(["method", "fold", "cell_type", "seed"]):
        if cell_type == "CD4+ T":
            best_layer = int(grp.loc[grp["R_full"].idxmax(), "layer"])
            recipe = "full-embed"
            cond_layer, cond_k = best_layer, 0
            row = grp[grp["layer"] == best_layer].iloc[0]
            R_holdout_cond = float(row["R_full"])
            MAE_holdout_cond = float(row["MAE_full"])
            R_aida_cond = float(row.get("R_aida_full", np.nan))
        else:  # NK
            idx = grp["R_residual"].idxmax()
            cond_layer = int(grp.loc[idx, "layer"])
            cond_k = int(grp.loc[idx, "k_pc"])
            recipe = "pc-residual"
            R_holdout_cond = float(grp.loc[idx, "R_residual"])
            MAE_holdout_cond = float(grp.loc[idx, "MAE_residual"])
            R_aida_cond = float(grp.loc[idx].get("R_aida_residual", np.nan))

        fixed_layer = int(grp.loc[grp["R_full"].idxmax(), "layer"])
        fixed_row = grp[grp["layer"] == fixed_layer].iloc[0]
        R_holdout_fixed = float(fixed_row["R_full"])
        MAE_holdout_fixed = float(fixed_row["MAE_full"])
        R_aida_fixed = float(fixed_row.get("R_aida_full", np.nan))

        ge_h = gene_en[(gene_en["fold"] == fold) & (gene_en["cell_type"] == cell_type) & (gene_en["eval_cohort"] != "aida")]
        ge_a = gene_en[(gene_en["fold"] == fold) & (gene_en["cell_type"] == cell_type) & (gene_en["eval_cohort"] == "aida")]
        R_ge_h = float(ge_h["pearson_r"].iloc[0]) if len(ge_h) else np.nan
        R_ge_a = float(ge_a["pearson_r"].iloc[0]) if len(ge_a) else np.nan

        rows.append({
            "method": method, "fold": fold, "cell_type": cell_type, "seed": seed,
            "cond_recipe": recipe, "cond_layer": cond_layer, "cond_k_pc": cond_k,
            "fixed_layer": fixed_layer,
            "R_holdout_cond": R_holdout_cond, "MAE_holdout_cond": MAE_holdout_cond,
            "R_holdout_fixed": R_holdout_fixed, "MAE_holdout_fixed": MAE_holdout_fixed,
            "deltaR_holdout": R_holdout_cond - R_holdout_fixed,
            "R_aida_cond": R_aida_cond, "R_aida_fixed": R_aida_fixed,
            "deltaR_aida": (R_aida_cond - R_aida_fixed) if not (np.isnan(R_aida_cond) or np.isnan(R_aida_fixed)) else np.nan,
            "R_geneEN_holdout": R_ge_h, "R_geneEN_aida": R_ge_a,
            "R_holdout_cond_vs_geneEN": R_holdout_cond - R_ge_h if not np.isnan(R_ge_h) else np.nan,
            "source": "F.5",
        })

    # === B-cell rows from H.1 (CV-honest, multi-seed) ===
    for _, h_row in h1.iterrows():
        fold = h_row["fold"]
        seed = int(h_row["seed"])
        ge_h = gene_en[(gene_en["fold"] == fold) & (gene_en["cell_type"] == "B") & (gene_en["eval_cohort"] != "aida")]
        ge_a = gene_en[(gene_en["fold"] == fold) & (gene_en["cell_type"] == "B") & (gene_en["eval_cohort"] == "aida")]
        R_ge_h = float(ge_h["pearson_r"].iloc[0]) if len(ge_h) else np.nan
        R_ge_a = float(ge_a["pearson_r"].iloc[0]) if len(ge_a) else np.nan
        rows.append({
            "method": f"geneformer_frozen_seed{seed}", "fold": fold, "cell_type": "B", "seed": seed,
            "cond_recipe": "pc-residual-CV", "cond_layer": int(h_row["cv_layer"]), "cond_k_pc": int(h_row["cv_k_pc"]),
            "fixed_layer": int(h_row["full_best_L_holdout"]),
            "R_holdout_cond": float(h_row["R_holdout_resid"]),
            "MAE_holdout_cond": float(h_row["MAE_holdout_resid"]),
            "R_holdout_fixed": float(h_row["full_best_R_holdout"]),  # full-embed best layer (post-hoc) on holdout
            "MAE_holdout_fixed": np.nan,  # not computed in H.1 (we have at cv_layer, not best_layer)
            "deltaR_holdout": float(h_row["R_holdout_resid"]) - float(h_row["full_best_R_holdout"]),
            "R_aida_cond": float(h_row["R_aida_resid"]),
            "R_aida_fixed": float(h_row["R_aida_full_at_cv_layer"]),  # AIDA full at cv_layer
            "deltaR_aida": float(h_row["R_aida_resid"]) - float(h_row["R_aida_full_at_cv_layer"]),
            "R_geneEN_holdout": R_ge_h, "R_geneEN_aida": R_ge_a,
            "R_holdout_cond_vs_geneEN": float(h_row["R_holdout_resid"]) - R_ge_h if not np.isnan(R_ge_h) else np.nan,
            "source": "H.1",
        })

    out = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False, float_format="%.4f")
    print(f"[H.1-G.3] wrote {len(out)} rows to {OUT_CSV}\n")
    print(out[[
        "method", "fold", "cell_type", "seed", "cond_recipe", "cond_layer", "cond_k_pc",
        "R_holdout_cond", "R_holdout_fixed", "deltaR_holdout",
        "R_geneEN_holdout", "R_holdout_cond_vs_geneEN", "source",
    ]].to_string(index=False, float_format="%.3f"))

    print("\n=== H.1-G.3 Aggregation per cell type ===")
    for ct in ["CD4+ T", "B", "NK"]:
        sub = out[out["cell_type"] == ct]
        dR_h = sub["deltaR_holdout"].mean()
        dR_a = sub["deltaR_aida"].dropna().mean() if sub["deltaR_aida"].notna().any() else np.nan
        n = len(sub)
        print(f"  {ct:8s} | n={n:2d} | mean ΔR_holdout = {dR_h:+.4f} | mean ΔR_aida = {dR_a:+.4f}")

    overall = out["deltaR_holdout"].mean()
    print(f"\n  Overall mean ΔR_holdout (all conditions): {overall:+.4f}")
    print(f"\n=== Decision rule ===")
    if overall >= 0.05:
        print(f"  Mean ΔR = {overall:+.4f} ≥ +0.05 → CONTRIBUTION: cell-type-conditional recipe is a primary methodology contribution.")
    elif overall >= 0:
        print(f"  Mean ΔR = {overall:+.4f} ∈ [0, +0.05) → REFINEMENT: small but consistent gains; supplement only.")
    else:
        print(f"  Mean ΔR = {overall:+.4f} < 0 → BIOLOGICAL ONLY: recipe doesn't generalize.")

    # B-cell-specific multi-seed mean ± std (the load-bearing claim)
    b = out[(out["cell_type"] == "B") & (out["fold"] == "loco_terekhova")]
    if len(b) >= 2:
        mean_R = b["R_holdout_cond"].mean()
        std_R = b["R_holdout_cond"].std()
        gene_en_R = float(b["R_geneEN_holdout"].iloc[0])
        print(f"\n=== B × loco_terekhova multi-seed (n={len(b)}) ===")
        print(f"  Mean R_holdout_cond = {mean_R:+.4f} ± {std_R:.4f}")
        print(f"  vs gene-EN R = {gene_en_R:.3f} → mean gap = {mean_R - gene_en_R:+.4f}")


if __name__ == "__main__":
    main()
