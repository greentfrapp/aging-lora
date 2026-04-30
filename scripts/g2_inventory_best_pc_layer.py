"""Find F.5-best (layer × k_pc) for B-cell across folds, for G.2 setup."""
import pandas as pd

df = pd.read_csv("results/phase3/f5_pc_residual.csv")

# B × loco_terekhova × frozen seed 0 — primary G.2 condition
sub = df[(df["cell_type"] == "B") & (df["fold"] == "loco_terekhova")
         & (df["method"] == "geneformer_frozen_seed0") & (df["seed"] == 0)]
print("=== B × loco_terekhova × frozen seed 0 (G.2 primary) ===")
print(sub.sort_values("R_residual", ascending=False).head(8)[
    ["layer", "k_pc", "R_full", "R_residual", "deltaR_holdout"]
].to_string(index=False, float_format="%.4f"))

# Also show B × loco_onek1k for context
sub2 = df[(df["cell_type"] == "B") & (df["fold"] == "loco_onek1k")
          & (df["method"] == "geneformer_frozen_seed0") & (df["seed"] == 0)]
print("\n=== B × loco_onek1k × frozen seed 0 (context) ===")
print(sub2.sort_values("R_residual", ascending=False).head(8)[
    ["layer", "k_pc", "R_full", "R_residual", "deltaR_holdout",
     "R_aida_full", "R_aida_residual", "deltaR_aida"]
].to_string(index=False, float_format="%.4f"))
