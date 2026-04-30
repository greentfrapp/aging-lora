"""Quick inventory of F.5 results: per-condition max delta-R."""
import pandas as pd

df = pd.read_csv("results/phase3/f5_pc_residual.csv")
g = df.groupby(["method", "fold", "cell_type", "seed"])["deltaR_holdout"].agg(["max", "min", "mean"]).reset_index()
g["class"] = g["max"].apply(lambda x: "IMPROVE" if x >= 0.05 else ("DEGRADE" if x <= -0.05 else "no_chg"))
g = g.sort_values("max", ascending=False)
print("=== Holdout deltaR per condition ===")
print(g.to_string(index=False, float_format="%.3f"))
print()
counts = g["class"].value_counts()
print(f"IMPROVE: {counts.get('IMPROVE', 0)}, no_chg: {counts.get('no_chg', 0)}, DEGRADE: {counts.get('DEGRADE', 0)}")

if "deltaR_aida" in df.columns:
    df_aida = df.dropna(subset=["deltaR_aida"])
    if len(df_aida):
        ga = df_aida.groupby(["method", "fold", "cell_type", "seed"])["deltaR_aida"].agg(["max", "mean"]).reset_index()
        ga["aida_class"] = ga["max"].apply(lambda x: "IMPROVE" if x >= 0.05 else ("DEGRADE" if x <= -0.05 else "no_chg"))
        ga = ga.sort_values("max", ascending=False)
        print()
        print("=== AIDA deltaR per condition ===")
        print(ga.to_string(index=False, float_format="%.3f"))
        ac = ga["aida_class"].value_counts()
        print(f"AIDA — IMPROVE: {ac.get('IMPROVE', 0)}, no_chg: {ac.get('no_chg', 0)}, DEGRADE: {ac.get('DEGRADE', 0)}")
