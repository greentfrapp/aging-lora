"""E.6 — Formal band-width quantification using K * SD-of-seed-variance.

Define "candidate band" as: layers with mean-across-seeds R within
K * SD_top of the best layer's mean R, where SD_top is the seed-SD at the
best layer. K=1.5 default.

For multi-seed conditions only (where seed-SD is meaningful):
  - NK frozen × loco_onek1k × 3 seeds
  - NK frozen × loco_terekhova × 3 seeds
  - rank-16 × CD4+T × loco_onek1k × 3 seeds
  - rank-32 × CD4+T × loco_onek1k × 3 seeds

Output: results/phase3/e6_band_width.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


E5_CSV = Path("results/phase3/e5_holdout_layer_flatness.csv")
OUT_CSV = Path("results/phase3/e6_band_width.csv")
K_VALUES = [1.0, 1.5, 2.0]


# Multi-seed groups: (method_base, fold, cell_type)
GROUPS = [
    ("geneformer_frozen", "loco_onek1k", "NK"),
    ("geneformer_frozen", "loco_terekhova", "NK"),
    ("geneformer_rank16", "loco_onek1k", "CD4+ T"),
    ("geneformer_rank32", "loco_onek1k", "CD4+ T"),
]


def _parse_list(s):
    """Parse a stringified list from CSV."""
    if pd.isna(s):
        return None
    return json.loads(s) if isinstance(s, str) else s


def main():
    e5 = pd.read_csv(E5_CSV)
    rows = []

    for method_base, fold_id, cell_type in GROUPS:
        # Find all multi-seed entries
        if method_base == "geneformer_frozen":
            method_pattern = "geneformer_frozen_seed"
        else:
            method_pattern = f"{method_base}_seed"

        sub = e5[(e5["method"].str.startswith(method_pattern)) &
                 (e5["fold"] == fold_id) &
                 (e5["cell_type"] == cell_type)].copy().reset_index(drop=True)
        if len(sub) < 2:
            continue

        # Stack per-seed holdout R curves
        R_curves = np.array([_parse_list(s) for s in sub["holdout_R_per_layer"]])
        MAE_curves = np.array([_parse_list(s) for s in sub["holdout_MAE_per_layer"]])
        aida_R_curves = None
        if "aida_R_per_layer" in sub.columns and sub["aida_R_per_layer"].notna().all():
            aida_R_curves = np.array([_parse_list(s) for s in sub["aida_R_per_layer"]])

        n_seeds, n_layers = R_curves.shape
        mean_R = R_curves.mean(axis=0)
        sd_R = R_curves.std(axis=0, ddof=1)
        mean_MAE = MAE_curves.mean(axis=0)
        sd_MAE = MAE_curves.std(axis=0, ddof=1)

        L_top_R = int(np.argmax(mean_R))
        L_top_MAE = int(np.argmin(mean_MAE))

        # Use SD at the top layer for the band
        sd_top_R = float(sd_R[L_top_R])
        sd_top_MAE = float(sd_MAE[L_top_MAE])

        for K in K_VALUES:
            R_threshold = mean_R[L_top_R] - K * sd_top_R
            band_R = sorted([int(l) for l in np.where(mean_R >= R_threshold)[0]])
            MAE_threshold = mean_MAE[L_top_MAE] + K * sd_top_MAE
            band_MAE = sorted([int(l) for l in np.where(mean_MAE <= MAE_threshold)[0]])

            row = {
                "method_base": method_base,
                "fold": fold_id,
                "cell_type": cell_type,
                "n_seeds": n_seeds,
                "K": K,
                "L_top_R": L_top_R,
                "mean_R_top": float(mean_R[L_top_R]),
                "sd_R_top": sd_top_R,
                "R_threshold": float(R_threshold),
                "band_R_layers": str(band_R),
                "band_R_width": len(band_R),
                "L_top_MAE": L_top_MAE,
                "mean_MAE_top": float(mean_MAE[L_top_MAE]),
                "sd_MAE_top": sd_top_MAE,
                "MAE_threshold": float(MAE_threshold),
                "band_MAE_layers": str(band_MAE),
                "band_MAE_width": len(band_MAE),
            }

            if aida_R_curves is not None:
                mean_aida_R = aida_R_curves.mean(axis=0)
                sd_aida_R = aida_R_curves.std(axis=0, ddof=1)
                L_top_aida = int(np.argmax(mean_aida_R))
                sd_top_aida = float(sd_aida_R[L_top_aida])
                aida_threshold = mean_aida_R[L_top_aida] - K * sd_top_aida
                band_aida = sorted([int(l) for l in np.where(mean_aida_R >= aida_threshold)[0]])
                row.update({
                    "L_top_aida_R": L_top_aida,
                    "mean_aida_R_top": float(mean_aida_R[L_top_aida]),
                    "sd_aida_R_top": sd_top_aida,
                    "aida_R_threshold": float(aida_threshold),
                    "band_aida_R_layers": str(band_aida),
                    "band_aida_R_width": len(band_aida),
                })

            rows.append(row)

        # Print per-condition summary at K=1.5
        print(f"\n=== {method_base} | {fold_id} × {cell_type} ({n_seeds} seeds) ===", flush=True)
        print(f"  Mean R per layer: {[f'{x:+.3f}' for x in mean_R]}")
        print(f"  SD R per layer:   {[f'{x:.3f}' for x in sd_R]}")
        print(f"  L_top_R = L{L_top_R} (mean R = {mean_R[L_top_R]:+.3f}, SD = {sd_top_R:.3f})")
        for K in K_VALUES:
            R_threshold = mean_R[L_top_R] - K * sd_top_R
            band_R = sorted([int(l) for l in np.where(mean_R >= R_threshold)[0]])
            print(f"  K={K}: R band = {band_R} (width {len(band_R)}, threshold R >= {R_threshold:+.3f})")

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, float_format="%.4f")
    print(f"\n[E.6] wrote {len(df)} rows to {OUT_CSV}")

    print("\n=== Band-width summary at K=1.5 (R) ===")
    summary = df[df["K"] == 1.5][["method_base", "fold", "cell_type", "L_top_R",
                                   "mean_R_top", "sd_R_top", "band_R_width", "band_R_layers"]]
    print(summary.to_string(index=False, float_format="%.3f"))

    if "L_top_aida_R" in df.columns:
        print("\n=== AIDA band-width at K=1.5 ===")
        aida_summary = df[df["K"] == 1.5][["method_base", "fold", "cell_type",
                                            "L_top_aida_R", "mean_aida_R_top", "sd_aida_R_top",
                                            "band_aida_R_width", "band_aida_R_layers"]]
        print(aida_summary.dropna().to_string(index=False, float_format="%.3f"))


if __name__ == "__main__":
    main()
