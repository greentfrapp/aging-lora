"""D.31 — Mechanistic donor-cluster analysis on NK L3 vs CD4+T L12 vs B layer embeddings.

Tests the hypothesis that early-layer NK signal captures coarse compositional
shifts (donor-level subset distributions) while late-layer CD4+T signal
captures cell-state abstractions (activation programs).

For each (cell_type, best_layer, eval_cohort) in the §31 + D.26 analysis:
  1. Load donor mean-pooled embeddings at the best layer.
  2. Compute pairwise donor distances (cosine).
  3. Compute donor age vs nearest-neighbor age (does an "age-NN R" exist at
     this layer?).
  4. UMAP projection coloured by age — qualitative check.

Output: results/phase3/d31_donor_cluster_metrics.csv + UMAP plots.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from scipy.spatial.distance import pdist, squareform


EMB_DIR = Path("results/phase3/embeddings_layered")
OUT_CSV = Path("results/phase3/d31_donor_cluster_metrics.csv")
TAG = "frozen_base_alllayers"


def _slug(cell_type: str) -> str:
    return cell_type.replace("+", "p").replace(" ", "_")


def _load_npz(cohort: str, cell_type: str):
    path = EMB_DIR / f"{cohort}_{_slug(cell_type)}_{TAG}.npz"
    z = np.load(path, allow_pickle=True)
    return z["donor_ids"], z["ages"].astype(np.float32), z["embeddings_per_layer"].astype(np.float32)


def _knn_age_correlation(emb, ages, k=5):
    """For each donor, compute mean age of k nearest neighbours by cosine; correlate with own age."""
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms < 1e-9] = 1.0
    emb_n = emb / norms
    sim = emb_n @ emb_n.T
    np.fill_diagonal(sim, -np.inf)  # exclude self
    knn_idx = np.argpartition(-sim, kth=k, axis=1)[:, :k]
    knn_ages = ages[knn_idx].mean(axis=1)
    r, _ = pearsonr(knn_ages, ages)
    return float(r), knn_ages


def main():
    rows = []
    # Conditions: (cell, eval_cohort, layer)
    # Use the L_best identified in D.26 (best-R layer per condition)
    conditions = [
        ("CD4+ T", "onek1k", 12),
        ("CD4+ T", "terekhova", 5),
        ("CD4+ T", "aida", 12),  # both AIDA folds; same cohort here
        ("NK", "onek1k", 3),
        ("NK", "terekhova", 2),
        ("NK", "aida", 5),
        ("B", "onek1k", 7),
        ("B", "terekhova", 9),
        ("B", "aida", 11),
    ]

    for cell_type, cohort, layer in conditions:
        try:
            _, ages, emb_LDH = _load_npz(cohort, cell_type)
        except SystemExit as e:
            print(f"  skip {cohort} × {cell_type}: {e}", flush=True)
            continue
        emb = emb_LDH[layer]  # (n_donors, H)

        # Compare with L12 baseline
        emb12 = emb_LDH[12]

        # kNN age correlation at best layer
        k = min(5, len(ages) - 1)
        r_best, _ = _knn_age_correlation(emb, ages, k=k)
        r_12, _ = _knn_age_correlation(emb12, ages, k=k)

        # Donor-distance variance (within-cohort spread of donors in embedding space)
        d_best = pdist(emb, metric="cosine")
        d_12 = pdist(emb12, metric="cosine")

        rows.append({
            "cell_type": cell_type, "eval_cohort": cohort,
            "layer_best": layer, "n_donors": len(ages),
            "kNN_age_R_best_layer": r_best,
            "kNN_age_R_L12": r_12,
            "kNN_age_R_delta": r_best - r_12,
            "donor_dist_mean_best": float(d_best.mean()),
            "donor_dist_std_best": float(d_best.std()),
            "donor_dist_mean_L12": float(d_12.mean()),
            "donor_dist_std_L12": float(d_12.std()),
        })
        print(f"[{cell_type} × {cohort} × L{layer}] kNN-age R: best={r_best:+.3f}, L12={r_12:+.3f}, Δ={r_best-r_12:+.3f}", flush=True)

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, float_format="%.6g")
    print(f"\n[d31] wrote {len(df)} rows to {OUT_CSV}")
    print()
    print(df[["cell_type", "eval_cohort", "layer_best", "kNN_age_R_best_layer", "kNN_age_R_L12", "kNN_age_R_delta"]].to_string(index=False, float_format="%.3f"))


if __name__ == "__main__":
    main()
