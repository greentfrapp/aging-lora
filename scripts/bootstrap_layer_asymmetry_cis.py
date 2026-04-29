"""D.26 — Bootstrap 95% CIs on §31 layer-asymmetry numbers.

For each (cell_type × eval_cohort) condition, refit ridge per layer using
existing layered embeddings, then:
  1. Bootstrap-resample donors (n_boot=1000) and recompute Pearson R for
     L_best (per-condition) and L12 on each resample.
  2. Report median + 95% CI for ΔR(L_best vs L12).
  3. Flag conditions where the CI on ΔR includes 0 (i.e., L_best is not
     significantly better than L12 at the 5% level).

Output: `results/phase3/layer_asymmetry_cis.csv`
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge, RidgeCV


EMB_DIR = Path("results/phase3/embeddings_layered")
OUT_CSV = Path("results/phase3/layer_asymmetry_cis.csv")
ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
SEED = 0
N_BOOT = 1000
TAG = "frozen_base_alllayers"


def _slug(cell_type: str) -> str:
    return cell_type.replace("+", "p").replace(" ", "_")


def _load_npz(cohort: str, cell_type: str):
    path = EMB_DIR / f"{cohort}_{_slug(cell_type)}_{TAG}.npz"
    z = np.load(path, allow_pickle=True)
    return z["donor_ids"], z["ages"].astype(np.float32), z["embeddings_per_layer"].astype(np.float32)


def _fit_ridge(train_X, train_y):
    cv = RidgeCV(alphas=ALPHAS, cv=3, scoring="neg_mean_absolute_error")
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(train_y))
    cv.fit(train_X[perm], train_y[perm])
    return float(cv.alpha_), Ridge(alpha=float(cv.alpha_)).fit(train_X, train_y)


def main():
    folds = json.loads(Path("data/loco_folds.json").read_text())["folds"]
    fold_map = {f["fold_id"]: f for f in folds}

    # All conditions where layered ridge ran
    conditions = [
        ("loco_onek1k", "CD4+ T", True),
        ("loco_terekhova", "CD4+ T", False),
        ("loco_onek1k", "B", True),
        ("loco_terekhova", "B", False),
        ("loco_onek1k", "NK", True),
        ("loco_terekhova", "NK", False),
    ]

    rows = []
    for fold_id, cell_type, also_aida in conditions:
        f = fold_map[fold_id]
        train_cohorts = f["train_cohorts"]
        eval_cohort = f["holdout_cohort"]

        train_X_per_layer, train_y_all = [], []
        for tc in train_cohorts:
            _, ages, emb_LDH = _load_npz(tc, cell_type)
            train_X_per_layer.append(emb_LDH)
            train_y_all.append(ages)
        train_X_layered = np.concatenate(train_X_per_layer, axis=1)
        train_y = np.concatenate(train_y_all)

        eval_donors, eval_y, eval_X_layered = _load_npz(eval_cohort, cell_type)

        n_layers = train_X_layered.shape[0]

        # Pre-fit ridge per layer + collect predictions
        layer_preds_eval = np.zeros((n_layers, len(eval_y)))
        for layer in range(n_layers):
            _, model = _fit_ridge(train_X_layered[layer], train_y)
            layer_preds_eval[layer] = model.predict(eval_X_layered[layer])

        # Identify best-R layer on full eval
        layer_R = np.array([pearsonr(layer_preds_eval[L], eval_y)[0] for L in range(n_layers)])
        L_best = int(np.argmax(layer_R))
        L12 = n_layers - 1  # L12 is index 12 (13 layers including embedding output → indices 0..12)

        # Bootstrap CI on ΔR
        rng = np.random.default_rng(SEED)
        deltas = []
        n = len(eval_y)
        for _ in range(N_BOOT):
            idx = rng.integers(0, n, size=n)
            try:
                r_best = pearsonr(layer_preds_eval[L_best][idx], eval_y[idx])[0]
                r_12 = pearsonr(layer_preds_eval[L12][idx], eval_y[idx])[0]
                deltas.append(r_best - r_12)
            except Exception:
                continue
        deltas = np.array(deltas)
        ci_lo, ci_hi = float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))
        median_delta = float(np.median(deltas))

        rows.append({
            "fold": fold_id, "eval_cohort": eval_cohort, "cell_type": cell_type,
            "L_best": L_best, "L_best_R": float(layer_R[L_best]),
            "L12_R": float(layer_R[L12]),
            "delta_R_median": median_delta,
            "delta_R_ci_lo": ci_lo, "delta_R_ci_hi": ci_hi,
            "ci_excludes_zero": ci_lo > 0,
            "n_eval_donors": int(n),
        })
        print(f"[{fold_id} × {cell_type} × {eval_cohort}] L_best={L_best} R={layer_R[L_best]:+.3f}  L12 R={layer_R[L12]:+.3f}  ΔR median={median_delta:+.3f} CI=[{ci_lo:+.3f}, {ci_hi:+.3f}] {'EXCLUDES 0' if ci_lo > 0 else 'includes 0'}", flush=True)

        if also_aida:
            _, aida_y, aida_X_layered = _load_npz("aida", cell_type)
            aida_preds = np.zeros((n_layers, len(aida_y)))
            for layer in range(n_layers):
                _, model = _fit_ridge(train_X_layered[layer], train_y)
                aida_preds[layer] = model.predict(aida_X_layered[layer])
            aida_R = np.array([pearsonr(aida_preds[L], aida_y)[0] for L in range(n_layers)])
            L_best_a = int(np.argmax(aida_R))

            rng_a = np.random.default_rng(SEED)
            deltas_a = []
            n_a = len(aida_y)
            for _ in range(N_BOOT):
                idx = rng_a.integers(0, n_a, size=n_a)
                try:
                    r_best = pearsonr(aida_preds[L_best_a][idx], aida_y[idx])[0]
                    r_12 = pearsonr(aida_preds[L12][idx], aida_y[idx])[0]
                    deltas_a.append(r_best - r_12)
                except Exception:
                    continue
            deltas_a = np.array(deltas_a)
            aci_lo, aci_hi = float(np.percentile(deltas_a, 2.5)), float(np.percentile(deltas_a, 97.5))
            amedian_delta = float(np.median(deltas_a))

            rows.append({
                "fold": fold_id, "eval_cohort": "aida", "cell_type": cell_type,
                "L_best": L_best_a, "L_best_R": float(aida_R[L_best_a]),
                "L12_R": float(aida_R[L12]),
                "delta_R_median": amedian_delta,
                "delta_R_ci_lo": aci_lo, "delta_R_ci_hi": aci_hi,
                "ci_excludes_zero": aci_lo > 0,
                "n_eval_donors": int(n_a),
            })
            print(f"[{fold_id} × {cell_type} × AIDA] L_best={L_best_a} R={aida_R[L_best_a]:+.3f}  L12 R={aida_R[L12]:+.3f}  ΔR median={amedian_delta:+.3f} CI=[{aci_lo:+.3f}, {aci_hi:+.3f}] {'EXCLUDES 0' if aci_lo > 0 else 'includes 0'}", flush=True)

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, float_format="%.6g")
    print(f"\n[bootstrap-CI] wrote {len(df)} rows to {OUT_CSV}")
    print()
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
