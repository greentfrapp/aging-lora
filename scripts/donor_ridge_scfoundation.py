"""Per-donor ridge readout on scFoundation 3072-d cell embeddings.

Mirror donor_ridge_layered.py protocol: LOCO folds × cell type × eval cohort
× RidgeCV alpha selection. Single "layer" (the canonical scFoundation cell
embedding, not per-encoder-layer hidden states), so we don't iterate layers.

Usage:
    uv run python scripts/donor_ridge_scfoundation.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from scipy.stats import pearsonr


EMB_DIR = Path("results/phase3/embeddings_scfoundation")
OUT_CSV = Path("results/phase3/ridge_summary_scfoundation.csv")
ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
SEED = 0
TAG = "scfoundation_t4"


def _slug(cell_type: str) -> str:
    return cell_type.replace("+", "p").replace(" ", "_")


def _load_npz(cohort: str, cell_type: str, tag: str = TAG):
    path = EMB_DIR / f"{cohort}_{_slug(cell_type)}_{tag}.npz"
    if not path.exists():
        raise SystemExit(f"missing scFoundation embedding file: {path}")
    z = np.load(path, allow_pickle=True)
    return z["donor_ids"], z["ages"].astype(np.float32), z["embeddings"].astype(np.float32)


def _bootstrap_pearson_ci(pred, y, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(pred)
    rs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if pred[idx].std() == 0 or y[idx].std() == 0:
            continue
        rs.append(pearsonr(pred[idx], y[idx])[0])
    rs = np.asarray(rs)
    return float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))


def main():
    folds = json.loads(Path("data/loco_folds.json").read_text())["folds"]
    fmap = {f["fold_id"]: f for f in folds}

    runs = [
        # (fold_id, cell_type, also_aida, label)
        ("loco_onek1k",    "CD4+ T", True,  "scf_CD4pT_loco_onek1k"),
        ("loco_terekhova", "CD4+ T", True,  "scf_CD4pT_loco_terekhova"),
        ("loco_onek1k",    "B",      True,  "scf_B_loco_onek1k"),
        ("loco_terekhova", "B",      False, "scf_B_loco_terekhova"),
        ("loco_onek1k",    "NK",     True,  "scf_NK_loco_onek1k"),
    ]

    rows = []
    for fold_id, cell_type, also_aida, label in runs:
        f = fmap[fold_id]
        train_X, train_y = [], []
        for tc in f["train_cohorts"]:
            _, ages, emb = _load_npz(tc, cell_type)
            train_X.append(emb)
            train_y.append(ages)
        train_X = np.concatenate(train_X, axis=0)
        train_y = np.concatenate(train_y)
        eval_donors, eval_y, eval_X = _load_npz(f["holdout_cohort"], cell_type)
        aida_X = aida_y = None
        if also_aida:
            _, aida_y, aida_X = _load_npz("aida", cell_type)

        print(f"\n[{label}] fold={fold_id} cell={cell_type} | {len(train_y)} train, {len(eval_y)} eval")
        cv = RidgeCV(alphas=ALPHAS, cv=3, scoring="neg_mean_absolute_error")
        rng = np.random.default_rng(SEED)
        perm = rng.permutation(len(train_y))
        cv.fit(train_X[perm], train_y[perm])
        alpha = float(cv.alpha_)
        final = Ridge(alpha=alpha).fit(train_X, train_y)

        pred = final.predict(eval_X)
        r, p_val = pearsonr(pred, eval_y)
        mae = float(np.median(np.abs(pred - eval_y)))
        ci_lo, ci_hi = _bootstrap_pearson_ci(pred, eval_y, seed=SEED)
        pred_mean = float(pred.mean())
        eval_mean = float(eval_y.mean())
        print(f"  HOLDOUT  R={r:+.3f} ({ci_lo:+.3f}, {ci_hi:+.3f})  MAE={mae:.2f}  "
              f"alpha={alpha:.2g}  pred_mean={pred_mean:.2f} eval_mean={eval_mean:.2f}")
        rows.append({
            "label": label, "fold": fold_id, "tag": TAG,
            "eval_cohort": f["holdout_cohort"], "cell_type": cell_type,
            "alpha": alpha, "n_train_donors": int(len(train_y)),
            "n_eval_donors": int(len(eval_y)),
            "pearson_r": float(r), "pearson_p": float(p_val), "mae_y": mae,
            "pearson_ci_lo": ci_lo, "pearson_ci_hi": ci_hi,
            "pred_mean": pred_mean, "eval_mean": eval_mean,
        })
        if also_aida and aida_X is not None:
            aida_pred = final.predict(aida_X)
            ar, ap = pearsonr(aida_pred, aida_y)
            amae = float(np.median(np.abs(aida_pred - aida_y)))
            aci_lo, aci_hi = _bootstrap_pearson_ci(aida_pred, aida_y, seed=SEED)
            apm = float(aida_pred.mean())
            aem = float(aida_y.mean())
            print(f"  AIDA     R={ar:+.3f} ({aci_lo:+.3f}, {aci_hi:+.3f})  MAE={amae:.2f}  "
                  f"pred_mean={apm:.2f} eval_mean={aem:.2f}")
            rows.append({
                "label": label, "fold": fold_id, "tag": TAG,
                "eval_cohort": "aida", "cell_type": cell_type,
                "alpha": alpha, "n_train_donors": int(len(train_y)),
                "n_eval_donors": int(len(aida_y)),
                "pearson_r": float(ar), "pearson_p": float(ap), "mae_y": amae,
                "pearson_ci_lo": aci_lo, "pearson_ci_hi": aci_hi,
                "pred_mean": apm, "eval_mean": aem,
            })

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False, float_format="%.6g")
    print(f"\n[ridge-scf] wrote {len(df)} rows to {OUT_CSV}")

    print("\n=== summary by (label × eval_cohort) ===")
    print(df[["label", "eval_cohort", "n_train_donors", "n_eval_donors",
              "pearson_r", "pearson_ci_lo", "pearson_ci_hi", "mae_y"]].to_string(index=False))


if __name__ == "__main__":
    main()
