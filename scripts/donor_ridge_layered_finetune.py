"""Ridge fits for fine-tuned-checkpoint layered extractions (Variant 3 follow-up).

Compares per-layer R + MAE between frozen-base and fine-tuned representations
on the same fold. Tests the §26.7 question: does fine-tuning preserve or
destroy the layer-1 signal that the frozen base achieves?

Reads `.npz` files written by `extract_embeddings_layered.py --checkpoint ...`
with `output-tag = {fold}_e5b_alllayers`. Writes one row per (fold × layer ×
eval_cohort) to `results/phase3/ridge_summary_layered_finetune.csv`.

Usage:
    uv run python scripts/donor_ridge_layered_finetune.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from scipy.stats import pearsonr


EMB_DIR = Path("results/phase3/embeddings_layered")
OUT_CSV = Path("results/phase3/ridge_summary_layered_finetune.csv")
ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
SEED = 0


def _slug(cell_type: str) -> str:
    return cell_type.replace("+", "p").replace(" ", "_")


def _load_npz(cohort: str, cell_type: str, tag: str):
    path = EMB_DIR / f"{cohort}_{_slug(cell_type)}_{tag}.npz"
    if not path.exists():
        raise SystemExit(f"missing layered embedding file: {path}")
    z = np.load(path, allow_pickle=True)
    return z["donor_ids"], z["ages"].astype(np.float32), z["embeddings_per_layer"].astype(np.float32)


def main():
    folds = json.loads(Path("data/loco_folds.json").read_text())["folds"]
    fold_map = {f["fold_id"]: f for f in folds}

    # (fold_id, output_tag, cell_type, also_aida)
    runs = [
        ("loco_terekhova", "loco_terekhova_e5b_alllayers", "CD4+ T", True),
        ("loco_onek1k", "loco_onek1k_e5b_alllayers", "CD4+ T", True),
    ]

    rows = []
    for fold_id, tag, cell_type, also_aida in runs:
        f = fold_map[fold_id]
        train_cohorts = f["train_cohorts"]
        eval_cohort = f["holdout_cohort"]

        train_X_per_layer, train_y_all = [], []
        for tc in train_cohorts:
            _, ages, emb_LDH = _load_npz(tc, cell_type, tag)
            train_X_per_layer.append(emb_LDH)
            train_y_all.append(ages)
        train_X_layered = np.concatenate(train_X_per_layer, axis=1)
        train_y = np.concatenate(train_y_all)

        _, eval_y, eval_X_layered = _load_npz(eval_cohort, cell_type, tag)
        aida_X_layered = aida_y = None
        if also_aida:
            _, aida_y, aida_X_layered = _load_npz("aida", cell_type, tag)

        n_layers = train_X_layered.shape[0]
        print(f"\n[ridge-FT] fold={fold_id} tag={tag} | {n_layers} layers, {len(train_y)} train, {len(eval_y)} eval")

        for layer in range(n_layers):
            cv = RidgeCV(alphas=ALPHAS, cv=3, scoring="neg_mean_absolute_error")
            rng = np.random.default_rng(SEED)
            perm = rng.permutation(len(train_y))
            cv.fit(train_X_layered[layer][perm], train_y[perm])
            alpha = float(cv.alpha_)
            final = Ridge(alpha=alpha).fit(train_X_layered[layer], train_y)

            pred = final.predict(eval_X_layered[layer])
            r, p = pearsonr(pred, eval_y)
            mae = float(np.median(np.abs(pred - eval_y)))
            print(f"  layer={layer:>2d} alpha={alpha:>8.2f} R={r:+.3f} p={p:.2e} MAE={mae:.2f}")
            rows.append({
                "fold": fold_id,
                "tag": tag,
                "eval_cohort": eval_cohort,
                "cell_type": cell_type,
                "layer": layer,
                "alpha": alpha,
                "n_train_donors": int(len(train_y)),
                "n_eval_donors": int(len(eval_y)),
                "pearson_r": float(r),
                "pearson_p": float(p),
                "mae_y": mae,
            })
            if also_aida and aida_X_layered is not None:
                aida_pred = final.predict(aida_X_layered[layer])
                aida_r, aida_p = pearsonr(aida_pred, aida_y)
                aida_mae = float(np.median(np.abs(aida_pred - aida_y)))
                print(f"           AIDA: R={aida_r:+.3f} p={aida_p:.2e} MAE={aida_mae:.2f}")
                rows.append({
                    "fold": fold_id,
                    "tag": tag,
                    "eval_cohort": "aida",
                    "cell_type": cell_type,
                    "layer": layer,
                    "alpha": alpha,
                    "n_train_donors": int(len(train_y)),
                    "n_eval_donors": int(len(aida_y)),
                    "pearson_r": float(aida_r),
                    "pearson_p": float(aida_p),
                    "mae_y": aida_mae,
                })

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False, float_format="%.6g")
    print(f"\n[ridge-FT] wrote {len(df)} rows to {OUT_CSV}")

    # Side-by-side print: layer-1 frozen vs layer-1 fine-tuned vs layer-12 fine-tuned.
    print("\n=== best layer per (fold × eval_cohort) by R ===")
    best = df.loc[df.groupby(["fold", "eval_cohort"])["pearson_r"].idxmax()]
    print(best[["fold", "eval_cohort", "layer", "pearson_r", "mae_y"]].to_string(index=False))


if __name__ == "__main__":
    main()
