"""Layer-wise ridge fits for B / NK fine-tune checkpoints + CD4+T seed 1 / seed 2.

Reads `.npz` from extract_embeddings_layered.py with various output_tags
and fits ridge per (fold × cell × layer × eval). Writes one row per
condition to `results/phase3/ridge_summary_post_finetune.csv`.

Usage:
    uv run python scripts/donor_ridge_layered_post_finetune.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from scipy.stats import pearsonr


EMB_DIR = Path("results/phase3/embeddings_layered")
OUT_CSV = Path("results/phase3/ridge_summary_post_finetune.csv")
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
    fmap = {f["fold_id"]: f for f in folds}

    # (fold_id, output_tag, cell_type, also_aida, label_for_summary)
    runs = [
        ("loco_onek1k",    "loco_onek1k_B_e5b_alllayers",          "B",     True,  "B_loco_onek1k_s0"),
        ("loco_onek1k",    "loco_onek1k_NK_e5b_alllayers",         "NK",    True,  "NK_loco_onek1k_s0"),
        ("loco_terekhova", "loco_terekhova_B_e5b_alllayers",       "B",     False, "B_loco_terekhova_s0"),
        ("loco_onek1k",    "loco_onek1k_CD4pT_e5b_seed1_alllayers", "CD4+ T", True,  "CD4pT_loco_onek1k_s1"),
        ("loco_onek1k",    "loco_onek1k_CD4pT_e5b_seed2_alllayers", "CD4+ T", True,  "CD4pT_loco_onek1k_s2"),
    ]

    rows = []
    for fold_id, tag, cell_type, also_aida, label in runs:
        f = fmap[fold_id]
        train_X_per_layer, train_y_all = [], []
        for tc in f["train_cohorts"]:
            _, ages, emb_LDH = _load_npz(tc, cell_type, tag)
            train_X_per_layer.append(emb_LDH)
            train_y_all.append(ages)
        train_X_layered = np.concatenate(train_X_per_layer, axis=1)
        train_y = np.concatenate(train_y_all)
        eval_donors, eval_y, eval_X_layered = _load_npz(f["holdout_cohort"], cell_type, tag)
        aida_X_layered = aida_y = None
        if also_aida:
            _, aida_y, aida_X_layered = _load_npz("aida", cell_type, tag)

        n_layers = train_X_layered.shape[0]
        print(f"\n[{label}] fold={fold_id} cell={cell_type} | {n_layers} layers, {len(train_y)} train, {len(eval_y)} eval")

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
            rows.append({
                "label": label, "fold": fold_id, "tag": tag,
                "eval_cohort": f["holdout_cohort"], "cell_type": cell_type, "layer": layer,
                "alpha": alpha, "n_train_donors": int(len(train_y)),
                "n_eval_donors": int(len(eval_y)),
                "pearson_r": float(r), "pearson_p": float(p), "mae_y": mae,
            })
            print(f"  L{layer:>2d}  R={r:+.3f}  MAE={mae:>6.2f}", end="")
            if also_aida and aida_X_layered is not None:
                aida_pred = final.predict(aida_X_layered[layer])
                ar, ap = pearsonr(aida_pred, aida_y)
                amae = float(np.median(np.abs(aida_pred - aida_y)))
                print(f"  | AIDA: R={ar:+.3f} MAE={amae:>6.2f}")
                rows.append({
                    "label": label, "fold": fold_id, "tag": tag,
                    "eval_cohort": "aida", "cell_type": cell_type, "layer": layer,
                    "alpha": alpha, "n_train_donors": int(len(train_y)),
                    "n_eval_donors": int(len(aida_y)),
                    "pearson_r": float(ar), "pearson_p": float(ap), "mae_y": amae,
                })
            else:
                print("")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False, float_format="%.6g")
    print(f"\n[ridge] wrote {len(df)} rows to {OUT_CSV}")

    print("\n=== best layer per (label × eval_cohort) ===")
    best = df.loc[df.groupby(["label", "eval_cohort"])["pearson_r"].idxmax()]
    print(best[["label", "eval_cohort", "layer", "pearson_r", "mae_y"]].to_string(index=False))

    # Layer-12 specifically (the new headline pattern):
    print("\n=== layer-12 by (label × eval_cohort) — head-equivalent layer ===")
    L12 = df[df.layer == 12]
    print(L12[["label", "eval_cohort", "pearson_r", "mae_y"]].to_string(index=False))


if __name__ == "__main__":
    main()
