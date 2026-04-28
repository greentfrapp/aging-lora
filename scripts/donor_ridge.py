"""Ridge regression on per-donor mean Geneformer embeddings.

Reads `.npz` files written by `scripts/extract_embeddings.py`, fits ridge
(nested 3-fold CV on training cohorts to choose alpha), evaluates Pearson R
and MAE on the held-out cohort. Appends one row to
`results/phase3/ridge_summary.csv`.

Usage:
    uv run python scripts/donor_ridge.py \\
        --fold loco_onek1k --cell-type "CD4+ T" --tag frozen_base
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from scipy.stats import pearsonr


def _load_fold(fold_id: str) -> dict:
    folds = json.loads(Path("data/loco_folds.json").read_text())["folds"]
    for f in folds:
        if f.get("fold_id") == fold_id:
            return f
    raise SystemExit(f"unknown fold {fold_id}")


def _slug(cell_type: str) -> str:
    return cell_type.replace("+", "p").replace(" ", "_")


def _load_npz(emb_dir: Path, cohort: str, cell_type: str, tag: str):
    path = emb_dir / f"{cohort}_{_slug(cell_type)}_{tag}.npz"
    if not path.exists():
        raise SystemExit(f"missing embedding file: {path}")
    z = np.load(path, allow_pickle=True)
    return z["donor_ids"], z["ages"].astype(np.float32), z["embeddings"].astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fold", required=True, help="fold_id from data/loco_folds.json (e.g. loco_onek1k)")
    p.add_argument("--cell-type", required=True)
    p.add_argument("--tag", required=True,
                   help="output_tag passed to extract_embeddings.py (e.g. frozen_base, loco_onek1k_seed0_e5b)")
    p.add_argument("--emb-dir", default="results/phase3/embeddings")
    p.add_argument("--summary-csv", default="results/phase3/ridge_summary.csv")
    p.add_argument("--alphas", default="0.01,0.1,1,10,100,1000,10000",
                   help="comma-separated regularization grid for nested CV")
    p.add_argument("--cv-folds", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--also-eval-aida", action="store_true",
                   help="if set, additionally evaluate the trained ridge on AIDA embeddings (must exist)")
    args = p.parse_args()

    fold = _load_fold(args.fold)
    train_cohorts = fold["train_cohorts"]
    eval_cohort = fold["holdout_cohort"]
    emb_dir = Path(args.emb_dir)

    # Load training embeddings (concatenate across train cohorts)
    train_donors_all, train_ages_all, train_emb_all = [], [], []
    for tc in train_cohorts:
        ids, ages, emb = _load_npz(emb_dir, tc, args.cell_type, args.tag)
        train_donors_all.append(ids)
        train_ages_all.append(ages)
        train_emb_all.append(emb)
    train_donors = np.concatenate(train_donors_all)
    train_ages = np.concatenate(train_ages_all)
    train_emb = np.concatenate(train_emb_all, axis=0)

    # Eval embeddings
    eval_donors, eval_ages, eval_emb = _load_npz(emb_dir, eval_cohort, args.cell_type, args.tag)

    print(f"[ridge] fold={args.fold} cell_type={args.cell_type} tag={args.tag}")
    print(f"[ridge] train: {len(train_donors)} donors × {train_emb.shape[1]}-dim from {train_cohorts}")
    print(f"[ridge] eval: {len(eval_donors)} donors from {eval_cohort}")

    alphas = [float(a) for a in args.alphas.split(",")]

    t0 = time.time()
    # Nested CV: pick alpha by RidgeCV on training data
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(train_donors))
    train_emb_p = train_emb[perm]
    train_ages_p = train_ages[perm]

    cv = RidgeCV(alphas=alphas, cv=args.cv_folds, scoring="neg_mean_absolute_error")
    cv.fit(train_emb_p, train_ages_p)
    chosen_alpha = float(cv.alpha_)

    # Refit on all train, evaluate on holdout
    final = Ridge(alpha=chosen_alpha).fit(train_emb, train_ages)
    eval_pred = final.predict(eval_emb)
    mae = float(np.median(np.abs(eval_pred - eval_ages)))
    r, p_value = pearsonr(eval_pred, eval_ages)
    elapsed = time.time() - t0
    print(f"[ridge] alpha={chosen_alpha} mae={mae:.3f} r={r:.3f} p={p_value:.2e} wall={elapsed:.1f}s")

    rows = [{
        "fold": args.fold,
        "eval_cohort": eval_cohort,
        "cell_type": args.cell_type,
        "tag": args.tag,
        "n_train_donors": int(len(train_donors)),
        "n_eval_donors": int(len(eval_donors)),
        "alpha": chosen_alpha,
        "mae_y": mae,
        "pearson_r": float(r),
        "pearson_p": float(p_value),
        "wall_s": float(elapsed),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }]

    if args.also_eval_aida:
        try:
            aida_donors, aida_ages, aida_emb = _load_npz(emb_dir, "aida", args.cell_type, args.tag)
            aida_pred = final.predict(aida_emb)
            aida_mae = float(np.median(np.abs(aida_pred - aida_ages)))
            aida_r, aida_p = pearsonr(aida_pred, aida_ages)
            print(f"[ridge] aida: mae={aida_mae:.3f} r={aida_r:.3f} p={aida_p:.2e}")
            rows.append({
                "fold": args.fold,
                "eval_cohort": "aida",
                "cell_type": args.cell_type,
                "tag": args.tag,
                "n_train_donors": int(len(train_donors)),
                "n_eval_donors": int(len(aida_donors)),
                "alpha": chosen_alpha,
                "mae_y": aida_mae,
                "pearson_r": float(aida_r),
                "pearson_p": float(aida_p),
                "wall_s": float(elapsed),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            })
        except SystemExit as e:
            print(f"[ridge] AIDA eval skipped: {e}")

    summary_csv = Path(args.summary_csv)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame(rows)
    if summary_csv.exists():
        df = pd.concat([pd.read_csv(summary_csv), new_df], ignore_index=True)
    else:
        df = new_df
    df.to_csv(summary_csv, index=False)
    print(f"[ridge] appended {len(rows)} row(s) to {summary_csv}")


if __name__ == "__main__":
    main()
