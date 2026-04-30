"""E.1 + E.4 — Multi-seed modal-layer ensemble + end-to-end deployment.

E.1: Aggregate D.37 cv_R_per_layer arrays across seeds within each multi-seed
group (NK frozen × 3 seeds × 2 folds; rank-16 × 3 seeds; rank-32 × 3 seeds).
Compute mean-CV-R per layer across seeds; modal_layer = argmax.

E.4: Refit ridge at modal_layer per seed on full train, evaluate on holdout +
AIDA. Compare R/MAE penalties (vs oracle) for ensemble vs single-seed CV.

Decision rules baked in (per roadmap):
  E.1 modal-layer matches per-seed oracle in:
    - >=2/3 conditions → ensembling is the deployment recipe
    - 1/3            → marginal
    - <=0/3 across seeds in >=2 conditions → ensembling does not help

  E.4 ensemble R penalty drop vs single-seed mean R penalty:
    - >=50% → recommend ensemble deployment
    - 25-50% → ensemble helps with caveat
    - <25%  → single-seed CV is the recipe

Outputs:
  - results/phase3/e1_modal_layer_ensemble.csv
  - results/phase3/e4_ensemble_deployment.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from scipy.stats import pearsonr


D37_CSV = Path("results/phase3/d37_cv_layer_selection.csv")
EMB_DIR = Path("results/phase3/embeddings_layered")
OUT_E1 = Path("results/phase3/e1_modal_layer_ensemble.csv")
OUT_E4 = Path("results/phase3/e4_ensemble_deployment.csv")
ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
SEED = 0


# Multi-seed group definition: (method_base, fold, cell_type, [(seed, tag)])
GROUPS = [
    ("geneformer_frozen", "loco_onek1k", "NK", [
        (0, "frozen_base_alllayers"),
        (1, "frozen_base_seed1_alllayers"),
        (2, "frozen_base_seed2_alllayers"),
    ], True),
    ("geneformer_frozen", "loco_terekhova", "NK", [
        (0, "frozen_base_alllayers"),
        (1, "frozen_base_seed1_alllayers"),
        (2, "frozen_base_seed2_alllayers"),
    ], False),
    ("geneformer_rank16", "loco_onek1k", "CD4+ T", [
        (0, "loco_onek1k_e5b_alllayers"),
        (1, "loco_onek1k_CD4pT_e5b_seed1_alllayers"),
        (2, "loco_onek1k_CD4pT_e5b_seed2_alllayers"),
    ], True),
    ("geneformer_rank32", "loco_onek1k", "CD4+ T", [
        (0, "loco_onek1k_CD4pT_e5b_r32_alllayers"),
        (1, "loco_onek1k_CD4pT_e5b_r32_seed1_alllayers"),
        (2, "loco_onek1k_CD4pT_e5b_r32_seed2_alllayers"),
    ], True),
]


def _slug(cell_type: str) -> str:
    return cell_type.replace("+", "p").replace(" ", "_")


def _load_npz(cohort: str, cell_type: str, tag: str):
    path = EMB_DIR / f"{cohort}_{_slug(cell_type)}_{tag}.npz"
    if not path.exists():
        return None
    z = np.load(path, allow_pickle=True)
    return z["donor_ids"], z["ages"].astype(np.float32), z["embeddings_per_layer"].astype(np.float32)


def _fit_ridge_eval(X_train, y_train, X_eval, y_eval):
    cv = RidgeCV(alphas=ALPHAS, cv=3, scoring="neg_mean_absolute_error")
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(y_train))
    cv.fit(X_train[perm], y_train[perm])
    final = Ridge(alpha=float(cv.alpha_)).fit(X_train, y_train)
    pred = final.predict(X_eval)
    if np.std(pred) > 0 and np.std(y_eval) > 0 and len(y_eval) > 1:
        r, _ = pearsonr(pred, y_eval)
    else:
        r = 0.0
    mae = float(np.median(np.abs(pred - y_eval)))
    return float(r), mae


def main():
    folds = json.loads(Path("data/loco_folds.json").read_text())["folds"]
    fmap = {f["fold_id"]: f for f in folds}
    d37 = pd.read_csv(D37_CSV)

    e1_rows, e4_rows = [], []

    for method_base, fold_id, cell_type, seed_tags, also_aida in GROUPS:
        # Pull per-seed cv_R arrays from D.37 CSV
        per_seed_cv_R = []
        per_seed_cv_picks = []
        per_seed_oracles = []
        per_seed_holdout_at_cv = []
        per_seed_holdout_at_oracle = []
        per_seed_holdout_MAE_at_cv = []
        per_seed_holdout_MAE_at_oracle = []
        per_seed_aida_R_at_cv = []
        per_seed_aida_oracle_R = []
        per_seed_aida_oracle_layer = []

        for seed, _ in seed_tags:
            method_label = f"{method_base}_seed{seed}"
            row = d37[(d37["method"] == method_label) & (d37["fold"] == fold_id) &
                     (d37["cell_type"] == cell_type) & (d37["seed"] == seed)]
            if len(row) == 0:
                print(f"  [SKIP] no D.37 row for {method_label} {fold_id} {cell_type}", flush=True)
                continue
            row = row.iloc[0]
            cv_R = json.loads(row["cv_R_per_layer"])
            per_seed_cv_R.append(cv_R)
            per_seed_cv_picks.append(int(row["L_cv_R_selected"]))
            per_seed_oracles.append(int(row["L_oracle_holdout_R"]))
            per_seed_holdout_at_cv.append(float(row["holdout_R_at_cv_R"]))
            per_seed_holdout_at_oracle.append(float(row["holdout_R_at_oracle"]))
            per_seed_holdout_MAE_at_cv.append(float(row["holdout_MAE_at_cv_R"]))
            per_seed_holdout_MAE_at_oracle.append(float(row["holdout_MAE_at_oracle_MAE"]))
            if also_aida:
                per_seed_aida_R_at_cv.append(float(row["aida_R_at_cv_R"]))
                per_seed_aida_oracle_R.append(float(row["aida_R_at_aida_oracle"]))
                per_seed_aida_oracle_layer.append(int(row["aida_oracle_layer"]))

        if not per_seed_cv_R:
            continue

        per_seed_cv_R = np.array(per_seed_cv_R)  # (n_seeds, n_layers)
        ensemble_mean = per_seed_cv_R.mean(axis=0)
        ensemble_median = np.median(per_seed_cv_R, axis=0)
        modal_mean = int(np.argmax(ensemble_mean))
        modal_median = int(np.argmax(ensemble_median))

        # E.1 row: agreement with per-seed oracles
        n_seeds = len(per_seed_oracles)
        agreement_mean = sum(modal_mean == o for o in per_seed_oracles)
        agreement_median = sum(modal_median == o for o in per_seed_oracles)
        agreement_with_cv_picks = sum(modal_mean == p for p in per_seed_cv_picks)

        e1_rows.append({
            "method_base": method_base,
            "fold": fold_id,
            "cell_type": cell_type,
            "n_seeds": n_seeds,
            "per_seed_cv_picks": str(per_seed_cv_picks),
            "per_seed_oracles": str(per_seed_oracles),
            "modal_layer_mean": modal_mean,
            "modal_layer_median": modal_median,
            "agreement_modal_mean_vs_oracle": agreement_mean,
            "agreement_modal_median_vs_oracle": agreement_median,
            "agreement_modal_mean_vs_cv_picks": agreement_with_cv_picks,
            "ensemble_cv_R_per_layer": ensemble_mean.tolist(),
        })

        # E.4: refit ridge at modal_mean layer on full train per seed, evaluate
        f = fmap[fold_id]
        modal_holdout_R = []
        modal_holdout_MAE = []
        modal_aida_R = []
        modal_aida_MAE = []

        for seed, tag in seed_tags:
            train_X_per_layer, train_y_all = [], []
            skip = False
            for tc in f["train_cohorts"]:
                ret = _load_npz(tc, cell_type, tag)
                if ret is None:
                    print(f"  [E.4 SKIP] missing {tc}_{_slug(cell_type)}_{tag}")
                    skip = True
                    break
                _, ages, emb_LDH = ret
                train_X_per_layer.append(emb_LDH)
                train_y_all.append(ages)
            if skip:
                continue
            train_X_layered = np.concatenate(train_X_per_layer, axis=1)
            train_y = np.concatenate(train_y_all)

            eval_ret = _load_npz(f["holdout_cohort"], cell_type, tag)
            if eval_ret is None:
                continue
            _, eval_y, eval_X_layered = eval_ret

            r, mae = _fit_ridge_eval(
                train_X_layered[modal_mean], train_y,
                eval_X_layered[modal_mean], eval_y,
            )
            modal_holdout_R.append(r)
            modal_holdout_MAE.append(mae)

            if also_aida:
                aida_ret = _load_npz("aida", cell_type, tag)
                if aida_ret is None:
                    modal_aida_R.append(None)
                    modal_aida_MAE.append(None)
                else:
                    _, aida_y, aida_X_layered = aida_ret
                    ar, amae = _fit_ridge_eval(
                        train_X_layered[modal_mean], train_y,
                        aida_X_layered[modal_mean], aida_y,
                    )
                    modal_aida_R.append(ar)
                    modal_aida_MAE.append(amae)

            print(f"  [E.4] {method_base} {fold_id} {cell_type} seed{seed} @L{modal_mean}: "
                  f"holdout R={r:+.3f} MAE={mae:.2f}"
                  + (f", AIDA R={modal_aida_R[-1]:+.3f}" if also_aida and modal_aida_R[-1] is not None else ""),
                  flush=True)

        # Aggregate E.4 stats: penalty drop
        single_R_penalty = float(np.mean(np.array(per_seed_holdout_at_oracle) - np.array(per_seed_holdout_at_cv)))
        ensemble_R_penalty = float(np.mean(np.array(per_seed_holdout_at_oracle) - np.array(modal_holdout_R)))
        single_MAE_penalty = float(np.mean(np.array(per_seed_holdout_MAE_at_cv) - np.array(per_seed_holdout_MAE_at_oracle)))
        ensemble_MAE_penalty = float(np.mean(np.array(modal_holdout_MAE) - np.array(per_seed_holdout_MAE_at_oracle)))

        if abs(single_R_penalty) > 1e-9:
            R_penalty_drop_pct = 100.0 * (single_R_penalty - ensemble_R_penalty) / single_R_penalty
        else:
            R_penalty_drop_pct = 0.0

        if abs(single_MAE_penalty) > 1e-9:
            MAE_penalty_drop_pct = 100.0 * (single_MAE_penalty - ensemble_MAE_penalty) / single_MAE_penalty
        else:
            MAE_penalty_drop_pct = 0.0

        if R_penalty_drop_pct >= 50:
            recommendation = "ensemble (modal-layer-across-seeds)"
        elif R_penalty_drop_pct >= 25:
            recommendation = "ensemble helps with caveat"
        else:
            recommendation = "single-seed CV is the recipe"

        e4_rows.append({
            "method_base": method_base,
            "fold": fold_id,
            "cell_type": cell_type,
            "n_seeds": n_seeds,
            "modal_layer": modal_mean,
            "single_seed_holdout_R_at_cv_mean": float(np.mean(per_seed_holdout_at_cv)),
            "single_seed_holdout_R_at_oracle_mean": float(np.mean(per_seed_holdout_at_oracle)),
            "ensemble_holdout_R_at_modal_mean": float(np.mean(modal_holdout_R)),
            "single_seed_R_penalty": single_R_penalty,
            "ensemble_R_penalty": ensemble_R_penalty,
            "R_penalty_drop_pct": R_penalty_drop_pct,
            "single_seed_holdout_MAE_at_cv_mean": float(np.mean(per_seed_holdout_MAE_at_cv)),
            "single_seed_holdout_MAE_at_oracle_mean": float(np.mean(per_seed_holdout_MAE_at_oracle)),
            "ensemble_holdout_MAE_at_modal_mean": float(np.mean(modal_holdout_MAE)),
            "single_seed_MAE_penalty": single_MAE_penalty,
            "ensemble_MAE_penalty": ensemble_MAE_penalty,
            "MAE_penalty_drop_pct": MAE_penalty_drop_pct,
            "aida_R_at_cv_mean": float(np.mean(per_seed_aida_R_at_cv)) if per_seed_aida_R_at_cv else None,
            "aida_R_at_modal_mean": float(np.mean([r for r in modal_aida_R if r is not None])) if modal_aida_R else None,
            "aida_R_at_aida_oracle_mean": float(np.mean(per_seed_aida_oracle_R)) if per_seed_aida_oracle_R else None,
            "recommendation": recommendation,
        })

    # Save
    OUT_E1.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(e1_rows).to_csv(OUT_E1, index=False, float_format="%.4f")
    pd.DataFrame(e4_rows).to_csv(OUT_E4, index=False, float_format="%.4f")
    print(f"\n[E.1] wrote {len(e1_rows)} rows to {OUT_E1}")
    print(f"[E.4] wrote {len(e4_rows)} rows to {OUT_E4}")

    # Decision summary
    print("\n=== E.1 Decision summary ===\n")
    df_e1 = pd.DataFrame(e1_rows)
    if len(df_e1) > 0:
        print(df_e1[["method_base", "fold", "cell_type", "per_seed_cv_picks", "per_seed_oracles",
                    "modal_layer_mean", "agreement_modal_mean_vs_oracle"]].to_string(index=False))
        n_pass = (df_e1["agreement_modal_mean_vs_oracle"] >= 2).sum()
        print(f"\n  Conditions where modal-layer matches oracle in >=2/3 seeds: {n_pass}/{len(df_e1)}")
        n_marg = (df_e1["agreement_modal_mean_vs_oracle"] == 1).sum()
        print(f"  Conditions where modal-layer matches oracle in 1/3 seeds: {n_marg}/{len(df_e1)}")
        n_fail = (df_e1["agreement_modal_mean_vs_oracle"] == 0).sum()
        print(f"  Conditions where modal-layer matches oracle in 0/3 seeds: {n_fail}/{len(df_e1)}")
        if n_pass / len(df_e1) >= 2/3:
            print("  → DECISION: ensembling is the deployment recipe")
        elif n_pass / len(df_e1) >= 1/3:
            print("  → DECISION: ensembling helps marginally")
        else:
            print("  → DECISION: ensembling does not robustly help")

    print("\n=== E.4 Decision summary ===\n")
    df_e4 = pd.DataFrame(e4_rows)
    if len(df_e4) > 0:
        print(df_e4[["method_base", "fold", "cell_type", "modal_layer",
                    "single_seed_R_penalty", "ensemble_R_penalty", "R_penalty_drop_pct",
                    "recommendation"]].to_string(index=False, float_format="%.3f"))


if __name__ == "__main__":
    main()
