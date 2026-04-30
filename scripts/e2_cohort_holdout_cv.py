"""E.2 — Cohort-holdout inner CV.

For each (fold × cell × seed × method-tag):
  Train cohorts = [c1, c2].
  For each inner-validation cohort c_inner ∈ {c1, c2}:
    - Inner-train = the OTHER cohort
    - For each layer: fit ridge on inner-train, evaluate on c_inner → R per layer
    - L_cv = argmax R over layers
    - Refit ridge at L_cv on FULL train (c1+c2), evaluate on actual holdout +AIDA

Compare cohort-holdout CV picks to D.37 K-fold CV picks AND oracle.

Decision rule (pre-commit):
  cohort-holdout-CV pick == K-fold-CV pick in:
    >=75% conditions → recipe survives
    50-75% → cohort-specific caveat (D.22 already partial)
    <50%  → layer is cohort-specific, not generalizable

Output: results/phase3/e2_cohort_holdout_cv.csv
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
OUT_CSV = Path("results/phase3/e2_cohort_holdout_cv.csv")
ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
SEED = 0


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


# Same configs as D.37
CONFIGS = []
for fold_id, also_aida in [("loco_onek1k", True), ("loco_terekhova", False)]:
    for cell_type in ["CD4+ T", "B", "NK"]:
        CONFIGS.append((fold_id, cell_type, 0, "frozen_base_alllayers", "geneformer_frozen_seed0", also_aida))

for fold_id, also_aida in [("loco_onek1k", True), ("loco_terekhova", False)]:
    for seed, tag in [(1, "frozen_base_seed1_alllayers"), (2, "frozen_base_seed2_alllayers")]:
        CONFIGS.append((fold_id, "NK", seed, tag, f"geneformer_frozen_seed{seed}", also_aida))

rank16_seed_tags = [
    (0, "loco_onek1k_e5b_alllayers"),
    (1, "loco_onek1k_CD4pT_e5b_seed1_alllayers"),
    (2, "loco_onek1k_CD4pT_e5b_seed2_alllayers"),
]
for seed, tag in rank16_seed_tags:
    CONFIGS.append(("loco_onek1k", "CD4+ T", seed, tag, f"geneformer_rank16_seed{seed}", True))

rank32_seed_tags = [
    (0, "loco_onek1k_CD4pT_e5b_r32_alllayers"),
    (1, "loco_onek1k_CD4pT_e5b_r32_seed1_alllayers"),
    (2, "loco_onek1k_CD4pT_e5b_r32_seed2_alllayers"),
]
for seed, tag in rank32_seed_tags:
    CONFIGS.append(("loco_onek1k", "CD4+ T", seed, tag, f"geneformer_rank32_seed{seed}", True))


def main():
    folds = json.loads(Path("data/loco_folds.json").read_text())["folds"]
    fmap = {f["fold_id"]: f for f in folds}
    d37 = pd.read_csv(D37_CSV)

    rows = []
    for fold_id, cell_type, seed, tag, method_label, also_aida in CONFIGS:
        f = fmap[fold_id]
        train_cohorts = f["train_cohorts"]
        if len(train_cohorts) < 2:
            continue

        # Load each train cohort separately
        per_cohort_data = {}
        for tc in train_cohorts:
            ret = _load_npz(tc, cell_type, tag)
            if ret is None:
                print(f"  [SKIP] missing {tc}_{_slug(cell_type)}_{tag}")
                per_cohort_data = None
                break
            per_cohort_data[tc] = ret  # (donor_ids, ages, emb_LDH)
        if per_cohort_data is None:
            continue

        eval_ret = _load_npz(f["holdout_cohort"], cell_type, tag)
        if eval_ret is None:
            continue
        _, eval_y, eval_X_layered = eval_ret
        n_layers = eval_X_layered.shape[0]

        aida_ret = _load_npz("aida", cell_type, tag) if also_aida else None

        # D.37 reference
        ref = d37[(d37["method"] == method_label) & (d37["fold"] == fold_id) &
                  (d37["cell_type"] == cell_type) & (d37["seed"] == seed)]
        kfold_pick = int(ref["L_cv_R_selected"].iloc[0]) if len(ref) > 0 else -1
        oracle_layer = int(ref["L_oracle_holdout_R"].iloc[0]) if len(ref) > 0 else -1

        # Full train (for final eval)
        full_train_X_per_layer = []
        full_train_y_all = []
        for tc in train_cohorts:
            _, ages, emb_LDH = per_cohort_data[tc]
            full_train_X_per_layer.append(emb_LDH)
            full_train_y_all.append(ages)
        full_train_X_layered = np.concatenate(full_train_X_per_layer, axis=1)
        full_train_y = np.concatenate(full_train_y_all)

        print(f"\n=== {method_label} | {fold_id} × {cell_type} × seed{seed} ===", flush=True)

        # For each cohort-holdout configuration
        for c_inner in train_cohorts:
            inner_train_cohorts = [c for c in train_cohorts if c != c_inner]
            if len(inner_train_cohorts) == 0:
                continue
            c_inner_train = inner_train_cohorts[0]  # only one in 2-cohort case
            inner_train_X = per_cohort_data[c_inner_train][2]
            inner_train_y = per_cohort_data[c_inner_train][1]
            inner_val_X = per_cohort_data[c_inner][2]
            inner_val_y = per_cohort_data[c_inner][1]

            # Layer selection by inner-validation R
            R_per_layer = np.zeros(n_layers)
            for layer in range(n_layers):
                r, _ = _fit_ridge_eval(
                    inner_train_X[layer], inner_train_y,
                    inner_val_X[layer], inner_val_y,
                )
                R_per_layer[layer] = r
            L_cv = int(np.argmax(R_per_layer))

            # Refit at L_cv on full train, evaluate on holdout + AIDA
            r_holdout, mae_holdout = _fit_ridge_eval(
                full_train_X_layered[L_cv], full_train_y,
                eval_X_layered[L_cv], eval_y,
            )
            r_aida, mae_aida = (None, None)
            if aida_ret is not None:
                _, aida_y, aida_X_layered = aida_ret
                r_aida, mae_aida = _fit_ridge_eval(
                    full_train_X_layered[L_cv], full_train_y,
                    aida_X_layered[L_cv], aida_y,
                )

            # Also compute oracle holdout R/MAE for reference at this L_cv
            agree_kfold = (L_cv == kfold_pick)
            agree_oracle = (L_cv == oracle_layer)

            rows.append({
                "method": method_label,
                "fold": fold_id,
                "cell_type": cell_type,
                "seed": seed,
                "inner_validation_cohort": c_inner,
                "inner_train_cohort": c_inner_train,
                "L_cohort_holdout_cv": L_cv,
                "L_kfold_cv_d37": kfold_pick,
                "L_oracle_holdout_d37": oracle_layer,
                "agree_with_kfold_cv": agree_kfold,
                "agree_with_oracle": agree_oracle,
                "holdout_R_at_L_cv": r_holdout,
                "holdout_MAE_at_L_cv": mae_holdout,
                "aida_R_at_L_cv": r_aida,
                "aida_MAE_at_L_cv": mae_aida,
                "R_per_layer": R_per_layer.tolist(),
            })
            print(f"  inner_val={c_inner} (train={c_inner_train}) → L_cv={L_cv} | "
                  f"k-fold-CV={kfold_pick} oracle={oracle_layer} | "
                  f"holdout R={r_holdout:+.3f} MAE={mae_holdout:.2f}"
                  + (f" | AIDA R={r_aida:+.3f}" if r_aida is not None else ""),
                  flush=True)

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, float_format="%.4f")
    print(f"\n[E.2] wrote {len(df)} rows to {OUT_CSV}")

    # Decision summary
    n_total = len(df)
    n_agree_kfold = df["agree_with_kfold_cv"].sum()
    n_agree_oracle = df["agree_with_oracle"].sum()
    print(f"\n=== E.2 Decision summary ===")
    print(f"  Total cohort-holdout configurations: {n_total}")
    print(f"  Agreement with K-fold CV (D.37 pick): {n_agree_kfold}/{n_total} ({100*n_agree_kfold/n_total:.1f}%)")
    print(f"  Agreement with oracle (test-best):    {n_agree_oracle}/{n_total} ({100*n_agree_oracle/n_total:.1f}%)")
    pct_kf = n_agree_kfold / n_total
    if pct_kf >= 0.75:
        print("  → DECISION: cohort-holdout CV agrees with K-fold CV; recipe survives")
    elif pct_kf >= 0.50:
        print("  → DECISION: 50-75% agreement; cohort-specific caveat")
    else:
        print("  → DECISION: <50% agreement; layer choice is cohort-specific")

    print("\n=== Per-method breakdown ===")
    for m, g in df.groupby("method"):
        n = len(g)
        a_kf = g["agree_with_kfold_cv"].sum()
        a_or = g["agree_with_oracle"].sum()
        print(f"  {m}: agree-K-fold={a_kf}/{n}, agree-oracle={a_or}/{n}")


if __name__ == "__main__":
    main()
