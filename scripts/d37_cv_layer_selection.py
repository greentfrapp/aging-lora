"""D.37 — Inner-CV layer selection for cell-type-conditional layer methodology.

Removes the test-set-selection bias from the §31/D.22 layer-of-best-readout
finding. For each (fold × cell_type × seed × method-tag):

1. Run K-fold CV on TRAIN donors only — for each layer, fit ridge on K-1
   folds, evaluate on the held-out fold; report CV-mean R per layer.
2. Pick CV-best layer by argmax(CV-mean R) on train.
3. Refit ridge at CV-selected layer on full train, evaluate on holdout cohort
   + AIDA cross-ancestry.
4. Compare to "oracle" (test-best) layer: would deployment R/MAE differ much
   from the post-hoc characterization?

Output: results/phase3/d37_cv_layer_selection.csv

Outcome interpretation (per the user query 2026-04-30):
- CV-selected matches oracle → cell-type-conditional finding is deployable
- CV-selected differs but R/MAE is close → recipe works at small cost
- CV-selected unstable across seeds → finding is post-hoc-only, not deployable
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import KFold
from scipy.stats import pearsonr


EMB_DIR = Path("results/phase3/embeddings_layered")
OUT_CSV = Path("results/phase3/d37_cv_layer_selection.csv")
ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
SEED = 0
N_INNER_CV = 5  # K-fold on train donors for layer selection


def _slug(cell_type: str) -> str:
    return cell_type.replace("+", "p").replace(" ", "_")


def _load_npz(cohort: str, cell_type: str, tag: str):
    path = EMB_DIR / f"{cohort}_{_slug(cell_type)}_{tag}.npz"
    if not path.exists():
        return None
    z = np.load(path, allow_pickle=True)
    return z["donor_ids"], z["ages"].astype(np.float32), z["embeddings_per_layer"].astype(np.float32)


def _fit_ridge(X_train, y_train, X_eval):
    cv = RidgeCV(alphas=ALPHAS, cv=3, scoring="neg_mean_absolute_error")
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(y_train))
    cv.fit(X_train[perm], y_train[perm])
    final = Ridge(alpha=float(cv.alpha_)).fit(X_train, y_train)
    return float(cv.alpha_), final.predict(X_eval)


def _cv_select_layer(train_X_layered, train_y, k=N_INNER_CV):
    """K-fold CV on train donors → CV-mean R per layer → return argmax + scores."""
    n_layers = train_X_layered.shape[0]
    n_donors = train_X_layered.shape[1]
    if n_donors < k * 2:
        # Reduce k for small donor counts (e.g., Stephenson)
        k = max(2, n_donors // 5)
    kf = KFold(n_splits=k, shuffle=True, random_state=SEED)
    fold_indices = list(kf.split(np.arange(n_donors)))
    layer_cv_R_means = np.zeros(n_layers)
    layer_cv_R_stds = np.zeros(n_layers)
    layer_cv_MAE_means = np.zeros(n_layers)
    for layer in range(n_layers):
        rs, maes = [], []
        for tr_idx, te_idx in fold_indices:
            X_tr = train_X_layered[layer][tr_idx]
            y_tr = train_y[tr_idx]
            X_te = train_X_layered[layer][te_idx]
            y_te = train_y[te_idx]
            _, pred = _fit_ridge(X_tr, y_tr, X_te)
            if np.std(pred) > 0 and np.std(y_te) > 0 and len(y_te) > 1:
                r, _ = pearsonr(pred, y_te)
                rs.append(r)
                maes.append(float(np.median(np.abs(pred - y_te))))
        layer_cv_R_means[layer] = np.mean(rs)
        layer_cv_R_stds[layer] = np.std(rs)
        layer_cv_MAE_means[layer] = np.mean(maes)
    L_cv_R = int(np.argmax(layer_cv_R_means))
    L_cv_MAE = int(np.argmin(layer_cv_MAE_means))
    return {
        "L_cv_R_best": L_cv_R, "L_cv_MAE_best": L_cv_MAE,
        "cv_R_per_layer": layer_cv_R_means.tolist(),
        "cv_R_std_per_layer": layer_cv_R_stds.tolist(),
        "cv_MAE_per_layer": layer_cv_MAE_means.tolist(),
        "k_folds_used": k,
    }


def _evaluate_at_layer(train_X_layered, train_y, eval_X_layered, eval_y, layer):
    _, pred = _fit_ridge(train_X_layered[layer], train_y, eval_X_layered[layer])
    if np.std(pred) > 0 and np.std(eval_y) > 0:
        r, _ = pearsonr(pred, eval_y)
    else:
        r = 0.0
    mae = float(np.median(np.abs(pred - eval_y)))
    return float(r), mae


def main():
    folds = json.loads(Path("data/loco_folds.json").read_text())["folds"]
    fmap = {f["fold_id"]: f for f in folds}

    # Configurations to test
    # Each config: (fold_id, cell_type, seed, tag, method_label, also_aida)
    configs = []

    # Frozen Geneformer × all 3 cell types × 2 folds × seed 0 (pre-D.22 baseline)
    for fold_id, also_aida_a in [("loco_onek1k", True), ("loco_terekhova", False)]:
        for cell_type in ["CD4+ T", "B", "NK"]:
            configs.append((fold_id, cell_type, 0, "frozen_base_alllayers", "geneformer_frozen_seed0", also_aida_a))

    # NK frozen × 3 seeds (D.22 verification)
    for fold_id, also_aida_a in [("loco_onek1k", True), ("loco_terekhova", False)]:
        for seed, tag in [(1, "frozen_base_seed1_alllayers"), (2, "frozen_base_seed2_alllayers")]:
            configs.append((fold_id, "NK", seed, tag, f"geneformer_frozen_seed{seed}", also_aida_a))

    # Rank-16 LoRA × CD4+T × loco_onek1k × 3 seeds
    rank16_seed_tags = [
        (0, "loco_onek1k_e5b_alllayers"),
        (1, "loco_onek1k_CD4pT_e5b_seed1_alllayers"),
        (2, "loco_onek1k_CD4pT_e5b_seed2_alllayers"),
    ]
    for seed, tag in rank16_seed_tags:
        configs.append(("loco_onek1k", "CD4+ T", seed, tag, f"geneformer_rank16_seed{seed}", True))

    # Rank-32 LoRA × CD4+T × loco_onek1k × 3 seeds
    rank32_seed_tags = [
        (0, "loco_onek1k_CD4pT_e5b_r32_alllayers"),
        (1, "loco_onek1k_CD4pT_e5b_r32_seed1_alllayers"),
        (2, "loco_onek1k_CD4pT_e5b_r32_seed2_alllayers"),
    ]
    for seed, tag in rank32_seed_tags:
        configs.append(("loco_onek1k", "CD4+ T", seed, tag, f"geneformer_rank32_seed{seed}", True))

    rows = []
    for fold_id, cell_type, seed, tag, method_label, also_aida_a in configs:
        f = fmap[fold_id]
        train_X_per_layer, train_y_all = [], []
        skip = False
        for tc in f["train_cohorts"]:
            ret = _load_npz(tc, cell_type, tag)
            if ret is None:
                print(f"  [skip {method_label} {fold_id} {cell_type} seed{seed}] missing {tc}", flush=True)
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
        aida_ret = _load_npz("aida", cell_type, tag) if also_aida_a else None

        n_layers = train_X_layered.shape[0]
        print(f"\n=== {method_label} | {fold_id} × {cell_type} × seed{seed} | {n_layers} layers, {len(train_y)} train, {len(eval_y)} eval ===", flush=True)

        # Inner CV layer selection
        cv_result = _cv_select_layer(train_X_layered, train_y)
        L_cv_R = cv_result["L_cv_R_best"]
        L_cv_MAE = cv_result["L_cv_MAE_best"]
        print(f"  CV-selected layer (by R): L{L_cv_R}", flush=True)
        print(f"  CV-selected layer (by MAE): L{L_cv_MAE}", flush=True)

        # Evaluate at each layer on holdout + AIDA → identify oracle
        holdout_R_per_layer = np.zeros(n_layers)
        holdout_MAE_per_layer = np.zeros(n_layers)
        aida_R_per_layer = np.zeros(n_layers) if aida_ret else None
        aida_MAE_per_layer = np.zeros(n_layers) if aida_ret else None
        for layer in range(n_layers):
            r, mae = _evaluate_at_layer(train_X_layered, train_y, eval_X_layered, eval_y, layer)
            holdout_R_per_layer[layer] = r
            holdout_MAE_per_layer[layer] = mae
            if aida_ret:
                _, aida_y, aida_X_layered = aida_ret
                ar, amae = _evaluate_at_layer(train_X_layered, train_y, aida_X_layered, aida_y, layer)
                aida_R_per_layer[layer] = ar
                aida_MAE_per_layer[layer] = amae

        L_oracle_R = int(np.argmax(holdout_R_per_layer))
        L_oracle_MAE = int(np.argmin(holdout_MAE_per_layer))
        print(f"  Oracle layer (best holdout R): L{L_oracle_R} (R={holdout_R_per_layer[L_oracle_R]:+.3f})", flush=True)
        print(f"  Oracle layer (best holdout MAE): L{L_oracle_MAE} (MAE={holdout_MAE_per_layer[L_oracle_MAE]:.2f})", flush=True)
        print(f"  CV-selected (by R) on holdout: R={holdout_R_per_layer[L_cv_R]:+.3f}, MAE={holdout_MAE_per_layer[L_cv_R]:.2f}", flush=True)
        if aida_ret:
            L_aida_oracle_R = int(np.argmax(aida_R_per_layer))
            print(f"  AIDA oracle layer (best R): L{L_aida_oracle_R} (R={aida_R_per_layer[L_aida_oracle_R]:+.3f})", flush=True)
            print(f"  AIDA at CV-selected (by R): L{L_cv_R} R={aida_R_per_layer[L_cv_R]:+.3f} MAE={aida_MAE_per_layer[L_cv_R]:.2f}", flush=True)

        rows.append({
            "method": method_label, "fold": fold_id, "cell_type": cell_type, "seed": seed,
            "n_train": int(len(train_y)), "n_eval": int(len(eval_y)),
            "k_folds_inner_cv": cv_result["k_folds_used"],
            "L_cv_R_selected": L_cv_R, "L_cv_MAE_selected": L_cv_MAE,
            "L_oracle_holdout_R": L_oracle_R, "L_oracle_holdout_MAE": L_oracle_MAE,
            "holdout_R_at_cv_R": float(holdout_R_per_layer[L_cv_R]),
            "holdout_MAE_at_cv_R": float(holdout_MAE_per_layer[L_cv_R]),
            "holdout_R_at_oracle": float(holdout_R_per_layer[L_oracle_R]),
            "holdout_MAE_at_oracle_MAE": float(holdout_MAE_per_layer[L_oracle_MAE]),
            "holdout_R_at_L12": float(holdout_R_per_layer[12]),
            "holdout_MAE_at_L12": float(holdout_MAE_per_layer[12]),
            "aida_R_at_cv_R": float(aida_R_per_layer[L_cv_R]) if aida_ret else None,
            "aida_MAE_at_cv_R": float(aida_MAE_per_layer[L_cv_R]) if aida_ret else None,
            "aida_R_at_aida_oracle": float(aida_R_per_layer.max()) if aida_ret else None,
            "aida_oracle_layer": int(np.argmax(aida_R_per_layer)) if aida_ret else None,
            "cv_R_per_layer": cv_result["cv_R_per_layer"],
        })

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, float_format="%.4f")
    print(f"\n[d37] wrote {len(df)} rows to {OUT_CSV}")

    # Summary tables
    print("\n=== Summary: CV-selected vs Oracle layer per (method × cell_type × eval) ===\n")
    cols = ["method", "fold", "cell_type", "seed", "L_cv_R_selected", "L_oracle_holdout_R",
            "holdout_R_at_cv_R", "holdout_R_at_oracle", "holdout_R_at_L12",
            "aida_R_at_cv_R", "aida_R_at_aida_oracle", "aida_oracle_layer"]
    print(df[cols].to_string(index=False, float_format="%.3f"))

    # Cross-seed stability: do same CV layers come out for the same condition?
    print("\n=== Cross-seed CV-layer stability (rank-16, rank-32, NK frozen) ===\n")
    multi_seed = df[df["method"].str.contains("seed", regex=False)].copy()
    multi_seed["method_base"] = multi_seed["method"].str.replace(r"_seed\d+", "", regex=True)
    for (m_base, fold, ct), g in multi_seed.groupby(["method_base", "fold", "cell_type"]):
        if len(g) >= 2:
            cv_layers = g["L_cv_R_selected"].tolist()
            oracle_layers = g["L_oracle_holdout_R"].tolist()
            print(f"  {m_base} | {fold} × {ct}: CV-layers across seeds = {cv_layers}, oracle = {oracle_layers}")


if __name__ == "__main__":
    main()
