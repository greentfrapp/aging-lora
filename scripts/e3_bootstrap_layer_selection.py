"""E.3 — Bootstrap CIs on layer selection itself.

For each (fold × cell × seed × method-tag):
  Bootstrap-resample train donors with replacement N times.
  Per bootstrap: run K-fold CV per layer → record CV-best layer.
  Output: distribution over "which layer wins" across bootstraps.

Quantifies layer-choice uncertainty under donor sampling.

Decision rule (pre-commit):
  Single layer wins:
    >=70% of bootstraps → layer choice is robust
    40-70%             → moderate uncertainty (top-2 layer band)
    <40%               → noisy under donor sampling (layer band only)

Output: results/phase3/e3_bootstrap_layer_selection.csv

Optimization:
- Pre-load layered embeddings once per condition
- Use a fixed alpha (median of D.37-selected alphas) to avoid RidgeCV per fit
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from scipy.stats import pearsonr


D37_CSV = Path("results/phase3/d37_cv_layer_selection.csv")
EMB_DIR = Path("results/phase3/embeddings_layered")
OUT_CSV = Path("results/phase3/e3_bootstrap_layer_selection.csv")
N_BOOT = 200
N_INNER_CV = 5
SEED = 0
ALPHA_FIXED = 10.0  # Robust default for ridge across the layered embeddings (per D.37 alphas selected)


def _slug(cell_type: str) -> str:
    return cell_type.replace("+", "p").replace(" ", "_")


def _load_npz(cohort: str, cell_type: str, tag: str):
    path = EMB_DIR / f"{cohort}_{_slug(cell_type)}_{tag}.npz"
    if not path.exists():
        return None
    z = np.load(path, allow_pickle=True)
    return z["donor_ids"], z["ages"].astype(np.float32), z["embeddings_per_layer"].astype(np.float32)


def _bootstrap_layer_pick(train_X_layered, train_y, n_boot=N_BOOT, k=N_INNER_CV, alpha=ALPHA_FIXED):
    """For each bootstrap: sample donors w/ replacement, run K-fold CV, pick argmax(layer)."""
    n_layers, n_donors, _ = train_X_layered.shape
    if n_donors < k * 2:
        k = max(2, n_donors // 5)

    rng = np.random.default_rng(SEED)
    layer_wins = np.zeros(n_layers, dtype=np.int64)
    cv_R_per_layer_acc = np.zeros((n_boot, n_layers))

    for b in range(n_boot):
        boot_idx = rng.integers(0, n_donors, size=n_donors)
        boot_y = train_y[boot_idx]

        kf = KFold(n_splits=k, shuffle=True, random_state=int(rng.integers(1, 10**6)))
        layer_R_means = np.zeros(n_layers)

        for layer in range(n_layers):
            X_layer_boot = train_X_layered[layer][boot_idx]
            rs = []
            for tr_idx, te_idx in kf.split(np.arange(n_donors)):
                X_tr, y_tr = X_layer_boot[tr_idx], boot_y[tr_idx]
                X_te, y_te = X_layer_boot[te_idx], boot_y[te_idx]
                if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
                    continue
                m = Ridge(alpha=alpha).fit(X_tr, y_tr)
                pred = m.predict(X_te)
                if np.std(pred) > 0 and np.std(y_te) > 0:
                    r, _ = pearsonr(pred, y_te)
                    rs.append(r)
            layer_R_means[layer] = np.mean(rs) if rs else 0.0

        L_pick = int(np.argmax(layer_R_means))
        layer_wins[L_pick] += 1
        cv_R_per_layer_acc[b] = layer_R_means

    return layer_wins, cv_R_per_layer_acc


# Same configs as D.37 (12 multi-seed conditions + 4 frozen non-NK conditions = 16 total)
CONFIGS = []
for fold_id, also_aida in [("loco_onek1k", True), ("loco_terekhova", False)]:
    for cell_type in ["CD4+ T", "B", "NK"]:
        CONFIGS.append((fold_id, cell_type, 0, "frozen_base_alllayers", "geneformer_frozen_seed0"))

for fold_id, also_aida in [("loco_onek1k", True), ("loco_terekhova", False)]:
    for seed, tag in [(1, "frozen_base_seed1_alllayers"), (2, "frozen_base_seed2_alllayers")]:
        CONFIGS.append((fold_id, "NK", seed, tag, f"geneformer_frozen_seed{seed}"))

rank16_seed_tags = [
    (0, "loco_onek1k_e5b_alllayers"),
    (1, "loco_onek1k_CD4pT_e5b_seed1_alllayers"),
    (2, "loco_onek1k_CD4pT_e5b_seed2_alllayers"),
]
for seed, tag in rank16_seed_tags:
    CONFIGS.append(("loco_onek1k", "CD4+ T", seed, tag, f"geneformer_rank16_seed{seed}"))

rank32_seed_tags = [
    (0, "loco_onek1k_CD4pT_e5b_r32_alllayers"),
    (1, "loco_onek1k_CD4pT_e5b_r32_seed1_alllayers"),
    (2, "loco_onek1k_CD4pT_e5b_r32_seed2_alllayers"),
]
for seed, tag in rank32_seed_tags:
    CONFIGS.append(("loco_onek1k", "CD4+ T", seed, tag, f"geneformer_rank32_seed{seed}"))


def main():
    folds = json.loads(Path("data/loco_folds.json").read_text())["folds"]
    fmap = {f["fold_id"]: f for f in folds}
    d37 = pd.read_csv(D37_CSV)

    rows = []
    t_start = time.time()

    for fold_id, cell_type, seed, tag, method_label in CONFIGS:
        f = fmap[fold_id]
        train_X_per_layer, train_y_all = [], []
        skip = False
        for tc in f["train_cohorts"]:
            ret = _load_npz(tc, cell_type, tag)
            if ret is None:
                skip = True
                break
            _, ages, emb_LDH = ret
            train_X_per_layer.append(emb_LDH)
            train_y_all.append(ages)
        if skip:
            continue
        train_X_layered = np.concatenate(train_X_per_layer, axis=1)
        train_y = np.concatenate(train_y_all)

        ref = d37[(d37["method"] == method_label) & (d37["fold"] == fold_id) &
                  (d37["cell_type"] == cell_type) & (d37["seed"] == seed)]
        kfold_pick = int(ref["L_cv_R_selected"].iloc[0]) if len(ref) > 0 else -1
        oracle_pick = int(ref["L_oracle_holdout_R"].iloc[0]) if len(ref) > 0 else -1

        t0 = time.time()
        layer_wins, cv_R_per_layer_acc = _bootstrap_layer_pick(train_X_layered, train_y)
        dt = time.time() - t0
        n_total = layer_wins.sum()
        win_rate = layer_wins / n_total

        # Top-1 and top-2 layer
        sorted_layers = np.argsort(-win_rate)
        top1_layer = int(sorted_layers[0])
        top1_rate = float(win_rate[top1_layer])
        top2_layer = int(sorted_layers[1])
        top2_rate = float(win_rate[top2_layer])
        top1_top2_combined = top1_rate + top2_rate

        if top1_rate >= 0.70:
            stability = "robust"
        elif top1_rate >= 0.40:
            stability = "moderate"
        else:
            stability = "noisy"

        # Mean CV-R at each layer across bootstraps + std
        cv_R_mean = cv_R_per_layer_acc.mean(axis=0)
        cv_R_std = cv_R_per_layer_acc.std(axis=0)

        # Per-layer rows
        for layer in range(len(layer_wins)):
            rows.append({
                "method": method_label,
                "fold": fold_id,
                "cell_type": cell_type,
                "seed": seed,
                "layer": layer,
                "n_bootstraps": int(n_total),
                "wins": int(layer_wins[layer]),
                "win_rate": float(win_rate[layer]),
                "mean_cv_R": float(cv_R_mean[layer]),
                "std_cv_R": float(cv_R_std[layer]),
                "kfold_cv_pick_d37": kfold_pick,
                "oracle_pick_d37": oracle_pick,
                "top1_layer": top1_layer,
                "top1_win_rate": top1_rate,
                "top2_layer": top2_layer,
                "top2_win_rate": top2_rate,
                "stability": stability,
            })

        elapsed = time.time() - t_start
        print(f"[E.3] {method_label} {fold_id} {cell_type} seed{seed}: "
              f"top1=L{top1_layer} ({top1_rate:.1%}) top2=L{top2_layer} ({top2_rate:.1%}) "
              f"stability={stability} | k-fold CV={kfold_pick} oracle={oracle_pick} | "
              f"{dt:.1f}s | total {elapsed/60:.1f}min", flush=True)

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, float_format="%.4f")
    print(f"\n[E.3] wrote {len(df)} rows to {OUT_CSV}")

    # Decision summary
    summary = df.drop_duplicates(subset=["method", "fold", "cell_type", "seed"])[
        ["method", "fold", "cell_type", "seed", "top1_layer", "top1_win_rate",
         "top2_layer", "top2_win_rate", "kfold_cv_pick_d37", "oracle_pick_d37", "stability"]
    ].copy()
    print("\n=== E.3 Per-condition summary ===")
    print(summary.to_string(index=False, float_format="%.3f"))

    n_total = len(summary)
    n_robust = (summary["stability"] == "robust").sum()
    n_moderate = (summary["stability"] == "moderate").sum()
    n_noisy = (summary["stability"] == "noisy").sum()
    print(f"\n=== E.3 Decision summary ({n_total} conditions) ===")
    print(f"  robust   (top1 >= 70%):  {n_robust}/{n_total}")
    print(f"  moderate (40-70%):        {n_moderate}/{n_total}")
    print(f"  noisy    (<40%):          {n_noisy}/{n_total}")


if __name__ == "__main__":
    main()
