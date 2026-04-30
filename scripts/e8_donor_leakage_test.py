"""E.8 — Donor-identity-leakage test on frozen × NK × Terekhova × 3 seeds.

Hypothesis: bootstrap-with-replacement allows the same donor in train and test
folds within a single bootstrap, biasing toward layers that encode donor
identity. If true, switching to subsampling-WITHOUT-replacement should change
the layer pick.

For each (fold × cell × seed × method-tag), do TWO bootstrap variants:
  1. WITH replacement (matches E.3): rng.integers(0, n, size=n) — duplicates allowed
  2. WITHOUT replacement: rng.choice(n, size=int(0.8*n), replace=False) — subsample 80%

For each: K-fold CV per resample, distribution of CV-best layer.

Compare top-1 layer pick + win-rate between the two methods.

Decision rule (post-hoc, since this is exploratory):
  - If WITHOUT-replacement picks a different layer (e.g., L2 instead of L3 for
    frozen × NK × Terekhova) with ≥70% confidence, donor-identity-leakage is
    the explanation.
  - If WITHOUT-replacement picks the same layer, leakage is NOT the issue.

Focus: frozen × NK × loco_terekhova × seeds 0/1/2 (the persistent
bootstrap-vs-oracle gap).

Optional comparison: frozen × NK × loco_onek1k × seeds 0/1/2 (where E.3 was
already noisy).

Output: results/phase3/e8_donor_leakage_test.csv
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from scipy.stats import pearsonr


EMB_DIR = Path("results/phase3/embeddings_layered")
OUT_CSV = Path("results/phase3/e8_donor_leakage_test.csv")
N_BOOT = 200
N_INNER_CV = 5
SEED = 0
ALPHA_FIXED = 10.0
SUBSAMPLE_FRACTION = 0.8


def _slug(cell_type: str) -> str:
    return cell_type.replace("+", "p").replace(" ", "_")


def _load_npz(cohort: str, cell_type: str, tag: str):
    path = EMB_DIR / f"{cohort}_{_slug(cell_type)}_{tag}.npz"
    if not path.exists():
        return None
    z = np.load(path, allow_pickle=True)
    return z["donor_ids"], z["ages"].astype(np.float32), z["embeddings_per_layer"].astype(np.float32)


def _bootstrap_layer_pick(train_X_layered, train_y, replacement: bool,
                           n_boot=N_BOOT, k=N_INNER_CV, alpha=ALPHA_FIXED):
    """Bootstrap-resample donors, run K-fold CV per resample, pick argmax layer."""
    n_layers, n_donors, _ = train_X_layered.shape
    if n_donors < k * 2:
        k = max(2, n_donors // 5)
    if not replacement:
        n_sample = max(int(SUBSAMPLE_FRACTION * n_donors), k * 2)
    else:
        n_sample = n_donors

    rng = np.random.default_rng(SEED)
    layer_wins = np.zeros(n_layers, dtype=np.int64)

    for b in range(n_boot):
        if replacement:
            boot_idx = rng.integers(0, n_donors, size=n_sample)
        else:
            boot_idx = rng.choice(n_donors, size=n_sample, replace=False)

        boot_y = train_y[boot_idx]
        kf = KFold(n_splits=k, shuffle=True, random_state=int(rng.integers(1, 10**6)))
        layer_R_means = np.zeros(n_layers)

        for layer in range(n_layers):
            X_layer_boot = train_X_layered[layer][boot_idx]
            rs = []
            for tr_idx, te_idx in kf.split(np.arange(n_sample)):
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

    return layer_wins


# Focus configs: frozen × NK × Terekhova (the persistent bootstrap-vs-oracle gap)
# Plus loco_onek1k for comparison
CONFIGS = [
    ("loco_terekhova", "NK", 0, "frozen_base_alllayers", "geneformer_frozen_seed0"),
    ("loco_terekhova", "NK", 1, "frozen_base_seed1_alllayers", "geneformer_frozen_seed1"),
    ("loco_terekhova", "NK", 2, "frozen_base_seed2_alllayers", "geneformer_frozen_seed2"),
    ("loco_onek1k", "NK", 0, "frozen_base_alllayers", "geneformer_frozen_seed0"),
    ("loco_onek1k", "NK", 1, "frozen_base_seed1_alllayers", "geneformer_frozen_seed1"),
    ("loco_onek1k", "NK", 2, "frozen_base_seed2_alllayers", "geneformer_frozen_seed2"),
]


def main():
    folds = json.loads(Path("data/loco_folds.json").read_text())["folds"]
    fmap = {f["fold_id"]: f for f in folds}

    rows = []
    for fold_id, cell_type, seed, tag, method_label in CONFIGS:
        f = fmap[fold_id]
        train_X_per_layer, train_y_all = [], []
        for tc in f["train_cohorts"]:
            ret = _load_npz(tc, cell_type, tag)
            _, ages, emb_LDH = ret
            train_X_per_layer.append(emb_LDH)
            train_y_all.append(ages)
        train_X_layered = np.concatenate(train_X_per_layer, axis=1)
        train_y = np.concatenate(train_y_all)
        n_donors = train_X_layered.shape[1]
        n_layers = train_X_layered.shape[0]

        print(f"\n=== {method_label} | {fold_id} × {cell_type} × seed{seed} | {n_donors} train donors ===", flush=True)

        for label, replacement in [("with_repl", True), ("without_repl", False)]:
            t0 = time.time()
            layer_wins = _bootstrap_layer_pick(train_X_layered, train_y, replacement=replacement)
            dt = time.time() - t0
            n_total = layer_wins.sum()
            win_rate = layer_wins / n_total
            sorted_layers = np.argsort(-win_rate)
            top1, top1_rate = int(sorted_layers[0]), float(win_rate[sorted_layers[0]])
            top2, top2_rate = int(sorted_layers[1]), float(win_rate[sorted_layers[1]])

            rows.append({
                "method": method_label,
                "fold": fold_id,
                "cell_type": cell_type,
                "seed": seed,
                "n_donors_train": n_donors,
                "resample_method": label,
                "n_boot": int(n_total),
                "top1_layer": top1,
                "top1_win_rate": top1_rate,
                "top2_layer": top2,
                "top2_win_rate": top2_rate,
                "win_rates_per_layer": win_rate.tolist(),
            })
            print(f"  {label:13s}: top1=L{top1} ({top1_rate:.1%}), top2=L{top2} ({top2_rate:.1%}) | {dt:.1f}s", flush=True)

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, float_format="%.4f")
    print(f"\n[E.8] wrote {len(df)} rows to {OUT_CSV}")

    # Decision summary
    print("\n=== E.8 Layer-pick comparison: with-replacement vs without ===")
    for (method, fold_id, ct, seed), grp in df.groupby(["method", "fold", "cell_type", "seed"]):
        with_row = grp[grp["resample_method"] == "with_repl"].iloc[0]
        wo_row = grp[grp["resample_method"] == "without_repl"].iloc[0]
        same = with_row["top1_layer"] == wo_row["top1_layer"]
        marker = "SAME LAYER" if same else "DIFFERENT LAYER"
        print(f"  {method:25s} {fold_id:14s} seed{seed}: with-repl=L{with_row['top1_layer']} ({with_row['top1_win_rate']:.1%}) | without-repl=L{wo_row['top1_layer']} ({wo_row['top1_win_rate']:.1%}) → {marker}")

    print("\n=== Decision (donor-identity-leakage hypothesis) ===")
    n_diff = sum(1 for (_, fold_id, _, _), grp in df.groupby(["method", "fold", "cell_type", "seed"])
                 if grp[grp["resample_method"] == "with_repl"]["top1_layer"].iloc[0] !=
                    grp[grp["resample_method"] == "without_repl"]["top1_layer"].iloc[0])
    print(f"  {n_diff}/6 conditions changed top-1 layer pick when switching to without-replacement.")
    if n_diff >= 4:
        print("  → STRONG SUPPORT for donor-identity-leakage hypothesis.")
    elif n_diff >= 2:
        print("  → PARTIAL support; layer choice is partially leakage-driven.")
    else:
        print("  → NO SUPPORT; bootstrap pick is stable across resampling methods (leakage NOT the explanation).")


if __name__ == "__main__":
    main()
