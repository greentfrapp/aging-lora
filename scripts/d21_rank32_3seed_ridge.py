"""D.21 — Rank-32 LoRA × 3-seed L9 AIDA verification ridge analysis.

Reads layered embeddings for seeds 0, 1, 2 of rank-32 LoRA on CD4+T × loco_onek1k,
fits ridge per layer, and reports per-seed + 3-seed mean for L9 AIDA cross-ancestry.

Usage:
    uv run python scripts/d21_rank32_3seed_ridge.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from scipy.stats import pearsonr


EMB_DIR = Path("results/phase3/embeddings_layered")
OUT_CSV = Path("results/phase3/d21_rank32_3seed_layered_ridge.csv")
ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
SEED = 0  # ridge alpha-selection RNG, not the cell-sampling seed


def _slug(cell_type: str) -> str:
    return cell_type.replace("+", "p").replace(" ", "_")


def _load_npz(cohort: str, cell_type: str, tag: str):
    path = EMB_DIR / f"{cohort}_{_slug(cell_type)}_{tag}.npz"
    if not path.exists():
        return None
    z = np.load(path, allow_pickle=True)
    return z["donor_ids"], z["ages"].astype(np.float32), z["embeddings_per_layer"].astype(np.float32)


def _bootstrap_pearson_ci(pred, y, seed=0, n_boot=1000):
    rng = np.random.default_rng(seed)
    n = len(y)
    rs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if np.std(pred[idx]) > 0 and np.std(y[idx]) > 0:
            rs.append(pearsonr(pred[idx], y[idx])[0])
    return float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))


def main():
    folds = json.loads(Path("data/loco_folds.json").read_text())["folds"]
    fmap = {f["fold_id"]: f for f in folds}

    # rank-32 tags per seed
    seed_tags = [
        (0, "loco_onek1k_CD4pT_e5b_r32_alllayers"),
        (1, "loco_onek1k_CD4pT_e5b_r32_seed1_alllayers"),
        (2, "loco_onek1k_CD4pT_e5b_r32_seed2_alllayers"),
    ]
    fold_id = "loco_onek1k"
    cell_type = "CD4+ T"
    f = fmap[fold_id]
    rows = []
    available_seeds = []
    for seed, tag in seed_tags:
        train_X_per_layer, train_y_all = [], []
        skip = False
        for tc in f["train_cohorts"]:
            ret = _load_npz(tc, cell_type, tag)
            if ret is None:
                print(f"[seed{seed}] missing embeddings for {tc} ({tag}); skipping")
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
        aida_ret = _load_npz("aida", cell_type, tag)
        if eval_ret is None or aida_ret is None:
            print(f"[seed{seed}] missing eval/aida embeddings; skipping")
            continue
        _, eval_y, eval_X_layered = eval_ret
        _, aida_y, aida_X_layered = aida_ret

        n_layers = train_X_layered.shape[0]
        print(f"\n=== seed {seed} | {n_layers} layers, train n={len(train_y)} eval n={len(eval_y)} aida n={len(aida_y)} ===")

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
            apred = final.predict(aida_X_layered[layer])
            ar, ap = pearsonr(apred, aida_y)
            amae = float(np.median(np.abs(apred - aida_y)))
            aci_lo, aci_hi = _bootstrap_pearson_ci(apred, aida_y, seed=0)
            print(f"  L{layer:>2d}  onek1k R={r:+.3f} MAE={mae:>6.2f} | AIDA R={ar:+.3f} MAE={amae:>6.2f} (CI [{aci_lo:+.3f},{aci_hi:+.3f}])")
            rows.append({
                "seed": seed, "layer": layer, "alpha": alpha,
                "onek1k_r": float(r), "onek1k_mae": mae,
                "aida_r": float(ar), "aida_mae": amae,
                "aida_ci_lo": aci_lo, "aida_ci_hi": aci_hi,
                "n_train": int(len(train_y)), "n_eval": int(len(eval_y)), "n_aida": int(len(aida_y)),
            })
        available_seeds.append(seed)

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, float_format="%.6g")
    print(f"\n[d21] wrote {len(df)} rows to {OUT_CSV}")
    print(f"[d21] available seeds: {available_seeds}")

    if len(available_seeds) >= 2:
        print("\n=== 3-seed (or N-seed) mean ± std per layer ===")
        agg = df.groupby("layer").agg(
            onek1k_r_mean=("onek1k_r", "mean"), onek1k_r_std=("onek1k_r", "std"),
            onek1k_mae_mean=("onek1k_mae", "mean"), onek1k_mae_std=("onek1k_mae", "std"),
            aida_r_mean=("aida_r", "mean"), aida_r_std=("aida_r", "std"),
            aida_mae_mean=("aida_mae", "mean"), aida_mae_std=("aida_mae", "std"),
            n_seeds=("seed", "count"),
        ).reset_index()
        print(agg.to_string(index=False, float_format="%.3f"))

        # Decision rule check on L9 AIDA
        l9 = agg[agg["layer"] == 9].iloc[0] if (agg["layer"] == 9).any() else None
        if l9 is not None:
            mae_mean = float(l9["aida_mae_mean"])
            mae_std = float(l9["aida_mae_std"])
            r_mean = float(l9["aida_r_mean"])
            r_std = float(l9["aida_r_std"])
            n = int(l9["n_seeds"])
            print(f"\n=== L9 AIDA {n}-seed: R = {r_mean:+.3f} ± {r_std:.3f}, MAE = {mae_mean:.2f}y ± {mae_std:.2f}y ===")
            if mae_mean <= 7.5:
                band = "≤ 7.5y → matched-splits parity headline survives, outline (a) viable"
            elif mae_mean <= 8.5:
                band = "7.5y–8.5y → competitive within ~1y, outline (a) hedged"
            else:
                band = "> 8.5y → drop AIDA-parity from headline, outline (b)"
            print(f"  Decision band per notes/decision_rules_phase3.md §D.21: {band}")
            if r_mean < 0.55:
                print(f"  Decision rule on R: r_mean={r_mean:.3f} < 0.55 → §32 parity narrative weakened regardless of MAE")
            if mae_std > 2.0:
                print(f"  Robustness: σ(MAE)={mae_std:.2f}y > 2.0y → finding too noisy to anchor a paper claim")

        agg_csv = OUT_CSV.parent / "d21_rank32_3seed_aggregated.csv"
        agg.to_csv(agg_csv, index=False, float_format="%.6g")
        print(f"\n[d21] wrote aggregated to {agg_csv}")


if __name__ == "__main__":
    main()
