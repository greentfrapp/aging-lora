"""D.36 — Bootstrap MAE CI on rank-32 3-seed L9 AIDA vs gene-EN matched AIDA.

Tests the strictest version of the matched-splits parity claim: do the
bootstrap MAE distributions overlap directly?

For each method:
  1. Reload predictions (or recompute from embeddings).
  2. Bootstrap-resample donors (n=1000), compute MAE per resample.
  3. Compare distributions: 95% CI overlap, Wilcoxon test, etc.

Output: results/phase3/d36_mae_ci_overlap.csv + stdout summary.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, mannwhitneyu
from sklearn.linear_model import Ridge, RidgeCV, ElasticNetCV, ElasticNet
from sklearn.preprocessing import StandardScaler

import anndata as ad
import scipy.sparse as sp


SEED = 0
N_BOOT = 1000


def _bootstrap_MAE(pred, y, seed=0, n_boot=N_BOOT):
    rng = np.random.default_rng(seed)
    n = len(y)
    maes = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        maes.append(np.median(np.abs(pred[idx] - y[idx])))
    return np.array(maes)


def _slug(cell_type: str) -> str:
    return cell_type.replace("+", "p").replace(" ", "_")


def _load_npz(cohort: str, cell_type: str, tag: str):
    path = Path("results/phase3/embeddings_layered") / f"{cohort}_{_slug(cell_type)}_{tag}.npz"
    z = np.load(path, allow_pickle=True)
    return z["donor_ids"], z["ages"].astype(np.float32), z["embeddings_per_layer"].astype(np.float32)


def main():
    folds = json.loads(Path("data/loco_folds.json").read_text())["folds"]
    fmap = {f["fold_id"]: f for f in folds}

    fold_id = "loco_onek1k"
    cell_type = "CD4+ T"
    f = fmap[fold_id]

    # 1. Rank-32 3-seed: re-fit + bootstrap-MAE per seed at L9 AIDA
    seed_tags = [
        (0, "loco_onek1k_CD4pT_e5b_r32_alllayers"),
        (1, "loco_onek1k_CD4pT_e5b_r32_seed1_alllayers"),
        (2, "loco_onek1k_CD4pT_e5b_r32_seed2_alllayers"),
    ]
    rank32_l9_aida_maes = []
    for seed, tag in seed_tags:
        train_X_per_layer, train_y_all = [], []
        for tc in f["train_cohorts"]:
            _, ages, emb_LDH = _load_npz(tc, cell_type, tag)
            train_X_per_layer.append(emb_LDH)
            train_y_all.append(ages)
        train_X_layered = np.concatenate(train_X_per_layer, axis=1)
        train_y = np.concatenate(train_y_all)
        _, aida_y, aida_X_layered = _load_npz("aida", cell_type, tag)

        # Fit at L9
        cv = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0], cv=3, scoring="neg_mean_absolute_error")
        rng = np.random.default_rng(SEED)
        perm = rng.permutation(len(train_y))
        cv.fit(train_X_layered[9][perm], train_y[perm])
        alpha = float(cv.alpha_)
        final = Ridge(alpha=alpha).fit(train_X_layered[9], train_y)
        apred = final.predict(aida_X_layered[9])
        maes = _bootstrap_MAE(apred, aida_y, seed=0)
        rank32_l9_aida_maes.append(maes)
        print(f"[rank-32 L9 AIDA seed{seed}] MAE: median={np.median(maes):.2f}, CI=[{np.percentile(maes, 2.5):.2f}, {np.percentile(maes, 97.5):.2f}]")

    rank32_pooled = np.concatenate(rank32_l9_aida_maes)
    print(f"\nrank-32 3-seed pooled bootstrap (n=3000): median={np.median(rank32_pooled):.2f}, CI=[{np.percentile(rank32_pooled, 2.5):.2f}, {np.percentile(rank32_pooled, 97.5):.2f}]")

    # 2. Gene-EN matched: refit + bootstrap-MAE on AIDA
    print("\n[gene-EN matched AIDA loco_onek1k]")
    # Reuse the matched-splits gene-EN protocol
    from src.finetune.data_loader import select_indices

    def _per_donor_log1p_mean(h5ad_path, idx, donors, ages, max_cells=None):
        a = ad.read_h5ad(h5ad_path, backed="r")
        n_genes = a.var.shape[0]
        unique_donors = np.unique(donors)
        X = np.zeros((len(unique_donors), n_genes), dtype=np.float32)
        y = np.zeros(len(unique_donors), dtype=np.float32)
        for i, d in enumerate(unique_donors):
            m = donors == d
            sub_idx = idx[m]
            rows = a.X[sub_idx]
            if sp.issparse(rows):
                rows_dense = rows.toarray().astype(np.float32)
            else:
                rows_dense = np.asarray(rows, dtype=np.float32)
            sums = rows_dense.sum(axis=1, keepdims=True)
            sums[sums < 1e-9] = 1
            cp10k = (rows_dense * 1e4) / sums
            X[i] = np.log1p(cp10k).mean(axis=0)
            y[i] = ages[m][0]
        a.file.close()
        syms_col = a.var.get("gene_symbol", a.var.get("feature_name", pd.Series(a.var.index, index=a.var.index)))
        syms = np.asarray(syms_col.astype(str).tolist())
        return unique_donors, X, y, syms

    def _build(cohort, max_cells):
        h5ad = Path("data/cohorts/aida_eval/CD4p_T.h5ad") if cohort == "aida" else Path("data/cohorts/integrated/CD4p_T.h5ad")
        cohorts_filter = None if cohort == "aida" else [cohort]
        # AIDA donors filter for cross-ancestry
        if cohort == "aida":
            include_donors = json.loads(Path("data/aida_split.json").read_text())["ancestry_shift_mae_donors"]
            include_donors = [d if d.startswith("aida:") else f"aida:{d}" for d in include_donors]
        else:
            include_donors = None
        idx, ages, donors = select_indices(
            h5ad, cell_type="CD4+ T", cohorts=cohorts_filter,
            include_donors=include_donors, max_cells_per_donor=max_cells, rng_seed=SEED,
        )
        return _per_donor_log1p_mean(h5ad, idx, donors, ages)

    train_X_list, train_y_list, syms_ref = [], [], None
    for tc in f["train_cohorts"]:
        _, X_tc, y_tc, syms_tc = _build(tc, 100)
        if syms_ref is None:
            syms_ref = syms_tc
            train_X_list.append(X_tc)
        else:
            # Align gene columns
            tcmap = {s: i for i, s in enumerate(syms_tc)}
            new_X = np.zeros((X_tc.shape[0], len(syms_ref)), dtype=np.float32)
            for j, s in enumerate(syms_ref):
                if s in tcmap:
                    new_X[:, j] = X_tc[:, tcmap[s]]
            train_X_list.append(new_X)
        train_y_list.append(y_tc)
    train_X = np.concatenate(train_X_list, axis=0)
    train_y = np.concatenate(train_y_list)

    # Build AIDA matrix
    _, aida_X_raw, aida_y, aida_syms = _build("aida", 200)
    aida_map = {s: i for i, s in enumerate(aida_syms)}
    aida_X = np.zeros((aida_X_raw.shape[0], len(syms_ref)), dtype=np.float32)
    for j, s in enumerate(syms_ref):
        if s in aida_map:
            aida_X[:, j] = aida_X_raw[:, aida_map[s]]

    # HVG + standardize
    train_var = train_X.var(axis=0)
    top_idx = np.argsort(-train_var)[:5000]
    train_X = train_X[:, top_idx]
    aida_X = aida_X[:, top_idx]
    scaler = StandardScaler().fit(train_X)
    train_X_s = scaler.transform(train_X).astype(np.float32)
    aida_X_s = scaler.transform(aida_X).astype(np.float32)

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(train_y))
    cv = ElasticNetCV(l1_ratio=[0.3, 0.5, 0.7, 0.9], alphas=[0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0], cv=3, max_iter=5000, n_jobs=-1, random_state=SEED, selection="cyclic")
    cv.fit(train_X_s[perm], train_y[perm])
    final = ElasticNet(alpha=float(cv.alpha_), l1_ratio=float(cv.l1_ratio_), max_iter=5000).fit(train_X_s, train_y)
    apred = final.predict(aida_X_s)
    gene_en_maes = _bootstrap_MAE(apred, aida_y, seed=0)
    print(f"  median={np.median(gene_en_maes):.2f}, CI=[{np.percentile(gene_en_maes, 2.5):.2f}, {np.percentile(gene_en_maes, 97.5):.2f}]")

    # Comparison
    print("\n=== STRICT MAE PARITY TEST (rank-32 L9 vs gene-EN, AIDA loco_onek1k) ===\n")
    print(f"  rank-32 LoRA L9 (3-seed pooled bootstrap, n=3000):")
    print(f"    median MAE = {np.median(rank32_pooled):.2f}y, 95% CI = [{np.percentile(rank32_pooled, 2.5):.2f}, {np.percentile(rank32_pooled, 97.5):.2f}]")
    print(f"  gene-EN matched (1-seed bootstrap, n=1000):")
    print(f"    median MAE = {np.median(gene_en_maes):.2f}y, 95% CI = [{np.percentile(gene_en_maes, 2.5):.2f}, {np.percentile(gene_en_maes, 97.5):.2f}]")

    rank32_lo, rank32_hi = np.percentile(rank32_pooled, [2.5, 97.5])
    gene_lo, gene_hi = np.percentile(gene_en_maes, [2.5, 97.5])
    overlap_lo = max(rank32_lo, gene_lo)
    overlap_hi = min(rank32_hi, gene_hi)
    overlapping = overlap_lo <= overlap_hi
    print(f"\n  CI overlap range: [{overlap_lo:.2f}, {overlap_hi:.2f}] ({'OVERLAP' if overlapping else 'NO OVERLAP'})")

    diff = rank32_pooled.mean() - gene_en_maes.mean()
    print(f"  Mean MAE difference (rank-32 - gene-EN): {diff:+.2f}y")

    # Mann-Whitney U test
    u_stat, p = mannwhitneyu(rank32_pooled, gene_en_maes, alternative="greater")
    print(f"  Mann-Whitney U test (rank-32 > gene-EN): p = {p:.3e}")
    print(f"  Interpretation: p<0.001 means rank-32 MAE is significantly larger; p>0.05 means parity")

    # Save
    out = Path("results/phase3/d36_mae_ci_overlap.csv")
    pd.DataFrame({
        "method": ["rank32_l9_3seed_pooled", "gene_en_matched"],
        "n_bootstrap": [len(rank32_pooled), len(gene_en_maes)],
        "median_mae": [float(np.median(rank32_pooled)), float(np.median(gene_en_maes))],
        "mean_mae": [float(rank32_pooled.mean()), float(gene_en_maes.mean())],
        "ci_lo_2.5": [float(rank32_lo), float(gene_lo)],
        "ci_hi_97.5": [float(rank32_hi), float(gene_hi)],
        "ci_overlap": [overlapping, overlapping],
    }).to_csv(out, index=False, float_format="%.4f")
    print(f"\n[d36] wrote {out}")


if __name__ == "__main__":
    main()
