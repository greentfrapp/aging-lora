"""I.6 — gene-EN 3-seed across caps {50, 100, 500, 1000}.

For each (cap, seed, fold), build per-donor log1p-mean pseudobulks at
that cap and seed, fit ElasticNetCV (HVG-5000 + StandardScaler), and
evaluate on the holdout cohort + AIDA cross-ancestry.

Output: results/phase3/i6_gene_en_3seed_caps.csv
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import pearsonr
from sklearn.linear_model import ElasticNetCV, ElasticNet
from sklearn.preprocessing import StandardScaler

from src.finetune.data_loader import select_indices


CELL_TYPE_TO_FILE = {"CD4+ T": "CD4p_T.h5ad"}
INTEGRATED_DIR = Path("data/cohorts/integrated")
AIDA_DIR = Path("data/cohorts/aida_eval")
ALPHAS = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
L1_RATIOS = [0.3, 0.5, 0.7, 0.9]
TOP_N_HVG = 5000
CAPS = [50, 100, 500, 1000]
SEEDS = [0, 1, 2]
OUT_CSV = Path("results/phase3/i6_gene_en_3seed_caps.csv")


def _h5ad(cohort: str, cell_type: str) -> Path:
    base = AIDA_DIR if cohort == "aida" else INTEGRATED_DIR
    return base / CELL_TYPE_TO_FILE[cell_type]


def _aida_donors():
    raw = json.loads(Path("data/aida_split.json").read_text())["ancestry_shift_mae_donors"]
    return [d if d.startswith("aida:") else f"aida:{d}" for d in raw]


def _resolve_var_symbols(var):
    syms = np.asarray(var.get("gene_symbol", pd.Series(np.nan, index=var.index)).astype(object))
    idx = np.asarray(var.index.astype(str))
    return np.where(pd.isna(syms) | (syms == "nan"), idx, syms.astype(object).astype(str)).astype(str)


def _per_donor_log1p_mean(h5ad_path, idx, donors, ages):
    a = ad.read_h5ad(h5ad_path, backed="r")
    var_syms = _resolve_var_symbols(a.var)
    n_genes = len(var_syms)
    unique_donors = np.unique(donors)
    n_d = len(unique_donors)
    X_donor = np.zeros((n_d, n_genes), dtype=np.float32)
    y_donor = np.zeros(n_d, dtype=np.float32)
    for i, d in enumerate(unique_donors):
        m = donors == d
        sub_idx = idx[m]
        rows = a.X[sub_idx]
        if sp.issparse(rows):
            rows_dense = rows.toarray().astype(np.float32)
        else:
            rows_dense = np.asarray(rows, dtype=np.float32)
        totals = rows_dense.sum(axis=1, keepdims=True)
        safe = np.where(totals > 0, totals, 1.0)
        per_cell = np.log1p(rows_dense / safe * 1e4)
        X_donor[i] = per_cell.mean(axis=0)
        y_donor[i] = ages[m][0]
    a.file.close()
    return unique_donors, X_donor, y_donor, var_syms


def _align(train_syms, eval_syms, eval_X):
    train_set = {s: i for i, s in enumerate(train_syms)}
    out = np.zeros((eval_X.shape[0], len(train_syms)), dtype=np.float32)
    for j, s in enumerate(eval_syms):
        i = train_set.get(s)
        if i is not None:
            out[:, i] = eval_X[:, j]
    return out


def _bootstrap_ci(pred, y, seed, n_boot=500):
    rng = np.random.default_rng(seed + 9999)
    n = len(pred)
    rs = []
    for _ in range(n_boot):
        i = rng.integers(0, n, size=n)
        if pred[i].std() == 0 or y[i].std() == 0:
            continue
        rs.append(pearsonr(pred[i], y[i])[0])
    return float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))


def _build(cohort, cell_type, max_cells, seed):
    h5ad = _h5ad(cohort, cell_type)
    if cohort == "aida":
        idx, ages, donors = select_indices(
            h5ad, cell_type=cell_type, cohorts=None,
            include_donors=_aida_donors(), max_cells_per_donor=max_cells, rng_seed=seed,
        )
    else:
        idx, ages, donors = select_indices(
            h5ad, cell_type=cell_type, cohorts=[cohort],
            max_cells_per_donor=max_cells, rng_seed=seed,
        )
    return _per_donor_log1p_mean(h5ad, idx, donors, ages)


def main():
    folds = json.loads(Path("data/loco_folds.json").read_text())["folds"]
    fmap = {f["fold_id"]: f for f in folds}
    cell_type = "CD4+ T"

    rows = []
    for seed in SEEDS:
        for cap in CAPS:
            print(f"\n=== seed={seed} CAP={cap} ===", flush=True)
            for fold_id in ["loco_onek1k", "loco_terekhova"]:
                f = fmap[fold_id]
                eval_cohort = f["holdout_cohort"]
                t0 = time.time()
                train_X_list, train_y_list, ref_syms = [], [], None
                for tc in f["train_cohorts"]:
                    _, X_tc, y_tc, syms_tc = _build(tc, cell_type, cap, seed)
                    if ref_syms is None:
                        ref_syms = syms_tc
                        train_X_list.append(X_tc)
                    else:
                        train_X_list.append(_align(ref_syms, syms_tc, X_tc))
                    train_y_list.append(y_tc)
                train_X = np.concatenate(train_X_list, axis=0)
                train_y = np.concatenate(train_y_list)

                _, eval_X_raw, eval_y, eval_syms = _build(eval_cohort, cell_type, cap, seed)
                eval_X = _align(ref_syms, eval_syms, eval_X_raw)

                train_var = train_X.var(axis=0)
                top_idx = np.argsort(-train_var)[:TOP_N_HVG]
                train_X = train_X[:, top_idx]
                eval_X = eval_X[:, top_idx]

                scaler = StandardScaler().fit(train_X)
                train_X_s = scaler.transform(train_X).astype(np.float32)
                eval_X_s = scaler.transform(eval_X).astype(np.float32)

                rng = np.random.default_rng(seed)
                perm = rng.permutation(len(train_y))
                cv = ElasticNetCV(
                    l1_ratio=L1_RATIOS, alphas=ALPHAS, cv=3, max_iter=5000,
                    n_jobs=-1, random_state=seed, selection="cyclic",
                )
                cv.fit(train_X_s[perm], train_y[perm])
                alpha = float(cv.alpha_)
                l1 = float(cv.l1_ratio_)
                final = ElasticNet(alpha=alpha, l1_ratio=l1, max_iter=5000).fit(train_X_s, train_y)
                n_nz = int(np.sum(final.coef_ != 0))

                pred = final.predict(eval_X_s)
                r, _ = pearsonr(pred, eval_y)
                mae = float(np.median(np.abs(pred - eval_y)))
                ci = _bootstrap_ci(pred, eval_y, seed)
                elapsed = time.time() - t0

                rows.append({
                    "seed": seed, "cap": cap, "fold": fold_id, "eval_cohort": eval_cohort,
                    "n_train": int(len(train_y)), "n_eval": int(len(eval_y)),
                    "alpha": alpha, "l1_ratio": l1, "n_nonzero_genes": n_nz,
                    "R": float(r), "MAE": mae, "R_ci_lo": ci[0], "R_ci_hi": ci[1],
                })
                print(f"  seed={seed} cap={cap:5d} {fold_id:14s} → {eval_cohort:12s}: R={r:+.3f} MAE={mae:.2f} ({elapsed:.0f}s)", flush=True)

                _, aida_X_raw, aida_y, aida_syms = _build("aida", cell_type, cap, seed)
                aida_X = _align(ref_syms, aida_syms, aida_X_raw)
                aida_X = aida_X[:, top_idx]
                aida_X_s = scaler.transform(aida_X).astype(np.float32)
                apred = final.predict(aida_X_s)
                ar, _ = pearsonr(apred, aida_y)
                amae = float(np.median(np.abs(apred - aida_y)))
                aci = _bootstrap_ci(apred, aida_y, seed)
                rows.append({
                    "seed": seed, "cap": cap, "fold": fold_id, "eval_cohort": "aida",
                    "n_train": int(len(train_y)), "n_eval": int(len(aida_y)),
                    "alpha": alpha, "l1_ratio": l1, "n_nonzero_genes": n_nz,
                    "R": float(ar), "MAE": amae, "R_ci_lo": aci[0], "R_ci_hi": aci[1],
                })
                print(f"  seed={seed} cap={cap:5d} {fold_id:14s} → AIDA       : R={ar:+.3f} MAE={amae:.2f}", flush=True)

            # Snapshot CSV after each cap-fold completes (resumable).
            df = pd.DataFrame(rows)
            OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(OUT_CSV, index=False, float_format="%.4f")

    df = pd.DataFrame(rows)
    print(f"\n[I.6 gene-EN] wrote {len(df)} rows to {OUT_CSV}")

    print("\n=== I.6 gene-EN 3-seed mean AIDA R per cap × fold ===")
    aida = df[df["eval_cohort"] == "aida"]
    for fold in ["loco_onek1k", "loco_terekhova"]:
        sub = aida[aida["fold"] == fold]
        if len(sub) == 0:
            continue
        print(f"\nFold: {fold}")
        for cap in sorted(sub["cap"].unique()):
            vals = sub[sub["cap"] == cap]["R"]
            print(f"  cap={cap:5d}: 3-seed mean R = {vals.mean():+.3f} ± {vals.std():.3f}  (n={len(vals)})")


if __name__ == "__main__":
    main()
