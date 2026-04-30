"""I.1 — Gene-EN cap-sweep on CD4+T (matched-splits ElasticNet at varying caps).

Tests whether F.3's cap=20 → cap=100 effect is FM-specific or shared with the
gene-EN bulk baseline.

For each cap ∈ {20, 100, 500, 5000 (≈full)}:
  Build per-donor log1p-mean pseudobulks for CD4+T at cap cells/donor.
  Fit ElasticNetCV; evaluate on AIDA + holdout cohort.

Output: results/phase3/i1_gene_en_cap_sweep.csv

Decision rule (pre-commit):
  Gene-EN cap=20 R ~ 0.55-0.60 on AIDA → bulk also benefits from cap; FM
    cap=100 advantage shrinks (0.71 - gene-EN 0.62 = +0.09 R, modest).
  Gene-EN cap=20 R ~ 0.40-0.50 on AIDA → bulk also gains but FM gains more;
    FM-vs-bulk gap widens at cap=100 → FM headline supported.
  Gene-EN at cap=full R >= 0.70 on AIDA → bulk plateau equals or exceeds FM
    cap=100; methodology contribution must reframe around layer choice.
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


CELL_TYPE_TO_FILE = {
    "CD4+ T": "CD4p_T.h5ad",
}
INTEGRATED_DIR = Path("data/cohorts/integrated")
AIDA_DIR = Path("data/cohorts/aida_eval")
ALPHAS = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
L1_RATIOS = [0.3, 0.5, 0.7, 0.9]
TOP_N_HVG = 5000
SEED = 0
CAPS = [20, 100, 500, 5000]
OUT_CSV = Path("results/phase3/i1_gene_en_cap_sweep.csv")


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


def _bootstrap_ci(pred, y, n_boot=500):
    rng = np.random.default_rng(SEED)
    n = len(pred)
    rs = []
    for _ in range(n_boot):
        i = rng.integers(0, n, size=n)
        if pred[i].std() == 0 or y[i].std() == 0:
            continue
        rs.append(pearsonr(pred[i], y[i])[0])
    return float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))


def _build(cohort, cell_type, max_cells):
    h5ad = _h5ad(cohort, cell_type)
    if cohort == "aida":
        idx, ages, donors = select_indices(
            h5ad, cell_type=cell_type, cohorts=None,
            include_donors=_aida_donors(), max_cells_per_donor=max_cells, rng_seed=SEED,
        )
    else:
        idx, ages, donors = select_indices(
            h5ad, cell_type=cell_type, cohorts=[cohort],
            max_cells_per_donor=max_cells, rng_seed=SEED,
        )
    return _per_donor_log1p_mean(h5ad, idx, donors, ages)


def main():
    folds = json.loads(Path("data/loco_folds.json").read_text())["folds"]
    fmap = {f["fold_id"]: f for f in folds}
    cell_type = "CD4+ T"

    rows = []
    for cap in CAPS:
        print(f"\n=== CAP={cap} ===", flush=True)
        for fold_id, also_aida in [("loco_onek1k", True), ("loco_terekhova", True)]:
            f = fmap[fold_id]
            eval_cohort = f["holdout_cohort"]
            t0 = time.time()
            train_X_list, train_y_list, ref_syms = [], [], None
            for tc in f["train_cohorts"]:
                _, X_tc, y_tc, syms_tc = _build(tc, cell_type, cap)
                if ref_syms is None:
                    ref_syms = syms_tc
                    train_X_list.append(X_tc)
                else:
                    train_X_list.append(_align(ref_syms, syms_tc, X_tc))
                train_y_list.append(y_tc)
            train_X = np.concatenate(train_X_list, axis=0)
            train_y = np.concatenate(train_y_list)

            _, eval_X_raw, eval_y, eval_syms = _build(eval_cohort, cell_type, cap)
            eval_X = _align(ref_syms, eval_syms, eval_X_raw)

            train_var = train_X.var(axis=0)
            top_idx = np.argsort(-train_var)[:TOP_N_HVG]
            train_X = train_X[:, top_idx]
            eval_X = eval_X[:, top_idx]
            ref_syms_kept = ref_syms[top_idx]

            scaler = StandardScaler().fit(train_X)
            train_X_s = scaler.transform(train_X).astype(np.float32)
            eval_X_s = scaler.transform(eval_X).astype(np.float32)

            rng = np.random.default_rng(SEED)
            perm = rng.permutation(len(train_y))
            cv = ElasticNetCV(
                l1_ratio=L1_RATIOS, alphas=ALPHAS, cv=3, max_iter=5000,
                n_jobs=-1, random_state=SEED, selection="cyclic",
            )
            cv.fit(train_X_s[perm], train_y[perm])
            alpha = float(cv.alpha_)
            l1 = float(cv.l1_ratio_)
            final = ElasticNet(alpha=alpha, l1_ratio=l1, max_iter=5000).fit(train_X_s, train_y)
            n_nz = int(np.sum(final.coef_ != 0))

            pred = final.predict(eval_X_s)
            r, _ = pearsonr(pred, eval_y)
            mae = float(np.median(np.abs(pred - eval_y)))
            ci = _bootstrap_ci(pred, eval_y)
            elapsed = time.time() - t0

            rows.append({
                "cap": cap, "fold": fold_id, "eval_cohort": eval_cohort,
                "n_train": int(len(train_y)), "n_eval": int(len(eval_y)),
                "alpha": alpha, "l1_ratio": l1, "n_nonzero_genes": n_nz,
                "R": float(r), "MAE": mae,
                "R_ci_lo": ci[0], "R_ci_hi": ci[1],
            })
            print(f"  cap={cap:5d} {fold_id:14s} → {eval_cohort:12s}: R={r:+.3f} [CI {ci[0]:+.3f}, {ci[1]:+.3f}] MAE={mae:.2f} ({elapsed:.0f}s)", flush=True)

            if also_aida:
                _, aida_X_raw, aida_y, aida_syms = _build("aida", cell_type, cap)
                aida_X = _align(ref_syms, aida_syms, aida_X_raw)
                aida_X = aida_X[:, top_idx]
                aida_X_s = scaler.transform(aida_X).astype(np.float32)
                apred = final.predict(aida_X_s)
                ar, _ = pearsonr(apred, aida_y)
                amae = float(np.median(np.abs(apred - aida_y)))
                aci = _bootstrap_ci(apred, aida_y)
                rows.append({
                    "cap": cap, "fold": fold_id, "eval_cohort": "aida",
                    "n_train": int(len(train_y)), "n_eval": int(len(aida_y)),
                    "alpha": alpha, "l1_ratio": l1, "n_nonzero_genes": n_nz,
                    "R": float(ar), "MAE": amae,
                    "R_ci_lo": aci[0], "R_ci_hi": aci[1],
                })
                print(f"  cap={cap:5d} {fold_id:14s} → AIDA       : R={ar:+.3f} [CI {aci[0]:+.3f}, {aci[1]:+.3f}] MAE={amae:.2f}", flush=True)

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, float_format="%.4f")
    print(f"\n[I.1] wrote {len(df)} rows to {OUT_CSV}")

    print("\n=== I.1 Summary table (gene-EN AIDA R per cap × fold) ===")
    aida_pivot = df[df["eval_cohort"] == "aida"].pivot(index="cap", columns="fold", values="R")
    print(aida_pivot.to_string(float_format="%.3f"))

    print("\n=== Decision rule ===")
    aida_onek1k = df[(df["eval_cohort"] == "aida") & (df["fold"] == "loco_onek1k")]
    if len(aida_onek1k) > 0:
        gene_en_cap20 = aida_onek1k[aida_onek1k["cap"] == 20]["R"].iloc[0] if 20 in aida_onek1k["cap"].values else None
        gene_en_cap100 = aida_onek1k[aida_onek1k["cap"] == 100]["R"].iloc[0] if 100 in aida_onek1k["cap"].values else None
        gene_en_full = aida_onek1k[aida_onek1k["cap"] == 5000]["R"].iloc[0] if 5000 in aida_onek1k["cap"].values else None

        fm_cap100 = 0.706  # F.3 result
        if gene_en_cap20 is not None and gene_en_cap100 is not None:
            print(f"  Gene-EN cap=20 AIDA R = {gene_en_cap20:+.3f}")
            print(f"  Gene-EN cap=100 AIDA R = {gene_en_cap100:+.3f}")
            print(f"  FM cap=100 AIDA R = {fm_cap100:+.3f} (F.3)")
            gap = fm_cap100 - gene_en_cap100
            print(f"  FM-vs-gene-EN gap at cap=100: {gap:+.3f}")
            if gene_en_full is not None and gene_en_full >= 0.70:
                print(f"  → gene-EN at cap=full R={gene_en_full:.3f} ≥ 0.70: bulk plateau exceeds/matches FM cap=100; reframe methodology around layer choice.")
            elif gap >= 0.10:
                print("  → FM cap=100 exceeds gene-EN cap=100 by >=0.10: FM headline supported.")
            elif gap >= 0.05:
                print("  → FM cap=100 exceeds gene-EN cap=100 by 0.05-0.10: modest FM advantage; report both.")
            else:
                print("  → FM cap=100 vs gene-EN cap=100 gap <0.05: FM at cap=100 has minimal advantage; reframe.")


if __name__ == "__main__":
    main()
