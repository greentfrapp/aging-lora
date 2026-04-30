"""G.1 — Composition-additive ensemble.

Tests whether composition (5-d cell-type-fraction vector) adds R beyond what
gene-EN and FM-ridge already capture — answering F.1 reviewer concern that
the borderline +0.298 AIDA composition R may be silently driving part of the
"FM matched-splits parity with gene-EN" headline.

For CD4+T (the matched-splits headline cell):
  Method 1: gene-EN (HVG-5000 log1p-mean-pseudobulk) — 2 folds × {alone, +comp}
  Method 2: FM rank-32 LoRA × L12 (deployment recipe per E.4) — 3 seeds × {alone, +comp}, loco_onek1k only

Decision rule (pre-commit, applied to ΔR = (method+comp) − (method-alone) on AIDA):
  ΔR ≥ +0.05 on either method → method misses composition; additive claim.
  0 ≤ ΔR < +0.05 → methods already capture most of composition; disclose as confounder.
  ΔR < 0 → composition dominated by methods; strict baseline only.

Output: results/phase3/g1_composition_additive.csv
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
from sklearn.linear_model import ElasticNetCV, ElasticNet, Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler

from src.finetune.data_loader import select_indices


CELL_TYPE_FILES = {
    "B": "B.h5ad",
    "CD4+ T": "CD4p_T.h5ad",
    "CD8+ T": "CD8p_T.h5ad",
    "NK": "NK.h5ad",
    "Monocyte": "Monocyte.h5ad",
}
INTEGRATED_DIR = Path("data/cohorts/integrated")
AIDA_DIR = Path("data/cohorts/aida_eval")
AIDA_SPLIT = Path("data/aida_split.json")
EMB_DIR = Path("results/phase3/embeddings_layered")
OUT_CSV = Path("results/phase3/g1_composition_additive.csv")
TOP_N_HVG = 5000
ALPHAS_EN = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
L1_RATIOS = [0.3, 0.5, 0.7, 0.9]
ALPHAS_RIDGE = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
MAX_CELLS = 100
SEED = 0
DEPLOYMENT_LAYER = 12  # rank-32 modal layer per e4_ensemble_deployment.csv


def _aida_donors() -> set[str]:
    raw = json.loads(AIDA_SPLIT.read_text())["ancestry_shift_mae_donors"]
    return set(d if d.startswith("aida:") else f"aida:{d}" for d in raw)


def _build_composition_table() -> pd.DataFrame:
    """Donor-id -> 5-d cell-type-frequency vector + age. Same logic as F.1."""
    aida_donor_set = _aida_donors()
    rows = {}  # donor_id -> {cell_type: count, age: ..., cohort: ...}
    for cohort_label, base in [("integrated", INTEGRATED_DIR), ("aida", AIDA_DIR)]:
        for ct, fname in CELL_TYPE_FILES.items():
            a = ad.read_h5ad(base / fname, backed="r")
            obs = a.obs
            if cohort_label == "aida":
                mask = obs["donor_id"].isin(aida_donor_set)
            else:
                mask = pd.Series([True] * len(obs), index=obs.index)
            sub = obs[mask]
            for d, sub_d in sub.groupby("donor_id", observed=True):
                if len(sub_d) == 0:
                    continue
                d_str = str(d)
                if d_str not in rows:
                    rows[d_str] = {ct_: 0 for ct_ in CELL_TYPE_FILES}
                    rows[d_str]["age"] = float(sub_d["age"].iloc[0])
                rows[d_str][ct] = int(len(sub_d))
            a.file.close()
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "donor_id"
    cell_type_cols = list(CELL_TYPE_FILES.keys())
    totals = df[cell_type_cols].sum(axis=1)
    for ct in cell_type_cols:
        df[f"frac_{ct}"] = df[ct] / totals.replace(0, 1)
    return df.reset_index()


def _resolve_var_symbols(var: pd.DataFrame) -> np.ndarray:
    syms = np.asarray(var.get("gene_symbol", pd.Series(np.nan, index=var.index)).astype(object))
    idx = np.asarray(var.index.astype(str))
    return np.where(pd.isna(syms) | (syms == "nan"), idx, syms.astype(object).astype(str)).astype(str)


def _align_columns(train_syms, eval_syms, eval_X):
    train_set = {s: i for i, s in enumerate(train_syms)}
    out = np.zeros((eval_X.shape[0], len(train_syms)), dtype=np.float32)
    for j, s in enumerate(eval_syms):
        i = train_set.get(s)
        if i is not None:
            out[:, i] = eval_X[:, j]
    return out


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
        rows = a.X[idx[m]]
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


def _build_gene_en_donor_matrix(cohort, cell_type):
    base = AIDA_DIR if cohort == "aida" else INTEGRATED_DIR
    h5ad = base / CELL_TYPE_FILES[cell_type]
    if cohort == "aida":
        idx, ages, donors = select_indices(
            h5ad, cell_type=cell_type, cohorts=None,
            include_donors=list(_aida_donors()),
            max_cells_per_donor=MAX_CELLS, rng_seed=SEED,
        )
    else:
        idx, ages, donors = select_indices(
            h5ad, cell_type=cell_type, cohorts=[cohort],
            max_cells_per_donor=MAX_CELLS, rng_seed=SEED,
        )
    return _per_donor_log1p_mean(h5ad, idx, donors, ages)


def _r_mae(pred, y):
    if np.std(pred) > 0 and np.std(y) > 0 and len(y) > 1:
        r, _ = pearsonr(pred, y)
    else:
        r = 0.0
    return float(r), float(np.median(np.abs(pred - y)))


def _fit_en(X_tr, y_tr):
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(y_tr))
    cv = ElasticNetCV(l1_ratio=L1_RATIOS, alphas=ALPHAS_EN, cv=3, max_iter=5000,
                      n_jobs=-1, random_state=SEED, selection="cyclic")
    cv.fit(X_tr[perm], y_tr[perm])
    final = ElasticNet(alpha=float(cv.alpha_), l1_ratio=float(cv.l1_ratio_), max_iter=5000).fit(X_tr, y_tr)
    return final, float(cv.alpha_), float(cv.l1_ratio_)


def _fit_ridge(X_tr, y_tr):
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(y_tr))
    cv = RidgeCV(alphas=ALPHAS_RIDGE, cv=3, scoring="neg_mean_absolute_error")
    cv.fit(X_tr[perm], y_tr[perm])
    final = Ridge(alpha=float(cv.alpha_)).fit(X_tr, y_tr)
    return final, float(cv.alpha_)


def _slug(ct):
    return ct.replace("+", "p").replace(" ", "_")


def _load_fm_layer(cohort, cell_type, tag, layer):
    """Return (donor_ids, ages, X_at_layer)."""
    p = EMB_DIR / f"{cohort}_{_slug(cell_type)}_{tag}.npz"
    z = np.load(p, allow_pickle=True)
    donor_ids = np.asarray([str(d) for d in z["donor_ids"]])
    ages = z["ages"].astype(np.float32)
    X = z["embeddings_per_layer"][layer].astype(np.float32)
    return donor_ids, ages, X


def _add_composition(donor_ids, comp_df):
    """Lookup composition_5d per donor_id; missing → zeros (warn)."""
    cell_type_cols = [f"frac_{ct}" for ct in CELL_TYPE_FILES]
    comp_lookup = comp_df.set_index("donor_id")[cell_type_cols]
    n = len(donor_ids)
    C = np.zeros((n, len(cell_type_cols)), dtype=np.float32)
    missing = 0
    for i, d in enumerate(donor_ids):
        if d in comp_lookup.index:
            C[i] = comp_lookup.loc[d].values.astype(np.float32)
        else:
            missing += 1
    if missing:
        print(f"    [warn] {missing}/{n} donors missing composition; filled with zeros")
    return C


def _eval_method(name, X_tr, y_tr, train_donors, eval_donors_list, eval_y_list, comp_df, fitter):
    """Fit on X_tr (already-built train features), predict on each eval set with and without comp."""
    Xc_tr = _add_composition(train_donors, comp_df)
    X_tr_concat = np.concatenate([X_tr, Xc_tr], axis=1)

    # Standardize: separate scalers for X-part and concat
    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr).astype(np.float32)
    scaler_concat = StandardScaler().fit(X_tr_concat)
    X_tr_concat_s = scaler_concat.transform(X_tr_concat).astype(np.float32)

    model_alone, *_ = fitter(X_tr_s, y_tr)
    model_concat, *_ = fitter(X_tr_concat_s, y_tr)

    out_rows = []
    for label, eval_donors, eval_X_raw, eval_y in eval_donors_list:
        X_ev_s = scaler.transform(eval_X_raw).astype(np.float32)
        Xc_ev = _add_composition(eval_donors, comp_df)
        X_ev_concat = np.concatenate([eval_X_raw, Xc_ev], axis=1)
        X_ev_concat_s = scaler_concat.transform(X_ev_concat).astype(np.float32)

        pred_alone = model_alone.predict(X_ev_s)
        pred_concat = model_concat.predict(X_ev_concat_s)
        r_alone, mae_alone = _r_mae(pred_alone, eval_y)
        r_concat, mae_concat = _r_mae(pred_concat, eval_y)
        delta_r = r_concat - r_alone
        delta_mae = mae_concat - mae_alone
        out_rows.append({
            "method": name, "eval": label,
            "n_train": len(y_tr), "n_eval": len(eval_y),
            "R_alone": r_alone, "MAE_alone": mae_alone,
            "R_concat": r_concat, "MAE_concat": mae_concat,
            "deltaR": delta_r, "deltaMAE": delta_mae,
        })
        print(f"    {name} | {label}: R_alone={r_alone:+.3f} R_concat={r_concat:+.3f} ΔR={delta_r:+.3f} ΔMAE={delta_mae:+.2f}", flush=True)
    return out_rows


def main():
    print("[G.1] building composition table from integrated + AIDA atlases...", flush=True)
    comp_df = _build_composition_table()
    print(f"  composition table: {len(comp_df)} donors", flush=True)

    folds = json.loads(Path("data/loco_folds.json").read_text())["folds"]
    fmap = {f["fold_id"]: f for f in folds}

    all_rows = []

    # ===== Method 1: gene-EN × CD4+T × {loco_onek1k, loco_terekhova} → AIDA =====
    cell_type = "CD4+ T"
    for fold_id in ["loco_onek1k", "loco_terekhova"]:
        f = fmap[fold_id]
        eval_cohort = f["holdout_cohort"]
        print(f"\n=== gene-EN × {cell_type} × {fold_id} ===", flush=True)
        t0 = time.time()
        # Build per-cohort pseudobulks
        train_X_list, train_y_list, train_donors_list, train_syms = [], [], [], None
        for tc in f["train_cohorts"]:
            d_ids, X, y, syms = _build_gene_en_donor_matrix(tc, cell_type)
            if train_syms is None:
                train_syms = syms
                train_X_list.append(X)
            else:
                train_X_list.append(_align_columns(train_syms, syms, X))
            train_y_list.append(y)
            train_donors_list.append(d_ids)
        train_X = np.concatenate(train_X_list, axis=0)
        train_y = np.concatenate(train_y_list)
        train_donors = np.concatenate(train_donors_list)
        # HVG filter on train
        var = train_X.var(axis=0)
        top = np.argsort(-var)[:TOP_N_HVG]
        train_X_hvg = train_X[:, top]
        train_syms_hvg = train_syms[top]

        eval_donors, eval_X_raw, eval_y, eval_syms = _build_gene_en_donor_matrix(eval_cohort, cell_type)
        eval_X = _align_columns(train_syms_hvg, eval_syms, eval_X_raw)

        aida_donors, aida_X_raw, aida_y, aida_syms = _build_gene_en_donor_matrix("aida", cell_type)
        aida_X = _align_columns(train_syms_hvg, aida_syms, aida_X_raw)

        eval_list = [
            (f"{eval_cohort}_holdout", eval_donors, eval_X, eval_y),
            ("aida", aida_donors, aida_X, aida_y),
        ]
        rows = _eval_method(
            f"gene-EN_{fold_id}", train_X_hvg, train_y, train_donors,
            eval_list, [eval_y, aida_y], comp_df, _fit_en,
        )
        for r in rows:
            r["fold"] = fold_id
            r["cell_type"] = cell_type
            r["seed"] = 0
        all_rows.extend(rows)
        print(f"  ({time.time() - t0:.0f}s)", flush=True)

    # ===== Method 2: FM rank-32 × CD4+T × loco_onek1k × L12 × 3 seeds → AIDA =====
    fold_id = "loco_onek1k"
    f = fmap[fold_id]
    seed_tags = [
        (0, "loco_onek1k_CD4pT_e5b_r32_alllayers"),
        (1, "loco_onek1k_CD4pT_e5b_r32_seed1_alllayers"),
        (2, "loco_onek1k_CD4pT_e5b_r32_seed2_alllayers"),
    ]
    for seed, tag in seed_tags:
        print(f"\n=== FM-rank32 × {cell_type} × {fold_id} × seed{seed} × L{DEPLOYMENT_LAYER} ===", flush=True)
        train_X_list, train_y_list, train_donors_list = [], [], []
        for tc in f["train_cohorts"]:
            d_ids, ages, X = _load_fm_layer(tc, cell_type, tag, DEPLOYMENT_LAYER)
            train_X_list.append(X)
            train_y_list.append(ages)
            train_donors_list.append(d_ids)
        train_X = np.concatenate(train_X_list, axis=0)
        train_y = np.concatenate(train_y_list)
        train_donors = np.concatenate(train_donors_list)

        eval_donors, eval_y, eval_X = _load_fm_layer(f["holdout_cohort"], cell_type, tag, DEPLOYMENT_LAYER)
        aida_donors, aida_y, aida_X = _load_fm_layer("aida", cell_type, tag, DEPLOYMENT_LAYER)

        eval_list = [
            (f"{f['holdout_cohort']}_holdout", eval_donors, eval_X, eval_y),
            ("aida", aida_donors, aida_X, aida_y),
        ]
        rows = _eval_method(
            f"FM-rank32-L{DEPLOYMENT_LAYER}_{fold_id}_seed{seed}", train_X, train_y, train_donors,
            eval_list, [eval_y, aida_y], comp_df, _fit_ridge,
        )
        for r in rows:
            r["fold"] = fold_id
            r["cell_type"] = cell_type
            r["seed"] = seed
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, float_format="%.4f")
    print(f"\n[G.1] wrote {len(df)} rows to {OUT_CSV}\n")
    print(df.to_string(index=False, float_format="%.3f"))

    # === Decision summary on AIDA ΔR, averaged over seeds ===
    aida = df[df["eval"] == "aida"]
    print("\n=== Decision summary (AIDA ΔR by method, seed-averaged) ===")
    for method_prefix, label in [
        ("gene-EN_loco_onek1k", "gene-EN × loco_onek1k → AIDA"),
        ("gene-EN_loco_terekhova", "gene-EN × loco_terekhova → AIDA"),
        ("FM-rank32-L12_loco_onek1k", "FM-rank32-L12 × loco_onek1k → AIDA (3-seed mean)"),
    ]:
        sub = aida[aida["method"].str.startswith(method_prefix)]
        if len(sub) == 0:
            continue
        dR_mean = sub["deltaR"].mean()
        dMAE_mean = sub["deltaMAE"].mean()
        if dR_mean >= 0.05:
            verdict = "ADDITIVE — method misses composition"
        elif dR_mean >= 0:
            verdict = "CONFOUNDER — method already captures composition; disclose"
        else:
            verdict = "DOMINATED — composition strict baseline only"
        print(f"  {label}: ΔR={dR_mean:+.3f}, ΔMAE={dMAE_mean:+.2f} → {verdict}")


if __name__ == "__main__":
    main()
