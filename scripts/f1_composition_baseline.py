"""F.1 — Composition-only baseline.

For each donor, build a cell-type-frequency vector (counts of each cell type
divided by total cells across the 5 cell-type h5ad files in the integrated
atlas: B, CD4+T, CD8+T, NK, Monocyte). Fit LASSO/ElasticNet on these vectors
alone for age prediction. Evaluate on AIDA + loco-holdout cohorts.

Tests the additional_concerns.md #1 hypothesis: PBMC composition shifts
(naive-T loss, monocyte expansion, etc.) may be the dominant signal in our
age predictions, with within-cell-type expression a small residual.

Decision rule (pre-commit):
  R >= 0.5 on AIDA → composition explains substantial fraction of signal;
                     paper must reframe to 'within-cell-type expression
                     beyond composition' with composition as strong baseline.
  0.3 <= R < 0.5  → meaningful but not dominant; report composition baseline.
  R < 0.3         → composition is not the main signal; existing cell-type-
                     specific framing stands.

Output: results/phase3/f1_composition_baseline.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV, ElasticNet
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr


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
OUT_CSV = Path("results/phase3/f1_composition_baseline.csv")
SEED = 0


def _build_composition_matrix(cohort: str):
    """Return (donor_ids, ages, freq_matrix). freq_matrix is (n_donors, n_cell_types).

    For 'aida', uses AIDA cross-ancestry split. For other cohorts, uses the integrated atlas
    filtered to cohort_id == cohort.
    """
    if cohort == "aida":
        base = AIDA_DIR
        aida_donors_raw = json.loads(AIDA_SPLIT.read_text())["ancestry_shift_mae_donors"]
        target_donors = set(d if d.startswith("aida:") else f"aida:{d}" for d in aida_donors_raw)
    else:
        base = INTEGRATED_DIR
        target_donors = None

    # Aggregate cells per donor per cell type
    donor_counts = {}  # {donor_id: {cell_type: count}}
    donor_ages = {}

    for ct, fname in CELL_TYPE_FILES.items():
        h5ad = base / fname
        a = ad.read_h5ad(h5ad, backed="r")
        obs = a.obs
        if cohort == "aida":
            # AIDA file has all aida donors; filter by target_donors
            mask = obs["donor_id"].isin(target_donors)
        else:
            mask = obs["cohort_id"] == cohort
        sub = obs[mask]
        for d, sub_d in sub.groupby("donor_id", observed=True):
            if len(sub_d) == 0:
                continue
            d_str = str(d)
            if d_str not in donor_counts:
                donor_counts[d_str] = {ct: 0 for ct in CELL_TYPE_FILES}
                donor_ages[d_str] = float(sub_d["age"].iloc[0])
            donor_counts[d_str][ct] += len(sub_d)
        a.file.close()

    if not donor_counts:
        return None, None, None

    donor_ids = sorted(donor_counts.keys())
    cell_types = list(CELL_TYPE_FILES.keys())
    freq = np.zeros((len(donor_ids), len(cell_types)), dtype=np.float32)
    ages = np.zeros(len(donor_ids), dtype=np.float32)
    for i, d in enumerate(donor_ids):
        total = sum(donor_counts[d].values())
        if total == 0:
            continue
        for j, ct in enumerate(cell_types):
            freq[i, j] = donor_counts[d][ct] / total
        ages[i] = donor_ages[d]
    return donor_ids, ages, freq


def _fit_eval(X_train, y_train, X_eval, y_eval):
    scaler = StandardScaler().fit(X_train)
    Xs_train = scaler.transform(X_train)
    Xs_eval = scaler.transform(X_eval)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(y_train))
    cv = ElasticNetCV(
        l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
        alphas=[0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
        cv=5, max_iter=10000, n_jobs=-1, random_state=SEED, selection="cyclic",
    )
    cv.fit(Xs_train[perm], y_train[perm])
    final = ElasticNet(alpha=float(cv.alpha_), l1_ratio=float(cv.l1_ratio_), max_iter=10000).fit(Xs_train, y_train)
    pred = final.predict(Xs_eval)
    if np.std(pred) > 0 and np.std(y_eval) > 0 and len(y_eval) > 1:
        r, _ = pearsonr(pred, y_eval)
    else:
        r = 0.0
    mae = float(np.median(np.abs(pred - y_eval)))
    return float(r), mae, float(cv.alpha_), float(cv.l1_ratio_)


def main():
    folds = json.loads(Path("data/loco_folds.json").read_text())["folds"]
    fmap = {f["fold_id"]: f for f in folds}

    # Build composition matrix per cohort
    comp_data = {}
    for cohort in ["onek1k", "stephenson", "terekhova", "aida"]:
        donor_ids, ages, freq = _build_composition_matrix(cohort)
        if donor_ids is None:
            print(f"  [SKIP] {cohort}: no donors found")
            continue
        comp_data[cohort] = (donor_ids, ages, freq)
        print(f"[F.1] {cohort}: n_donors={len(donor_ids)}, freq matrix shape={freq.shape}", flush=True)
        print(f"  Mean freq: B={freq.mean(0)[0]:.3f}, CD4T={freq.mean(0)[1]:.3f}, CD8T={freq.mean(0)[2]:.3f}, NK={freq.mean(0)[3]:.3f}, Mono={freq.mean(0)[4]:.3f}")

    rows = []
    for fold_id in ["loco_onek1k", "loco_terekhova"]:
        f = fmap[fold_id]
        train_X, train_y = [], []
        for tc in f["train_cohorts"]:
            if tc not in comp_data:
                continue
            _, ages, freq = comp_data[tc]
            train_X.append(freq)
            train_y.append(ages)
        if not train_X:
            continue
        train_X = np.concatenate(train_X, axis=0)
        train_y = np.concatenate(train_y)

        # Evaluate on actual holdout
        if f["holdout_cohort"] in comp_data:
            _, eval_y, eval_X = comp_data[f["holdout_cohort"]]
            r, mae, alpha, l1 = _fit_eval(train_X, train_y, eval_X, eval_y)
            rows.append({
                "fold": fold_id, "eval_cohort": f["holdout_cohort"],
                "n_train": len(train_y), "n_eval": len(eval_y),
                "R": r, "MAE": mae, "alpha": alpha, "l1_ratio": l1,
            })
            print(f"[F.1] {fold_id} → {f['holdout_cohort']}: R={r:+.3f} MAE={mae:.2f} (alpha={alpha:.3f}, l1={l1:.2f})", flush=True)

        # Evaluate on AIDA cross-ancestry
        if "aida" in comp_data:
            _, aida_y, aida_X = comp_data["aida"]
            r, mae, alpha, l1 = _fit_eval(train_X, train_y, aida_X, aida_y)
            rows.append({
                "fold": fold_id, "eval_cohort": "aida",
                "n_train": len(train_y), "n_eval": len(aida_y),
                "R": r, "MAE": mae, "alpha": alpha, "l1_ratio": l1,
            })
            print(f"[F.1] {fold_id} → AIDA: R={r:+.3f} MAE={mae:.2f} (alpha={alpha:.3f}, l1={l1:.2f})", flush=True)

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, float_format="%.4f")
    print(f"\n[F.1] wrote {len(df)} rows to {OUT_CSV}")

    # Decision summary
    print("\n=== F.1 Decision summary ===")
    print(df.to_string(index=False, float_format="%.3f"))
    aida_rows = df[df["eval_cohort"] == "aida"]
    if len(aida_rows) > 0:
        max_aida_R = aida_rows["R"].max()
        print(f"\n  Max AIDA R from composition alone: {max_aida_R:+.3f}")
        if max_aida_R >= 0.5:
            print("  → DECISION: composition explains substantial fraction; paper must reframe.")
        elif max_aida_R >= 0.3:
            print("  → DECISION: composition is meaningful baseline; report alongside.")
        else:
            print("  → DECISION: composition is not the main signal; existing framing stands.")


if __name__ == "__main__":
    main()
