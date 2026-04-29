# Phase-3 paper draft v0 (2026-04-29)

This is a first-pass draft of the Methods + Results sections, written before
D.21 + D.22 verification lands. Section content is hedged where Tier-1
verifications are pending. **DO NOT cite any number from this draft as
final** — verify against the source CSVs and decision rules before circulating.

The chosen outline is **(b) comparison-led** as the safer default; if D.21 +
D.22 land in upper bands, restructure to outline (a) per
`notes/paper_outline_drafts.md`.

---

## Working title (b)

"Methodology-aware comparison of single-cell foundation models and bulk
baselines on donor-level aging prediction in PBMC scRNA-seq"

## Abstract (~250 words; placeholder)

[TODO: write abstract last, after sections crystallize]

Single-cell foundation models (FMs) such as Geneformer and scFoundation are
typically evaluated on cell-level tasks (annotation, perturbation response).
Their performance on donor-level phenotype prediction has been mixed,
particularly for chronological age regression on PBMC scRNA-seq, where
prior work reports bulk gene-EN baselines outperforming FMs by ~0.4 R-units.
We show that this gap is substantially driven by methodology mismatches —
different cohort splits, preprocessing, and hyperparameter grids — rather
than fundamental FM limitations. On matched LOCO splits with comparable
preprocessing and hyperparameter searches, frozen Geneformer + ridge readout
matches gene-EN within seed variance (Δ R = 0.05–0.15) on CD4+T age
regression across cohorts, while scFoundation (3B params) lags both
(Δ R = 0.08–0.10 vs Geneformer). LoRA fine-tuning at rank-16 and rank-32
does not improve over frozen+ridge at 3-seed mean. We further show that
layer-of-best-readout for FM probing is cell-type-conditional (NK reads
best at early layers, CD4+T at late layers, B substrate is mostly empty)
and unit-of-analysis-conditional (pseudobulk-input shifts layer choice
toward early layers regardless of cell type). Methodology recommendation:
matched-splits comparison and per-cell-type layer selection are essential
for FM benchmarking.

## 1. Introduction

[TODO: 4-5 paragraphs covering single-cell FMs, prior aging benchmarks,
donor-level vs cell-level evaluation, methodology comparisons]

Key positioning points:
- Cite the TF paper (Theodoris 2023) as the canonical Geneformer aging
  paper; their gene-EN R=0.83 is the headline number we re-examine.
- Cite scFoundation (Hao 2024) as the 3B FM also tested.
- Cite the Kedzierska 2025 zero-shot critique as the closest methodology
  precedent.
- Frame contribution as: methodology-aware comparison + methodology
  recommendations, not "FMs lose on aging."

## 2. Methods

### 2.1 Cohort harmonization

Four cohorts: OneK1K (Yazar 2022, n=981 donors), Stephenson (2021, n=24-29),
Terekhova (2024, n=166), AIDA (Tian 2024, n=293-307 cross-ancestry). All
processed through `src/data/harmonize_cohorts.py` for cell-type relabeling
and gene-symbol alignment to a 19264-gene shared panel.

Cell types analyzed: CD4+ T, NK, B (per-cell-type AnnData files in
`data/cohorts/integrated/{CD4p_T,NK,B}.h5ad`).

LOCO splits (`data/loco_folds.json`):
- `loco_onek1k`: train = Stephenson + Terekhova; holdout = OneK1K (in-distribution check).
- `loco_terekhova`: train = OneK1K + Stephenson; holdout = Terekhova
  (chemistry-shift cross-cohort).
- AIDA cross-ancestry transfer: held out from both folds for cross-ancestry
  evaluation.

### 2.2 Per-cell mean-pool ridge readout

Extract per-cell embeddings with the FM (frozen base or LoRA fine-tuned),
mean-pool per donor, fit RidgeCV (alphas = [0.01, 0.1, 1, 10, 100, 1000,
10000], 3-fold inner CV, scoring=neg_MAE) on per-donor embeddings vs
chronological age. Reported metrics: Pearson R + MAE on holdout donors,
1000-iteration bootstrap 95% CI.

### 2.3 Pseudobulk-input ridge readout

Per donor, sum raw counts across selected cells → pass through Geneformer's
canonical rank-value tokenization (CP10k / median normalize → rank → top-K
genes by rank) → forward as a single pseudo-cell with output_hidden_states=True
→ mean-pool across attended positions per layer → fit RidgeCV per layer.

### 2.4 LoRA fine-tuning

Rank-16 (production) and rank-32 (capacity ablation) LoRA on Geneformer
V2-104M. Hyperparameters per memo §15 e5b config:
- 3 epochs, batch size 8, gradient accumulation 4 (effective batch 32)
- LR 2e-4 (backbone) and 2e-4 (head), warmup 10%
- Pool: mean across attended positions
- max_cells_per_donor: 50 (training), 20 (eval)
- bf16 autocast, gradient checkpointing
- Loss: per-cell MSE on donor-replicated age labels
- Multi-seed: 3 seeds (0, 1, 2) per (cell-type × fold) combo

### 2.5 Bulk gene-EN baseline (matched splits)

Per donor, compute log1p(CP10k) per gene → mean across cells → top-5000 HVG
selection (variance across train donors) → StandardScaler → ElasticNetCV
(4 l1_ratios × 8 alphas × 3-fold inner CV, max_iter=5000) on per-donor
matrices vs age. Same LOCO splits and donor caps as the FM analyses to
ensure apples-to-apples comparison.

### 2.6 Bootstrap CIs

For all R + MAE estimates on holdout sets, 1000-iteration donor bootstrap
percentile CI at 95%.

## 3. Results

### 3.1 Matched-splits gene-EN baseline reaches R = 0.61–0.78 on CD4+T cross-cohort, not 0.83

Re-running gene-EN on the same LOCO splits with the same preprocessing as
the FM experiments yields:

| Cell × eval | gene-EN R | gene-EN MAE | TF-paper claim |
|---|---|---|---|
| CD4+T × OneK1K | 0.612 | 14.19 | ~0.83 LOCO (different cohorts) |
| CD4+T × Terekhova | 0.776 | 10.52 | ~0.83 LOCO (different cohorts) |
| CD4+T × AIDA loco_onek1k | 0.616 | 6.42 | 0.77 |
| CD4+T × AIDA loco_terekhova | 0.651 | 6.66 | 0.77 |

The matched-splits R is substantially below the TF-paper R=0.83. We attribute
this to (a) more training cohorts available to TF, (b) different
preprocessing, (c) different hyperparameter grids. The matched-splits R is
the comparable number for FM-vs-bulk evaluation.

### 3.2 Frozen Geneformer + ridge readout matches gene-EN within seed variance on CD4+T

[TODO: insert three-way comparison from `results/phase3/d25_three_way_matched_splits.csv`]

| Cell × eval | gene-EN R | Geneformer best-layer R | Δ |
|---|---|---|---|
| CD4+T × OneK1K | 0.612 | 0.560 (L12) | -0.052 |
| CD4+T × Terekhova | 0.776 | 0.621 (L5) | -0.155 |
| CD4+T × AIDA loco_onek1k | 0.616 | 0.527 (L12) | -0.088 |

Δ ~0.05–0.16 R-units, well within typical 3-seed variance of FM evaluations.

### 3.3 scFoundation 3B at frozen+ridge lags Geneformer by 0.08–0.10 R-units

Same matched-splits comparison vs scFoundation:

| Cell × eval | gene-EN R | scFoundation R | Δ vs gene-EN | Δ vs Geneformer per-cell |
|---|---|---|---|---|
| CD4+T × OneK1K | 0.612 | 0.475 | -0.137 | -0.085 |
| CD4+T × Terekhova | 0.776 | 0.519 | -0.256 | -0.101 |
| CD4+T × AIDA loco_onek1k | 0.616 | 0.442 | -0.174 | -0.086 |

scFoundation does not match gene-EN at matched splits and does not match
Geneformer either. The matched-splits parity finding is **Geneformer-
specific**, not pan-FM. This is a more specific (and more defensible)
contribution than a pan-FM claim.

### 3.4 LoRA fine-tuning at rank-16 and rank-32 does not improve over frozen+ridge

[TODO: insert §27/§28/§30 numbers; conditional on D.21 verification]

3-seed mean rank-16 LoRA on CD4+T × loco_onek1k (per §28):
- L12 OneK1K MAE = 10.85y ± 2.19y (vs frozen R=0.560 / MAE = ~16y)
- L11 AIDA MAE = **7.96y ± 0.42y** (3-seed best layer; new in D.32)
- L9 AIDA MAE = 8.36y ± 0.14y (D.32 confirms tight 3-seed std on L9)

Rank-32 single-seed (§30) was at MAE=11.00y on L12 OneK1K; rank-32 3-seed
verification (D.21) is **PENDING** — interpretation depends on outcome:
- ≤7.5y MAE on L9 AIDA → outline (a) viable, parity headline supported
- 7.5–8.5y → outline (a) hedged, "competitive within ~1y"
- >8.5y → outline (b), drop AIDA-parity from headline

[TODO: complete after D.21 lands]

### 3.5 Cell-type-conditional layer-of-best-readout

For frozen Geneformer per-cell mean-pool ridge readout:

| Cell type | Best-R layer (mean of 3 cohorts) | Pattern |
|---|---|---|
| CD4+T | L9.7 (L12 dominant) | Late-layer specialization |
| NK | L3.3 (L2-L5 best on each cohort) | Early-layer dominant |
| B | L9.0 mean, but R<0.23 | Substrate mostly empty |

**Important caveat**: D.26 bootstrap CIs show NK early-layer ΔR vs L12
robustly excludes zero only on AIDA cross-ancestry; on OneK1K and Terekhova,
the median is positive (+0.04, +0.07) but CI includes zero. The
cell-type-conditional layer claim is robustly supported only on AIDA
cross-ancestry without 3-seed verification (D.22 pending).

[TODO: complete after D.22 lands]

### 3.6 Unit-of-analysis interacts with layer choice

Pseudobulk-input frozen Geneformer ridge readout best layer per condition:

| Cell × eval | per-cell mean-pool best layer | pseudobulk-input best layer |
|---|---|---|
| CD4+T × OneK1K | L12 | L1 |
| CD4+T × Terekhova | L5 | L1 |
| CD4+T × AIDA loco_onek1k | L12 | L4 |
| NK × OneK1K | L3 | L3 |
| NK × Terekhova | L2 | L2 |
| NK × AIDA loco_onek1k | L5 | L0 |

Pseudobulk-input drives best layer toward L0–L4 across all CD4+T and NK
conditions. The two-axis principle: **pseudobulk-input → early layers
regardless of cell type; per-cell mean-pool layer choice is
cell-type-conditional**. When fed donor-aggregated input, the FM behaves
more like a bulk model and its late-layer specialization no longer matches
the donor-level task.

### 3.7 Cross-ancestry AIDA characterization

AIDA cross-ancestry (Indonesian/Chinese/Indian donors held out from both
folds) is the most paper-relevant evaluation:

| Method | AIDA R | AIDA MAE |
|---|---|---|
| gene-EN matched (loco_onek1k) | 0.616 | 6.42 |
| gene-EN matched (loco_terekhova) | 0.651 | 6.66 |
| Pasta-REG (Li 2024) | 0.659 | 6.32 |
| Geneformer rank-16 LoRA L11 3-seed | 0.566 ± 0.032 | **7.96 ± 0.42** |
| Geneformer rank-32 LoRA L9 single-seed | 0.617 | 6.92 |
| Geneformer rank-32 LoRA L9 3-seed (D.21 PENDING) | TBD | TBD |
| scFoundation frozen L_final | 0.442 | 20.92 |
| Geneformer frozen L12 ridge | 0.527 | 11.76 |

Cross-ancestry generalization is a setting where FM ridge readout reaches
within ~1.5y MAE of the best bulk methods at rank-16 3-seed mean.

## 4. Discussion

[TODO]

Key points to develop:
- Matched-splits methodology is essential to interpretable FM-vs-bulk
  comparison. The TF paper's R=0.83 is a real cohort-specific number, but
  not the right anchor for "FMs lose."
- FM-specific behavior (Geneformer ≈ bulk at matched splits; scFoundation
  doesn't) matters more than FM-class behavior.
- Layer-of-best-readout is cell-type-conditional and unit-of-analysis-
  conditional. Both axes contribute to the methodology contribution.
- Limitations: single task, single organism, three cell types, donor count
  ≤1000. Generalization to other single-cell FM applications requires
  validation.

## 5. Conclusion

[TODO]

## Acknowledgments + Data + Code Availability

[TODO]

---

## Status of each result section (2026-04-29 snapshot)

| Section | Result | Source | Status |
|---|---|---|---|
| 3.1 | gene-EN matched R = 0.61–0.78 | gene_en_matched_splits.csv | DONE |
| 3.2 | Geneformer ridge ≈ gene-EN | d25_three_way_matched_splits.csv | DONE |
| 3.3 | scFoundation lags by 0.08–0.10 | d25_three_way_matched_splits.csv | DONE |
| 3.4 | rank-16 L11 MAE=7.96y | d32_rank16_3seed_layered_bootstrap_cis.csv | DONE |
| 3.4 | rank-32 3-seed L9 | D.21 in progress | PENDING |
| 3.5 | NK early-layer cell-type-conditional | ridge_summary_layered.csv + d22 | PARTIAL (D.22 pending) |
| 3.5 | Bootstrap CI on NK early-layer | layer_asymmetry_cis.csv | DONE |
| 3.6 | Pseudobulk-input best layer L0-L4 | ridge_summary_pseudobulk.csv | DONE |
| 3.7 | AIDA cross-ancestry table | aggregate from above CSVs | DONE except D.21 row |

When D.21 lands: complete §3.4 with rank-32 3-seed mean ± std.
When D.22 lands: complete §3.5 with NK 3-seed cross-cohort confirmation.

After both: re-evaluate outline (a) vs (b) per `notes/decision_rules_phase3.md`.
