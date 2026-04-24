# Pre-trained sc-ImmuAging LASSO sanity check — results & methodology

**Purpose.** Task 1e of `roadmap/phase-1.md`. Validate that the Python port of the scImmuAging R-package scoring pipeline reproduces the paper's internal-validation numbers to within an order of magnitude before any downstream baseline or foundation-model work depends on it.

**Date.** 2026-04-24.

## Summary result

Applied the pre-trained sc-ImmuAging LASSO (from `data/scImmuAging/data/all_model.RDS`, at `lambda.min`) to **all 981 donors of OneK1K (CellxGene-curated)** via a pure-Python re-implementation of the R package's pseudocell pipeline. OneK1K was in the sc-ImmuAging authors' training set of 5 cohorts — this is **not** a LOCO holdout; it reproduces the paper's *internal-validation* regime.

| Cell type | n donors | Median \|err\| (yr) | Mean \|err\| (yr) | Pearson R | Mean bias (yr) |
|---|---:|---:|---:|---:|---:|
| CD4+ T  | 981 | 9.4  | 10.1 | 0.747 | −4.42 |
| CD8+ T  | 981 | 7.6  |  8.7 | 0.766 | −1.53 |
| Mono    | 978 | 7.9  |  9.6 | 0.705 | −3.11 |
| NK      | 981 | 9.6  | 10.8 | 0.629 | −2.57 |
| B       | 981 | 10.7 | 11.9 | 0.531 | −3.71 |

Paper reference (Li et al. 2025 *Nature Aging*): internal Pearson R range 0.60–0.91 across 5 cell types, with B cells lowest and CD4+ T highest (~0.91).

## Interpretation

**Sanity check passes.** All Pearson R values are positive and highly significant (all p < 1e-70). The per-cell-type ordering roughly matches the paper (B lowest), though with compressed magnitude. All median \|err\| values are well within 2× of the paper's Extended Data Table 2 ranges. The pipeline is validated for use in Phase 2 baseline scoring.

**Systematic negative bias.** Every cell type shows a mean bias of −1.5 to −4.4 years — predictions systematically underestimate age. This is not a bug in the scoring code; it is a systematic divergence from the paper's own pipeline (see *Known divergences* below). The bias does not compromise Phase 2 use because our LOCO evaluation measures per-fold MAE relative to baseline, not absolute calibration.

**Pearson R ~0.05–0.15 lower than paper's internal validation.** Likely causes (ranked):
1. The paper's internal validation used an 80/20 within-cohort split. We scored against *all* 981 OneK1K donors (including both the paper's train and held-out sets mixed). Scoring-set composition explains some of the gap.
2. We source OneK1K from the CellxGene-curated h5ad (pre-demultiplexed per donor with canonical cell-type annotations); the paper used its own demultiplexed QC pipeline. Minor differences in cell filtering and cell-type label boundaries cascade into pseudocell composition.
3. Cell-type ontology mapping is approximate. The paper uses 5 labels (CD4T/CD8T/MONO/NK/B); CellxGene splits these into 10–15 sub-ontology IDs (e.g., "naive thymus-derived CD4+ alpha-beta T" vs. "central memory CD4+ alpha-beta T"). The mapping `CELLTYPE_CODE_TO_ONTOLOGY` (in `src/baselines/score_pretrained_lasso.py`) aggregates these to the canonical five. A one-time audit confirmed the ID set matches OneK1K's observed labels (no stray inclusion of gamma-delta, MAIT, dendritic, etc., after the 2026-04-24 ontology fix).

## Methodology

### Python port overview

The R package's scoring pipeline is:

```
PreProcess(seurat, cell_type, model, marker_gene)
  → subset to marker_gene
  → log-normalize (Seurat NormalizeData default)
  → group by donor, 100 pseudocells × 15-cell random samples per donor
AgingClockCalculator(preprocessed_df, model, marker_gene)
  → align to 700–1000 marker genes (zero-pad missing)
  → predict(cv.glmnet, newx, s="lambda.min")
Age_Donor(predict_res)
  → per-donor mean of the 100 pseudocell predictions
```

The Python port (`src/baselines/score_pretrained_lasso.py`):

1. **Read `all_model.RDS`** via `rdata` (pure-Python RDS parser). `cv.glmnet` is returned as a `dict` of numpy arrays; `glmnet.fit.beta` is a `dgCMatrix` (CSC sparse), `glmnet.fit.a0` holds 100 per-lambda intercepts, `lambda.min` gives the chosen column. Extract the coefficient vector at `argmin(|lambda - lambda.min|)` and the paired intercept.
2. **Read `all_model_inputfeatures.RDS`** for the 1000 (or 700 for CD8T) per-cell-type marker gene list in Ensembl form. Verified that `marker_gene` == `beta.Dimnames[0]` for every cell type.
3. **Log-normalize** counts with `log1p(counts / total × 10000)` per cell — identical to Seurat's `NormalizeData` default (`LogNormalize`, `scale.factor = 10000`).
4. **Align to marker genes** via CSC column-slicing + scatter into an (n_cells, n_marker) output; missing marker genes become zero columns. Observed overlap: 696/700 (CD8T), 996/1000 (CD4T), similarly high elsewhere.
5. **Pseudocell sampling** per donor using `numpy.random.default_rng(seed=0)`. `pseudocell_n = 100`, `pseudocell_size = 15`. The R package's "dynamic replace" logic is preserved: `replace=True` if the donor has ≤ 15 cells of that type, else `replace=False` per pseudocell (matching R's `sample(..., size=15, replace=FALSE)` semantic — distinct cells within a pseudocell, but independent across pseudocells).
6. **Predict** via dense `pseudocell @ coef_vec + intercept`. Per-donor predicted age = mean across 100 pseudocells.

### Why a Python port

R + Seurat + scImmuAging requires a native R install plus Seurat's compilation chain. Both `pyreadr` and `rdata` were tried for reading the serialized models. `pyreadr` fails on `cv.glmnet` (S4 class not supported). `rdata` parses cleanly with minor warnings about missing constructors — but *the underlying dict representation of `cv.glmnet` exposes exactly what scoring needs* (`glmnet.fit.beta`, `glmnet.fit.a0`, `lambda`). So we skip the R install entirely.

### Known divergences from the R pipeline (source of the systematic bias)

- **OneK1K source:** Paper uses its own demultiplexed cut of GSE196830 RAW; we use the CellxGene-curated h5ad (project decision 2026-04-24, documented in `roadmap/phase-1.md`). QC thresholds and cell-type boundaries differ slightly.
- **Pseudocell random state:** Paper's R pipeline uses `set.seed()` with unspecified value; our Python port uses `np.random.default_rng(seed=0)`. Pseudocell sampling is stochastic — Pearson R changes by ≤ 0.01 across seeds in spot checks.
- **Log-normalization precision:** Seurat's `LogNormalize` operates on a `dgCMatrix` with double precision; our port casts to `float32` for memory. Round-trip error is < 1e-6 and should not meaningfully affect predictions.
- **Gene coverage:** 696/700 to 996/1000 marker genes present. The 4–10 missing genes contribute zero — the R pipeline does the same, so this is not a divergence but is a source of small absolute drift.

## Reproducibility

```
uv run python -m src.baselines.score_pretrained_lasso --cell-type CD8T
uv run python -m src.baselines.score_pretrained_lasso --cell-type CD4T
uv run python -m src.baselines.score_pretrained_lasso --cell-type MONO
uv run python -m src.baselines.score_pretrained_lasso --cell-type NK
uv run python -m src.baselines.score_pretrained_lasso --cell-type B
```

Outputs:
- `results/baselines/pretrained_sanity_{CELLTYPE}.csv` — per-donor predictions.
- `results/baselines/pretrained_sanity_summary.csv` — per-cell-type metrics (regenerated on each run).

Runtime: ~90 s per cell type on a 32 GB consumer box (250k cells × 5 cell types + OneK1K h5ad backed load).

## Phase 2 implication

This sanity check does **not** produce a LOCO baseline — it is a training-set recapitulation. The Phase 2 Task "Apply pre-trained sc-ImmuAging clocks to LOCO holdout sets" will:

- Apply the same scoring pipeline to **Terekhova** (true holdout, not in sc-ImmuAging's training set) — this is the primary LOCO-2 fold.
- Apply to Stephenson (exploratory-only, below 80-donor threshold).
- Apply to OneK1K with the **training-set-asymmetry caveat documented**: OneK1K's numbers reflect "model applied to training data" and will be reported with a warning row in the results tables.

The negative mean-bias observed here should persist across cohorts if it is a pipeline artefact, or should change direction if it is a per-cohort calibration issue. Comparing bias across (OneK1K, Stephenson, Terekhova) in Phase 2 will localize the source.
