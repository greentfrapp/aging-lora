# MAE detectability floor for LOCO folds — results & interpretation

**Purpose.** Success criterion in `roadmap/phase-1.md` L35: compute the minimum per-fold donor count needed to detect a 10% relative MAE reduction at 80% power, α=0.05. Any fold below its per-cell-type floor is flagged exploratory-only so it doesn't carry the paper's headline numbers.

**Date.** 2026-04-24.

## Inputs

- Per-donor absolute errors from the Task 1e pre-trained LASSO sanity check: `results/baselines/pretrained_sanity_{CD4T,CD8T,MONO,NK,B}.csv` (OneK1K, 981 donors, per cell type).
- Baseline MAE and residual SD per cell type are taken from these CSVs rather than the paper's Ext Data Table 2 (which we don't have verbatim). The Task 1e sanity check reproduces paper internal R within 0.05–0.15 per cell type, so this is a reasonable proxy.

## Methodology

Paired one-sided Wilcoxon signed-rank power calculation with pairing correlation ρ:

```
σ_d = σ_|err| × √(2 × (1 − ρ))
n   = ceil( ((z_{1−α} + z_{1−β}) × σ_d / δ)² / WILCOXON_ARE )
```

where `δ = 0.10 × median(|err|)` (the relative-effect target in years), and `WILCOXON_ARE = 0.955` is the asymptotic efficiency of the one-sample Wilcoxon signed-rank relative to the paired t-test under normality.

`σ_|err|` is the per-donor absolute-error standard deviation from the sanity check. ρ is unobservable (we haven't scored any candidate method yet), so we report a sensitivity sweep.

## Results — sensitivity over ρ

`n_required` for 10% MAE reduction at 80% power, α=0.05, per cell type:

| ρ | CD4T | CD8T | MONO | NK | B |
|---|---:|---:|---:|---:|---:|
| 0.3 | 460 | 630 | 800 | 544 | 541 |
| 0.5 | 328 | 450 | 571 | 389 | 386 |
| 0.7 | 197 | 270 | 343 | 233 | 232 |
| **0.8 (planning assumption)** | **132** | **180** | **229** | **156** | **155** |
| 0.9 | 66 | 90 | 115 | 78 | 78 |

**Choice of ρ = 0.8 for planning.** Two clocks scored on the same donors share a lot of variance — sequencing depth, donor age, technical batch, cell-composition shifts all drive errors in parallel. For aging clocks specifically we expect ρ ≈ 0.7–0.9. ρ = 0.8 is the middle-of-range conservative planning choice.

## Mapping to the three LOCO folds (at ρ = 0.8)

| Fold | n held out | CD4T primary | CD8T primary | MONO primary | NK primary | B primary |
|---|---:|:-:|:-:|:-:|:-:|:-:|
| loco_onek1k | 981 | ✅ | ✅ | ✅ | ✅ | ✅ |
| loco_terekhova | 166 | ✅ | ❌ (180 needed) | ❌ (229 needed) | ✅ | ✅ |
| loco_stephenson | 29 | ❌ | ❌ | ❌ | ❌ | ❌ |

**Headline takeaways:**

- `loco_onek1k` is adequately powered for every cell type — but carries the training-set asymmetry caveat for scGPT / UCE per the leakage audit. Use with asterisk-footnote in paper result tables.
- `loco_terekhova` (our clean-for-all-FMs gold-standard fold) is adequately powered for **CD4+ T, NK, and B** cell types only. For **CD8+ T and Monocyte** it falls short at ρ = 0.8. Report CD8T and MONO on loco_terekhova as secondary / exploratory with a note.
- `loco_stephenson` is exploratory-only across the board (as expected, given 29 donors).

**If ρ turns out higher (≥ 0.9) in practice.** At ρ = 0.9 the per-cell-type floor drops to 66–115 donors and loco_terekhova becomes adequately powered for all five. We'll know the empirical ρ after the first head-to-head scoring in Phase 2 — at that point, revisit this analysis with measured ρ and update `loco_folds.json`.

## Implementation

`src/data/detectability_floor.py` computes and writes `data/detectability_floor.json`. `src/data/freeze_splits.py` consumes that file and emits per-cell-type `primary` flags per LOCO fold in `data/loco_folds.json`.

To regenerate with different assumptions:

```
uv run python -m src.data.detectability_floor --relative-effect 0.15   # easier target
uv run python -m src.data.detectability_floor --pairing-rho 0.9         # optimistic ρ
uv run python -m src.data.detectability_floor --power 0.9               # 90% power
```

## Planning implication for Phase 3/4

When loco_terekhova is underpowered for a given cell type at our observed ρ (after Phase 2), we have three options in order of preference:

1. **Empirical ρ is higher than 0.8.** Update assumptions; possibly loco_terekhova becomes adequately powered.
2. **Pool loco_terekhova + loco_onek1k via meta-analysis** for CD8+ T and Monocyte specifically; the leakage confound on OneK1K is noted but the combined effect size is still a valid estimate if we weight per-fold appropriately.
3. **Explicit exploratory label** in the main result table; the combined (OneK1K + Terekhova) primary fold count remains 1 (OneK1K only) for those two cell types.

Option 1 is the expected outcome; aging clocks typically show very high per-donor error correlation.

## Phase 2 Task 2.8 update (2026-04-25): empirical pairing-ρ from baseline residuals

The Phase-1 floor used a sensitivity sweep over ρ ∈ {0.3, 0.5, 0.7, 0.8, 0.9}. Phase-2 produced per-donor predictions for three baselines (LASSO, scAgeClock, Pasta-REG) on the same 3 cohorts × 5 cell types, so we can now **measure** the actual pairing-ρ between baseline residuals as an empirical proxy for what Phase 3's baseline-vs-FM ρ will look like.

**Per-cell-type empirical ρ** (median across 9 cohort × baseline-pair values, see `results/baselines/empirical_pairing_rho.csv`):

| Cell type | Empirical median ρ | Range across pairs | n_required at empirical ρ (Wilcoxon) |
|---|---|---|---|
| CD4T | **0.23** | −0.42 to 0.73 | **504** |
| CD8T | **0.16** | −0.09 to 0.61 | **753** |
| MONO | **0.06** | −0.86 to 0.66 | **1,075** |
| NK | **0.28** | −0.27 to 0.66 | **557** |
| B | **0.35** | 0.07 to 0.63 | **502** |

These are far below the Phase-1 ρ=0.8 planning value. Under the conservative empirical-ρ interpretation, the only adequately-powered LOCO fold is **OneK1K (981 donors)** — and only for CD4T, NK, B. Terekhova (166 donors) is underpowered for *all five cell types*; Stephenson (29) underpowered everywhere. The Phase-1 floor was overoptimistic by 2–7×.

**Important caveat.** The ρ values above are between BASELINE PAIRS (LASSO vs Pasta-REG, etc.), not between BASELINE and FM. The Phase-3 LoRA fine-tunes are expected to share more residual structure with the baselines than the baselines do with each other (both are predicting the same chronological-age signal from the same gene-expression data). The empirical baseline-pair ρ here is therefore a **conservative lower bound** on the Phase-3-measured baseline-vs-FM ρ. The truth is somewhere between 0.06–0.35 (this measurement) and 0.8 (Phase-1 planning value); Phase 3 will measure it directly and append a `post_phase3_override` block to `data/loco_folds.json`.

**Practical implication for Phase 3 reporting.** The preprint's headline detectability flags use the **post_phase3 measured ρ** (not these baseline-pair values, not the Phase-1 ρ=0.8). For the Phase-3 preprint that posts before Phase 4 is run, the empirical baseline-pair ρ table above goes in the supplementary as a transparency disclosure: "the Phase-1 detectability floor was overoptimistic; Phase 3 measured ρ=X, between the two extremes."

Empirical-ρ data: `results/baselines/empirical_pairing_rho.csv`. The detectability floor JSON now carries a `post_phase2_empirical_rho` block alongside the original Phase-1 fields.
