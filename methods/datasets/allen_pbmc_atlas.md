# Allen Institute PBMC atlas (REJECTED)

**Status**: Evaluated as a Case 2 alternative to Barreiro during the cohort-selection phase; **rejected** after closer inspection because the cohort composition, age structure, and primary-data yield are unsuitable for our aging-clock target. Terekhova was ultimately chosen instead.

## Paper and accession

**Collection**: CellxGene `77f9d7e9-...` — Allen Institute "Single-cell multi-omics analysis of human peripheral blood cells" (exact collection UUID lookup via CellxGene curation API).

**Paper**: referenced in early research notes but not formally cited here since we never used the data. The collection description mentions proteogenomic / CITE-seq profiling of healthy PBMCs with paired cellular indexing of transcriptome and epitopes.

## Why it was evaluated

When Barreiro/Randolph 2021 was dropped (see `methods/datasets/barreiro_randolph.md`), we searched for a replacement cohort. The Allen PBMC atlas was attractive because:

1. **Same 10x 3' chemistry** as OneK1K and Stephenson — no chemistry batch correction needed if included
2. Large cell count (~1.82 million cells in the raw deposit)
3. Part of a well-curated institutional release (Allen Institute for Immunology)

## Why we rejected it

Closer inspection via the CellxGene curation API and the collection description revealed:

### 1. CMV-seropositive donors mixed in

The atlas intentionally includes a mix of cytomegalovirus (CMV)-seropositive and -seronegative donors to study CMV's effect on immune cell populations. For an aging clock, CMV status is a known strong confounder — CMV seropositivity drives accumulation of late-differentiated effector-memory CD8+ T cells that mimic aging-like transcriptional signatures. Including CMV+ donors without stratifying for CMV status would inflate age-prediction noise and/or shift per-cohort calibration in unpredictable ways.

### 2. 37–54 year age gap

The donor-age distribution has a ~17-year gap between roughly 37 and 54 years — deliberate for the CMV-vs-age factorial study design, but bad for a chronological-age regressor. The gap coincides with the steepest changes in immune cell composition (thymic involution, T-cell repertoire contraction), and leaving it unsampled creates a regression-target discontinuity that hurts both training and evaluation stability.

### 3. Pediatric contamination

16 of the listed donors are pediatric (under 18 yr). The project scope is adult immune aging; pediatric inclusion is a conceptual mismatch. Filterable, but reduces the effective adult donor pool.

### 4. Low `is_primary_data` yield

Of the 1,820,000 cells in the deposit, only **263,917 are marked `is_primary_data=True`**. The rest are CellxGene-integration-remapped cells from other collections that happen to share donors or samples with this atlas. Using only primary cells cuts the effective atlas by ~85% and introduces a non-trivial QC-selection step.

## Comparison with Terekhova (the alternative that won)

| Criterion | Allen PBMC atlas | Terekhova 2023 |
|---|---|---|
| Donors | ~(unclear post-filtering) | 166 |
| Age distribution | 17-yr gap (37–54) + pediatric contamination | Continuous 25–81 yr |
| CMV control | Mixed CMV+/– without stratification info in CellxGene obs | Paper does not mention CMV as a design factor; assumed standard healthy PBMC mix |
| Primary data yield | 14% (263k / 1.82M) | 100% |
| Chemistry | 10x 3' (compatible with training set) | 10x 5' v2 (incompatible → needs Task 1f chemistry correction) |
| Leakage-audit risk | High (CellxGene Census member → scGPT + UCE would be overlapping) | None (Synapse-only → all 4 FMs clean) |
| Net primary LOCO fold contribution | Unclear (filter loss + age gap makes this murky) | Second primary LOCO fold at 166 donors |

Terekhova wins on almost every axis except chemistry — and the chemistry concern is explicitly addressed as Task 1f (report naive and chemistry-corrected MAE side by side). The leakage-audit win in particular was decisive once we realized Terekhova is the only cohort clean for all four foundation models.

## Role in the study — none

Not downloaded. Not included. The evaluation notes above are retained to document why it was rejected, so the next reviewer asking "why not Allen?" has a written answer.

## Related references

- Early FUTURE_WORK.md entry for Allen was removed during the Terekhova-promotion edit cycle; replaced by the Barreiro-revival entry.
- `methods/datasets/barreiro_randolph.md` — the cohort this was considered as a replacement for.
- `methods/datasets/terekhova.md` — the cohort that ultimately replaced Barreiro.
