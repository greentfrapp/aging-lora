# Phase-3 paper outline drafts (D.28)

Two parallel outlines per the reframed-review verification gate. Pick at the
gate based on D.21–D.24 outcomes, not before.

---

## Outline (a): Methodology-led — "Cell-type-conditional layer selection in single-cell foundation model probing"

**Viable only if Tier 1 verification (D.21–D.24) lands in the upper bands.**
Specifically requires:
- D.21 L9 AIDA 3-seed mean MAE ≤ 7.5y (matched-splits parity claim survives)
- D.22 NK ΔR(best vs L12) > +0.05 across all 3 cohorts at 3-seed mean
- D.24 either band (already done; supports two-axis principle in the form: pseudobulk-input → early layers regardless, per-cell mean-pool layer choice cell-type-conditional)

### Title (working)
"Cell-type-conditional layer selection in single-cell foundation model probing
for donor-level aging prediction in PBMC scRNA-seq"

### Abstract structure
- Single-cell FMs are typically probed at the last layer or via mean-of-layers; we show this is suboptimal for donor-level cell-type-specific phenotypes.
- We characterize layer-of-best-readout for frozen Geneformer probing across CD4+T, NK, B cells × 3 cohort splits × cross-ancestry transfer to Indonesian/Chinese/Indian donors.
- Finding 1: NK aging signal lives in early layers (L2–L5) while CD4+T at L12; B substrate-empty across all probes.
- Finding 2: Pseudobulk-input shifts layer-of-best-readout to early layers regardless of cell type, while per-cell mean-pool layer choice is cell-type-conditional. Two-axis layer-selection principle.
- Finding 3: At matched splits, frozen Geneformer + ridge readout achieves R = 0.61–0.78 on CD4+T cross-cohort age regression, within ~0.10 R-units of bulk gene-EN baseline. Capacity (rank-16 vs 32 LoRA) and training (3 epochs) further don't change this.
- Methodology recommendation: per-cell-type layer selection + matched-splits comparison.

### Sections
1. **Introduction**
   - Single-cell FMs are typically probed at last layer; donor-level evaluations exist but methodology varies.
   - Prior work (TF paper, scImmuAging, Geneformer/scFoundation): mostly use last-layer or mean-of-layers; comparison vs bulk uses different splits.
   - Our contribution: characterize layer-by-layer probing across cell types and unit-of-analysis, with matched-splits comparison.

2. **Methods**
   - Cohorts: OneK1K (981), Stephenson (24-29), Terekhova (166), AIDA (293-307). LOCO splits: train cohorts → holdout cohort + AIDA cross-ancestry.
   - Per-cell mean-pool ridge readout: extract per-layer embeddings, mean-pool per donor, ridge regress age.
   - Pseudobulk-input ridge readout: aggregate raw counts per donor → rank-value tokenize as one pseudo-cell → frozen forward → per-layer extraction → ridge.
   - Bulk baseline: ElasticNetCV on log1p-mean per-donor pseudobulk, top-5000 HVG, standardized features.
   - LoRA fine-tuning: rank-16 (production), rank-32 (capacity ablation), 3 seeds, e5b config (3 epochs, mean-pool, 50 cells/donor).
   - Bootstrap CIs (n=1000) on Pearson R per layer.

3. **Results**
   - 3.1: Frozen Geneformer per-cell mean-pool: cell-type-conditional layer-of-best-readout. NK at L3.3, CD4+T at L9.7, B substrate-empty. Δ(best vs L12) largest on cross-ancestry AIDA.
   - 3.2: Pseudobulk-input shifts layer choice to early layers for CD4+T (L1–L4) and stays at early layers for NK (already L0–L3). Two-axis principle: pseudobulk-input always favors early layers; per-cell mean-pool layer choice is cell-type-conditional.
   - 3.3: Matched-splits comparison: gene-EN R = 0.61–0.78 on CD4+T cross-cohort, vs Geneformer ridge R = 0.53–0.62. Gap is ~0.05–0.15 R-units, not the 0.38 implied by TF paper splits.
   - 3.4: Cross-FM comparison: scFoundation 3B-param frozen+ridge lags Geneformer by 0.08–0.10 R-units across CD4+T. Matched-splits parity is Geneformer-specific.
   - 3.5: Capacity ablation: rank-32 LoRA does not improve over rank-16 at 3-seed mean (D.21 results).
   - 3.6: Cross-ancestry AIDA: best-performing FM configuration (rank-32 LoRA L9 ridge) achieves R=0.617/MAE=6.92 (D.21 verifies at 3-seed mean), comparable to Pasta-REG (R=0.659/MAE=6.32) and gene-EN matched (R=0.616/MAE=6.42).

4. **Discussion**
   - Layer specialization in single-cell FMs is cell-type-conditional. Late-layer specialization captures cell-state abstractions (CD4+T) but doesn't help for cell types where aging signal is coarser composition (NK).
   - Pseudobulk-input destroys late-layer specialization across all cell types.
   - The "FMs lose to bulk" framing in prior literature reflects methodology-comparison mismatches, not fundamental FM limitations — but only for Geneformer; scFoundation does lag.
   - Limitations: single task (PBMC aging), single organism, three cell types. Generalization to other single-cell FM applications requires further validation.

5. **Conclusion**
   - Methodology recommendations: per-cell-type layer selection, matched-splits comparison, multi-seed reporting.
   - Reproducibility: code/embeddings/results released.

### Risks if Tier 1 verifications collapse
- If D.21 L9 AIDA 3-seed > 8.5y → drop §3.6's parity claim, weaken to "FM is competitive within ~1.5y MAE."
- If D.22 NK ΔR collapses on any cohort → drop §3.1's "cross-cohort robustness" framing; demote to "preliminary observation."
- Both collapsing → outline (a) is not viable; switch to outline (b).

---

## Outline (b): Comparison-led — "Matched-splits comparison of single-cell foundation models against bulk baselines on PBMC aging prediction"

**Viable regardless of Tier 1 outcomes.** This is the safer outline that
treats cell-type-conditional layer selection as one of several contributions
rather than the headline.

### Title (working)
"Methodology-aware comparison of single-cell foundation models and bulk
baselines on donor-level aging prediction in PBMC scRNA-seq"

### Abstract structure
- Prior comparisons of single-cell FMs vs bulk gene-EN baselines on donor-level age regression have used different cohort splits, preprocessing, and hyperparameter grids — making the "FM loses by X" claims hard to interpret.
- We re-run gene-EN on the same LOCO splits used for FM evaluation, with the same preprocessing and a comparable hyperparameter search.
- Finding 1: Matched-splits gene-EN reaches R=0.61–0.78 on CD4+T cross-cohort, vs prior reports of R=0.83. The "FM loses" framing was substantially driven by methodology mismatch.
- Finding 2: Frozen Geneformer + ridge readout closes most of the matched gap on CD4+T (Δ ~0.05–0.15 R-units). LoRA fine-tuning at rank-16 and rank-32 does not improve over frozen+ridge at 3-seed mean.
- Finding 3: scFoundation 3B does not match Geneformer; matched-splits parity is Geneformer-specific.
- Finding 4 (incidental): Per-cell mean-pool layer-of-best-readout is cell-type-conditional (NK early-layer, CD4+T late-layer). Pseudobulk-input shifts layer choice toward early layers for all cell types.
- Conclusion: Methodology-aware comparison is essential. FM-specific behavior matters more than FM-class behavior.

### Sections
1. **Introduction** — same as outline (a), but framed around the FM-vs-bulk comparison question rather than the methodology contribution.

2. **Methods** — same.

3. **Results**
   - 3.1: Matched-splits gene-EN baseline. Comparison to TF-paper R=0.83 LOCO and 0.77 AIDA. Driver analysis: more cohorts + different preprocessing + different hyperparameter grids.
   - 3.2: Frozen Geneformer + ridge readout closes most of the matched gap.
   - 3.3: LoRA fine-tuning (rank-16, rank-32) does not improve over frozen+ridge at 3-seed mean.
   - 3.4: scFoundation lags Geneformer at matched splits.
   - 3.5: Cross-ancestry AIDA characterization (best FM config vs Pasta vs gene-EN matched).
   - 3.6: Layer-of-best-readout characterization — supplementary methodology contribution.

4. **Discussion** — emphasizes the methodology-comparison lesson over the layer-selection contribution.

5. **Conclusion** — methodology recommendations + cautious framing of FM-class behavior.

### Risks if Tier 1 verifications collapse
- If D.21 collapses → adjust §3.5 numbers; the framing is robust because it's already a "FM is competitive" rather than "FM is tied" claim.
- If D.22 collapses → demote §3.6 to a brief supplementary mention.
- Both collapsing → still publishable; the matched-splits comparison itself doesn't depend on either single-seed number.

---

## Selection criteria at the verification gate

Pick outline (a) if ALL of the following hold:
1. D.21 L9 AIDA 3-seed mean MAE ≤ 7.5y AND R ≥ 0.55
2. D.22 NK ΔR(best vs L12) > +0.05 at 3-seed mean across all 3 cohorts (or 2/3 with cohort-specific caveat)
3. D.24 NK pseudobulk-input best layer in L0–L4 (DONE — confirmed L0–L3)
4. D.25 — already done, scFoundation finding strengthens both outlines

Pick outline (b) if any of (1) or (2) collapses but the underlying matched-splits framing survives.

If both (1) and (2) collapse → pause and audit per the §28 base-rate concern.

---

## Shared infrastructure (do regardless of outline choice)

Whichever outline is chosen, the following are common contributions:
- **Methodology section** detailing per-cell mean-pool ridge readout vs pseudobulk-input ridge readout.
- **Bootstrap CI tables** for all positive numbers (D.26 done).
- **Three-way comparison table** (gene-EN | Geneformer | scFoundation) (D.25 done).
- **Cross-ancestry AIDA characterization** as a separate subsection (the best-supported positive in either outline).
- **B-substrate-empty cross-method confirmation** (gene-EN B R<0.2 across most matched-splits conditions per D.23 results-pending; check decision rule on completion).

---

## Decision-rule lookup at the gate

After D.21 + D.22 + D.23 land:

| D.21 (L9 AIDA MAE) | D.22 (NK ΔR cohorts) | D.23 (B-empty) | Outline | Rationale |
|---|---|---|---|---|
| ≤7.5 | All 3 >+0.05 | Bilateral | (a) | All three positives anchor-ready |
| ≤7.5 | 2/3 >+0.05 | Bilateral | (a) with caveat | NK finding limited but still cross-cohort |
| 7.5–8.5 | Any 3-seed | Any | (b) | Parity claim weakens; switch to "competitive" |
| >8.5 | Any 3-seed | Any | (b) | Drop AIDA-parity from headline |
| Any | ≤1/3 >+0.05 | Any | (b) | NK methodology contribution is supplementary |

---

## Pre-commitment metadata

- File written: 2026-04-29
- Both outlines drafted before Tier 1 verification (D.21–D.23) lands.
- Selection at the gate is rule-based, not vibes-based.
- Decision rules in `notes/decision_rules_phase3.md` and the table above are the binding criteria.
