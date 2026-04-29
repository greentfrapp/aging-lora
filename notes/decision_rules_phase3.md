# Phase-3-B reframed-review verification: pre-committed decision rules

Written 2026-04-29 BEFORE the D.21–D.24 verification runs land. The §28 audit
showed that single-seed near-headline numbers can collapse at 3-seed; the
prevention is pre-commitment to outcome interpretation so post-hoc
rationalization is harder. This file is the commitment device.

If a result lands outside its committed band, the framing implication
specified here is the one that applies. New framings post-hoc need explicit
acknowledgment that they are post-hoc.

## D.21 — L9 AIDA rank-32 LoRA × 3-seed verification

**Single-seed (seed 0) baseline**: L9 R = 0.617, MAE = 6.92. Currently
load-bearing for the matched-splits parity claim (vs gene-EN matched
R = 0.616, MAE = 6.42 on the same fold).

**Decision bands (3-seed mean MAE)**:

| 3-seed mean MAE | Implication |
|---|---|
| ≤ 7.5y | "Tied with gene-EN within seed variance" headline survives. AIDA cross-ancestry parity claim is supportable. Methodology-led paper outline (a) viable. |
| 7.5y – 8.5y | "Modestly behind gene-EN, within ~1y on AIDA" — publishable but weaker. Headline shifts from "tied" to "competitive within seed variance." Outline (a) needs hedging. |
| > 8.5y | Drop AIDA-parity from headline. Fall back to "matched gap is much smaller than TF framing implied" without claiming parity. Outline (b) is the writeup. |

**Decision rule on R**: if 3-seed mean R < 0.55, the §32 parity narrative is
weakened regardless of MAE — "FM ranks donors meaningfully worse than
gene-EN" becomes the honest read.

**Robustness check**: per-seed std must be reported. If σ(MAE) > 2.0y across
3 seeds, the L9 AIDA finding is too noisy to anchor a paper claim regardless
of mean.

## D.22 — NK best-layer 3-seed verification on frozen Geneformer

**Single-seed baseline**: NK best layers L3 (OneK1K), L2 (Terekhova), L5
(AIDA); mean L3.3. ΔR(best vs L12) by cohort: OneK1K +0.044, Terekhova
+0.067, AIDA +0.121.

**Decision bands (3-seed mean ΔR per cohort)**:

| ΔR(best vs L12), 3-seed mean | Implication |
|---|---|
| > +0.05 across **all 3 cohorts** | Cell-type-conditional finding anchor-ready. NK-at-early-layers is robust. Methodology section can lead with this. |
| > +0.05 on 2/3 cohorts | Partial support. Finding survives as "NK reads better at early layers in most cross-cohort conditions"; cohort-specific caveat required. |
| > +0.05 on ≤1 cohort | Single-seed artifact. The L3.3 NK best-layer claim is not a robust cross-cohort finding. Demote from headline; mention in supplementary as "preliminary observation requiring further validation." |

**Best-layer drift check**: if 3-seed best-layer per cohort shifts by >2
layers from single-seed (e.g., L3 → L7 on OneK1K), the cell-type-conditional
*specific layer* claim is not supportable; only the directional "NK reads
earlier than CD4+T" can be claimed.

**Critical**: do NOT relax these thresholds if results land just outside them.
The §28 close-MATCH-not-WIN lesson applies.

## D.23 — Matched-splits gene-EN on NK and B

**No prior single-seed baseline; this is the first run on these cell types.**

**Decision bands for B-substrate-empty interpretation**:

| Gene-EN B R (3 conditions) | Implication |
|---|---|
| All R < 0.20 | B-substrate-empty interpretation gets bilateral support. The paper claim "B aging signal is absent in PBMC scRNA-seq at this scale across both bulk and FM methods" is supportable (with TF-pseudocell caveat). |
| Any R ≥ 0.20 | B-empty is FM-specific OR method-specific to our gene-EN protocol. The "substrate-level finding" framing is wrong — revert to "B is harder for both methods we tested, signal extractable but absent at our protocol." Strengthens the case for trying TF's pseudocell augmentation. |
| Any R ≥ 0.40 | B is *not* substrate-empty for gene-EN. The contrast with FM-frozen R~0 becomes an FM-class failure on B specifically, which is a *different* paper claim than the substrate-level one. |

**Decision band for NK matched-splits**:

| Gene-EN NK R | Implication |
|---|---|
| R ≥ 0.50 across conditions | NK matched-splits is closer to TF-class than to FM-frozen NK (R=0.30). Gap on NK is real and the "FM matches gene-EN at matched splits" claim is CD4+T-specific, not pan-cell-type. |
| R 0.30–0.50 | NK at matched splits is similar to FM-frozen NK ridge readout. The matched-splits parity finding extends to NK. |
| R < 0.30 | NK is hard for both methods (matches §28's NK-substrate-weak narrative). |

## D.24 — Pseudobulk-input frozen Geneformer on NK and B

**Note**: pseudobulk embeddings already exist for NK and B from D.18. What's
new here is *ridge fits on the missing cross-cohort conditions* (NK ×
Terekhova, B × AIDA, etc., not yet in the pseudobulk ridge CSV) and the
analysis comparing pseudobulk-input layer profile to per-cell mean-pool
layer profile **for NK and B specifically**.

**Single-seed CD4+T baseline**: pseudobulk-input shifts best-R layer to L1–L4
(early), opposite of per-cell mean-pool which favors L12.

**Decision bands for the two-axis principle**:

| NK pseudobulk-input best layer | Implication |
|---|---|
| L0–L4 (matches CD4+T pseudobulk shift) | Unit-of-analysis effect is universal — pseudobulk-input → early layers regardless of cell type. Two-axis principle (cell-type × unit-of-analysis) **supported as: pseudobulk-input always favors early layers, but per-cell mean-pool layer choice is cell-type-conditional**. |
| L5–L8 (intermediate) | Unit-of-analysis effect is partial; cell-type still matters even in pseudobulk-input. Two-axis principle supported in weaker form. |
| L9–L12 (matches per-cell mean-pool L12 for CD4+T or L3 for NK pattern) | No unit-of-analysis effect on NK. The principle is CD4+T-specific. Headline two-axis claim collapses; finding becomes "pseudobulk-input shifts CD4+T to early layers, doesn't shift NK." |

**B pseudobulk-input best layer**: substrate-empty likely makes layer choice
ill-defined (very low R across all layers). If B pseudobulk-input has best R
> 0.30 at any layer, this is *unexpected* and would warrant a note —
substrate-empty in per-cell mean-pool but signal-bearing in pseudobulk-input
would be a methodologically-interesting finding (donor-aggregation
*recovers* B signal that per-cell-pool destroys).

## Verification gate (after D.21–D.24 land)

The reframed-review proposed three outcome paths:

1. **All four hold cleanly**: methodology-led headline (outline (a)) supported.
2. **Some hold, some don't** (most likely outcome by §28 base rate): partial framing — outline (b) is the writeup, with whichever findings survive as supporting contributions.
3. **Multiple regress**: structural framing-failure pattern. Pause and run §28-style audit before writing anything.

**Pre-committed mapping from results to outcome**:
- Outcome 1 requires: D.21 mean MAE ≤ 7.5y AND D.22 ΔR > +0.05 on all 3 cohorts AND D.23 + D.24 not contradicting.
- Outcome 2 is the default if any single Tier-1 task lands in its middle band.
- Outcome 3 triggers if 2+ Tier-1 tasks land in their bottom band, OR if any single result is so far outside its committed bands that the underlying assumption was wrong.

## What NOT to do at the gate

- **Do not invent new framings to fit results that don't match committed
  bands.** That is the post-hoc rationalization the §28 lesson warns against.
- **Do not relax thresholds** because the result was "close." If MAE = 7.6y,
  it's in the 7.5–8.5y band, not the ≤7.5y band.
- **Do not selectively report** seeds. Report 3-seed mean ± std for every
  3-seed task. If a particular seed is dramatically out-of-distribution
  (>3σ), report it but do not exclude it.
- **Do not commit to a venue tier** before the gate. The post-§28 lesson is
  that aspirational framing precedes verification failure.

## Pre-commitment metadata

- File written: 2026-04-29
- Author: Claude Opus 4.7 (autonomous mode, user offline 12h)
- Commits to find this file: search `notes/decision_rules_phase3.md` in git log
- Supersedes: nothing (first formal pre-commitment of decision rules in this project)
- Linked roadmap items: D.21, D.22, D.23, D.24, D.27 (this file partially fulfills D.27 by inventorying the load-bearing single-seed numbers that need verification)
