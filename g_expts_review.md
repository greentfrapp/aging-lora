# G.1, G.2, G.3: the headline holds, the cell-type-conditional story stays mostly contained

This batch is the cleanest stress-test sequence we've run. Three pre-committed decision rules, three rules fired in different ways, and the combined picture is more constrained than I expected. Let me work through what each does.

## G.1: composition is dominated, the parity claim is uncomplicated

This is the cleanest "pre-committed decision rule, decisive outcome" result in the conversation. ΔR ranges from −0.004 to +0.004 across all six rows; F.1's +0.298 composition AIDA signal is fully absorbed by both gene-EN and FM rank-32. The methods aren't recovering composition signal as an additive component — they're already capturing it as part of their broader recovery.

Two paper-relevant implications:

**The matched-splits parity claim doesn't need a confound disclosure.** I'd been worried last turn that the F.1 composition baseline at +0.298 might mean some fraction of the FM and gene-EN AIDA performance was monocyte-fraction-tracking-ancestry rather than within-cell-type aging. G.1 says no. The parity claim (FM 0.589 vs gene-EN 0.616 on AIDA) is over and above the composition baseline; both methods substantially exceed it (by ~0.3 R) and adding composition explicitly doesn't help.

**Composition-only baseline is reportable as a strict baseline.** The paper can include "composition-only ElasticNet R = 0.298 on AIDA cross-ancestry; both gene-EN and FM exceed this by ~0.3 R, and adding composition explicitly to either method does not improve performance." This is a clean and honest comparison that contextualizes the parity finding without complicating it.

I had flagged this as one of the more concerning structural questions two turns ago — the cell-type-frequency finding I said I should have surfaced earlier. The result is that the concern was real but the methods turn out to be robust to it. That's the outcome where running the test pays off precisely because the test could have come out the other way.

## G.2: B-cell within-cohort matches, B-cell cross-ancestry doesn't

The headline is that PC-residualized FM probing on B × Terekhova holdout reaches R = +0.281 with the CV-picked recipe (L9, k=5) — within 0.041 of gene-EN's R = 0.321. PC-residualization closes 72% of the gap. The decision rule fires "MATCHES gene-EN on Terekhova holdout."

But the AIDA result is a clear loss: PC-residual R = −0.072 vs gene-EN R = +0.168. Cross-ancestry transfer for B fails regardless of probing recipe.

Three observations worth pulling out:

**The CV-honest vs F.5-holdout-best distinction matters.** F.5's reported max ΔR = +0.144 was at (L0, k=10), a different point on the (layer × k_pc) surface than the CV-picked (L9, k=5). The CV-picked recipe is what would actually deploy; F.5's reported point was post-hoc-best on holdout. The honest deployable result is R = +0.281, not the F.5-headline of +0.290.

This is a methodologically important detail. The F.5 finding was reported at the holdout-best point, which is the upper bound of what the (layer × k_pc) surface achieves on test data. CV-honest deployment lands close to but slightly below that upper bound. The paper needs to report the CV-honest number (R = 0.281, gap 0.041 to gene-EN) as the deployable claim, with the F.5-holdout-best number (R = 0.290, gap 0.031) as the upper-bound reference. This is the same pattern as the bands-not-points and oracle-vs-deployment distinctions earlier — the honest deployment number is the relevant metric.

**Single-seed limitation is a real constraint.** G.2 ran on frozen seed 0 only because seeds 1/2 frozen B embeddings would require GPU re-extraction. This is reasonable given $0 CPU constraint, but the §28 lesson applies: a single-seed near-headline number is a correction-risk. The B-cell parity claim should be framed accordingly: "single-seed PC-residual probing on B × Terekhova achieves R = 0.281 vs gene-EN R = 0.321; multi-seed verification deferred." If multi-seed verification is feasible later (~$5-10 to extract two more seeds of frozen B embeddings), it's worth doing before final lockdown.

**Cross-ancestry on B-cells is a methodology limitation.** The R = −0.072 on AIDA isn't a small gap; it's a sign-flip relative to gene-EN's R = +0.168. PC-residual recovers within-cohort B-cell signal but doesn't transfer cross-ancestry. The paper should be honest about this: the cell-type-conditional probing recipe extends matched-splits parity to B-cell within-cohort but not to B-cell cross-ancestry. This is consistent with the F.5 mechanistic reading — B-cell age signal is low-variance residual, and low-variance residual signals are particularly vulnerable to cross-ancestry distribution shift because they're more cohort-specific than principal-axis signals.

## G.3: the global aggregator says refinement, the per-cell-type breakdown says something more nuanced

The headline-aggregator decision rule fires "REFINEMENT" with mean ΔR_holdout = +0.0086. But this aggregator is biased toward zero by construction: CD4+T's cell-type-conditional recipe is the same as its fixed-recipe baseline (full-embed at best layer), so 8/16 conditions have ΔR ≡ 0. The per-cell-type means tell the actually-informative story:

- CD4+T: ΔR ≡ 0 by construction (8 conditions)
- B: ΔR_holdout = +0.062, ΔR_aida = +0.138 (2 conditions)
- NK: ΔR_holdout = +0.002, ΔR_aida = +0.091 (6 conditions)

Under the per-cell-type aggregator, ΔR_aida = +0.076 and crosses the +0.05 threshold. Under the per-condition global aggregator, ΔR_holdout = +0.0086 doesn't.

This is the kind of "the pre-committed rule is technically pass/fail in different ways depending on aggregator choice" outcome that requires honest interpretation rather than mechanically applying the rule. Two competing readings:

**Reading 1 (refinement):** The condition-flattened global aggregator is the right one because it weights each (cell × fold × seed) condition equally. Cell-type-conditional probing helps on B specifically but doesn't move the global average meaningfully. Report as a refinement, headline stays "matched-splits parity on CD4+T."

**Reading 2 (cell-type-conditional contribution):** The per-cell-type aggregator is the right one because cell types are the entity the methodology is conditional on. B improves substantively, NK improves modestly on AIDA, CD4+T is unchanged. The recipe is "different cell types have different optimal probing strategies," which is itself a methodological recommendation.

The honest position is probably between these. The per-condition CD4+T results (rank-32 +0.011 to +0.033 above gene-EN, rank-16 +0.011 to +0.027) are slight wins within seed variance — the parity claim still stands but with FM marginally above gene-EN at full rank-32, slightly weakening the "FM trails gene-EN by 1.35y on average" framing from D.36. This is itself worth checking against D.36's bootstrap analysis.

The B-cell improvement is real and within-cohort-specific. The AIDA improvements for B and NK are conditional on within-cohort training and don't extend to robust cross-ancestry deployment.

## What this combined picture does to the paper

The Tier 1 pre-lockdown analyses have produced a substantially clearer picture than three turns ago. The paper's contributions list updates:

**1. Matched-splits parity on CD4+T** (D.21, D.32) — survives, strengthened by G.1 (composition is not a confounder) and slightly reframed by G.3 (rank-32 LoRA narrowly above gene-EN at the per-condition level; D.36's bootstrap analysis still applies for the bulk-distribution comparison).

**2. Cell-type-conditional FM age-encoding in variance hierarchy** (F.5) — survives. B-cell as low-variance residual; CD4+T as principal-axis. This is the substantive biological/representation finding.

**3. Cell-type-conditional probing recipe** (G.2, G.3) — survives as a refinement, with explicit scope: closes the FM-vs-gene-EN gap on B × within-cohort by 72% (single-seed); doesn't close the gap on B × cross-ancestry. NK gains are AIDA-only and cohort-conditional. CD4+T is unchanged. The recipe is a refinement contribution rather than a primary one, with B-within-cohort as the strongest demonstration.

**4. Composition baseline reported as strict baseline** (F.1, G.1) — uncomplicated. Both methods exceed composition by ~0.3 R; composition is not additive.

**5. Cell-type-conditional layer-band selection** (§31, D.22) — survives, mechanistically explained by F.5's variance-hierarchy finding.

**6. Bootstrap-at-small-N regularization-bias finding** (E.8) — survives.

**7. Methodology-comparison essentiality** (D.34) — survives.

**8. Forensic findings on per-cell-MSE-on-donor-labels** (§27, §28) — survives.

**9. Ridge-as-near-maximal-linear-recovery** (F.4) — survives at the linear-probe level. Analysis A (non-linear probes) still pending.

The paper structure is now substantially more constrained than two turns ago. F.5's biological finding is the substantive contribution. G.2 + G.3 confirm the F.5 mechanism translates to a deployable methodology refinement (with appropriate scope-bounding). G.1 closes the composition concern decisively.

## What still needs running

Three items remain on the priority list:

**Tier 1 (must do):**

1. **Multi-seed verification of G.2's B × Terekhova result.** ~$5-10, half a day for the GPU re-extraction plus CPU analysis. The single-seed R = 0.281 is the load-bearing number for the cell-type-conditional probing extension claim; §28 lesson applies. Without multi-seed verification, the B-cell parity claim is single-seed-near-headline.

2. **Analysis A: per-layer non-linear probe sweep.** ~1 day, no GPU. F.4 resolved the within-linear-probe question; Analysis A tests whether non-linear probes shift the layer ordering or recover signal beyond ridge. Pending. Determines whether the methodology contribution is "linear-probe layer selection" or "general layer selection." Material to the writeup framing.

**Tier 2 (defensive, do if time):**

3. **Donor-deduplication audit.** ~Half day, no GPU. Defends against an entire reviewer concern category. Cheap insurance.

4. **F.5 cross-method check on gene-EN feature variances.** ~Half day, no GPU. Tests whether F.5's variance-hierarchy finding is FM-specific (representation property) or holds for gene-EN (biological property). Determines how strong the biological framing in the paper can be.
