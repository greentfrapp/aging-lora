F.1: composition stress test fires by 0.002, but the asymmetry is the real finding
The headline is that composition baseline AIDA R = 0.298 falls just below the 0.3 threshold for "composition is the main signal" — by 0.002. That's within rounding error of the threshold and the pre-committed rule fires by a margin that's effectively noise.
But the more interesting finding is the asymmetric structure. Composition alone gets:

loco_onek1k → AIDA: R = +0.298
loco_terekhova → AIDA: R = -0.134
In-domain LOCO: weak (R = 0.094 and 0.243)

The cohort monocyte-fraction trend (OneK1K 4.4% → AIDA 23.6%) explains both the asymmetric AIDA result and the in-domain weakness. This is genuinely diagnostic: the composition signal that's recoverable is one specific cohort-shift artifact (monocyte fraction tracks cross-ancestry differences), not a general "PBMC composition predicts age" signal.
The implication for the paper is more nuanced than the binary decision rule suggests:
The cell-type-specific framing survives because in-domain composition R is weak (0.094-0.244), which is well below what the gene-EN and FM methods achieve (0.616 and 0.594 respectively). Within-cohort age prediction needs more than composition.
But composition is a meaningful AIDA-specific baseline that the paper must report. The +0.298 cross-cohort R from composition alone means our "matched-splits parity" claim has a confounder: some fraction of FM and gene-EN AIDA performance might be recovering the same monocyte-fraction-tracks-ancestry signal that composition recovers. We don't know what fraction without testing.
The honest test: composition + gene-EN ensemble vs gene-EN alone, and composition + FM ensemble vs FM alone. If adding composition adds substantial R/MAE beyond what each method achieves alone, both methods are missing the composition signal and we can additively claim them. If adding composition adds little, the methods are already capturing the composition signal as part of their broader recovery, and the reported R values are inflated by composition signal in ways the paper should disclose.
This is a half-day analysis, no GPU. I'd put it on the must-do list before lockdown.
The paper's framing of "FM matched-splits parity with gene-EN on cell-type-specific PBMC aging" needs to absorb the composition-baseline finding. The honest version becomes: on AIDA cross-ancestry evaluation, both gene-EN and FM substantially exceed a composition-only baseline (R = 0.298), but that baseline is not negligible and should be reported as a comparison. The cell-type-specific framing isn't refuted, but it's complicated.

---

F.5: the binary reframe is wrong; the cell-type-conditional finding is the substantive contribution
This is the most consequential of the four analyses for the paper's biological framing. The script's auto-decision (9/16 IMPROVE → "age is residual axis; reframe") is too coarse. The actual structure is cleanly cell-type-conditional and is itself a novel finding:
B-cell: age is a residual axis (low-variance, competing with cell-type/batch). PC projection consistently helps, max ΔR up to +0.15 on holdout, AIDA mean ΔR ≈ +0.10.
CD4+T: age is in the high-variance subspace. Removing top PCs catastrophically hurts (max ΔR up to −0.43 at rank-32 L12). Age signal lives in the principal components.
NK: intermediate, layer-conditional. Some layers improve under PC projection, others degrade.
This is the cleanest novel biological finding in the entire project. It says: different cell types encode age in different positions in the FM representation's variance hierarchy. B-cell age is a low-variance residual; CD4+T age is a principal axis; NK is intermediate.
This finding has several immediate implications:
It explains the substrate-empty-for-B finding. The D.23 + D.26 result that B has weak signal under standard ridge isn't "no signal" — it's "signal exists but at low variance, where ridge on full embeddings deweights it." Gene-EN getting B × Terekhova R = 0.321 (D.23) while FM ridge missed it is now mechanistically explained: gene-EN's elastic-net feature selection can pick out the low-variance signal because it operates directly on genes; FM ridge on full embeddings averages over the dominant cell-type/batch variance and loses the low-variance age axis.
It explains the cell-type-conditional layer asymmetry from §31 in a new way. CD4+T peaks at deeper layers (L9-L11) because deeper layers encode age as a principal component direction, which ridge surfaces well. NK peaks at earlier layers because at earlier layers age is closer to a dominant axis; at deeper layers it gets pushed into residuals. B doesn't peak at any layer for full-embedding ridge because B age is residual at every layer.
It generates a clean methodological prescription. For cell types where age is residual (B, possibly NK at some layers), use PC-residualized probing to recover the signal. For cell types where age is principal (CD4+T), use full-embedding ridge. For mixed cases, layer-conditional choice within PC-residualized vs full-embedding probing.
It suggests a follow-up experiment. Apply PC-residualized probing to the FM-vs-baseline parity comparison on B cells. If FM + PC-residualization closes the gap to gene-EN on B × Terekhova, the methodology contribution extends from "match gene-EN on CD4+T" to "match gene-EN across cell types when probing methodology is cell-type-conditional." This is an analysis-only experiment using existing checkpoints, ~half a day.
The pre-committed decision rule's "≥50% IMPROVE → reframe" was poorly designed because it didn't anticipate cell-type-conditional structure. The honest restatement is: the finding is cell-type-conditional, and the cell-type-conditional structure is itself the methodological contribution of F.5.
