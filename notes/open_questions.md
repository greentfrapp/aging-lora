# Open questions

Things we noticed but haven't resolved. Could feed `FUTURE_WORK.md` later, or be addressed in a later phase. Edit-in-place — remove or move when resolved.

---

## Phase-3-resolvable

**OQ-1. What is the actual baseline-vs-FM pairing-ρ?** Phase-2 measured baseline-pair ρ at 0.06–0.35. Phase-3 will measure baseline-vs-FM ρ on the CD4+T pilot. Expected to be higher (say 0.4–0.7) but could surprise us in either direction. The detectability floor flags depend on this.

**OQ-2. Does scAgeClock's persistent −13y PBMC bias have a structural cause?** Across all cohorts, all cell types except OneK1K NK, scAgeClock systematically predicts ~13 years younger than truth. Possible causes: (a) PBMC-tissue-specific underrepresentation in CELLxGENE Census training mix, (b) age-distribution skew in the training corpus (younger samples overweighted), (c) cell-type categorical embedding miscalibration for PBMC types. Worth a one-paragraph diagnostic in the preprint methods if the bias pattern persists into Phase 3 FM scoring.

**OQ-3. Why does the 3-cohort retrained LASSO fail on OneK1K-out × B cells?** `LassoCV` regularized to intercept-only (α=0.90, 0/1100 non-zero coefs, R=NaN). The 195-donor Stephenson+Terekhova training set was small and chemistry-mixed. Could be (a) genuine "no signal in this combination" (B-cell aging is harder to learn from <200 donors with chemistry mix), (b) `LassoCV`'s default α grid topped out, (c) numerical issue. Worth a one-line check before Phase 3 starts.

## Phase-4-relevant

**OQ-4. Is the FM win on B cells "rescue from LASSO collapse" or "rescue beyond Pasta"?** Pre-Phase-2, the chemistry-rescue narrative was "FMs rescue where LASSO collapses (R=0.08 → ?)". Post-Phase-2, Pasta already does some rescue (R=0.28). The interesting question is whether FMs add R *on top of* Pasta's rank-norm baseline, not just whether they beat the chemistry-collapsed LASSO. Phase-4 few-shot curve (now 3-line: LASSO + Pasta + FM) tests this.

**OQ-5. Does Pasta's chemistry-invariance generalize to other 5' chemistries (10x 5' v3, BD Rhapsody)?** Phase 2 tested only Terekhova + AIDA which are both 10x 5' v2. If the FM benchmark gets a future 5' v3 cohort, Pasta would be the natural baseline. Out of scope for this paper but worth flagging.

**OQ-6. Are scAgeClock's poor PBMC results model-architectural or training-data-distribution?** scAgeClock's training mix is dominated by non-PBMC tissues (per its paper). A PBMC-fine-tuned scAgeClock might compete with the FMs. We don't plan to do this fine-tune; flag as future work.

## Phase-5-relevant

**OQ-7. Does Pasta REG's high systematic bias on OneK1K (−23y) reflect age-distribution shift or library-prep shift?** OneK1K's 19–97yr range may differ from Pasta's training distribution. A simple mean-recentering (subtract median residual per cohort) would change MAE but not R. Worth a one-line mention if relevant for Phase-5 cross-cohort calibration.

**OQ-8. Do the per-cell-type empirical ρ values stabilize across more baselines?** Phase 2 has 9 baseline-pair ρ values per cell type (3 pairs × 3 cohorts). Phase 3 adds baseline-vs-FM ρ; Phase 4 adds FM-vs-FM ρ. If the per-cell ρ remains in the 0.1–0.4 range across additions, that's evidence for a genuine "low-ρ" regime in PBMC aging (i.e., different aging clocks make different mistakes per donor). Worth a Phase-5 sub-figure.

## Methodological

**OQ-9. Should we report leakage-flagged AND chemistry-flagged rows in the same forest plot?** Phase-4 reporting policy uses 3-way stratification (leakage_status × chemistry_match × detectability_flag). The forest plot can render up to 8 cell variants per fold (2³ combinations). If reviewer feedback says this is too busy, we collapse to a single "strict-clean ALL-three-green" headline + a leakage-and-chemistry-inclusive supplement.

**OQ-10. Bootstrap confidence intervals on per-cell MAE/R for Phase-3 win/match/loss thresholds.** Currently the win/match/loss decision is based on point estimates of MAE per cell. With 24-981 donors per cell, there's real variance. Bootstrap (1,000 resamples) per cell gives MAE 95% CIs; the win/match decision could be made on whether the FM CI excludes the best-baseline point. Easy add but not currently in scope.
