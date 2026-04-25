# Phase 3 — CD4+ T pilot fine-tune & preprint

## Goal

With baselines in hand, this phase produces the minimal viable result: LoRA fine-tuning of **Geneformer, scFoundation, and scGPT** on the CD4+ T cell LOCO folds, compared head-to-head against the Phase 2 baselines on identical splits. CD4+ T is chosen first because it has the strongest sc-ImmuAging internal R (≈0.91) and the largest per-cohort donor pool, making it the best-powered starting point. This phase also calibrates the true GPU-hours-per-run from measured timing (not the ≈4 h pilot estimate in the proposal), which gates the feasibility of Phase 4's full matrix. The phase ends with a bioRxiv preprint submission covering the CD4+ T LOCO result — the sprint-to-preprint mitigation for scoop risk.

### Model-panel decision (recorded 2026-04-24 after leakage audit)

The Phase 3 pilot launches with **three** models, not the original two, driven by the completed leakage audit (`data/leakage_audit.csv`, `methods/leakage_audit_notes.md`):

| Role | Model | Why |
|---|---|---|
| **Lead (headline)** | **Geneformer** | Only foundation model `clean` across *all* three training cohorts + AIDA. Carries the paper's primary "true-holdout generalization" claim. |
| Secondary (primary-fold contrast) | **scFoundation** | `clean` on OneK1K and Terekhova (the two primary LOCO folds). `overlapping` on Stephenson via HCA-Covid19PBMC ingestion; Stephenson rows carry the caveat footnote. Worth the extra LoRA run because it's the second "true-holdout" comparator on the 981-donor primary fold. |
| Comparator | **scGPT** | `overlapping` on OneK1K and Stephenson (CellxGene Census). `clean` only on Terekhova. Kept in the panel because it is the most-cited single-cell FM and readers expect a scGPT row; all non-Terekhova scGPT rows go into the leakage-flagged analysis pathway. |
| *Deferred to Phase 4* | UCE | Same overlap pattern as scGPT (CellxGene Census); adds no new info relative to scGPT at the Phase 3 pilot. Enters in Phase 4 for the full matrix. |

**Headline primary fold: `loco_terekhova` (166 donors, 10x 5' v2), the only fold that is leakage-`clean` for all four FMs.** `loco_onek1k` is a secondary primary fold — cleanest for Geneformer and scFoundation; scGPT/UCE rows on it are leakage-flagged. This is the inverse of the original plan, which had OneK1K as the main LOCO table.

### Chemistry-robustness framing (recorded 2026-04-24 after Task 1f)

Phase 1 Task 1f (`methods/terekhova_chemistry_shift.md`) established that the pre-trained LASSO's 3'→5' transfer is cell-type-selective: CD4T and CD8T retain R ≈ 0.7–0.8 on Terekhova; MONO/NK/B degrade substantially, with B cells collapsing to R = 0.08. The Phase 3 pilot uses CD4+ T — which is chemistry-robust for LASSO — so the pilot's FM-vs-LASSO contrast on `loco_terekhova` is a *clean* FM benefit test, not a chemistry-rescue test. This is deliberate: the chemistry-rescue story (Can FMs recover where LASSO fails?) is sharper on B and NK cells and is reserved for Phase 4's few-shot curve, where it becomes a secondary headline rather than a pilot distraction.

For CD4+ T itself, the Phase 3 pilot produces a **3-comparator × 3-chemistry-context** subfigure (LASSO-pretrained vs Pasta-REG vs Geneformer-LoRA × {3' OneK1K, 5' Terekhova European, 5' AIDA Asian}) so the preprint headline shows chemistry-generalization AND ancestry-generalization behaviour in a single composite figure. The baselines are already scored for all three contexts (Phase-2 Tasks 2.1, 2.4, 2.6); only the FM-on-5' and FM-on-AIDA numbers are new.

### Tri-headline preprint structure (revised 2026-04-25 after Phase-2 add-ons)

With Phase 2 closed (75-row `results/baselines/loco_baseline_table.csv` covering 4 baselines × 3 training cohorts × 5 cell types + AIDA × 3 baselines × 5 cell types), the preprint headline expands from one cell to **three CD4+ T headline cells**, each with explicit win/match/loss criteria computed against the **per-cell minimum of FOUR baselines: {LASSO-pretrained, LASSO-retrained-3cohort, scAgeClock, Pasta-REG}**:

| Headline cell | Best baseline / MAE | 10%-win FM target | Story if FM wins |
|---|---|---|---|
| **OneK1K CD4+T** (981 donors, 10x 3') | LASSO-pretrained 9.4y | ≤8.5y | "FMs match-or-beat the published 5-cohort LASSO at 3-cohort training" |
| **Terekhova CD4+T** (166 donors, 10x 5') | Pasta-REG 8.0y | ≤7.2y | "FMs beat the chemistry-invariant rank-norm bulk model under chemistry shift" |
| **AIDA CD4+T** (595 donors, 10x 5' v2 + Asian ancestry) | Pasta-REG **6.3y** | **≤5.7y** ★ | "FMs beat the strongest published baseline on the cross-ancestry headline cell" |

★ AIDA CD4+T's Pasta floor (6.3y) is the lowest MAE in the entire 75-row baseline matrix; this is the toughest cell. AIDA is leakage-`clean` for all 4 FMs (Phase-1 audit) AND for scAgeClock (Phase-2 Task 2.2 audit) AND for Pasta (bulk pretraining), making it the only cell where every comparator + every FM is strict-clean simultaneously.

The Terekhova bar is materially tighter than the original "beat LASSO 9.2y" gate. The AIDA bar is tighter still. Pre-Phase-2 estimate of beating-best-baseline by 10% on Terekhova CD4T was ~80%; revised post-Phase-2 estimate is **~50–60% on Terekhova** and **~30–40% on AIDA** (Pasta-REG is rank-norm-invariant + ancestry-invariant in a way that's hard to beat).

**Headline classification rules (per cell):**

- *Win*: FM clears the 10% margin against the per-cell best baseline. Cell-level claim: "FMs outperform the strongest published baseline."
- *Match*: FM matches the best baseline within ±5%. Cell-level claim: "FMs match Pasta-REG (or LASSO-pretrained) at lower training cost on three cohorts."
- *Loss*: FM loses to the best baseline. Cell-level: null finding for that cell.

**Aggregate preprint outcomes** based on count of cell-level wins across the three headline cells:

- **3/3 wins** → headline: "FMs outperform the strongest published baselines across chemistry shift and ancestry shift on the CD4+T LOCO-pilot."
- **2/3 wins** → headline: "FMs outperform on [winning cells]; [losing cell] is a Pasta-tie/loss" (still a strong primary result).
- **1/3 wins** → degraded claim: "FMs match-or-beat published baselines on [winning cell]; deeper investigation needed for ancestry/chemistry generalization."
- **0/3 wins** → pivot to evaluation-study framing: "Do scRNA-seq FMs improve on rank-normalized bulk-transcriptomic clocks for PBMC age prediction?" with the null finding as the headline.

The 4th comparator (LASSO-retrained-3cohort) is the **methodologically symmetric apples-to-apples baseline** to the FM fine-tunes — both see the same 3 cohorts. Including it in the gate directly addresses the "FM win is just from more training data" reviewer concern. Phase-2 data shows LASSO-retrained ≈ LASSO-pretrained for CD4+T/CD8+T, so the per-cell minimum bar is unchanged for those cells, but explicitly listing both LASSOs in the comparator panel pre-empts the criticism.

## Success criteria

- Geneformer, scFoundation, and scGPT LoRA (rank-16, attention + MLP layers, peft library) fine-tuned on all CD4+ T LOCO training folds; per-fold LOCO m.a.e. and Pearson R recorded for all three models and compared against the **Phase 2 panel (LASSO-pretrained + LASSO-retrained-3cohort + scAgeClock + Pasta-REG)** numbers on the same folds. Result rows carry the three Phase-4 stratification columns (`leakage_status` from `data/leakage_audit.csv`, `chemistry_match_to_baseline_training`, `detectability_flag`).
- **Beat-best-baseline gate evaluated** with the **per-cell min-of-4** rule: for each (cohort, cell type) cell, compute relative MAE reduction of each FM vs. the *minimum* of {LASSO-pretrained, LASSO-retrained-3cohort, scAgeClock, Pasta-REG} MAEs on that cell. Headline win on a cell requires the relative reduction to be ≥10% against this minimum-MAE baseline. The per-cell win/match/loss classification rule (above) is recorded for each (cohort, cell type) cell and aggregated across the three headline cells (OneK1K, Terekhova, AIDA) for the preprint's primary table. Including LASSO-retrained-3cohort as the 4th comparator addresses the "FM win is just from more training data" concern: the retrained LASSO sees the same 3-cohort corpus as the FMs.
- **AIDA scoring (added 2026-04-25):** every fine-tuned FM checkpoint produced in this phase is also scored on AIDA CD4+T (595 donors, 10x 5' v2, Asian ancestry) — one extra inference call per fine-tune, no additional training. AIDA is held back from training but scored at evaluation time to give the preprint a cross-ancestry headline cell *before* Phase 4 begins. AIDA × FM rows are leakage-`clean` for all 4 FMs (Phase-1 audit) so they enter the strict-clean meta-analysis directly. **AIDA pipeline note:** the trained checkpoint from each LOCO fold is applied to AIDA without retraining; AIDA donor splits from `data/aida_split.json` are honored (the ancestry_shift_mae half is used here; the age_axis_alignment half is reserved for Phase 5).
- **Chemistry+ancestry-robustness subfigure produced for CD4+ T**: 3-comparator × 3-chemistry-context panel showing Pearson R for {LASSO-pretrained, Pasta-REG, Geneformer LoRA} × {3' OneK1K, 5' Terekhova European, 5' AIDA Asian}. The 3×3 panel simultaneously tests chemistry robustness (3'→5' transfer) AND ancestry robustness (European→Asian transfer) in one figure. Pasta-REG is the chemistry-invariant + ancestry-invariant baseline (Phase-2 Pasta-B R=0.28 on Terekhova vs LASSO R=0.08; Pasta CD4T R=0.66 on AIDA vs LASSO R=0.65). The subfigure asks: "does single-cell FM pretraining add chemistry-invariance + ancestry-invariance *on top of* what rank-normalized bulk modelling already provides?" Committed as `results/phase3/cd4t_robustness_3x3.csv` + `fig_cd4t_robustness_3x3.pdf`. Headline element of the preprint.
- Measured GPU-hours per LoRA run (mean and SD across three random seeds) recorded in `compute/runtime_log.csv`; Phase 4 full-matrix time estimate updated from this measurement before Phase 4 begins.
- m.a.e.-detectability floor reviewed using CD4+ T pilot data: compute paired Wilcoxon power from observed per-donor absolute residuals (empirical baseline-vs-FM ρ measured from CD4+ T pilot). The preprint methods reports **all three ρ values bracketing the truth**: (i) Phase-1 planning ρ=0.8 (overoptimistic, n_required=132–229); (ii) Phase-2 empirical baseline-pair ρ=0.06–0.35 (conservative lower bound, n_required=502–1,075); (iii) the Phase-3-measured baseline-vs-FM ρ which is expected to fall between these. The post_phase3 ρ value drives the per-cell-type primary/exploratory flags via a `post_phase3_override` block in `data/loco_folds.json` (preserves Phase-1 planning flags as baseline; supersedes them for Phase 4 reporting).
- bioRxiv preprint posted within 10 weeks of project start; manuscript's headline figure is **the CD4+ T tri-cell result** (OneK1K + Terekhova + AIDA) for Geneformer + scFoundation + scGPT vs. **the per-cell minimum-MAE of {LASSO-pretrained, LASSO-retrained-3cohort, scAgeClock, Pasta-REG}**, with the full 20-row leakage-audit table (5 models × 4 cohorts), the frozen split design, the 3×3 chemistry+ancestry robustness subfigure, and the bracketed detectability-ρ disclosure. At this point the paper is "minimal viable" regardless of whether criterion (a) is met on all three headline cells.

## Tasks

- [ ] Task: Implement LoRA fine-tuning wrapper for Geneformer, scFoundation, and scGPT. Using peft (LoRA rank-16 applied to attention + MLP projection layers), build training scripts that accept a cell-type label, a LOCO fold index, and a random seed; output a model checkpoint and per-donor age predictions on the held-out cohort. Run three seeds per (model, fold) combination for CD4+ T cells on the **two primary LOCO folds** (`loco_onek1k`, `loco_terekhova`); Stephenson LOCO is exploratory-only and can be deferred. Record wall-clock time and peak GPU memory per run. Done when reproducible per-fold predictions exist for CD4+ T × {Geneformer, scFoundation, scGPT} × {loco_onek1k, loco_terekhova} × 3 seeds.

- [ ] Task: Score every fine-tuned FM checkpoint on AIDA CD4+T. For each (model, LOCO-fold, seed) checkpoint produced by the previous task, run an additional inference pass on `data/cohorts/aida_eval/CD4p_T.h5ad` filtered to the `ancestry_shift_mae` donor half from `data/aida_split.json` (307 donors). Append per-donor predictions to `results/phase3/cd4t_aida_predictions.csv` and per-(model, source-fold, seed) summary rows to `results/phase3/cd4t_aida_summary.csv`. The AIDA evaluation has no extra training cost — it's just inference. AIDA is leakage-clean for all 4 FMs + scAgeClock + Pasta, so its rows enter the strict-clean meta-analysis directly. Done when AIDA CD4+T predictions exist for all 18 (3 FMs × 2 source folds × 3 seeds) checkpoints.

- [ ] Task: Evaluate CD4+ T pilot results and update detectability floor. Compute per-fold LOCO m.a.e. (median absolute error) and Pearson R for each of the three FMs **on each of the three headline cells** (OneK1K, Terekhova, AIDA). Compare against the **Phase 2 panel (LASSO-pretrained + LASSO-retrained-3cohort + scAgeClock + Pasta-REG) on the same cells**. For each cell, compute relative MAE reduction of each FM vs. the minimum of the four baseline MAEs (the "beat-best-baseline" min-of-4 gate); classify the cell win/match/loss; aggregate to a 3-cell tri-headline outcome. Compute the *empirical* per-donor absolute-error correlation ρ between each baseline and each fine-tuned model (the Phase 1 floor assumed ρ=0.8; Phase 2 measured baseline-pair ρ ≈ 0.06–0.35; Phase 3 measures the actual baseline-vs-FM value). Run paired Wilcoxon signed-rank test per cell; append a `post_phase3_override` block to `data/loco_folds.json` with the updated primary/exploratory flags implied by the empirical baseline-vs-FM ρ (do NOT overwrite the frozen Phase-1 flags). Write results to `results/phase3/cd4t_loco_table.csv` with `leakage_status`, `chemistry_match_to_baseline_training`, `vs_best_baseline_pct`, and `headline_outcome ∈ {win, match, loss}` columns per (model, cell). Done when the table is committed, the `post_phase3_override` block is populated, and the methods section reports the bracketed ρ disclosure (Phase-1 planning ρ=0.8, Phase-2 baseline-pair ρ=0.06–0.35, Phase-3 measured ρ).

- [ ] Task: Produce CD4+T chemistry+ancestry-robustness 3×3 subfigure. Assemble the 3-comparator × 3-chemistry-context panel (LASSO-pretrained vs Pasta-REG vs Geneformer LoRA × {3' OneK1K, 5' Terekhova European, 5' AIDA Asian}), reusing the already-scored LASSO and Pasta numbers in `results/baselines/loco_baseline_table.csv` plus the new FM-on-AIDA scores from the previous task. **Pasta-REG is the chemistry-invariant + ancestry-invariant comparator** (Phase-2: Pasta-B R=0.28 on Terekhova vs LASSO R=0.08; Pasta CD4T R=0.66 on AIDA vs LASSO R=0.65). The 3×3 figure simultaneously tests two robustness dimensions in one panel: "does single-cell FM pretraining add chemistry-invariance + ancestry-invariance *on top of* what rank-normalized bulk modelling already provides?" Write `results/phase3/cd4t_robustness_3x3.csv` and render `fig_cd4t_robustness_3x3.pdf`. Done when the PDF is committed and cited in the preprint headline figure list.

- [ ] Task: Prepare and submit bioRxiv preprint. Draft manuscript sections covering: (1) the LOCO split design and the full 20-row leakage audit (5 models × 4 cohorts; highlight the HCA/ArrayExpress mirror lesson — scFoundation × Stephenson was missed by direct-accession search and recovered via HCA Project ID cross-reference); (2) the **CD4+T tri-headline result** with per-cell win/match/loss classification — OneK1K (3-cohort vs 5-cohort training-data control), Terekhova (chemistry-shift headline), AIDA (cross-ancestry headline) — vs. the **min-of-4 baseline panel** (LASSO-pretrained, LASSO-retrained-3cohort, scAgeClock, Pasta-REG); (3) the 3×3 chemistry+ancestry robustness subfigure; (4) the bracketed detectability-ρ disclosure (Phase-1 ρ=0.8 → Phase-2 ρ=0.06–0.35 → Phase-3 measured); (5) GPU compute envelope and runtime calibration. Include the frozen split files (including the `post_phase3_override` block), the AIDA `ancestry_shift_mae` donor half used here, and checkpoint hashes as supplementary data. Submit preprint to bioRxiv. Done when the bioRxiv DOI is confirmed (within 10 weeks of project start).

## References

```references
[
  {
    "title": "Single-cell immune aging clocks reveal inter-individual heterogeneity during infection and vaccination",
    "url": "https://www.nature.com/articles/s43587-025-00819-z",
    "authors": "Li et al.",
    "year": 2025,
    "venue": "Nature Aging"
  },
  {
    "title": "scGPT: toward building a foundation model for single-cell multi-omics using generative AI",
    "url": "https://www.nature.com/articles/s41592-024-02201-0",
    "authors": "Cui et al.",
    "year": 2024,
    "venue": "Nature Methods"
  },
  {
    "title": "Transfer learning enables predictions in network biology (Geneformer)",
    "url": "https://www.nature.com/articles/s41586-023-06139-9",
    "authors": "Theodoris et al.",
    "year": 2023,
    "venue": "Nature"
  },
  {
    "title": "Large-scale foundation model on single-cell transcriptomics (scFoundation)",
    "url": "https://www.nature.com/articles/s41592-024-02305-7",
    "authors": "Hao et al.",
    "year": 2024,
    "venue": "Nature Methods"
  },
  {
    "title": "Zero-shot evaluation reveals limitations of single-cell foundation models",
    "url": "https://link.springer.com/article/10.1186/s13059-025-03574-x",
    "authors": "Kedzierska et al.",
    "year": 2025,
    "venue": "Genome Biology"
  }
]
```
