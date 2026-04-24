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

For CD4+ T itself, the Phase 3 pilot should nonetheless produce a four-panel comparison plot (Geneformer × {3', 5'} × {pre-trained LASSO, fine-tuned FM}) so the manuscript's preprint headline shows the chemistry-generalization behaviour explicitly. This is cheap — the data are already scored for LASSO; only the FM-on-5' number is new.

## Success criteria

- Geneformer, scFoundation, and scGPT LoRA (rank-16, attention + MLP layers, peft library) fine-tuned on all CD4+ T LOCO training folds; per-fold LOCO m.a.e. and Pearson R recorded for all three models and compared against Phase 2 LASSO/RF/PointNet numbers on the same folds. Result rows carry `leakage_status` columns sourced from `data/leakage_audit.csv` and a `chemistry_match_to_baseline_training` column (10x 3' baseline-training rows = `match`; 10x 5' Terekhova rows = `shifted`).
- Chemistry-robustness subfigure produced for CD4+ T: 2 × 2 panel showing Pearson R for {pre-trained LASSO, Geneformer LoRA} × {3'-chemistry fold (OneK1K), 5'-chemistry fold (Terekhova)}. Committed as `results/phase3/cd4t_chemistry_robustness.csv` + `fig_cd4t_chemistry_robustness.pdf`. This subfigure is a preprint headline element, not a supplement.
- Measured GPU-hours per LoRA run (mean and SD across three random seeds) recorded in `compute/runtime_log.csv`; Phase 4 full-matrix time estimate updated from this measurement before Phase 4 begins.
- m.a.e.-detectability floor reviewed using CD4+ T pilot data: compute paired Wilcoxon power from observed per-donor absolute residuals (empirical ρ measured from CD4+ T pilot — replaces the ρ=0.8 planning assumption baked into `data/detectability_floor.json` at Phase 1). Updates are written to `data/loco_folds.json` as a **`post_phase3_override` field** keyed on (cohort, cell_type), preserving the Phase-1 planning assumptions as the baseline. The original primary/exploratory flags stay as frozen; the override field supersedes them for Phase 4 reporting. This avoids "refreezing frozen splits" while still letting empirical ρ refine underpowered-fold calls.
- bioRxiv preprint posted within 10 weeks of project start; manuscript's headline figure is **the CD4+ T loco_terekhova result** for Geneformer + scFoundation + scGPT vs. sc-ImmuAging baselines, with the full leakage-audit table, the frozen split design, and the chemistry-robustness subfigure. At this point the paper is "minimal viable" regardless of whether criterion (a) is met.

## Tasks

- [ ] Task: Implement LoRA fine-tuning wrapper for Geneformer, scFoundation, and scGPT. Using peft (LoRA rank-16 applied to attention + MLP projection layers), build training scripts that accept a cell-type label, a LOCO fold index, and a random seed; output a model checkpoint and per-donor age predictions on the held-out cohort. Run three seeds per (model, fold) combination for CD4+ T cells on the **two primary LOCO folds** (`loco_onek1k`, `loco_terekhova`); Stephenson LOCO is exploratory-only and can be deferred. Record wall-clock time and peak GPU memory per run. Done when reproducible per-fold predictions exist for CD4+ T × {Geneformer, scFoundation, scGPT} × {loco_onek1k, loco_terekhova} × 3 seeds.

- [ ] Task: Evaluate CD4+ T pilot results and update detectability floor. Compute per-fold LOCO m.a.e. (median absolute error) and Pearson R for each of the three FMs; compare against Phase 2 LASSO baseline on the same folds. Compute the *empirical* per-donor absolute-error correlation ρ between baseline and each fine-tuned model (the Phase 1 floor assumed ρ=0.8 — Phase 3 measures it). Run paired Wilcoxon signed-rank test per fold; append a `post_phase3_override` block to `data/loco_folds.json` with the updated primary/exploratory flags implied by the empirical ρ (do NOT overwrite the frozen Phase-1 flags). Write results to `results/phase3/cd4t_loco_table.csv` with `leakage_status` and `chemistry_match_to_baseline_training` columns per (model, fold). Done when the results table is committed and the `post_phase3_override` block is populated.

- [ ] Task: Produce CD4+ T chemistry-robustness subfigure. Assemble the 2 × 2 panel (pre-trained LASSO vs Geneformer LoRA × 3' OneK1K vs 5' Terekhova), reusing the already-scored LASSO numbers in `results/baselines/pretrained_sanity_summary.csv` (OneK1K CD4T) and `results/baselines/terekhova_chemistry_shift_naive.csv` (Terekhova CD4T). Add the two FM points from the LoRA fine-tune output. Write `results/phase3/cd4t_chemistry_robustness.csv` and render `fig_cd4t_chemistry_robustness.pdf`. Done when the PDF is committed and cited in the preprint headline figure list.

- [ ] Task: Prepare and submit bioRxiv preprint. Draft manuscript sections covering: (1) the LOCO split design and the full 16-row leakage audit (highlight the HCA/ArrayExpress mirror lesson — scFoundation × Stephenson was missed by direct-accession search and recovered via HCA Project ID cross-reference); (2) CD4+ T loco_terekhova result (Geneformer as headline, scFoundation + scGPT as comparators) vs. LASSO/RF/PointNet baselines; (3) CD4+ T loco_onek1k as the secondary table, with the clean rows (Geneformer, scFoundation) separated from the leakage-flagged rows (scGPT); (4) the CD4+ T chemistry-robustness subfigure as a supporting-primary result; (5) GPU compute envelope and runtime calibration. Include the frozen split files (including the `post_phase3_override` block) and checkpoint hashes as supplementary data. Submit preprint to bioRxiv. Done when the bioRxiv DOI is confirmed (within 10 weeks of project start).

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
