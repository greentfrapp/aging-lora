# Phase 4 — Full LOCO matrix, few-shot curve, and zero-shot transfer

## Goal

Phase 3 produced the CD4+ T pilot with Geneformer + scFoundation + scGPT; this phase extends the fine-tuning sweep to **all four foundation models** (scGPT, Geneformer, scFoundation, UCE — UCE enters in Phase 4), **all five PBMC cell types**, and **all surviving primary LOCO folds**. It then runs the two regime-specific experiments that form the paper's secondary claims: the few-shot downsampling curve targeting the low-data B- and NK-cell regime (success criterion b), and the zero-shot cell-type transfer experiment with joint leave-one-donor × leave-one-cell-type holdout design (success criterion c). A single full fine-tune ablation (one model × one cell type × one fold) completes the LoRA-as-regularizer control. All results are aggregated via random-effects meta-analysis over per-fold effect sizes for the headline claim.

### Leakage-stratified reporting policy (recorded 2026-04-24)

Every result row in `results/phase4/full_loco_matrix.csv` carries a `leakage_status ∈ {clean, overlapping}` column sourced from `data/leakage_audit.csv`. Aggregation is reported in **two modes**, both always shown side by side:

- **Strict-clean pooled estimate** — random-effects meta-analysis restricted to rows with `leakage_status = clean`. This is the paper's **headline effect size**. For criterion (a), the strict-clean mode determines whether the ≥10% relative MAE reduction claim holds.
- **Inclusive pooled estimate** — meta-analysis over all rows, with overlapping rows downweighted (1/2 weight) AND reported with 95% CI bounds from the strict-clean subset so readers can see the shift the overlap introduces.

When the two modes disagree in direction or cross the 10% threshold, the paper reports the disagreement explicitly rather than picking one. The leakage audit (`data/leakage_audit.csv` + `methods/leakage_audit_notes.md`) is cited as the source for every `leakage_status` value.

### Per-cell-type primary vs. exploratory assignments (from the ρ=0.8 detectability floor)

Using `data/detectability_floor.json` (Phase 1 output) with the empirical ρ measured in Phase 3 substituted:

| Cell type | loco_onek1k (981 donors) | loco_terekhova (166 donors) | loco_stephenson (29 donors) |
|---|:---:|:---:|:---:|
| CD4+ T | primary | primary | exploratory |
| CD8+ T | primary | **exploratory** (under-powered at ρ=0.8; may flip to primary if Phase 3 measures ρ ≥ 0.9) | exploratory |
| Monocyte | primary | **exploratory** (under-powered) | exploratory |
| NK | primary | primary | exploratory |
| B | primary | primary | exploratory |

`loco_onek1k` ×  {scGPT, UCE} cells are `overlapping` — included in the inclusive pooled estimate only. `loco_terekhova` cells are all `clean`. The headline meta-analysis strict-clean mode therefore has: Geneformer+scFoundation on both primary folds for 3/5 cell types, plus all 4 FMs on Terekhova for CD4+ T / NK / B — 10 strict-clean primary cells, not the 20 the original plan assumed.

## Success criteria

- LoRA fine-tune result matrix complete for all four models × five cell types × surviving primary LOCO folds (≥3 seeds per cell); per-fold per-cell-type m.a.e. recorded in `results/phase4/full_loco_matrix.csv` **with a `leakage_status` column** on every row; forest plot produced showing per-fold Fisher-z Pearson R effect sizes, with clean and overlapping rows rendered in distinct glyphs.
- Few-shot downsampling curve produced for at least two cell types (B cells and CD4+ T cells, where CD4+ T donor count is downsampled to match B-cell training set size); curve shows foundation-model vs. LASSO m.a.e. as a function of number of labelled training donors, identifying the crossover threshold (if any) for criterion (b).
- Zero-shot cell-type transfer: at least three surviving joint LOCO folds (train cell type A on donor set A, evaluate cell type B on disjoint donor set B, both sets ≥80 donors); Pearson R and m.a.e. reported for each surviving fold; results compared against sc-ImmuAging's inability to transfer by construction.
- Full fine-tune ablation complete on one foundation model × one cell type × one LOCO fold, reporting whether full fine-tuning materially changes ranking relative to LoRA; decision documented on whether to promote full fine-tune to primary reporting.

## Tasks

- [ ] Task: Extend LoRA fine-tuning to all models × cell types × LOCO folds. Using the training scripts from Phase 3, run LoRA rank-16 fine-tunes for scFoundation and UCE (in addition to the scGPT and Geneformer runs already complete) across all five PBMC cell types and all surviving primary LOCO folds, three seeds each. Schedule runs according to the Phase 3–calibrated GPU-hours estimate. Restrict (model, cohort) combinations flagged as overlapping in `data/leakage_audit.csv` to the leakage-restricted analysis pathway. Write all per-fold results to `results/phase4/full_loco_matrix.csv`. Done when all surviving primary-fold cells are populated and the forest plot is generated.

- [ ] Task: Run few-shot downsampling curve for B-cell and CD4+ T cell types. Downsample CD4+ T training donors to match the B-cell training set size (the weakest sc-ImmuAging clock), and run additional downsampling to 75%, 50%, 25%, and 10% of the full training set. For each downsampled level, fit LASSO and the best-performing LoRA foundation model from the full LOCO matrix; record per-fold m.a.e. Run the equivalent downsampling natively on B cells. Plot m.a.e. vs. number of labelled donors for both model classes, identifying any crossover point. Done when downsampling curves are saved to `results/phase4/fewshot_curves/` and the crossover threshold (or its absence) is documented.

- [ ] Task: Run zero-shot cell-type transfer with joint holdout. For each surviving (source cell type, target cell type, cohort) triple with ≥80 donors on both sides of the donor split: train a ridge probe on frozen foundation-model embeddings of the source cell type on donor set A; evaluate age prediction on the target cell type on disjoint donor set B. Compute Pearson R and m.a.e. per fold. Apply the permutation null (N = 1,000 donor-label shuffles within each group) to establish significance. Record results in `results/phase4/zero_shot_transfer.csv`; report aggregated effect sizes via random-effects meta-analysis across surviving folds. Done when at least three primary folds are evaluated and the result table is committed.

- [ ] Task: Full fine-tune ablation and meta-analysis. Run full parameter fine-tuning (not LoRA) on one foundation model (the best-performing from the full LOCO matrix), one cell type (CD4+ T), one LOCO fold, three seeds. Compare full fine-tune vs. LoRA m.a.e. and document the finding in `results/phase4/full_finetune_ablation.md`; if full fine-tune changes the ranking across baselines by more than 5% relative m.a.e., escalate to primary reporting for that model. Separately, aggregate all primary LOCO folds via random-effects meta-analysis (Fisher-z R for criterion c; paired log-ratio of m.a.e. for criterion a) in **both** the strict-clean and the inclusive modes (see Leakage-stratified reporting policy above); produce the headline forest plot in `results/phase4/forest_plot.pdf` with clean and overlapping glyphs distinguished. Done when ablation is complete, both meta-analysis modes are run, and the forest plot is committed.

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
    "title": "scAgeClock: a single-cell transcriptome-based human aging clock model using gated multi-head attention neural networks",
    "url": "https://www.nature.com/articles/s41514-026-00379-5",
    "authors": "Xie",
    "year": 2026,
    "venue": "npj Aging"
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
    "title": "Universal Cell Embeddings: A Foundation Model for Cell Biology (UCE)",
    "url": "https://www.biorxiv.org/content/10.1101/2023.11.28.568918v2",
    "authors": "Rosen, Roohani, Agrawal, Samotorcan, Tabula Sapiens Consortium, Quake, Leskovec",
    "year": 2023,
    "venue": "bioRxiv"
  },
  {
    "title": "Zero-shot evaluation reveals limitations of single-cell foundation models",
    "url": "https://link.springer.com/article/10.1186/s13059-025-03574-x",
    "authors": "Kedzierska et al.",
    "year": 2025,
    "venue": "Genome Biology"
  },
  {
    "title": "Discovering Candidate Anti-Aging Perturbations Using a Foundation Model for Gene Expression",
    "url": "https://www.mdpi.com/1422-0067/26/24/11977",
    "authors": "Tadevosyan, Efimov, Kriukov, Khrameeva",
    "year": 2025,
    "venue": "International Journal of Molecular Sciences"
  }
]
```
