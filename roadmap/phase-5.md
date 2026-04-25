# Phase 5 — Biological readouts & benchmark harness release

## Goal

The primary methods contribution is established by Phases 3–4. This phase delivers the biological results that lift the paper from a horse-race comparison to a substantive biological contribution — specifically the age-axis rotation analysis on foundation-model latent geometry, SHAP attribution to surface cell-type-specific age-associated gene programs, and an ancestry-shift readout using the frozen AIDA holdout. It closes with the release of the v1 benchmark harness: frozen splits, checkpoint hashes, and a `pytest`-runnable evaluation suite, so future immune-aging foundation-model work can plug into the same scaffold. The harness is a one-time v1 release; ongoing maintenance beyond that is not committed.

## Success criteria

- Age-axis cosine-similarity analysis complete for all four foundation models × five PBMC cell types: ridge probe fitted on frozen per-donor mean cell embeddings (training split, nested CV for ridge λ); cosine similarity between cell-type pairs and between sex/ancestry groups computed and tested against permutation null (N = 10,000 donor-age shuffles); results in `results/phase5/age_axis_cosine.csv` with permutation p-values.
- AIDA cross-ancestry alignment readout complete using the age-axis alignment half of the frozen AIDA split (the ancestry-shift m.a.e. half was used for evaluation in Phase 4); cosine similarity of AIDA-inferred age-axis vs. European-cohort age-axis reported per foundation model × cell type.
- SHAP attribution table produced for all four fine-tuned foundation models × five cell types: top-50 age-associated genes per (model, cell type) compared against sc-ImmuAging's published LASSO coefficient rankings and iAge cytokine signature genes (literature comparison only; no matched cytokine data available); written to `results/phase5/shap_attribution_table.csv`.
- In-silico perturbation run for the top-5 SHAP-attributed genes per cell type (for at least one foundation model); direction of embedding shift toward younger/older phenotype reported; results positioned explicitly as a replication and extension of Tadevosyan 2025's perturbation approach, not a novel method.
- Benchmark harness v1 released alongside the revised preprint: `pytest`-runnable suite covering frozen LOCO splits, checkpoint hashes, scanpy/PyTorch/peft version pins, and one end-to-end training + evaluation run per cell type on a subset of donors. **The harness ships the four fine-tuned FMs alongside the three Phase-2 baselines (LASSO, scAgeClock, Pasta-REG)** so external users can reproduce both the FM-vs-baseline contrast and the per-cell minimum-MAE benchmark used in Phase 4. GitHub release tag `v1.0.0` with DOI (Zenodo or equivalent).

## Tasks

- [ ] Task: Age-axis rotation analysis across cell types, sexes, and ancestries. For each (foundation model, cell type) pair, compute the per-donor mean embedding over the training donors, then fit a ridge regression of donor chronological age on this mean embedding (nested CV for λ on the training split). Extract the normalized weight vector as the age-axis. Compute pairwise cosine similarities between all age-axis pairs within the same model (across cell types) and across the two AIDA holdout groups (European-trained vs. AIDA age-axis inferred from the alignment half of the AIDA split). Test each cosine similarity against a permutation null (N = 10,000 donor-age shuffles within group before re-fitting the probe). Run PLS-based age-axis as a sensitivity check. Write all results to `results/phase5/age_axis_cosine.csv`. Done when all model × cell-type pairs are populated and permutation p-values are computed.

- [ ] Task: SHAP attribution and in-silico perturbation. For each fine-tuned foundation model × cell type combination, compute SHAP values on the regression head using a random sample of 500 held-out cells (background = 100-cell reference); extract the top-50 genes by mean |SHAP|. Cross-reference top genes against sc-ImmuAging's published LASSO non-zero coefficient genes (from the paper's supplementary tables) and against iAge's top cytokine drivers (CXCL9 and co-drivers, from literature; no matched data required). For at least one foundation model, run in-silico perturbation on the top-5 SHAP genes by zeroing each gene's expression and measuring the shift in the model's predicted age and in the embedding's projection onto the age-axis; report direction (toward younger or older predicted phenotype). Write attribution table to `results/phase5/shap_attribution_table.csv` and perturbation results to `results/phase5/insilico_perturbation.csv`. Done when all model × cell-type SHAP tables are produced and at least one model's perturbation run is committed.

- [ ] Task: Package and release v1 benchmark harness. Assemble a self-contained evaluation package containing: (a) frozen LOCO and AIDA split files from Phase 1 (`data/loco_folds.json`, `data/aida_split.json`); (b) `data/checkpoint_hashes.txt` with SHA-256 for all four FM checkpoints **plus the scAgeClock and Pasta-REG checkpoints**; (c) a `requirements.txt` pinning scanpy, PyTorch, peft, scikit-learn, and R 4.5+ with `pasta` + `glmnet` to the exact versions used in the project; (d) a `pytest`-runnable test suite that, given downloaded data and checkpoints, verifies one end-to-end LoRA training + LOCO evaluation run produces m.a.e. within 5% of the logged result for one (model, cell type, fold) combination, **plus one round-trip baseline check (LASSO + scAgeClock + Pasta-REG numbers all reproduce within 1% of `results/baselines/loco_baseline_table.csv`)**. Create a GitHub release tag `v1.0.0`, upload to Zenodo for a permanent DOI, and link the DOI in the revised preprint. Done when the Zenodo DOI is confirmed and the `pytest` suite passes on a clean environment.

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
    "title": "An inflammatory aging clock (iAge) based on deep learning tracks multimorbidity, immunosenescence, frailty and cardiovascular aging",
    "url": "https://www.nature.com/articles/s43587-021-00082-y",
    "authors": "Sayed, Huang et al.",
    "year": 2021,
    "venue": "Nature Aging"
  },
  {
    "title": "Asian diversity in human immune cells (AIDA)",
    "url": "https://www.cell.com/cell/fulltext/S0092-8674(25)00202-8",
    "authors": "Kock et al.",
    "year": 2025,
    "venue": "Cell"
  },
  {
    "title": "Discovering Candidate Anti-Aging Perturbations Using a Foundation Model for Gene Expression",
    "url": "https://www.mdpi.com/1422-0067/26/24/11977",
    "authors": "Tadevosyan, Efimov, Kriukov, Khrameeva",
    "year": 2025,
    "venue": "International Journal of Molecular Sciences"
  },
  {
    "title": "scGPT: toward building a foundation model for single-cell multi-omics using generative AI",
    "url": "https://www.nature.com/articles/s41592-024-02201-0",
    "authors": "Cui et al.",
    "year": 2024,
    "venue": "Nature Methods"
  },
  {
    "title": "Universal Cell Embeddings: A Foundation Model for Cell Biology (UCE)",
    "url": "https://www.biorxiv.org/content/10.1101/2023.11.28.568918v2",
    "authors": "Rosen, Roohani, Agrawal, Samotorcan, Tabula Sapiens Consortium, Quake, Leskovec",
    "year": 2023,
    "venue": "bioRxiv"
  }
]
```
