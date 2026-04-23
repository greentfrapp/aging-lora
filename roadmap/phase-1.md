# Phase 1 — Data ingestion & infrastructure

## Goal

Before any model is trained, the project needs a clean, reproducible data stack and a leakage-audited split design. This phase downloads and harmonizes the five sc-ImmuAging training cohorts using the authors' published integration pipeline, downloads all four foundation-model checkpoints, audits pretraining-corpus overlap with each cohort, and freezes the fold assignments that all subsequent phases will use. Nothing downstream (baselines, fine-tunes, biological readouts) is valid without this ground being solid first.

## Success criteria

- All five sc-ImmuAging cohorts (GSE158055, GSE214534, GSE155673, Stephenson COVID-19 Cell Portal, OneK1K GSE196830) processed to per-cell-type AnnData objects using the sc-ImmuAging GitHub pipeline; cell-type label distribution and donor count per cohort logged and matches the paper's reported 864 training / 217 validation donor split.
- Leakage-audit table produced: for each (foundation model, cohort) pair, classification as *clean*, *overlapping*, or *unknown* based on cross-referencing published pretraining dataset lists against GEO/EGA/CELLxGENE accession IDs; table published in `data/leakage_audit.csv` alongside the code that generated it.
- LOCO fold matrix frozen in `data/loco_folds.json`: five leave-one-cohort-out folds plus leave-one-chemistry-out (10x 3' v2 vs v3.1), each annotated with donor count; folds with fewer than 80 held-out donors flagged as exploratory-only and excluded from the primary result matrix.
- AIDA donors split 50/50 (stratified by age decile and self-reported ancestry subgroup) before any model is trained; split frozen in `data/aida_split.json`; one half designated for ancestry-shift m.a.e., the other for age-axis alignment.
- m.a.e.-detectability floor computed from sc-ImmuAging Extended Data Table 2 baseline values (paired Wilcoxon power calculation, 80% power, α = 0.05) and recorded; if any LOCO fold is underpowered under this criterion, it is promoted to exploratory-only in the fold matrix.

## Tasks

- [ ] Task: Download and QC all five sc-ImmuAging cohorts. Clone `https://github.com/CiiM-Bioinformatics-group/scImmuAging` and run its integration pipeline. Verify donor counts and cell-type label distributions against the paper's Methods (1,081 donors total; five PBMC cell types: CD4+ T, CD8+ T, monocytes, NK, B). Log per-cohort donor × cell-type counts to `data/cohort_summary.csv`. Done when all five cohorts are in a single harmonized AnnData with consistent cell-type labels and donor age metadata.
  - [x] Task 1a: Set up Python project infrastructure — uv project, pyproject.toml with all dependencies (scanpy, anndata, torch, peft, transformers, scikit-learn, GEOparse), directory tree (`src/`, `data/`, `results/baselines/`, `methods/`, `notes/`). (Completed 2026-04-23)
  - [ ] Task 1b: Write `src/data/download_cohorts.py` — GEO download + QC script that fetches all five cohort accessions, validates donor/cell-type counts, and writes `data/cohort_summary.csv`.
  - [~] Task 1c: Execute download and run scImmuAging integration pipeline. Blocked on human: requires downloading ~50 GB of raw GEO data and running a multi-hour Seurat/scanpy integration.

- [ ] Task: Download foundation-model checkpoints and verify integrity. Retrieve scGPT whole-human checkpoint (~1.3 GB; https://github.com/bowang-lab/scGPT), Geneformer (~500 MB; https://huggingface.co/ctheodoris/Geneformer), scFoundation (~500–1 GB; https://github.com/biomap-research/scFoundation), and UCE (~2 GB; https://github.com/snap-stanford/UCE). Record SHA-256 hash for each checkpoint file in `data/checkpoint_hashes.txt`. Confirm each loads without error under the project's pinned Python/PyTorch environment. Done when all four models produce embeddings on a 50-cell PBMC toy input.

- [ ] Task: Run pretraining-corpus leakage audit. Extract each model's published pretraining dataset manifest (at GEO/EGA/CELLxGENE collection-ID granularity). Cross-reference against all five sc-ImmuAging training cohorts and against AIDA (CELLxGENE ced320a1-29f3-47c1-a735-513c7084d508) and OneK1K (GSE196830). Write the result to `data/leakage_audit.csv` with columns: model, cohort_id, cohort_name, status ∈ {clean, overlapping, unknown}, evidence_url. Identify any (model, cohort) pairs that are overlapping or unknown; flag the corresponding LOCO folds for leakage-restricted reporting. Done when the CSV is complete and committed.

- [ ] Task: Freeze LOCO fold assignments and AIDA split. Compute and write `data/loco_folds.json` enumerating all five LOCO folds, the leave-one-chemistry-out fold, and the joint leave-one-cell-type × LOCO fold candidates (targeting ~30–60 primary folds after the 80-donor pruning). Extract per-cell-type m.a.e. baseline values from sc-ImmuAging Extended Data Table 2; compute the paired Wilcoxon detectability floor for a 10% relative m.a.e. reduction at 80% power (α = 0.05) using baseline m.a.e. and residual SD estimates; mark underpowered folds as exploratory-only. Write `data/aida_split.json` with the 50/50 stratified AIDA donor assignments. All split files are treated as immutable after this task.

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
    "title": "Single-cell eQTL mapping identifies cell type-specific genetic control of autoimmune disease (OneK1K)",
    "url": "https://www.science.org/doi/10.1126/science.abf3041",
    "authors": "Yazar et al.",
    "year": 2022,
    "venue": "Science"
  },
  {
    "title": "Asian diversity in human immune cells (AIDA)",
    "url": "https://www.cell.com/cell/fulltext/S0092-8674(25)00202-8",
    "authors": "Kock et al.",
    "year": 2025,
    "venue": "Cell"
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
    "title": "PBMCpedia: a harmonized PBMC scRNA-seq database with unified mapping and enhanced celltype annotation",
    "url": "https://academic.oup.com/nar/article/54/D1/D1216/8340979",
    "authors": "PBMCpedia Consortium",
    "year": 2026,
    "venue": "Nucleic Acids Research"
  },
  {
    "title": "AgeAnno: a knowledgebase of single-cell annotation of aging in human",
    "url": "https://academic.oup.com/nar/article/51/D1/D805/6749541",
    "authors": "Huang, Gong, Guan, Zhang, Hu, Zhao, Huang, Zhang, Kim, Zhou",
    "year": 2023,
    "venue": "Nucleic Acids Research"
  }
]
```
