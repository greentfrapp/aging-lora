# Phase 1 — Data ingestion & infrastructure

## Goal

Before any model is trained, the project needs a clean, reproducible data stack and a leakage-audited split design. This phase downloads and harmonizes three publicly available PBMC training cohorts (see cohort decisions below), downloads all four foundation-model checkpoints, audits pretraining-corpus overlap with each cohort, and freezes the fold assignments that all subsequent phases will use. Nothing downstream (baselines, fine-tunes, biological readouts) is valid without this ground being solid first.

### Cohort decisions (recorded 2026-04-24)

The original sc-ImmuAging paper (Li et al. 2025) used five cohorts, two of which require EGA controlled-access approval (EGAS00001005529, EGAS00001006990). After auditing confirmed accessions and performing a sanity run on the downloaded data (2026-04-24), the project proceeds with **Case 1 — three public cohorts, Terekhova promoted in place of Barreiro**:

| Cohort | Accession | Source path | Healthy donors | Age range | Chemistry |
|---|---|---|---|---|---|
| OneK1K / Yazar 2022 | CellxGene dde06e0f → dataset 3faad104 (h5ad 4.2 GB) | CellxGene-curated h5ad | 981 | 19–97 yr (complete) | 10x 3' v2 |
| Stephenson 2021 | CellxGene ddfad306 (h5ad 6.6 GB) | CellxGene-curated h5ad | 29 (healthy controls only) | 21–73 yr (11 exact + 18 decade-bin) | 10x 3' |
| Terekhova 2023 *Immunity* | Synapse `syn49637038` | Synapse h5ad (size TBD) | 166 | 25–85 yr (continuous) | 10x 5' v2 |

Total: ~1,176 healthy donors. OneK1K fully covers the 19–97 yr range with every year represented.

**Cohort decision changes (recorded 2026-04-24, superseding the earlier Case 1):**

1. **Barreiro/Randolph 2021 dropped** from primary. GEO `GSE162632` `characteristics_ch1` contains no donor ages, and the release is genotype-multiplexed (each capture pools mock + IAV cells from multiple donors) with no demux files in the archive. Unblocking requires: (i) emailing the Barreiro lab for ages, (ii) obtaining genotype VCFs, (iii) running Demuxlet/Vireo demux — weeks of work for a cohort that was already below the 80-donor LOCO-primary threshold. Tracked in `FUTURE_WORK.md`.
2. **Terekhova 2023 promoted from Case 3 to Case 1.** 166 healthy donors > 80-donor threshold, so it becomes a **second primary LOCO fold** alongside OneK1K. Continuous 25–85 yr age span.
3. **Stephenson sanity-run result**: 29 healthy donors harmonized cleanly (21–73 yr range, 11 exact ages + 18 decade-bin ages flagged via `obs['age_precision']`). Below the 80-donor threshold so it remains exploratory-only.

**Chemistry heterogeneity.** OneK1K + Stephenson are 10x 3'; Terekhova is 10x 5'. The pre-trained sc-ImmuAging LASSO baseline was trained on 3' only — applying it to 5' Terekhova introduces a domain-shift confound. Mitigation: report Terekhova LOCO results twice, (a) naive (no chemistry correction, measures real-world generalization) and (b) chemistry-corrected via Harmony or scran (isolates aging-specific signal). For foundation-model fine-tuning, a 3'/5'-keyed batch-correction step is added to the integrated training pipeline.

**OneK1K source decision (recorded 2026-04-24):** Use the CellxGene-curated h5ad (`3faad104-…`, pre-demultiplexed per donor with canonical cell-type annotations), not the GEO RAW tar (75 multiplexed CSV pools requiring genotype demultiplexing). Rationale: the primary baseline is the pre-trained sc-ImmuAging LASSO applied to holdouts (we do not retrain on these cohorts for the primary comparison), so genotype demux work adds no scientific value. Minor QC/annotation divergence from the original paper is acceptable and must be documented in `methods/cohort_sources.md`. The 26 GB `GSE196830_RAW.tar` and its extracted CSVs remain on disk as a fallback; schedule for deletion once harmonization runs clean.

## Success criteria

- Three training cohorts (CellxGene `3faad104` OneK1K, CellxGene `ddfad306` Stephenson, Synapse `syn49637038` Terekhova) harmonized to per-cell-type AnnData objects with consistent cell-type labels and donor age metadata; donor counts logged to `data/cohort_summary.csv`.
- [x] Leakage-audit table produced 2026-04-24, fully resolved 2026-04-24: `data/leakage_audit.csv` + `methods/leakage_audit_notes.md`. 16 rows (4 models × 4 cohorts); all rows are now `clean` or `overlapping`, no `unknown`. Final table:

  | Model × Cohort | OneK1K | Stephenson | Terekhova | AIDA |
  |---|:---:|:---:|:---:|:---:|
  | scGPT | overlapping | overlapping | **clean** | clean |
  | Geneformer | **clean** | **clean** | **clean** | clean |
  | scFoundation | **clean** | overlapping | **clean** | clean |
  | UCE | overlapping | overlapping | **clean** | clean |

  **Key findings:**
  - `loco_terekhova` is clean for all four models — the project's gold-standard primary LOCO fold and the paper's generalization headline.
  - `loco_onek1k` is clean for **Geneformer and scFoundation** (the true-holdout contrast at the 981-donor fold); scGPT and UCE carry training-set overlap via CellxGene Census.
  - `loco_stephenson` is clean only for Geneformer; scFoundation additionally overlaps here via HCA-Covid19PBMC (ingested by scFoundation under the HCA project ID rather than the ArrayExpress E-MTAB-10026 mirror — direct-accession search originally missed this).
  - AIDA is clean across all models (2025 publication postdates every checkpoint).
- LOCO fold matrix frozen in `data/loco_folds.json`: three leave-one-cohort-out folds, each annotated with donor count. Folds with fewer than 80 held-out donors flagged as exploratory-only and excluded from the primary result matrix. **Primary LOCO folds (≥80 donors held out): OneK1K (981), Terekhova (166).** Exploratory-only: Stephenson (29). For the two primary folds, report naive (chemistry-uncorrected) and chemistry-corrected MAE side-by-side since OneK1K+Stephenson are 10x 3' while Terekhova is 10x 5'. **Pair each result row with the `leakage_status` from `data/leakage_audit.csv`.** Asterisk-footnote rules: `loco_onek1k` → scGPT and UCE carry the overlap caveat (Geneformer + scFoundation are clean); `loco_stephenson` → scGPT, UCE, and scFoundation all carry it (only Geneformer is clean). This makes `loco_terekhova` the headline generalization result of the paper.
- AIDA donors split 50/50 (stratified by age decile and self-reported ancestry subgroup) before any model is trained; split frozen in `data/aida_split.json`; one half designated for ancestry-shift m.a.e., the other for age-axis alignment.
- m.a.e.-detectability floor computed from sc-ImmuAging Extended Data Table 2 baseline values (paired Wilcoxon power calculation, 80% power, α = 0.05) and recorded; if any LOCO fold is underpowered under this criterion, it is promoted to exploratory-only in the fold matrix.

## Tasks

- [ ] Task: Download and QC three training cohorts. Harmonize to per-cell-type AnnData objects with canonical cell-type labels (CD4+ T, CD8+ T, Monocyte, NK, B) and donor age metadata. Log per-cohort donor × cell-type counts to `data/cohort_summary.csv`. Done when all three cohorts are in a single harmonized AnnData.
  - [x] Task 1a: Set up Python project infrastructure — uv project, pyproject.toml with all dependencies (scanpy, anndata, torch, peft, transformers, scikit-learn, GEOparse), directory tree (`src/`, `data/`, `results/baselines/`, `methods/`, `notes/`). (Completed 2026-04-23)
  - [x] Task 1b: Write `src/data/download_cohorts.py` — download + QC script for the three confirmed cohorts; validates donor/cell-type counts and writes `data/cohort_summary.csv`. (Completed 2026-04-23)
  - [x] Task 1c-v1: (Sanity run, 2026-04-24) Draft `src/data/harmonize_cohorts.py`; validated end-to-end on Stephenson alone — 78,850 cells × 5 cell types written correctly, `age_precision` flag working, `data/cohort_summary.csv` produced. OneK1K schema inspection confirmed exact numeric ages, single assay, `normal` disease.
  - [x] Task 1d: Downloaded OneK1K CellxGene h5ad. Source: `https://datasets.cellxgene.cziscience.com/a3f5651f-cd1a-4d26-8165-74964b79b4f2.h5ad`. On disk: `data/cohorts/raw/onek1k_cellxgene/a3f5651f-cd1a-4d26-8165-74964b79b4f2.h5ad` (4.2 GB, 1,248,980 cells × 35,528 genes, 981 donors ages 19–97 yr, all `normal`, 10x 3' v2).
  - [~] Task 1d-terekhova: Download Terekhova 2023 *Immunity* cohort (Synapse `syn49637038`, 166 donors, 25–85 yr, 10x 5' v2). Blocked on human: requires a free Synapse account + DUC acceptance. See `HUMAN_TASKS.md` #2. Destination: `data/cohorts/raw/terekhova/`. Once on disk, inspect schema and tighten `load_terekhova()` in harmonize_cohorts.py.
  - [ ] Task 1c-v2: Extend harmonize_cohorts.py to three cohorts — add `load_terekhova()`, run end-to-end on OneK1K + Stephenson + Terekhova, producing per-cell-type h5ads with all three cohorts concatenated and `cohort_summary.csv` with the three-cohort matrix. Expected scale: ~2M cells total across 1,176 donors. Expect 30–60 min run time on a 32 GB box.
  - [x] Task 1e: (Completed 2026-04-24) Pre-trained sc-ImmuAging LASSO sanity check on OneK1K. Implemented as a pure-Python port via `rdata` (avoids R/Seurat install). Results in `results/baselines/pretrained_sanity_summary.csv` (5 cell types, 981 donors). Per-cell-type Pearson R: CD4T 0.75, CD8T 0.77, MONO 0.71, NK 0.63, B 0.53 — all positive and highly significant (p < 1e-70). MAE values 7.6–10.7 yr, within 2× of paper's Ext Data Table 2. Systematic −1.5 to −4.4 yr negative bias observed; interpretation in `methods/pretrained_lasso_sanity_check.md`.
  - [ ] Task 1f: Chemistry batch-correction decision. Before freezing LOCO folds, evaluate whether the pre-trained LASSO can be applied directly to 10x 5' Terekhova or whether chemistry correction (Harmony or scran keyed on `obs['assay']`) is required to get interpretable MAE. Report both naive and corrected MAE in `results/baselines/terekhova_chemistry_shift.csv`.

- [ ] Task: Download foundation-model checkpoints and verify integrity. Retrieve scGPT whole-human checkpoint (~1.3 GB; https://github.com/bowang-lab/scGPT), Geneformer (~500 MB; https://huggingface.co/ctheodoris/Geneformer), scFoundation (~500–1 GB; https://github.com/biomap-research/scFoundation), and UCE (~2 GB; https://github.com/snap-stanford/UCE). Record SHA-256 hash for each checkpoint file in `data/checkpoint_hashes.txt`. Confirm each loads without error under the project's pinned Python/PyTorch environment. Done when all four models produce embeddings on a 50-cell PBMC toy input.

- [x] Task (2026-04-24): Run pretraining-corpus leakage audit. 16 rows (4 models × 4 cohorts) written to `data/leakage_audit.csv` with full methodology in `methods/leakage_audit_notes.md`. Overlap discovered in both the scGPT/UCE CellxGene-Census path (expected, `overlapping` on OneK1K + Stephenson) and a non-obvious scFoundation × Stephenson overlap via the HCA-Covid19PBMC project ID (direct E-MTAB-10026 accession search missed it; caught by cross-referencing HCA Project IDs in Hao 2024 Nat Methods Supp Table 5 row 81). **Lesson:** leakage audits must check *all* deposit mirrors (HCA / ArrayExpress / CellxGene / GEO), not just one canonical accession. No `unknown` rows remain.

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
