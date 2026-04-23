# Landscape: Foundation-Model Fine-Tuning for Cell-Type-Specific Immune Aging Clocks

*Survey date: 2026-04-23. All claims independently verified via web search or page fetch unless marked `[UNVERIFIED]`.*

---

## 1. Related Work

### 1.1 Direct task-specific prior art

**sc-ImmuAging** (Li et al., *Nature Aging* 5:607–621, 2025; DOI: 10.1038/s43587-025-00819-z)  
The primary baseline. Trains cell-type-specific PBMC aging clocks — LASSO, random forest, and PointNet — on 1,081 European donors (864 training / 217 internal validation) aged 18–97 from five cohorts (GEO: GSE158055, GSE214534, GSE155673, Stephenson COVID-19 Cell Portal, OneK1K GSE196830). Internal Pearson R spans 0.60–0.91 across five PBMC cell types (CD4+ T, CD8+ T, monocytes, NK, B cells), with B cells at the low end. External validation confirms reliable prediction on 42 independent donors from five separate published studies, though B- and NK-cell external performance is the weakest. Per-cell-type median absolute error (m.a.e.) values are reported in Extended Data Table 2 of the paper (not reproduced in accessible main text). LASSO outperforms RF and PointNet on internal splits. Code released at https://github.com/CiiM-Bioinformatics-group/scImmuAging.

**scAgeClock** (Xie, *npj Aging*, 2026; DOI: 10.1038/s41514-026-00379-5; preprint bioRxiv 2025.08.29.673183)  
The current state-of-the-art dedicated single-cell aging-clock architecture trained from scratch (not fine-tuned from a foundation model). Uses a gated multi-head attention network (five layers: embedding → feature gating → feature projection → multi-head attention → fully connected regression head) trained on >16 million single-cell transcriptome profiles from >40 human tissues and >400 cell types. Reported MAE as low as 2 years for the best-performing cell type (confirmed from abstract); approximately half of tissue-level cell types achieve MAE <10 years [UNVERIFIED: from abstract only; full table inaccessible]. Also introduces the Aging Deviation Index (ADI) for cell-level age-deviation scoring. The clear SOTA comparator for any continuous single-cell age regression claim.

**iAge** (Sayed, Huang et al., *Nature Aging* 1:598–615, 2021; DOI: 10.1038/s43587-021-00082-y)  
The canonical immune-system aging clock, predating single-cell methods. A guided auto-encoder (GAE) trained on plasma cytokine/protein patterns in 1,001 individuals aged 8–96. The strongest single driver is CXCL9, implicated in cardiac aging and arterial stiffness. iAge is the reference for immunosenescence phenotyping but requires matched serum cytokine data not present in PBMC scRNA-seq cohorts; direct metric comparison is therefore infeasible.

**Tabula Muris Senis** (Almanzar, Antony, Baghel et al., *Nature* 583:590–595, 2020; DOI: 10.1038/s41586-020-2496-1)  
The cross-tissue single-cell aging-regression precedent. A 23-tissue mouse atlas with 23 tissue types and age-regressor derivation per tissue. Not directly comparable (mouse, not human; tissue-level, not per-PBMC-cell-type), but establishes the conceptual prior for cell-type-specific aging clocks from scRNA-seq.

**scBayesAge / scMLEAge** (Hu, Pellegrini, bioRxiv 2025.12.04.692166, v1 as scBayesAge / v2 renamed scMLEAge)  
A Bayesian/maximum-likelihood generative framework that infers chronological age per cell from rank-order transcriptome counts, incorporating sequencing depth explicitly. Benchmarked on limb-muscle cell types from Tabula Muris Senis (mouse); achieves higher R² than ElasticNet in most tested cell types. Not a foundation-model approach, not fine-tuned, and benchmarked on mouse data rather than human PBMCs.

**scVI / scANVI** (Lopez, Regier, Cole, Jordan, Yosef; *Nature Methods* 15:1053–1058, 2018; DOI: 10.1038/s41592-018-0229-2; Xu, Lopez, Mehlman, Regier, Jordan, Yosef; *Molecular Systems Biology* 17(1):e9620, 2021; DOI: 10.15252/msb.20209620)  
VAE-based single-cell models producing continuous latent representations from count data. scVI models gene-expression counts directly; scANVI extends it with semi-supervised annotation. Relevant as deep-learning non-foundation baselines — particularly in the few-shot regime where amortised VAE inference may compete with fine-tuned foundation models. Neither paper benchmarks chronological-age regression on PBMCs; their inclusion in the project's comparison table should be evaluated against few-shot criterion (b) — not added to §2 pending project-start decision.

### 1.2 Foundation-model fine-tuning for aging (closest prior art)

**Tadevosyan et al.** ("Discovering Candidate Anti-Aging Perturbations Using a Foundation Model for Gene Expression," *IJMS* 26(24):11977, 2025; https://www.mdpi.com/1422-0067/26/24/11977)  
The closest precedent. Fine-tunes scGPT on the multi-tissue AgeAnno corpus (1.7M cells, 28 tissues) for **binary age-group classification** (mid-age 20–59 vs. old 60–100 years). Then applies the fine-tuned model for in-silico perturbation to nominate candidate anti-aging genes. ROC-AUC [UNVERIFIED: candidate document reports 0.91 for scGPT vs 0.89 logistic-regression baseline; full paper inaccessible during landscape audit — MDPI page returned 403] on AgeAnno test split. Key limitations relative to the proposed work: (a) binary classification, not continuous regression; (b) multi-tissue pooled, not per-PBMC-cell-type; (c) no LOCO or cohort-shift evaluation; (d) no head-to-head against sc-ImmuAging or scAgeClock on matched splits.

### 1.3 Single-cell foundation models

**scGPT** (Cui et al., *Nature Methods* 21:1470–1480, 2024; DOI: 10.1038/s41592-024-02201-0)  
Generative transformer pretrained on >33 million single-cell profiles (human whole-body checkpoint). Demonstrates state-of-the-art performance after fine-tuning on cell-type annotation, perturbation prediction, and multi-batch integration. The "whole-human" checkpoint (≈1.3 GB) is the standard starting point for human PBMC applications. Code and weights: https://github.com/bowang-lab/scGPT (MIT license, confirmed).

**Geneformer** (Theodoris et al., *Nature* 618:517–526, 2023; DOI: 10.1038/s41586-023-06139-9)  
BERT-style masked-learning model pretrained on ~30 million single-cell transcriptomes (Genecorpus). Encodes each cell as a rank-value ordering of gene expression, with 15% of genes masked during pretraining. Demonstrated for cardiac disease gene network perturbation and dosage-sensitive gene identification. Model weights on HuggingFace: https://huggingface.co/ctheodoris/Geneformer (Apache 2.0 license, confirmed).

**scFoundation** (Hao et al., *Nature Methods* 21:1481–1491, 2024; DOI: 10.1038/s41592-024-02305-7)  
100-million-parameter asymmetric transformer pretrained on >50 million human single-cell transcriptomic profiles covering ~20,000 genes. Achieves state-of-the-art on gene expression enhancement and tissue drug-response prediction. The standard fourth entry in current foundation-model benchmark tables for single-cell biology.

**UCE** (Rosen, Roohani, Agrawal, Samotorčan, Tabula Sapiens Consortium, Quake, Leskovec; bioRxiv 2023.11.28.568918; https://www.biorxiv.org/content/10.1101/2023.11.28.568918v2)  
650-million-parameter model producing universal zero-shot cell embeddings without any data annotations. Applied to construct the Integrated Mega-scale Atlas of 36 million cells spanning 1,000+ named cell types, dozens of tissues, and eight species. Outperforms scGPT and Geneformer in zero-shot cell-type clustering benchmarks (Single-Cell Integration Benchmark). Code: https://github.com/snap-stanford/UCE (license: check repo).

**CellFM** (Zeng et al., *Nature Communications* 16, Article 4679, 2025; DOI: 10.1038/s41467-025-59926-5)  
800-million-parameter model (RetNet architecture, MindSpore platform) pretrained on 100 million human cells. Outperforms UCE by 2.1% on average AvgBio score across standard cell-annotation benchmarks. The largest human-specific single-cell foundation model as of the landscape date. **Practical barrier:** MindSpore (Huawei) is not compatible with standard PyTorch pipelines and lacks first-class support on most Western GPU clusters; fine-tuning CellFM requires either a dedicated MindSpore install or a community PyTorch re-implementation. The project should decide up front whether to include CellFM in the fine-tuning sweep.

### 1.4 Counter-evidence

**Kedzierska et al.** ("Zero-shot evaluation reveals limitations of single-cell foundation models," *Genome Biology* 2025; DOI: 10.1186/s13059-025-03574-x; PMC12007350)  
Evaluates scGPT and Geneformer **in zero-shot only** on five datasets (Pancreas, PBMC 12k, PBMC 95k, Immune, Tabula Sapiens). Both foundation models underperform simple HVG, scVI, and Harmony baselines for cell-type clustering and batch integration. HVG achieves the best batch integration scores across all datasets. Geneformer ranks last in batch-mixing scores. scGPT shows advantage only on datasets seen during pretraining. **Important note:** The paper is titled "Zero-shot evaluation…" and available content (PMC full text, Microsoft Research summary) confirms the evaluation scope is exclusively zero-shot. The candidate's claim that Kedzierska also reports mixed-to-negative results for *fine-tuned* settings `[UNVERIFIED: this is not supported by the paper title, PMC text, or Microsoft Research summary; the paper explicitly defers to original authors' fine-tuning results rather than re-evaluating them]`. The counter-evidence from Kedzierska therefore applies specifically to zero-shot regimes; it does not directly speak to the proposed LoRA / full fine-tuning regimes.

### 1.5 Versatile transcriptomic clocks

**Pasta** (Salignon, Tsiokou, Marqués, Rodríguez-Diaz, Ang, Pietrocola, Riedel; bioRxiv 2025.06.04.657785v2; https://www.biorxiv.org/content/10.1101/2025.06.04.657785v2)  
An R package implementing a versatile transcriptomic aging clock (predicting relative age shift) that works across bulk RNA-seq, single-cell RNA-seq, microarrays, and L1000 data from multiple species. Age scores correlate with tumor grade and patient survival. Model coefficients enriched for p53 and DNA-damage-response pathways. Code: https://github.com/jsalignon/pasta. Proposed integration mode in this project: Pasta architecture retrained on sc-ImmuAging pseudobulk splits (architecture comparison), plus Pasta-as-released (out-of-box transfer baseline). **Methodological risk:** Pasta was designed for bulk RNA-seq and microarray data; adapting it to single-cell pseudobulks introduces non-trivial challenges (gene selection, sparsity, cell-count-dependent noise floor), and no precedent for this conversion has been identified in the literature.

---

## 2. Competitors — Existing Codebases

| System | Description | Link |
|---|---|---|
| **sc-ImmuAging** | LASSO/RF/PointNet PBMC aging clocks, the primary retrained baseline | https://github.com/CiiM-Bioinformatics-group/scImmuAging |
| **scAgeClock** | Gated multi-head attention SOTA architecture; paper notes code is publicly available | https://www.nature.com/articles/s41514-026-00379-5 (GitHub not surfaced in search; check paper supplementary). **Code availability unconfirmed; the project must independently verify before treating scAgeClock as a reproducible upper-bound comparator.** |
| **scGPT** | bowang-lab pretrained weights + fine-tuning tutorials | https://github.com/bowang-lab/scGPT |
| **Geneformer** | HuggingFace weights + fine-tuning code (Apache 2.0) | https://huggingface.co/ctheodoris/Geneformer |
| **UCE** | Stanford SNAP lab, universal cell embeddings | https://github.com/snap-stanford/UCE |
| **scFoundation** | 100M-parameter pretrained checkpoint | https://github.com/biomap-research/scFoundation |
| **Pasta** | R package transcriptomic clock | https://github.com/jsalignon/pasta |
| **scBayesAge / scMLEAge** | Bayesian/MLE single-cell age (preprint, no stable release found) | https://www.biorxiv.org/content/10.64898/2025.12.04.692166v2 |
| **scVI / scANVI** | VAE-based deep-learning non-foundation baselines; relevant to few-shot regime (criterion b); inclusion conditional on project-start evaluation | https://github.com/scverse/scvi-tools |

No existing codebase fine-tunes a general-purpose single-cell foundation model as a **continuous per-PBMC-cell-type age regressor** with LOCO evaluation and head-to-head comparison against sc-ImmuAging on matched splits. This gap is the central contribution.

---

## 3. Target Benchmarks and Current SOTA

### 3.1 sc-ImmuAging (primary baseline to beat)

| Metric | Value | Notes |
|---|---|---|
| Internal Pearson R (range across 5 cell types) | **0.60–0.91** | Internal 80/20 train/test split; 217 validation donors |
| Best-performing cell type (internal R) | ~0.91 | CD4+ T cell (confirmed from PMC12003178 Discussion: "R = 0.91 in CD4+ T cells") |
| Weakest cell type (internal R) | ~0.60 | B cells |
| External validation donors | 42 | 5 independent published datasets |
| External validation quality | Confirmed reliable; B- and NK-cell external performance weakest | Qualitative; exact external R not extracted from accessible text |
| Per-cell-type m.a.e. (absolute) | Reported in Extended Data Table 2 of Li et al. 2025 | Must be extracted at project start; not reproduced in main text |

Source: Li et al., *Nature Aging* 2025 (PMC12003178).

### 3.2 scAgeClock (SOTA architecture comparator)

| Metric | Value | Notes |
|---|---|---|
| Best-cell-type MAE | **~2 years** | Best-performing tissue-level cell type (confirmed from abstract: "mean absolute error for the best-performing cell type is remarkably low at 2 years") |
| Fraction of cell types with MAE <10 years | **~50%** | Across all tissue-level cell types evaluated [UNVERIFIED: from abstract only; full table inaccessible] |
| Training corpus | >16M cells, >40 tissues, >400 cell types | Trained from scratch, not fine-tuned |

Source: Xie, *npj Aging* 2026 (search-verified abstract; full table not accessible).

### 3.3 Single-cell foundation model zero-shot baselines (Kedzierska 2025)

Geneformer and scGPT both **underperform** HVG, scVI, and Harmony on cell-type clustering and batch integration across all five evaluated datasets in zero-shot. HVG is the strongest zero-shot baseline. This sets the expectation that zero-shot foundation-model embeddings will likely require fine-tuning to be competitive on the aging regression task.

### 3.4 Tadevosyan 2025 (binary aging classification with scGPT fine-tuning)

ROC-AUC: [UNVERIFIED: 0.91 (scGPT) vs. 0.89 (logistic regression); cited from candidate document; full paper inaccessible during audit]. This is a binary classification result, not continuous regression, so it is not a direct benchmark for m.a.e. comparison.

### 3.5 Success thresholds for this project

Per the study design:
- **(a)** ≥10% relative reduction in per-donor LOCO **median** absolute error vs. sc-ImmuAging retrained LASSO/RF/PointNet on matched splits (primary metric, matching sc-ImmuAging's Extended Data Table 2 baseline). Note: direct comparison against scAgeClock requires the full per-cell-type MAE table from Xie 2026 (§3.2), which is not yet accessible; if it remains inaccessible, criterion (a) is measurable only against sc-ImmuAging.
- **(b)** Few-shot crossover: foundation model fine-tunes win below some donor threshold where pretraining-free baselines win at full data.
- **(c)** Zero-shot cell-type transfer: Pearson R > 0.3 on held-out cell type (e.g., train CD4+ T, test B cells), where sc-ImmuAging cannot transfer by construction.

At least one of (a)/(b)/(c) positive in at least one cell type constitutes a publishable methods result.

---

## 4. Data Sources

### 4.1 Training corpus (sc-ImmuAging five cohorts)

| Dataset | GEO Accession | Content | Access |
|---|---|---|---|
| Cohort 1 | GSE158055 | PBMC scRNA-seq (COVID-19 study; healthy controls used) | Public, GEO (CC0-equivalent) |
| Cohort 2 | GSE214534 | PBMC scRNA-seq | Public, GEO |
| Cohort 3 | GSE155673 | PBMC scRNA-seq | Public, GEO |
| Stephenson COVID-19 Cell Portal | COVID-19 Human Cell Atlas | PBMC scRNA-seq from UK/European cohort | Public, COVID-19 Cell Atlas portal |
| OneK1K | GSE196830 | 1.27M PBMCs, 982 donors, 14 immune cell types | Public, GEO (also at www.onek1k.org) |

All five datasets are publicly available. The sc-ImmuAging GitHub repo provides integration and preprocessing scripts. OneK1K (Yazar et al., *Science* 2022; DOI: 10.1126/science.abf3041) is also used as an external validation cohort within the sc-ImmuAging framework.

### 4.2 External ancestry holdout

**Asian Immune Diversity Atlas (AIDA)** (Kock et al., *Cell* 2025; DOI: 10.1016/j.cell.2025.02.003)  
1,265,624 circulating immune cells (PBMCs) from **619 donors** spanning 7 population groups across 5 Asian countries. Available via CELLxGENE (collection ID: ced320a1-29f3-47c1-a735-513c7084d508) and Human Cell Atlas portal. Open access. The proposed study uses AIDA for ancestry-shift m.a.e. evaluation (50% of donors) and age-axis cross-ancestry alignment (remaining 50%), with the two halves stratified and frozen before any model training.

### 4.3 Cross-tissue pretraining-coverage check

**AgeAnno** (Huang, Gong, Guan, Zhang, Hu, Zhao, Huang, Zhang, Kim, Zhou; *Nucleic Acids Research* 51:D805–D814, 2023; DOI: 10.1093/nar/gkac791)  
1,678,610 cells from 28 healthy human tissues (152 scRNA cell types, 124 scATAC cell types), age range 0–110 years, 5,580 annotated aging-related genes. Freely available at https://relab.xidian.edu.cn/AgeAnno/#/ (note: candidate document cites an older gong_lab.hzau.edu.cn URL, which redirects; the confirmed current URL differs). Used in this project for auditing foundation-model pretraining-corpus coverage, not as a primary training set.

### 4.4 Foundation model weights

| Model | Source | License |
|---|---|---|
| scGPT (whole-human checkpoint, ≈1.3 GB) | https://github.com/bowang-lab/scGPT | **MIT** (confirmed) |
| Geneformer (≈500 MB) | https://huggingface.co/ctheodoris/Geneformer | **Apache 2.0** (confirmed) |
| scFoundation (≈500 MB–1 GB) | https://github.com/biomap-research/scFoundation | **Apache 2.0** code; separate Model License for weights (confirmed) |
| UCE (≈2 GB) | https://github.com/snap-stanford/UCE | **MIT** (confirmed) |

All four model checkpoints are publicly available. GPU memory envelopes: scGPT fine-tuning at batch-size 8, context 1,200 tokens fits in ≈18 GB at fp16; Geneformer fine-tuning fits in ≈12 GB at default 2,048 token context — both within a single 24 GB consumer GPU.

### 4.5 Potential leakage overlap

The pretraining manifests for all four foundation models are published at the dataset-ID level. Cross-referencing against the five sc-ImmuAging training cohorts by GEO/EGA/CELLxGENE collection ID is required before any LOCO fold is treated as leakage-free. Contingency: restrict LOCO to (model, cohort) pairs confirmed clean; publish full audit table.

### 4.6 Harmonized PBMC reference (auxiliary)

**PBMCpedia** (NAR 2026; https://academic.oup.com/nar/article/54/D1/D1216/8340979)  
4.3M cells from 519 samples across 24 publicly available PBMC scRNA-seq studies, uniformly re-processed. Useful for leakage-audit cross-referencing and for verifying cell-type annotation consistency across sc-ImmuAging cohorts, but not a primary training or evaluation source for this study.

---

## 5. References

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
    "title": "scAgeClock: a single-cell transcriptome based human aging clock model using gated multi-head attention neural networks (preprint)",
    "url": "https://www.biorxiv.org/content/10.1101/2025.08.29.673183v1",
    "authors": "Xie",
    "year": 2025,
    "venue": "bioRxiv"
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
    "title": "Universal Cell Embeddings: A Foundation Model for Cell Biology",
    "url": "https://www.biorxiv.org/content/10.1101/2023.11.28.568918v2",
    "authors": "Rosen, Roohani, Agrawal, Samotorcan, Tabula Sapiens Consortium, Quake, Leskovec",
    "year": 2023,
    "venue": "bioRxiv"
  },
  {
    "title": "CellFM: a large-scale foundation model pre-trained on transcriptomics of 100 million human cells",
    "url": "https://www.nature.com/articles/s41467-025-59926-5",
    "authors": "Zeng et al.",
    "year": 2025,
    "venue": "Nature Communications"
  },
  {
    "title": "An inflammatory aging clock (iAge) based on deep learning tracks multimorbidity, immunosenescence, frailty and cardiovascular aging",
    "url": "https://www.nature.com/articles/s43587-021-00082-y",
    "authors": "Sayed, Huang et al.",
    "year": 2021,
    "venue": "Nature Aging"
  },
  {
    "title": "A single-cell transcriptomic atlas characterizes ageing tissues in the mouse (Tabula Muris Senis)",
    "url": "https://www.nature.com/articles/s41586-020-2496-1",
    "authors": "Almanzar, Antony, Baghel et al. (Tabula Muris Consortium)",
    "year": 2020,
    "venue": "Nature"
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
  },
  {
    "title": "Determining the age of single cells using scMLEAge",
    "url": "https://www.biorxiv.org/content/10.64898/2025.12.04.692166v2",
    "authors": "Hu, Pellegrini",
    "year": 2025,
    "venue": "bioRxiv"
  },
  {
    "title": "Pasta, a versatile transcriptomic clock, maps the chemical and genetic determinants of aging and rejuvenation",
    "url": "https://www.biorxiv.org/content/10.1101/2025.06.04.657785v2",
    "authors": "Salignon, Tsiokou, Marques, Rodriguez-Diaz, Ang, Pietrocola, Riedel",
    "year": 2025,
    "venue": "bioRxiv"
  },
  {
    "title": "Asian diversity in human immune cells",
    "url": "https://www.cell.com/cell/fulltext/S0092-8674(25)00202-8",
    "authors": "Kock et al.",
    "year": 2025,
    "venue": "Cell"
  },
  {
    "title": "Single-cell eQTL mapping identifies cell type-specific genetic control of autoimmune disease",
    "url": "https://www.science.org/doi/10.1126/science.abf3041",
    "authors": "Yazar et al.",
    "year": 2022,
    "venue": "Science"
  },
  {
    "title": "AgeAnno: a knowledgebase of single-cell annotation of aging in human",
    "url": "https://academic.oup.com/nar/article/51/D1/D805/6749541",
    "authors": "Huang, Gong, Guan, Zhang, Hu, Zhao, Huang, Zhang, Kim, Zhou",
    "year": 2023,
    "venue": "Nucleic Acids Research"
  },
  {
    "title": "PBMCpedia: a harmonized PBMC scRNA-seq database with unified mapping and enhanced celltype annotation",
    "url": "https://academic.oup.com/nar/article/54/D1/D1216/8340979",
    "authors": "PBMCpedia Consortium",
    "year": 2026,
    "venue": "Nucleic Acids Research"
  },
  {
    "title": "Deep generative modeling for single-cell transcriptomics (scVI)",
    "url": "https://www.nature.com/articles/s41592-018-0229-2",
    "authors": "Lopez, Regier, Cole, Jordan, Yosef",
    "year": 2018,
    "venue": "Nature Methods"
  },
  {
    "title": "Probabilistic harmonization and annotation of single-cell transcriptomics data with deep generative models (scANVI)",
    "url": "https://www.embopress.org/doi/abs/10.15252/msb.20209620",
    "authors": "Xu, Lopez, Mehlman, Regier, Jordan, Yosef",
    "year": 2021,
    "venue": "Molecular Systems Biology"
  }
]
```
