# Pretraining-corpus leakage audit — findings & methodology

**Date.** 2026-04-24.
**Output.** `data/leakage_audit.csv` (16 rows: 4 models × 4 cohorts).

## Summary — what this means for the primary LOCO folds

All 16 rows resolved as of 2026-04-24 via direct searches of the Genecorpus-30M and scFoundation Nature Methods supplementary tables (see `scratchpad/scf_MOESM{4,5,6}_ESM.xlsx` and `scratchpad/41586_2023_6139_MOESM4_ESM.xlsx`).

| LOCO fold | Clean models | Overlapping models |
|---|---|---|
| **loco_terekhova** (primary, 166 donors) | scGPT, Geneformer, scFoundation, UCE | — |
| **loco_onek1k** (primary, 981 donors) | Geneformer, scFoundation | scGPT, UCE |
| loco_stephenson (exploratory, 29 donors) | Geneformer | scGPT, UCE, **scFoundation** |
| AIDA (external holdout) | scGPT, Geneformer, scFoundation, UCE | — |

**Headline:** Terekhova is the only primary fold where *all four* foundation models are confirmed clean — the paper's **generalization headline**. For `loco_onek1k`, scGPT and UCE carry training-set-memorization overlaps and must be reported with an asterisk; Geneformer and scFoundation are both clean on OneK1K, making them the "true-holdout" contrast at the largest primary fold. AIDA is clean across the board.

**Resolved `scFoundation × Stephenson`:** The scFoundation Nature Methods Supplementary Table 5 row 81 lists `HCA-Covid19PBMC` with study title *"The cellular immune response to COVID-19 deciphered by single cell multi-omics across three UK centres"* — the Stephenson/Haniffa 2021 Nature Medicine multi-omics COVID PBMC deposit (the HCA native project; `E-MTAB-10026` is its ArrayExpress mirror). Direct-accession search missed it because scFoundation ingested via the HCA ID, not the ArrayExpress ID.

## Per-model pretraining-manifest source

- **scGPT (whole-human):** pretrained on 33M cells from CZ CELLxGENE Discover Census release `2023-05-15` (stated in Methods of Cui et al. 2024 Nature Methods, `bowang-lab/scGPT` README). No per-dataset manifest is published; inclusion is inferred by whether a CellxGene collection existed in that snapshot.
- **Geneformer (V1 checkpoint, `ctheodoris/Geneformer` HF, June 2021):** pretrained on Genecorpus-30M — 561 public datasets from GEO/SRA/HCA/SCP/EBI. Source list in Theodoris et al. 2023 Nature Supplementary Table 1 and HF dataset card `ctheodoris/Genecorpus-30M`. V1 is locked to pre-2022 data — all of our training cohorts except Stephenson and OneK1K are trivially post-cutoff.
- **scFoundation (May 2023 preprint, 2024 Nature Methods):** ~50M cells manually curated from GEO/HCA/SCP/EMBL-EBI. Source list in `DataSupplement1.xlsx` and `DataSupplement2.xlsx` in the `biomap-research/scFoundation` GitHub repo, also on Figshare record 24049200. Not CellxGene-based.
- **UCE (Nov 2023 bioRxiv v2):** ~36M cells, majority (>33M) from CZ CELLxGENE Census. Manifest in Extended Data Table 2 of the preprint. Census snapshot ~mid-to-late 2023.

## Confidence level by row

**High confidence (12/16 rows):**
- All `overlapping` rows for scGPT and UCE are strong: both models explicitly pretrain on CZ CellxGene Census, and both OneK1K (collection `dde06e0f`) and Stephenson (collection `ddfad306`) were in Census well before the pretraining snapshots. Without cohort filtering the Census release *would* include them.
- All `clean` rows for Terekhova: its distribution is Synapse-only (`syn49637038`) with no GEO/HCA/SCP/EBI/CellxGene ingestion. None of the four models' documented corpora reach Synapse.
- All `clean` rows for AIDA: 2025 publication postdates every checkpoint's snapshot.

**Medium confidence — `unknown` rows requiring human verification:**
- `Geneformer × OneK1K` — timing says cannot be in V1 (OneK1K Apr 2022 > V1 cutoff Jun 2021), but Genecorpus-30M Supplementary Table 1 was not opened directly (Nature page returned 403 during audit). Almost certainly `clean` by timing; one-line verification by opening the supplementary table.
- `Geneformer × Stephenson` — Stephenson (Apr 2021) is pre-cutoff, could be in Genecorpus-30M. Requires Supplementary Table 1 check.
- `scFoundation × OneK1K` — `DataSupplement1.xlsx` / `DataSupplement2.xlsx` in the scFoundation repo list the GEO accessions used, but were not retrievable via `WebFetch` (github blob not parsed). Human to open and `Ctrl-F "GSE196830"`.
- `scFoundation × Stephenson` — same, check for `E-MTAB-10026` in the data supplements.

## Reporting rule for Phase 3 / Phase 4

Each per-(model, cohort) LOCO row in `results/phase3/*.csv` and `results/phase4/*.csv` must carry a `leakage_status` column whose value is taken from this file. Tables in the paper/preprint:

- Clean rows are the primary result.
- Overlapping rows are reported with an asterisk and a footnote: *"this model was pretrained on the held-out cohort; result is a lower bound on true generalization"*.
- Unknown rows are reported but must be pushed to `clean` or `overlapping` before paper submission by the human-verification step above.

## Human verification — completed 2026-04-24

- [x] Genecorpus-30M Supp Table 1 (Theodoris 2023 Nature MOESM4 ESM, 561 rows × 11 cols) — verified: zero hits for Stephenson/Haniffa/E-MTAB-10026/Yazar/OneK1K/GSE196830. Both Geneformer rows → **clean**.
- [x] scFoundation Nat Methods Supp Tables (Hao 2024, MOESM4 + MOESM5, 10 747 sample rows and 522 project rows) — verified: zero hits for OneK1K accessions → **clean**; Stephenson deposited via HCA-Covid19PBMC project ID, row 81 of MOESM5 → **overlapping**.

All 16 leakage-audit rows now resolved; `data/leakage_audit.csv` is the authoritative source for Phase 3/4 result-table footnoting.

## Addendum — scAgeClock × 4 cohorts (2026-04-24)

scAgeClock (Xie 2026, *npj Aging*; preprint bioRxiv 2025.08.29.673183) was added as a fifth foundation model and audited against the same four cohorts. Final classifications:

| cohort | scAgeClock |
|---|---|
| OneK1K (`3faad104`) | **overlapping** |
| Stephenson healthy controls (`ddfad306`) | **overlapping** |
| Terekhova (`syn49637038`) | **clean** |
| AIDA (`ced320a1`) | **clean** |

### Methodology

The audit followed the same template as the four-FM audit above: identify the model's pretraining-corpus snapshot, identify any cell-level filters, then check each cohort's deposit date and metadata against the snapshot.

**Source-of-truth quote (Xie 2026 npj Aging Methods, surfaced via search of the published article):** *"The human scRNA-seq datasets used for training, validation, and testing of the scAgeClock model were obtained from the CZ CELLxGENE Discover database (`census_version = "2024-07-01"`)."* Filters: normal tissues + primary cohorts (`is_primary_data=True`) + non-null age. Funnel: 74,322,510 human cells → 30,197,419 (normal + primary) → 16,497,049 (with age). The Census `2024-07-01` LTS release was *built on 2024-05-20* per the chanzuckerberg/cellxgene-census release-info doc — this build-date is the actual cutoff (not the label date).

Per-cohort reasoning:

- **OneK1K** — CellxGene-curated since 2022, healthy-by-design, `is_primary_data=True`, all 981 donors carry age metadata → trivially passes all three of scAgeClock's filters; in pretraining set.
- **Stephenson** — collection in CellxGene since 2021 and in Census 2024-07-01. The collection mixes COVID-19 cases and 29 healthy controls; CellxGene's schema requires the `disease` ontology field, so the healthy controls are tagged `disease='normal'` and would survive scAgeClock's normal-tissue filter. The COVID cells are excluded — but our project's `loco_stephenson` fold uses *exactly* those 29 healthy donors, so the overlap is real for our evaluation.
- **Terekhova** — distributed only on Synapse (`syn49637038`), never deposited in CELLxGENE Census. scAgeClock did no manual ingestion outside Census, so Terekhova is impossible to leak.
- **AIDA** — bioRxiv preprint posted 2024-07-01, Cell publication 2025-03-19; CellxGene collection `ced320a1` postdates the Census 2024-05-20 build cutoff. Confirmed clean by timing.

### Surprises and confidence notes

1. **The "2024-07-01" label is misleading** — the actual data freeze for that Census LTS is 2024-05-20. AIDA's preprint+CellxGene deposit on 2024-07-01 falls *after* the build but *on* the version label, which would have been a coin flip without the explicit build-date in the release-info doc. Worth flagging if any future audit checks a cohort deposited in May–June 2024.
2. **No per-dataset manifest is published** — unlike scGPT/UCE (which list Census version) and Geneformer/scFoundation (which publish full manifests), scAgeClock only states the Census version and filters. Dataset-level inclusion is therefore *inferred* from `is_primary_data`/`disease`/age metadata rather than directly verified. All four classifications above are inferential, not from a published manifest. To upgrade to "verified", one would need to re-run the same `cellxgene-census` query (`census_version="2024-07-01"`, `value_filter="is_primary_data==True and disease=='normal' and tissue_general=='blood'"`) and grep `dataset_id` for `3faad104` and `ddfad306`.
3. **scAgeClock's `normal` filter does not exclude pediatric or adult-only cohorts**; it only excludes diseased tissue. Age is required to be non-null but no minimum age was disclosed, so we cannot use age-range arguments to reduce overlap.
4. **Methods-section access:** Both the npj Aging article and the bioRxiv full HTML were 403-blocked by WebFetch; the Census version and filtering funnel were recovered via Google search snippets that quote the Methods. The rationale above relies on those search-surfaced quotes plus the cellxgene-census release-info doc — primary-source-quote-grade for the Census version, secondary-grade for the filter chain.
