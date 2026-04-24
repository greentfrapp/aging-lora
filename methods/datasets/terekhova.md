# Terekhova 2023 PBMC aging atlas

## Paper and accessions

**Paper.** Terekhova et al., "Single-cell atlas of healthy human blood unveils age-related loss of NKG2C+GZMB-CD57+ adaptive NK cells and amplifies the current understanding of healthy aging," *Immunity* 56 (12), 2836–2854.e9, 2023. DOI: [10.1016/j.immuni.2023.10.013](https://www.cell.com/immunity/fulltext/S1074-7613(23)00472-X). Senior author: Maxim Artyomov (Washington University).

**Accessions**:
- **Synapse** `syn49637038` — project root with six top-level folders (`GEX_HTO_processed`, `BCR_Processed`, `Cytek`, `Demultiplexing`, `RAW`, `TCR_Processed`). Requires a free Synapse account + DUC acceptance.
- **Synapse** `syn56693935` — `raw_counts_h5ad.tar.gz` (9.8 GB). **Corrupt at source** — see "Known issues" below.
- **Synapse** `syn51197006` — `all_pbmcs.tar.gz` (15.9 GB) ← **the source we use**.

No CellxGene mirror. No GEO mirror. Synapse-only.

## Access

Requires a free Synapse account + DUC acceptance on `syn49637038`. See `HUMAN_TASKS.md #2` for the full step-by-step. Summary:

1. Create account at `https://www.synapse.org/register`.
2. Accept the data-use conditions on `syn49637038`.
3. Authenticate via `~/.synapseConfig` (canonical INI `[authentication]\nauthtoken = <PAT>`; script also tolerates a raw-PAT-on-line-1 file).
4. Run `uv run python scripts/download_terekhova.py --synapse-id syn51197006`.

Download uses `synapseclient.syncFromSynapse`, which is checksum-aware (resumes partial transfers). ~9.8 GB tar, ~15.9 GB if extracted in full.

## Two-tar mystery

The Terekhova project exposes two separate raw-counts deposits. They differ in contents:

### `raw_counts_h5ad.tar.gz` (syn56693935) — CORRUPT AT SOURCE

- Tar file MD5 `20fcbe2531a999c51ed03dfce5d29b7b` matches Synapse metadata ✅
- Tar archive header claims `raw_counts_h5ad/pbmc_gex_raw_with_var_obs.h5ad` is 77,016,305,915 bytes
- On-disk after extraction: 23,477,328,896 bytes — the tar content itself is incomplete
- `h5py.File(...)`: `OSError: truncated file: eof = 23477328896, stored_eof = 77016305915`
- Diagnosis: upload pipeline at the Terekhova lab truncated the file before tarring. The tar + MD5 are internally consistent but wrap a corrupted h5ad.

Tracked as `HUMAN_TASKS.md #5` for a bug report to the Artyomov lab.

### `all_pbmcs.tar.gz` (syn51197006) — THE WORKING SOURCE

- 15.9 GB tar → extracts to 28 GB of files
- Six entries; we only extract the two we need:

| Entry | Size | Used? |
|---|---:|:---:|
| `all_pbmcs/all_pbmcs_rna.h5ad` | 27 GB | ✅ |
| `all_pbmcs/all_pbmcs_metadata.csv` | 406 MB | ✅ |
| `all_pbmcs/all_pbmcs_rna_harmony.h5ad` | large | ❌ (Harmony-corrected; we want raw for FMs) |
| `all_pbmcs/all_pbmcs_umap.csv` | small | ❌ |
| `all_pbmcs/all_pbmcs_hto/{barcodes,genes,matrix}_hto.{tsv,mtx}` | ~160 MB | ❌ (HTO demux — not needed; already applied) |

Selective extraction:
```
tar --force-local -xzvf /c/.../all_pbmcs.tar.gz \
  -C /c/.../terekhova/ \
  all_pbmcs/all_pbmcs_rna.h5ad all_pbmcs/all_pbmcs_metadata.csv
```

(Must use `--force-local` on msys because tar otherwise interprets `C:/` as a remote `host:path`.)

## Schema (as extracted)

### `all_pbmcs_rna.h5ad` — 1,916,367 cells × 36,601 genes

- **`.obs` is empty** (no columns) — all metadata is in the separate CSV
- **`.var` is empty** (no columns) — index only
- **`var_names`**: HGNC gene symbols (`MIR1302-2HG`, `FAM138A`, `OR4F5`, ...). Not Ensembl.
- **`obs_names`**: per-cell barcodes shaped like `ALAW-AS044-1_AAACCTGAGAAGCCCA-1`
- **`.X`**: raw integer counts, dense or CSR depending on anndata version
- **`.raw`**: None

### `all_pbmcs_metadata.csv` — 1,916,367 rows × 20 columns

Key columns:
- First unnamed column = barcode (matches `obs_names`)
- `Donor_id` — e.g. `A20`, `E07`, `E15` — 166 unique
- **`Age`** — int64, 25–81 yr range, no NaN
- `Age_group` — coarse bucket (`A`, `E`, ...)
- `Sex` — `Male` / `Female` — **skewed 84% Male** (1,614,747 Male / 301,620 Female cells)
- `Batch` — 14 unique batches
- `Cluster_names` — top-level cell-type label (9 values, see below)
- Standard Seurat/QC columns (`nCount_RNA`, `nFeature_RNA`, `percent.mt`, `percent.ribo`, `Cluster_numbers`, etc.)

## Cell-type mapping

| Cluster_names label | n cells | Canonical | Notes |
|---|---:|---|---|
| CD4+ T cells | 901,152 | CD4+ T | |
| Myeloid cells | 336,935 | **Monocyte** | **Coarse label**: ~90% monocytes + ~10% dendritic cells. See caveat below. |
| TRAV1-2- CD8+ T cells | 313,343 | CD8+ T | TRAV1-2− distinguishes from MAIT (TRAV1-2+) |
| NK cells | 205,469 | NK | |
| B cells | 71,614 | B | |
| gd T cells | 60,325 | Other (dropped) | |
| MAIT cells | 24,245 | Other (dropped) | |
| Progenitor cells | 1,794 | Other (dropped) | |
| DN T cells | 1,490 | Other (dropped) | double-negative T |

**Myeloid-as-Monocyte caveat.** Terekhova's Cluster_names top-level "Myeloid" includes both monocytes and dendritic cells. scImmuAging's canonical "Monocyte" panel is narrower. The harmonizer maps Myeloid → Monocyte, accepting ~10% DC contamination. For a PBMC aging clock this contamination is orders of magnitude below the donor-age signal; documented in the `load_terekhova` docstring and this note. Terekhova does not provide a finer-grained Cluster_names that would separate monocytes from DCs at the top level.

## Gene-symbol → Ensembl alignment

Terekhova's h5ad uses HGNC symbols as `var_names`; OneK1K and Stephenson use Ensembl. The harmonizer builds a symbol→Ensembl dict from OneK1K's var (35,528 symbol↔Ensembl pairs) and applies it to Terekhova. Unmapped symbols (a minority — mostly lncRNA / orphaned IDs) retain only the symbol; `ensembl_id = symbol` for those rows with a warning logged. Map is constructed in `src/data/harmonize_cohorts.py::main::_harvest_symbol_map`.

## Harmonization applied

`src/data/harmonize_cohorts.py::load_terekhova`:

1. Read `all_pbmcs_metadata.csv`, re-index to barcode.
2. Open `all_pbmcs_rna.h5ad` in `backed='r'` mode (critical — 27 GB file).
3. Align metadata to `adata.obs_names` via `meta.reindex(adata.obs_names)`.
4. Build a single boolean keep-mask:
   - `Age` not NaN
   - `Age >= 18` (adult-only; redundant with Terekhova's 25–81 range but enforced)
   - `Cluster_names` maps to one of the canonical five
5. Single materialization via `adata[keep].to_memory()` — only the ~1.83M kept cells are loaded into RAM.
6. Build var: `gene_symbol = var.index`, `ensembl_id = symbol_to_ensembl.get(sym, sym)`.
7. Return AnnData with the project's canonical obs columns + an extra `batch` column preserved from the Terekhova metadata (useful for downstream batch-correction analyses).

Output schema: `cohort_id='terekhova'`, `donor_id='terekhova:<Donor_id>'`, `age` (float), `age_precision='exact'`, `sex`, `assay="10x 5' v2"`, `cell_type`, `batch`.

### Expected per-cell-type output sizes (post-harmonization)

| Cell type | n cells (Terekhova) | ~h5ad size |
|---|---:|---:|
| CD4+ T | 901,152 | ~2.5 GB |
| Monocyte | 336,935 | ~1.2 GB |
| CD8+ T | 313,343 | ~880 MB |
| NK | 205,469 | ~580 MB |
| B | 71,614 | ~190 MB |

Total ~5.3 GB of per-cell-type output.

## Known issues / caveats

1. **The `raw_counts_h5ad.tar.gz` (syn56693935) is truncated at source.** See the Two-tar mystery section. We work around it via `all_pbmcs.tar.gz`.
2. **Myeloid→Monocyte coarsening** (see Cell-type mapping).
3. **Sex imbalance 84% Male.** Reflects the cohort Terekhova recruited; not a QC artefact. The aging-clock target is `age`, so sex is a covariate rather than the target, but calibration and ancestry/sex analyses should weight accordingly. Consider stratified reporting in Phase 4/5.
4. **10x 5' v2 chemistry differs from OneK1K + Stephenson (both 3').** The pre-trained sc-ImmuAging LASSO was trained on 3' only — applying it to 5' Terekhova introduces a domain-shift confound. Mitigation planned as `Task 1f`: report Terekhova LOCO both naive (measures real-world generalization) and chemistry-corrected via Harmony/scran (isolates aging signal).
5. **Gene identifier is symbol, not Ensembl.** Handled via symbol→Ensembl map from OneK1K's var. If only Terekhova is loaded (no CellxGene cohort), ensembl_id falls back to symbol and a warning fires.
6. **`scripts/download_terekhova.py` defaults to `syn56693935`** (the corrupt primary) for historical reasons — override with `--synapse-id syn51197006` explicitly. After this finding a future cleanup should swap the default; tracked informally.

## Role in the study

**Primary training cohort — and the paper's generalization headline.** Per `data/leakage_audit.csv`, Terekhova is `clean` for all four foundation models (scGPT, Geneformer, scFoundation, UCE) — the only primary LOCO fold where no model carries a training-set-overlap asterisk. So `loco_terekhova` becomes the paper's main figure fold rather than OneK1K.

**Leakage status per foundation model** (all `clean` — verified 2026-04-24):
- scGPT: Terekhova is Synapse-only, not in CellxGene Census 2023-05-15
- Geneformer: November 2023 publication, post V1 June-2021 cutoff
- scFoundation: Synapse-only distribution; not in the MOESM4/MOESM5 GEO/HCA/SCP/EBI corpus
- UCE: November 2023 publication, post CellxGene Census mid-late 2023 snapshot

**Detectability-floor status** at ρ=0.8 planning assumption (from `methods/detectability_floor.md`):

| Cell type | n_required (10% MAE reduction) | Terekhova (166 donors) | Primary? |
|---|---:|---:|:---:|
| CD4+ T | 132 | 166 | ✅ |
| NK | 156 | 166 | ✅ |
| B | 155 | 166 | ✅ |
| CD8+ T | 180 | 166 | ❌ (exploratory unless Phase 3 measures ρ ≥ 0.9) |
| Monocyte | 229 | 166 | ❌ (exploratory unless Phase 3 measures ρ ≥ 0.9) |

## Used in

- Task 1c-v2: three-cohort harmonization (headline primary cohort)
- Task 1f: chemistry batch-correction decision (Terekhova's 10x 5' vs. training cohorts' 10x 3')
- Phase 3 preprint: CD4+ T `loco_terekhova` is the headline MAE figure
- Phase 4: every FM × CD4+ T / NK / B on `loco_terekhova` is strict-clean primary; CD8+ T and Monocyte on Terekhova are exploratory pending empirical ρ
