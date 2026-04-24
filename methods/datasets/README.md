# Dataset notes

Consolidated notes on every PBMC single-cell dataset evaluated for the immune-aging-clock project. Each dataset has a dedicated document in this folder. Split per-file so that future cohort decisions can edit one notes page without merge conflicts with the others.

## In the study (Case 1 baseline, recorded 2026-04-24)

| Dataset | File | Role | Accession | Donors | Age | Chemistry |
|---|---|---|---|---|---|---|
| OneK1K / Yazar 2022 | [`onek1k.md`](onek1k.md) | Primary training cohort (largest) | CellxGene `3faad104` | 981 | 19–97 yr | 10x 3' v2 |
| Stephenson 2021 | [`stephenson.md`](stephenson.md) | Exploratory training cohort (<80 donors) | CellxGene `ddfad306` | 29 healthy | 21–73 yr (mixed precision) | 10x 3' |
| Terekhova 2023 | [`terekhova.md`](terekhova.md) | Primary training cohort (5' fold) | Synapse `syn49637038` | 166 | 25–81 yr | 10x 5' v2 |
| AIDA (Kock 2025) | [`aida.md`](aida.md) | External ancestry holdout | CellxGene `ced320a1` | 619 (7 Asian groups) | mixed | 10x 3' / 5' |

## Considered and rejected

| Dataset | File | Rejection reason |
|---|---|---|
| Barreiro / Randolph 2021 | [`barreiro_randolph.md`](barreiro_randolph.md) | No donor ages in GEO; genotype-multiplexed without demux files in archive |
| Allen Institute PBMC atlas | [`allen_pbmc_atlas.md`](allen_pbmc_atlas.md) | CMV-seropositive donors mixed in; 37–54 yr age gap; pediatric contamination; low `is_primary_data` yield |

## Cross-dataset learnings

**Accession mirrors matter.** Single-cell datasets are routinely deposited to multiple portals under different IDs. A leakage audit must check every mirror — not just the first-discovered accession. Specifically:

- Stephenson 2021 has at least three IDs: CellxGene `ddfad306`, ArrayExpress `E-MTAB-10026`, HCA `HCA-Covid19PBMC`.
- OneK1K has CellxGene `dde06e0f` (collection) / `3faad104` (dataset) + GEO `GSE196830`.
- This caught scFoundation × Stephenson as `overlapping` — the direct `E-MTAB-10026` search returned zero hits but the HCA Project ID was in the Hao 2024 supplementary table. See [`methods/leakage_audit_notes.md`](../leakage_audit_notes.md) for the full reporting rule.

**Gene identifier conventions diverge.** CellxGene-curated releases use Ensembl IDs as `var_names` with symbols in `var['feature_name']`. Synapse/Seurat-derived releases (Terekhova) use HGNC symbols as `var_names` and carry no Ensembl column. The harmonizer materialises a symbol→Ensembl map from OneK1K's var (whichever CellxGene cohort loads first) and applies it to the Terekhova data so all integrated h5ads carry both identifier columns in `.var`.

**Age precision varies.** OneK1K and Terekhova ship exact-integer ages. Stephenson CellxGene mixes exact and decade-bin labels (11/29 exact, 18/29 decade-only at 5-year midpoints); the harmonizer flags each cell via `obs['age_precision']` ∈ {`exact`, `decade`, `none`}.

**Cohort size × FM leakage determines result-table role.** Per `data/leakage_audit.csv`: Terekhova is `clean` for all four foundation models (headline generalization fold); OneK1K is clean only for Geneformer + scFoundation; Stephenson is clean only for Geneformer. The paper's primary figure is the Terekhova LOCO table because it's the only fold where no asterisks are required.

## Conventions shared across dataset notes

Every per-dataset doc follows this structure:

1. **Paper + accession** — canonical citation + all known deposit mirrors
2. **Access** — how to download, auth required, expected file size
3. **Schema** — what's in the raw h5ad/CSV/MTX before our harmonization
4. **Harmonization** — filters applied, cell-type map, obs schema produced
5. **Known issues / caveats** — what we found that isn't obvious from the paper
6. **Role in the study** — primary training / exploratory / external holdout / rejected
