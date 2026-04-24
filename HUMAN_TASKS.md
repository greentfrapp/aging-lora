# Human-required tasks

Append-only log of project tasks that required human action to
unblock — dataset licenses, manual downloads, credentials, or
judgment calls the run agent couldn't safely make on its own. The
agent writes new entries here when it encounters a blocker and marks
the corresponding roadmap task `- [~]` instead of `- [ ]`.

To unblock: do the action the agent asked for, then strip the `[~]`
back to `[ ]` on the referenced task line. Entries in this file stay
as permanent history — don't delete them.

---

## 2026-04-23 #1 — Execute sc-ImmuAging cohort download and integration pipeline

**Related roadmap task:** `roadmap/phase-1.md` / "Task 1c: Execute download and run scImmuAging integration pipeline"
**Kind:** manual download + compute + EGA data access application

**⚠️ Cohort accessions corrected 2026-04-23** — the original scaffold had three wrong GEO accessions (GSE158055, GSE214534, GSE155673). Confirmed authoritative cohorts from Li et al. 2025 PMC full text (PMC12003178):

| Cohort | Accession | Access | Status |
|---|---|---|---|
| OneK1K / Yazar 2022 | GSE196830 | Public GEO | `scripts/download_geo.sh` handles this |
| Stephenson 2021 | CellxGene ddfad306 | Public | Download in progress (~7 GB) |
| Barreiro IAV / Nedelec 2022 | GSE162632 | Public GEO | `scripts/download_geo.sh` handles this |
| COVID-NL | EGAS00001005529 | **EGA controlled access** | Requires DAC application |
| BCG vaccination | EGAS00001006990 | **EGA controlled access** | Requires DAC application |

**What was needed / current status:**
1. ✅ Run `bash scripts/download_geo.sh` — GSE196830 (13 GB) and GSE162632 (789 MB) downloaded as RAW.tar.
2. ✅ Stephenson h5ad (`c17079d3-204f-487e-bc54-d63bb947a5a2.h5ad`, 6.6 GB) confirmed correct and in `data/cohorts/raw/stephenson_covid_portal/`.
3. ✅ No additional cohort needed (Case 1). Case 3 tracked in `FUTURE_WORK.md`.
4. ✅ Cloned scImmuAging repo to `data/scImmuAging/` (2026-04-24).

**⚠️ Correction (2026-04-24):** The scImmuAging repo contains no data integration pipeline. It is a pre-trained inference R package — `data/all_model.RDS` ships the five finished LASSO aging clocks (CD4T, CD8T, MONO, NK, B) ready for use as the baseline. `scripts/integration.R` does not exist and never did. The original step 6 was wrong.

**Remaining human task:**
5. ✅ Extracted both GEO RAW.tar archives (2026-04-24): `GSE196830_RAW.tar` → 75 OneK1K pool CSVs + barcode/genotype files; `GSE162632_RAW.tar` → 30 Barreiro 10x MTX triplets.

6. ✅ OneK1K source decision (2026-04-24): switched to CellxGene-curated h5ad (`3faad104-…`, 4.4 GB) instead of the 26 GB GEO RAW demux path. Rationale documented in `roadmap/phase-1.md`. The GEO RAW files remain on disk as a fallback; `FUTURE_WORK.md` tracks the demux path as a deferred option.

7. Task 1c is no longer blocked. Agent can now proceed to download the OneK1K CellxGene h5ad (Task 1d) and write the scanpy harmonization pipeline (Task 1c).

**Status:** This entry is complete — no further human action required.

---

## 2026-04-24 #2 — Download Terekhova 2023 cohort from Synapse

**Related roadmap task:** `roadmap/phase-1.md` / Task 1d-terekhova
**Kind:** account registration + manual download

**Context.** Terekhova et al. 2023 *Immunity* — 166 healthy donors, continuous 25–85 yr age span, 10x 5' v2 chemistry. Promoted from Case 3 to Case 1 on 2026-04-24 after Barreiro was dropped (see entry #1 and the `FUTURE_WORK.md` Barreiro-revival note). Terekhova is the second primary LOCO fold (166 > 80-donor threshold) — without it we have only a single primary fold.

**Accessions**
- Project root: `syn49637038`
- **Target file: `syn56693935`** = `raw_counts_h5ad.tar.gz` (9.8 GB) — single tar of the 166-donor raw-counts h5ad
- Alternative: `syn51197006` = `all_pbmcs.tar.gz` (15.9 GB, full Seurat atlas, requires R→Python conversion)

**Steps**
1. Create a free Synapse account at https://www.synapse.org/register (no institutional affiliation required).
2. Accept the data use conditions on the `syn49637038` project page (one-time).
3. Authenticate: set up `~/.synapseConfig` in canonical form:
   ```ini
   [authentication]
   authtoken = <your-PAT>
   ```
   (The download script also tolerates a file containing just the raw PAT on line 1,
   or env-variable `SYNAPSE_AUTH_TOKEN`.)
4. Run the download script:
   ```bash
   uv run python scripts/download_terekhova.py --dry-run        # confirm what will transfer
   uv run python scripts/download_terekhova.py                  # actual download (~9.8 GB)
   ```
   `syncFromSynapse` is checksum-aware — re-running after a partial transfer only
   re-downloads the missing/incomplete files.
5. After the download completes, extract the tar:
   ```bash
   tar -xzf data/cohorts/raw/terekhova/raw_counts_h5ad.tar.gz -C data/cohorts/raw/terekhova/
   ```
6. Strip `[~]` → `[ ]` on Task 1d-terekhova in `roadmap/phase-1.md`.

**Once on disk, the agent will:**
- Inspect the extracted h5ad's obs schema (age encoding, cell-type labels, donor_id column).
- Tighten `load_terekhova()` in `src/data/harmonize_cohorts.py` based on actual schema.
- Run the three-cohort end-to-end harmonization.

**Estimated human effort:** ~15 min (account + accept DUC + PAT setup); download ~9.8 GB at broadband speeds is ~15–60 min.

---

## 2026-04-24 #3 — scFoundation checkpoint download (SharePoint click-through)

**Related roadmap task:** `roadmap/phase-1.md` / "Download foundation-model checkpoints and verify integrity"
**Kind:** manual browser download

**Context.** scFoundation weights (~700 MB, `models.ckpt` / `01B-resolution.ckpt`) are hosted on a SharePoint anonymous share that requires a browser click-through (email + "continue as guest"). This cannot be scripted via curl/wget/gdown. scGPT, Geneformer, and UCE all download autonomously via `scripts/download_foundation_models.py`.

**Steps**
1. Open https://hopebio2020.sharepoint.com/:f:/s/PublicSharedfiles/IgBlEJ72TBE5Q76AmgXbgjXiAR69fzcrgzqgUYdSThPLrqk in a browser.
2. Choose "Continue as guest" when prompted (email + consent form).
3. Download `models.ckpt` (or `01B-resolution.ckpt`), ~700 MB.
4. Save to `save/scFoundation/models/`.
5. Re-run `uv run python scripts/download_foundation_models.py --hash` to register SHA-256 in `data/checkpoint_hashes.txt`.

**License caveat.** scFoundation code is Apache-2.0 but **weights are under a separate Model License** (per the repo's copyright notice). Verify the Model License before any redistribution (e.g. bundling into Docker images or shared cluster scratch).

**Estimated human effort:** ~5 min (browser click-through + download).

---

## 2026-04-24 #4 — Resolve two `unknown` rows in the leakage audit

**Related roadmap task:** `roadmap/phase-1.md` / "Run pretraining-corpus leakage audit"
**Kind:** paper supplementary-table lookup

**Context.** Four of the 16 rows in `data/leakage_audit.csv` are `unknown` because the supporting supplementary tables were not retrievable during the automated audit (Nature 403'd, GitHub blob view not parseable). All four need a one-line check each before Phase 3/4 paper submission.

**Status: RESOLVED 2026-04-24.** All four `unknown` rows updated in `data/leakage_audit.csv`:

- `Geneformer × OneK1K` → **clean** (Genecorpus-30M Supp Table 1 direct-search confirmed; see `scratchpad/41586_2023_6139_MOESM4_ESM.xlsx`).
- `Geneformer × Stephenson` → **clean** (same supplement, zero hits for Stephenson/Haniffa/E-MTAB-10026/COVID-PBMC citations).
- `scFoundation × OneK1K` → **clean** (Hao 2024 Nat Methods MOESM4 + MOESM5 direct-search, zero hits).
- `scFoundation × Stephenson` → **overlapping** (Hao 2024 Nat Methods MOESM5 row 81 `HCA-Covid19PBMC` is the Stephenson/Haniffa 2021 dataset, ingested via its HCA native project ID rather than the ArrayExpress mirror).

Scratchpad files retained for audit: `scratchpad/scf_MOESM{4,5,6}_ESM.xlsx` and `scratchpad/41586_2023_6139_MOESM4_ESM.xlsx`.

---

## 2026-04-24 #5 — Report corrupt Terekhova `raw_counts_h5ad.tar.gz` to authors

**Related roadmap task:** `roadmap/phase-1.md` / Task 1d-terekhova
**Kind:** upstream bug report (low priority; blocker already worked around via Task #13)

**Issue.** The Synapse-hosted `syn56693935` (`raw_counts_h5ad.tar.gz`, 9.8 GB) contains a truncated h5ad. Diagnostic evidence:

- Tar file MD5: `20fcbe2531a999c51ed03dfce5d29b7b` — matches Synapse's posted MD5 ✅
- Tar contents (per `tar -tzvf`): the archive header claims `raw_counts_h5ad/pbmc_gex_raw_with_var_obs.h5ad` is **77,016,305,915 bytes** (71.7 GiB).
- After `tar -xzf`, the extracted file on disk is **23,477,328,896 bytes** (21.86 GiB) — i.e. the tar content itself is incomplete: only ~30% of the expected 77 GB was written into the tar before upload.
- `h5py.File(...)` and `anndata.read_h5ad(...)` both refuse to open: `OSError: Unable to synchronously open file (truncated file: eof = 23477328896, sblock->base_addr = 0, stored_eof = 77016305915)`.

This suggests the upload pipeline at Terekhova's end truncated the file at the point it was packaged into the tar — not a corruption in download or decompression.

**What to do (low priority — we have a local workaround via `all_pbmcs.tar.gz`):**
1. Email the Terekhova lab data-custody contact (Terekhova / Artyomov lab at Washington University; `artyomov <at> wustl.edu` is the PI). Cc Synapse helpdesk (`synapseinfo <at> sagebionetworks.org`).
2. Reference `syn56693935` and attach the tar listing output above.
3. Request a re-upload of the raw-counts h5ad.

**Local workaround (done):** Task #13 downloads `all_pbmcs.tar.gz` (`syn51197006`, 15.9 GB full Seurat atlas) and converts R→h5ad to produce a working `load_terekhova()` input.

**Estimated human effort:** ~10 min for the email.
