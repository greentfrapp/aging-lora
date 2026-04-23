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
**Kind:** manual download + compute
**Why I can't do this:** The five GEO cohorts (GSE158055, GSE214534, GSE155673, GSE196830) total >50 GB of raw count matrices; the Stephenson COVID cohort requires manual download from the CellxGene portal. Running the scImmuAging Seurat/scanpy integration pipeline requires an R + Bioconductor environment with ~16–32 GB RAM and several CPU-hours. This cannot be executed in the current agent environment.
**What I need from you:**
1. Download raw count matrices for GSE158055, GSE214534, GSE155673, GSE196830 from GEO (use `scripts/download_geo.sh` or NCBI sra-toolkit `prefetch`).
2. Download the Stephenson COVID-19 PBMC dataset from CellxGene: `https://cellxgene.cziscience.com/collections/b9fc3d70-5a72-4479-a046-c2cc1ab19efc` — export as h5ad.
3. Clone the scImmuAging repo: `git clone https://github.com/CiiM-Bioinformatics-group/scImmuAging data/scImmuAging`.
4. Run the integration pipeline: `Rscript data/scImmuAging/scripts/integration.R --data-dir data/cohorts/raw --out-dir data/cohorts/integrated`.
5. Once integrated h5ads are in `data/cohorts/integrated/`, run: `uv run python src/data/download_cohorts.py --summary-only` to produce `data/cohort_summary.csv`.
6. Strip `[~]` → `[ ]` on Task 1c in `roadmap/phase-1.md`.
**Estimated unblock effort:** 2–4 hours of download + 4–8 hours of compute (more with a GPU node).
