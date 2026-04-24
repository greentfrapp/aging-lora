#!/usr/bin/env bash
# Download raw count matrices for all four GEO cohorts used by sc-ImmuAging.
#
# Each accession's supplemental files are fetched from NCBI GEO FTP and
# placed under OUT_DIR/<accession>/.  Already-present files are skipped via
# timestamping (wget) or --continue (aria2c).
#
# Usage:
#   bash scripts/download_geo.sh [OPTIONS]
#
# Options:
#   --out-dir DIR   Root directory for downloads   (default: data/cohorts/raw)
#   --jobs N        Parallel accession workers      (default: 2)
#   --dry-run       Print URLs without downloading
#
# Requirements (first found is used, in preference order):
#   aria2c   — fastest, parallel segments (brew/apt install aria2)
#   wget     — standard recursive downloader
#   curl     — fallback, lists FTP dir then fetches each file

set -euo pipefail

# ── defaults ──────────────────────────────────────────────────────────────────
OUT_DIR="data/cohorts/raw"
JOBS=2
DRY_RUN=false

# Public GEO accessions only.
# EGAS00001005529 (COVID-NL) and EGAS00001006990 (BCG) are EGA controlled-access
# and must be requested separately at https://ega-archive.org/.
ACCESSIONS=(GSE196830 GSE162632)

# Supplemental files we care about — count matrices and metadata
ACCEPT_GLOB="*.gz,*.h5,*.h5ad,*.loom,*.mtx.gz,*.tsv.gz,*.csv.gz,*.tar.gz,*.tar"

# ── arg parsing ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --jobs)    JOBS="$2";    shift 2 ;;
    --dry-run) DRY_RUN=true; shift   ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \?//'
      exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

LOG_DIR="${OUT_DIR}/../logs"
LOG_FILE="${LOG_DIR}/download_geo_$(date '+%Y%m%d_%H%M%S').log"

# ── helpers ───────────────────────────────────────────────────────────────────
log() {
  local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
  echo "$msg"
  echo "$msg" >> "$LOG_FILE"
}

# GSE158055 → GSE158nnn  (NCBI FTP series prefix)
geo_series_prefix() {
  local acc="${1#GSE}"            # strip "GSE" → 158055
  local n=${#acc}                 # 6
  echo "GSE${acc:0:$((n - 3))}nnn"
}

# HTTPS URL for downloading individual files
geo_suppl_url() {
  local acc="$1"
  local prefix
  prefix=$(geo_series_prefix "$acc")
  echo "https://ftp.ncbi.nlm.nih.gov/geo/series/${prefix}/${acc}/suppl/"
}

# FTP URL for directory listing (curl -sl only works with ftp://)
geo_suppl_ftp_url() {
  local acc="$1"
  local prefix
  prefix=$(geo_series_prefix "$acc")
  echo "ftp://ftp.ncbi.nlm.nih.gov/geo/series/${prefix}/${acc}/suppl/"
}

# ── per-accession download ────────────────────────────────────────────────────
download_accession() {
  local acc="$1"
  local dest="${OUT_DIR}/${acc}"
  local url
  url=$(geo_suppl_url "$acc")

  log "[$acc] URL  : $url"
  log "[$acc] Dest : $dest"

  if $DRY_RUN; then
    log "[$acc] [dry-run] skipping actual download"
    return 0
  fi

  mkdir -p "$dest"

  if command -v aria2c &>/dev/null; then
    log "[$acc] Using aria2c"
    aria2c \
      --dir="$dest" \
      --continue=true \
      --remote-time=true \
      --max-connection-per-server=4 \
      --split=4 \
      --file-allocation=none \
      --retry-wait=10 \
      --max-tries=8 \
      --connect-timeout=30 \
      --timeout=600 \
      --log="${dest}/aria2c.log" \
      --log-level=warn \
      --recursive=true \
      --follow-ftp=true \
      "$url" \
      2>&1 | tee -a "$LOG_FILE" \
      || { log "[$acc] ERROR: aria2c failed"; return 1; }
  elif command -v wget &>/dev/null; then
    log "[$acc] Using wget  (install aria2 for faster parallel downloads)"
    # --cut-dirs=5 strips: /geo/series/<prefix>/<acc>/suppl/  (5 components)
    wget \
      --recursive \
      --no-parent \
      --no-host-directories \
      --cut-dirs=5 \
      --timestamping \
      --tries=8 \
      --wait=2 \
      --random-wait \
      --timeout=60 \
      --read-timeout=300 \
      --directory-prefix="$dest" \
      --accept "$ACCEPT_GLOB" \
      --progress=bar:force \
      "$url" \
      2>&1 | tee -a "$LOG_FILE" \
      || { log "[$acc] ERROR: wget failed"; return 1; }
  elif command -v curl &>/dev/null; then
    log "[$acc] Using curl  (install aria2 or wget for better resume support)"
    # List via ftp:// (curl -l only works with FTP), download via https://
    local ftp_url
    ftp_url=$(geo_suppl_ftp_url "$acc")
    local files
    files=$(curl -sl --retry 5 --connect-timeout 30 "$ftp_url" 2>>"$LOG_FILE") \
      || { log "[$acc] ERROR: could not list directory $ftp_url"; return 1; }

    local downloaded=0
    while IFS= read -r fname; do
      [[ -z "$fname" ]] && continue
      # Filter to count matrix / metadata extensions only
      case "$fname" in
        *.gz|*.h5|*.h5ad|*.loom|*.tar|*.tar.gz|*.tsv.gz|*.csv.gz) ;;
        *) continue ;;
      esac
      local fdest="${dest}/${fname}"
      if [[ -f "$fdest" ]]; then
        log "[$acc] Skipping (exists): $fname"
        downloaded=$((downloaded + 1))
        continue
      fi
      log "[$acc] Downloading: $fname"
      curl \
        --continue-at - \
        --location \
        --retry 8 \
        --retry-delay 10 \
        --connect-timeout 30 \
        --max-time 7200 \
        --output "$fdest" \
        --progress-bar \
        "${url}${fname}" \
        2>&1 | tee -a "$LOG_FILE" \
        || { log "[$acc] WARNING: failed to download $fname — continuing"; rm -f "$fdest"; }
      downloaded=$((downloaded + 1))
    done <<< "$files"

    [[ $downloaded -eq 0 ]] && { log "[$acc] ERROR: no matching files found at $url"; return 1; }
  else
    log "[$acc] ERROR: no downloader found — install aria2, wget, or curl"; return 1
  fi

  local n_files
  n_files=$(find "$dest" -type f ! -name 'aria2c.log' | wc -l)
  log "[$acc] Done — ${n_files} file(s) in ${dest}"
}

export -f download_accession geo_suppl_url geo_suppl_ftp_url geo_series_prefix log
export OUT_DIR DRY_RUN LOG_FILE ACCEPT_GLOB

# ── main ──────────────────────────────────────────────────────────────────────
mkdir -p "$LOG_DIR"
log "=================================================="
log "GEO cohort download starting"
log "Accessions : ${ACCESSIONS[*]}"
log "Output dir : ${OUT_DIR}"
log "Jobs       : ${JOBS}"
log "Dry-run    : ${DRY_RUN}"
log "Log        : ${LOG_FILE}"
log "=================================================="

# Parallel workers — one per accession, bounded by --jobs
printf '%s\n' "${ACCESSIONS[@]}" \
  | xargs -P "$JOBS" -I{} bash -c 'download_accession "$@"' _ {}

log "=================================================="
log "All GEO downloads complete."
log ""
log "Next steps:"
log "  1. Download Stephenson cohort manually from CellxGene:"
log "       https://cellxgene.cziscience.com/collections/ddfad306-714d-4cc0-9985-d9072820c530"
log "     Direct h5ad (~7 GB): https://datasets.cellxgene.cziscience.com/c17079d3-204f-487e-bc54-d63bb947a5a2.h5ad"
log "     Save the .h5ad to: ${OUT_DIR}/stephenson_covid_portal/"
log ""
log "  2. Apply for EGA controlled-access cohorts (see HUMAN_TASKS.md):"
log "       EGAS00001005529 (COVID-NL)     — https://ega-archive.org/studies/EGAS00001005529"
log "       EGAS00001006990 (BCG cohort)   — https://ega-archive.org/studies/EGAS00001006990"
log ""
log "  3. Clone and run the scImmuAging integration pipeline:"
log "       git clone https://github.com/CiiM-Bioinformatics-group/scImmuAging data/scImmuAging"
log "       Rscript data/scImmuAging/scripts/integration.R \\"
log "         --data-dir ${OUT_DIR} --out-dir data/cohorts/integrated"
log ""
log "  3. Build the cohort summary CSV:"
log "       uv run python src/data/download_cohorts.py --summary-only"
log "=================================================="
