#!/usr/bin/env bash
# I.2 — Cap=100 frozen for NK and B across 4 cohorts.

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=.venv/bin/python
COHORTS=("onek1k" "stephenson" "terekhova" "aida")
CELL_TYPES=("NK" "B")

for cell_type in "${CELL_TYPES[@]}"; do
  cell_slug=$(echo "$cell_type" | sed 's/+/p/g; s/ /_/g')
  for cohort in "${COHORTS[@]}"; do
    out_tag="frozen_base_cap100_alllayers"
    out_path="results/phase3/embeddings_layered/${cohort}_${cell_slug}_${out_tag}.npz"
    if [[ -f "$out_path" ]]; then
      echo "[I.2] SKIP existing $out_path"
      continue
    fi
    echo "[I.2] Extracting cohort=$cohort cell_type=$cell_type cap=100"
    $PYTHON scripts/extract_embeddings_layered.py \
      --cohort "$cohort" \
      --cell-type "$cell_type" \
      --max-cells-per-donor 100 \
      --bf16 \
      --output-tag "$out_tag" \
      2>&1 | tail -3
  done
done

echo
echo "[I.2] All extractions done. Running ridge readout..."
$PYTHON scripts/i2_nk_b_ridge.py
