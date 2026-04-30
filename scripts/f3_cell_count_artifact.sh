#!/usr/bin/env bash
# F.3 — cell-count artifact GPU re-extraction for CD4+T frozen base.
#
# Re-extract CD4+T layered embeddings at two per-donor cell caps:
#   cap=5   (small, max SNR pressure — does the L9-L12 pick still hold?)
#   cap=100 (large, NK-typical full count)
# Compares to existing cap=20 to determine whether the cell-type-conditional
# layer asymmetry is biology or per-donor-cell-count artifact.

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=.venv/bin/python
COHORTS=("onek1k" "stephenson" "terekhova" "aida")

for cap in 5 100; do
  for cohort in "${COHORTS[@]}"; do
    out_tag="frozen_base_cap${cap}_alllayers"
    out_path="results/phase3/embeddings_layered/${cohort}_CD4p_T_${out_tag}.npz"
    if [[ -f "$out_path" ]]; then
      echo "[F.3] SKIP existing $out_path"
      continue
    fi
    echo "[F.3] Extracting cohort=$cohort cell_type=CD4+T cap=$cap"
    $PYTHON scripts/extract_embeddings_layered.py \
      --cohort "$cohort" \
      --cell-type "CD4+ T" \
      --max-cells-per-donor "$cap" \
      --bf16 \
      --output-tag "$out_tag" \
      2>&1 | tail -10
  done
done

echo
echo "[F.3] All extractions done. Running ridge readout..."
$PYTHON scripts/f3_cell_count_ridge.py
