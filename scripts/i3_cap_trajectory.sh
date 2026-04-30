#!/usr/bin/env bash
# I.3 — Cap-trajectory plateau test on CD4+T frozen.
# Extracts cap=50, 200, 500 across 4 cohorts; combined with cap=5/20/100 from F.3
# gives a 6-point trajectory.

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=.venv/bin/python
COHORTS=("onek1k" "stephenson" "terekhova" "aida")
CAPS=(50 200 500)

for cap in "${CAPS[@]}"; do
  for cohort in "${COHORTS[@]}"; do
    out_tag="frozen_base_cap${cap}_alllayers"
    out_path="results/phase3/embeddings_layered/${cohort}_CD4p_T_${out_tag}.npz"
    if [[ -f "$out_path" ]]; then
      echo "[I.3] SKIP existing $out_path"
      continue
    fi
    echo "[I.3] Extracting cohort=$cohort cell_type=CD4+T cap=$cap"
    $PYTHON scripts/extract_embeddings_layered.py \
      --cohort "$cohort" \
      --cell-type "CD4+ T" \
      --max-cells-per-donor "$cap" \
      --bf16 \
      --output-tag "$out_tag" \
      2>&1 | tail -3
  done
done

echo
echo "[I.3] All extractions done. Running cap-trajectory ridge readout..."
$PYTHON scripts/i3_cap_trajectory_ridge.py
