#!/usr/bin/env bash
# I.4 — 3-seed verification of F.3 cap=100 CD4+T frozen.
# Re-extracts at seed=1 and seed=2 (seed=0 already done in F.3).

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=.venv/bin/python
COHORTS=("onek1k" "stephenson" "terekhova" "aida")

for seed in 1 2; do
  for cohort in "${COHORTS[@]}"; do
    out_tag="frozen_base_cap100_seed${seed}_alllayers"
    out_path="results/phase3/embeddings_layered/${cohort}_CD4p_T_${out_tag}.npz"
    if [[ -f "$out_path" ]]; then
      echo "[I.4] SKIP existing $out_path"
      continue
    fi
    echo "[I.4] Extracting cohort=$cohort cell_type=CD4+T cap=100 seed=$seed"
    $PYTHON scripts/extract_embeddings_layered.py \
      --cohort "$cohort" \
      --cell-type "CD4+ T" \
      --max-cells-per-donor 100 \
      --seed "$seed" \
      --bf16 \
      --output-tag "$out_tag" \
      2>&1 | tail -3
  done
done

echo
echo "[I.4] All extractions done. Running 3-seed ridge readout..."
$PYTHON scripts/i4_3seed_ridge.py
