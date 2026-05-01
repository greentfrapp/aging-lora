#!/usr/bin/env bash
# I.8 Phase D — fill the cap=20 seed=0 gap that Phase A's loop intentionally
# omitted. Adds 4 NPZs (one per cohort) so the FM matrix matches the gene-EN
# matrix at every cap × seed cell. Re-fires the combined ridge readout.
#
# Skip-existing logic so this is safe to re-run.

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=.venv/bin/python
COHORTS=("stephenson" "terekhova" "aida" "onek1k")

run_one() {
  local cohort=$1 cap=$2 seed=$3
  local out_tag
  if [[ $seed -eq 0 ]]; then
    out_tag="frozen_base_cap${cap}_alllayers"
  else
    out_tag="frozen_base_cap${cap}_seed${seed}_alllayers"
  fi
  local out_path="results/phase3/embeddings_layered/${cohort}_CD4p_T_${out_tag}.npz"
  if [[ -f "$out_path" ]]; then
    echo "[I.8-D] SKIP existing $out_path"
    return 0
  fi
  echo "[I.8-D] Extracting cohort=$cohort cell_type=CD4+T cap=$cap seed=$seed"
  $PYTHON scripts/extract_embeddings_layered.py \
    --cohort "$cohort" \
    --cell-type "CD4+ T" \
    --max-cells-per-donor "$cap" \
    --seed "$seed" \
    --bf16 \
    --output-tag "$out_tag" \
    2>&1 | tail -3
}

echo "[I.8-D] === Phase D: FM cap=20 seed=0 × 4 cohorts (gap-fill) ==="
for cohort in "${COHORTS[@]}"; do
  run_one "$cohort" 20 0
done

echo
echo "[I.8-D] Done. Re-running combined ridge readout with cap=20 seed=0 included..."
$PYTHON scripts/i6_combined_ridge.py
