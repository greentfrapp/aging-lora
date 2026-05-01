#!/usr/bin/env bash
# I.8 — manuscript-grade R-vs-cap curve completion (FM extractions).
#
# Phase A: cap=10 × 3 seeds × 4 cohorts + cap=20 × seeds {1, 2} × 4 cohorts (~2h GPU)
# Phase B: cap=500 × 3 seeds × onek1k (~24h GPU; relies on streaming-aggregation
#          fix in 4613816)
# Phase C: cap=1000 × seed 0 × onek1k + cap=1000 × seeds {1, 2} × 4 cohorts (~47h GPU)
#
# Skip-existing logic so re-launching after any interrupt resumes cleanly.
# Auto-fires combined ridge readout at the end.

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=.venv/bin/python
COHORTS=("stephenson" "terekhova" "aida" "onek1k")  # smaller cohorts first

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
    echo "[I.8] SKIP existing $out_path"
    return 0
  fi
  echo "[I.8] Extracting cohort=$cohort cell_type=CD4+T cap=$cap seed=$seed"
  $PYTHON scripts/extract_embeddings_layered.py \
    --cohort "$cohort" \
    --cell-type "CD4+ T" \
    --max-cells-per-donor "$cap" \
    --seed "$seed" \
    --bf16 \
    --output-tag "$out_tag" \
    2>&1 | tail -3
}

# Phase A: low-cap fillers
echo "[I.8] === Phase A: FM cap=10 × 3 seeds + cap=20 × seeds {1, 2} ==="
for seed in 0 1 2; do
  for cohort in "${COHORTS[@]}"; do
    run_one "$cohort" 10 "$seed"
  done
done
for seed in 1 2; do
  for cohort in "${COHORTS[@]}"; do
    run_one "$cohort" 20 "$seed"
  done
done

# Phase B: onek1k cap=500 × 3 seeds
echo "[I.8] === Phase B: FM cap=500 × 3 seeds × onek1k ==="
for seed in 0 1 2; do
  run_one "onek1k" 500 "$seed"
done

# Phase C: cap=1000 multi-seed expansion
echo "[I.8] === Phase C: FM cap=1000 × seed 0 × onek1k + seeds {1, 2} × 4 cohorts ==="
run_one "onek1k" 1000 0
for seed in 1 2; do
  for cohort in "${COHORTS[@]}"; do
    run_one "$cohort" 1000 "$seed"
  done
done

echo
echo "[I.8] All FM extractions done. Running combined ridge readout..."
$PYTHON scripts/i6_combined_ridge.py
