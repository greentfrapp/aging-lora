#!/usr/bin/env bash
# I.6 — FM extractions for the 3-seed cap-matrix.
#
#   cap=50  × seeds {0, 1, 2} × 4 cohorts  (~4.5h GPU)
#   cap=500 × seeds {0, 1, 2} × 4 cohorts  (~40h GPU)
#   cap=1000 × seed {0}      × 4 cohorts   (~18.5h GPU; pause for review)
#
# cap=100 already covered by F.3 (seed=0) + I.4 (seeds 1, 2).
#
# Skip-existing logic so re-launching resumes.

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=.venv/bin/python
# Two cohort lists:
#   COHORTS_CAP50: cap=50 includes onek1k (already complete on disk; skip-existing
#     covers it cheaply on rerun).
#   COHORTS_HIGHCAP: cap=500 and cap=1000 SKIP onek1k. The onek1k cap=500/1000
#     extractions are the long pole (~24h + ~10h) and not needed for the
#     loco_onek1k → AIDA matched-cap trajectory (which uses stephenson +
#     terekhova as train, aida as eval). Without onek1k we lose only the
#     loco_terekhova fold direction at cap=500/1000 (and the within-distribution
#     onek1k holdout, which is secondary).
COHORTS_CAP50=("stephenson" "terekhova" "aida" "onek1k")
COHORTS_HIGHCAP=("stephenson" "terekhova" "aida")

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
    echo "[I.6] SKIP existing $out_path"
    return 0
  fi
  echo "[I.6] Extracting cohort=$cohort cell_type=CD4+T cap=$cap seed=$seed"
  $PYTHON scripts/extract_embeddings_layered.py \
    --cohort "$cohort" \
    --cell-type "CD4+ T" \
    --max-cells-per-donor "$cap" \
    --seed "$seed" \
    --bf16 \
    --output-tag "$out_tag" \
    2>&1 | tail -3
}

# Phase A: cap=50 × 3 seeds (~4.5h GPU; all done on disk, skip-existing handles it)
echo "[I.6] === Phase A: FM cap=50 × 3 seeds ==="
for seed in 0 1 2; do
  for cohort in "${COHORTS_CAP50[@]}"; do
    run_one "$cohort" 50 "$seed"
  done
done

# Phase B: cap=500 × 3 seeds, NO ONEK1K (~5h GPU)
echo "[I.6] === Phase B: FM cap=500 × 3 seeds (skipping onek1k) ==="
for seed in 0 1 2; do
  for cohort in "${COHORTS_HIGHCAP[@]}"; do
    run_one "$cohort" 500 "$seed"
  done
done

# Phase C: cap=1000 × seed 0, NO ONEK1K (~8.5h GPU; pause for review after)
echo "[I.6] === Phase C: FM cap=1000 × seed 0 (skipping onek1k; review gate after) ==="
for cohort in "${COHORTS_HIGHCAP[@]}"; do
  run_one "$cohort" 1000 0
done

echo
echo "[I.6] All extractions done. Running combined ridge readout..."
$PYTHON scripts/i6_combined_ridge.py
