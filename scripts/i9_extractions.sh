#!/usr/bin/env bash
# I.9 — manuscript-grade R-vs-cap curves for NK and B cell types.
#
# Mirrors I.8's 3-phase structure but loops over CELL_TYPES=(NK B):
#   Phase A: cap=10/20 fillers (also cap=1, 5, 50 since these don't exist yet
#            for NK/B at any seed)
#   Phase B: cap=100 multi-seed (seeds 1, 2; seed=0 already done in I.2)
#   Phase C: cap=500 × 3 seeds (full)
#   Phase D: cap=1000 × 3 seeds (full)
#
# Skip-existing logic so re-launching after any interrupt resumes cleanly.
# Auto-fires combined ridge readout at the end.
#
# Existing NPZs (from F.3 + I.2):
#   {NK, B} cap=20 × seed=0 × 4 cohorts (8 NPZs)
#   {NK, B} cap=100 × seed=0 × 4 cohorts (8 NPZs)

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=.venv/bin/python
COHORTS=("stephenson" "terekhova" "aida" "onek1k")
CELL_TYPES=("NK" "B")

cell_type_slug() {
  echo "$1" | sed 's/+/p/g; s/ /_/g'
}

run_one() {
  local cell_type=$1 cohort=$2 cap=$3 seed=$4
  local slug
  slug=$(cell_type_slug "$cell_type")
  local out_tag
  if [[ $seed -eq 0 ]]; then
    out_tag="frozen_base_cap${cap}_alllayers"
  else
    out_tag="frozen_base_cap${cap}_seed${seed}_alllayers"
  fi
  local out_path="results/phase3/embeddings_layered/${cohort}_${slug}_${out_tag}.npz"
  if [[ -f "$out_path" ]]; then
    echo "[I.9] SKIP existing $out_path"
    return 0
  fi
  echo "[I.9] Extracting cell_type=$cell_type cohort=$cohort cap=$cap seed=$seed"
  $PYTHON scripts/extract_embeddings_layered.py \
    --cohort "$cohort" \
    --cell-type "$cell_type" \
    --max-cells-per-donor "$cap" \
    --seed "$seed" \
    --bf16 \
    --output-tag "$out_tag" \
    2>&1 | tail -3
}

for cell_type in "${CELL_TYPES[@]}"; do
  echo
  echo "[I.9] ============================================================"
  echo "[I.9] === Cell type: $cell_type ==="
  echo "[I.9] ============================================================"

  echo "[I.9] === Phase A ($cell_type): cap=1/5/10/20/50 × 3 seeds ==="
  for cap in 1 5 10 20 50; do
    for seed in 0 1 2; do
      for cohort in "${COHORTS[@]}"; do
        run_one "$cell_type" "$cohort" "$cap" "$seed"
      done
    done
  done

  echo "[I.9] === Phase B ($cell_type): cap=100 seeds {1, 2} (seed=0 from I.2) ==="
  for seed in 1 2; do
    for cohort in "${COHORTS[@]}"; do
      run_one "$cell_type" "$cohort" 100 "$seed"
    done
  done

  echo "[I.9] === Phase C ($cell_type): cap=500 × 3 seeds ==="
  for seed in 0 1 2; do
    for cohort in "${COHORTS[@]}"; do
      run_one "$cell_type" "$cohort" 500 "$seed"
    done
  done

  echo "[I.9] === Phase D ($cell_type): cap=1000 × 3 seeds ==="
  for seed in 0 1 2; do
    for cohort in "${COHORTS[@]}"; do
      run_one "$cell_type" "$cohort" 1000 "$seed"
    done
  done
done

echo
echo "[I.9] All FM extractions done. Running combined ridge readouts..."
$PYTHON scripts/i9_combined_ridge.py
