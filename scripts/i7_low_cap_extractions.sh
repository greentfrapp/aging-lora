#!/usr/bin/env bash
# I.7 — FM extractions for cap=1 and cap=5 × 3 seeds × 4 cohorts.
#
# Closes out the low-cap end of the matched-cap trajectory (cap=50 and
# above are done in I.6). Includes onek1k for free since these caps are
# tiny (~24 to ~981 cells per cohort × 1-5 cells/donor).
#
# Skip-existing logic: cap=5 seed=0 NPZs already exist from F.3 — only
# seeds 1, 2 at cap=5 plus all of cap=1 are new (20 extractions).

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=.venv/bin/python
COHORTS=("stephenson" "terekhova" "aida" "onek1k")
CAPS=(1 5)

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
    echo "[I.7] SKIP existing $out_path"
    return 0
  fi
  echo "[I.7] Extracting cohort=$cohort cell_type=CD4+T cap=$cap seed=$seed"
  $PYTHON scripts/extract_embeddings_layered.py \
    --cohort "$cohort" \
    --cell-type "CD4+ T" \
    --max-cells-per-donor "$cap" \
    --seed "$seed" \
    --bf16 \
    --output-tag "$out_tag" \
    2>&1 | tail -3
}

for cap in "${CAPS[@]}"; do
  for seed in 0 1 2; do
    for cohort in "${COHORTS[@]}"; do
      run_one "$cohort" "$cap" "$seed"
    done
  done
done

echo
echo "[I.7] All FM extractions done."
