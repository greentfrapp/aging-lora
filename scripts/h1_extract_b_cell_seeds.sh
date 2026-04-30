#!/usr/bin/env bash
# H.1 — Extract frozen Geneformer × B-cell layered embeddings for seeds 1, 2 across 4 cohorts.
# 8 sequential passes on local GPU. Outputs 8 npz files to results/phase3/embeddings_layered/.
set -euo pipefail

PY=".venv/Scripts/python.exe"
SCRIPT="scripts/extract_embeddings_layered.py"
OUTDIR="results/phase3/embeddings_layered"

for seed in 1 2; do
  for cohort in onek1k stephenson terekhova aida; do
    OUT="${OUTDIR}/${cohort}_B_frozen_base_seed${seed}_alllayers.npz"
    if [ -f "$OUT" ]; then
      echo "[H.1] SKIP $OUT (exists)"
      continue
    fi
    echo "[H.1] === extracting cohort=$cohort seed=$seed ==="
    PYTHONIOENCODING=utf-8 PYTHONWARNINGS=ignore "$PY" "$SCRIPT" \
      --cohort "$cohort" --cell-type "B" --seed "$seed" \
      --output-tag "frozen_base_seed${seed}_alllayers"
  done
done

echo "[H.1] DONE — all 8 frozen B × {seed1,seed2} × 4 cohorts extracted"
