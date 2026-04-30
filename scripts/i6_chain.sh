#!/usr/bin/env bash
# i6_chain.sh — wait for I.4 wrapper + ridge to finish, then launch I.6 FM extractions.
# Polls every 60s; survives Claude session swaps when launched with nohup ... &; disown.

cd /home/ubuntu/aging-lora

echo "[chain $(date -Iseconds)] waiting for I.4 wrapper..."
while pgrep -f "i4_cap100_3seed.sh" > /dev/null 2>&1; do
  sleep 60
done

echo "[chain $(date -Iseconds)] waiting for I.4 ridge readout (i4_3seed_ridge.py)..."
while pgrep -f "i4_3seed_ridge.py" > /dev/null 2>&1; do
  sleep 30
done

echo "[chain $(date -Iseconds)] I.4 done; launching I.6 FM extractions"
exec bash scripts/i6_fm_extractions.sh
