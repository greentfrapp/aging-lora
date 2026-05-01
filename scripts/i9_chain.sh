#!/usr/bin/env bash
# i9_chain.sh — wait for I.8 FM extractions to finish, then launch I.9.
# Survives Claude session swaps when started with nohup ... &; disown.

cd /home/ubuntu/aging-lora

echo "[chain $(date -Iseconds)] waiting for I.8 wrapper (i8_extractions.sh)..."
while pgrep -f "i8_extractions.sh" > /dev/null 2>&1; do
  sleep 60
done

echo "[chain $(date -Iseconds)] waiting for I.8 ridge readout (i6_combined_ridge.py from I.8 chain)..."
# i6_combined_ridge runs at the end of i8_extractions.sh; brief wait if any in flight
while pgrep -f "i6_combined_ridge.py" > /dev/null 2>&1; do
  sleep 30
done

echo "[chain $(date -Iseconds)] I.8 done; launching I.9 FM extractions"
exec bash scripts/i9_extractions.sh
