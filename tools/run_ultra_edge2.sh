#!/usr/bin/env bash
set -euo pipefail

# Run all ultra-edge-hp-tuned-all configs sequentially

echo "Running ultra-edge-hp-tuned-all configs..."

for cfg in config/ultra-edge-hp-tuned-all/*.yaml; do
  echo ""
  echo "========================================"
  echo "Running $cfg"
  echo "========================================"
  echo ""
  python experiments/run.py "$cfg"
done

echo ""
echo "All ultra-edge-hp-tuned-all configs completed!"
