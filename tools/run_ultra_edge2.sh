#!/usr/bin/env bash
set -euo pipefail

# Run all ultra-edge2 configs sequentially

echo "Running ultra-edge2 configs..."

for cfg in config/ultra-edge2/*.yaml; do
  echo ""
  echo "========================================"
  echo "Running $cfg"
  echo "========================================"
  echo ""
  python experiments/run.py "$cfg"
done

echo ""
echo "All ultra-edge2 configs completed!"
