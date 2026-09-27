#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Single-Phase Joint Training & Hardware Profiling Launcher
#
# Compares Two-Phase Knowledge Distillation vs. Single-Phase Joint Training
# on MedPix and Wound datasets across seeds.
#
# Usage:
#   ./scripts/run_joint_training.sh --dataset medpix --seed 42
#   ./scripts/run_joint_training.sh --dataset wound --seed 42
#   ./scripts/run_joint_training.sh --dataset all --seeds 42 43 44 45 46
#
# Overrides:
#   DEVICE=cuda:0 ./scripts/run_joint_training.sh --dataset all
# ==============================================================================

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN="$PYTHON"
elif [[ -x "/DATA1/rupesh_2421cs03/myenv/bin/python" ]]; then
  PYTHON_BIN="/DATA1/rupesh_2421cs03/myenv/bin/python"
else
  PYTHON_BIN="python"
fi

DEVICE="${DEVICE:-cuda:0}"

echo "=========================================================="
echo "Single-Phase Joint Training Engine"
echo "Python binary: $PYTHON_BIN"
echo "Device:        $DEVICE"
echo "=========================================================="

"$PYTHON_BIN" experiments/run_joint_training.py \
  --device "$DEVICE" \
  "$@"
