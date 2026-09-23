#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Run Ablation and Robustness Suite for MobileViT-xxs + BERT-mini
# on MedPix and Wound datasets across seeds 42, 43, 44, 45, 46.
#
# Settings:
#   - Unimodal Baselines (Trained from scratch): Image-Only, Text-Only
#   - Robustness Perturbations (Trained Multimodal Models): Mismatch-Text, Noise, Missing (30%)
#
# Usage:
#   ./scripts/run_ablations.sh --eval-robustness
#   ./scripts/run_ablations.sh --train-unimodal
#   ./scripts/run_ablations.sh --all
#
# Options / Overrides:
#   DEVICE=cuda:0 ./scripts/run_ablations.sh --eval-robustness
#   SEEDS="42 43 44 45 46" ./scripts/run_ablations.sh --eval-robustness
#   FORCE=1 ./scripts/run_ablations.sh --train-unimodal
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
FORCE="${FORCE:-0}"
DEFAULT_SEEDS="42 43 44 45 46"
SEEDS="${SEEDS:-$DEFAULT_SEEDS}"

EXTRA_ARGS=()
if [[ "$FORCE" == "1" ]]; then
  EXTRA_ARGS+=(--force)
fi

echo "=========================================================="
echo "Multimodal Ablation Suite: MobileViT-xxs + BERT-mini"
echo "Python binary: $PYTHON_BIN"
echo "Device:        $DEVICE"
echo "Seeds:         $SEEDS"
echo "=========================================================="

"$PYTHON_BIN" experiments/run_ablation.py \
  --device "$DEVICE" \
  --seeds $SEEDS \
  "${EXTRA_ARGS[@]}" \
  "$@"
