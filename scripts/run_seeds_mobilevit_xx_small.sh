#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Run KD experiments for medpix and wound mobilevit_xx_small-bert-mini
# across seeds: 42, 43, 44, 45, 46.
#
# Usage:
#   ./scripts/run_seeds_mobilevit_xx_small.sh
#   DEVICE=cuda:0 ./scripts/run_seeds_mobilevit_xx_small.sh
#   FORCE=1 ./scripts/run_seeds_mobilevit_xx_small.sh     # Rerun even if results exist
#   SEEDS="42 43" ./scripts/run_seeds_mobilevit_xx_small.sh
# ==============================================================================

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Select Python binary (prefer active virtualenv or local environment)
if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN="$PYTHON"
elif [[ -x "/DATA1/rupesh_2421cs03/myenv/bin/python" ]]; then
  PYTHON_BIN="/DATA1/rupesh_2421cs03/myenv/bin/python"
else
  PYTHON_BIN="python"
fi

# Configurable options
FORCE="${FORCE:-0}"
SEED_SUBDIR="${SEED_SUBDIR:-1}"  # 1: save to <base_log_dir>/seed_<seed>; 0: use base log_dir
DEFAULT_SEEDS=(42 43 44 45 46)

if [[ -n "${SEEDS:-}" ]]; then
  # Read space-separated seeds from environment variable
  read -r -a SEED_LIST <<< "$SEEDS"
else
  SEED_LIST=("${DEFAULT_SEEDS[@]}")
fi

# Target experiment configs
CONFIGS=(
  "config/ultra-edge-hp-tuned-all/medpix-mobilevit_xx_small-bert-mini.yaml"
  "config/ultra-edge-hp-tuned-all/wound-mobilevit_xx_small-bert-mini.yaml"
)

echo "=========================================================="
echo "Starting multi-seed experiment runner"
echo "Python binary: $PYTHON_BIN"
echo "Seeds:         ${SEED_LIST[*]}"
echo "Configs (${#CONFIGS[@]}):"
for cfg in "${CONFIGS[@]}"; do
  echo "  - $cfg"
done
echo "=========================================================="

for cfg in "${CONFIGS[@]}"; do
  if [[ ! -f "$cfg" ]]; then
    echo "ERROR: Config file not found: $cfg"
    exit 1
  fi

  # Extract base log directory from YAML
  base_log_dir="$($PYTHON_BIN - "$cfg" <<'PY'
import sys
import yaml

cfg_path = sys.argv[1]
with open(cfg_path, 'r') as f:
    cfg = yaml.safe_load(f) or {}

log_dir = (cfg.get('logging', {}) or {}).get('log_dir')
if not log_dir:
    import os
    base = os.path.splitext(os.path.basename(cfg_path))[0]
    log_dir = os.path.join('logs', 'multi-seed', base)

print(log_dir)
PY
)"

  for seed in "${SEED_LIST[@]}"; do
    if [[ "$SEED_SUBDIR" == "1" ]]; then
      run_log_dir="${base_log_dir}/seed_${seed}"
    else
      run_log_dir="${base_log_dir}"
    fi

    echo ""
    echo "----------------------------------------------------------"
    echo "Config:  $cfg"
    echo "Seed:    $seed"
    echo "Log Dir: $run_log_dir"
    echo "----------------------------------------------------------"

    # Skip if completed and not forced
    if [[ "$FORCE" != "1" && -f "$run_log_dir/results.json" ]]; then
      echo "SKIP: $cfg [seed $seed] (found $run_log_dir/results.json)"
      continue
    fi

    # Build command arguments
    CMD=("$PYTHON_BIN" "experiments/run.py" "$cfg" "--seed" "$seed" "--log_dir" "$run_log_dir")
    if [[ -n "${DEVICE:-}" ]]; then
      CMD+=("--device" "$DEVICE")
    fi

    echo "Running: ${CMD[*]}"
    "${CMD[@]}"
    echo "Finished: $cfg [seed $seed]"
  done
done

echo ""
echo "=========================================================="
echo "All multi-seed experiments completed successfully."
echo "=========================================================="
