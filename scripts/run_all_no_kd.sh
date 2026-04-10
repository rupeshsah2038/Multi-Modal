#!/usr/bin/env bash
set -euo pipefail

# Runs CE-only (No KD) student training for all configs in config/no-kd/.
# Default behavior: skip configs that already have <log_dir>/results.json.
#
# Usage:
#   ./scripts/run_all_no_kd.sh
#   DEVICE=cuda:0 SEED=0 ./scripts/run_all_no_kd.sh
#   CONFIG_DIR=config/no-kd DEVICE=cpu ./scripts/run_all_no_kd.sh
#   FORCE=1 ./scripts/run_all_no_kd.sh   # rerun even if results exist

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON:-python}"
CONFIG_DIR="${CONFIG_DIR:-config/no-kd}"
FORCE="${FORCE:-0}"

EXTRA_ARGS=()
if [[ -n "${DEVICE:-}" ]]; then
  EXTRA_ARGS+=(--device "$DEVICE")
fi
if [[ -n "${SEED:-}" ]]; then
  EXTRA_ARGS+=(--seed "$SEED")
fi

shopt -s nullglob
configs=("$CONFIG_DIR"/*.yaml)
shopt -u nullglob

if [[ ${#configs[@]} -eq 0 ]]; then
  echo "No config files found in $CONFIG_DIR"
  exit 1
fi

for cfg in "${configs[@]}"; do
  log_dir="$($PYTHON_BIN - "$cfg" <<'PY'
import os
import sys
import yaml

cfg_path = sys.argv[1]
with open(cfg_path, 'r') as f:
    cfg = yaml.safe_load(f)

log_dir = (cfg.get('logging', {}) or {}).get('log_dir')
if not log_dir:
    base = os.path.splitext(os.path.basename(cfg_path))[0]
    log_dir = os.path.join('logs', 'no-kd', base)

print(log_dir)
PY
)"

  if [[ "$FORCE" != "1" && -f "$log_dir/results.json" ]]; then
    echo "SKIP  $cfg (found $log_dir/results.json)"
    continue
  fi

  echo "RUN   $cfg"
  echo "      -> $log_dir"
  "$PYTHON_BIN" experiments/run_no_kd.py "$cfg" "${EXTRA_ARGS[@]}"
done

echo "Done."