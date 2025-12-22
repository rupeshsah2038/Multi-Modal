#!/usr/bin/env python3
"""Run experiments from configured folders and optionally re-run failures.

Features:
- Scans `CONFIG_DIRS` for YAML configs and runs `experiments/run.py` for each
- Overrides `device` in the config to the requested device
- Records failures to `failed_runs.txt` with exit codes
- Supports automatic re-runs with `--rerun` and a dry-run mode

Usage examples:
  python tools/run_and_rerun.py --gpu cuda:0           # run once on cuda:0
  python tools/run_and_rerun.py --gpu cuda:0 --rerun 2  # run and retry failures up to 2 times
  python tools/run_and_rerun.py --dry                   # show commands only
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
from datetime import datetime
import yaml


CONFIG_DIRS = [
    "config/ultra-edge-base-256",
    "config/ultra-edge-base-384",
    "config/ultra-edge-hp-tuned-all",
]

DEFAULT_DEVICE = "cuda:0"
FAILED_FILE = "failed_runs.txt"
RUN_LOG_DIR = "logs/run_attempts"


def list_configs():
    configs = []
    for d in CONFIG_DIRS:
        configs += sorted(glob.glob(os.path.join(d, "*.yaml")))
    return configs


def get_log_dir_from_cfg(cfg_path):
    try:
        with open(cfg_path, 'r') as f:
            doc = yaml.safe_load(f)
        log_dir = (doc.get('logging') or {}).get('log_dir')
        if log_dir:
            return log_dir
    except Exception:
        pass
    # fallback
    name = os.path.splitext(os.path.basename(cfg_path))[0]
    return os.path.join('logs', name)


def run_config(cfg_path, device, dry=False, attempt=1):
    # create temp config overriding device
    with open(cfg_path, 'r') as f:
        lines = f.readlines()
    new_lines = []
    found_device = False
    for line in lines:
        if line.strip().startswith('device:'):
            new_lines.append(f'device: "{device}"\n')
            found_device = True
        else:
            new_lines.append(line)
    if not found_device:
        new_lines.append(f"\ndevice: \"{device}\"\n")

    tmp_cfg = cfg_path + f".tmp.{attempt}.yaml"
    with open(tmp_cfg, 'w') as f:
        f.writelines(new_lines)

    cmd = [sys.executable, 'experiments/run.py', tmp_cfg]
    display_cmd = ' '.join(cmd)

    # prepare run log
    os.makedirs(RUN_LOG_DIR, exist_ok=True)
    base = os.path.splitext(os.path.basename(cfg_path))[0]
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_file = os.path.join(RUN_LOG_DIR, f"{base}.{attempt}.{timestamp}.log")

    print(f"Running: {display_cmd}")
    if dry:
        print(f"Dry run: would write temporary config {tmp_cfg} and log to {log_file}")
        os.remove(tmp_cfg)
        return 0

    with open(log_file, 'wb') as lf:
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
    ret = proc.returncode

    # cleanup temp config
    try:
        os.remove(tmp_cfg)
    except Exception:
        pass

    return ret


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, help='GPU device string (e.g. cuda:0)')
    parser.add_argument('--cpu', action='store_true', help='Run on CPU')
    parser.add_argument('--dry', action='store_true', help='Print commands but do not execute')
    parser.add_argument('--rerun', type=int, default=0, help='Number of times to retry failed runs')
    parser.add_argument('--failed-file', type=str, default=FAILED_FILE, help='Path to write failed runs list')
    args = parser.parse_args()

    device = 'cpu' if args.cpu else (args.gpu or DEFAULT_DEVICE)

    configs = list_configs()
    if not configs:
        print('No configs found in CONFIG_DIRS')
        return

    failed = {}

    # First pass
    for cfg in configs:
        ret = run_config(cfg, device, dry=args.dry, attempt=1)
        if ret != 0:
            failed[cfg] = ret

    # retries
    if args.rerun > 0 and failed and not args.dry:
        for attempt in range(2, args.rerun + 2):
            if not failed:
                break
            print(f"Retry attempt {attempt-1} for {len(failed)} failed configs")
            to_retry = list(failed.keys())
            failed = {}
            for cfg in to_retry:
                ret = run_config(cfg, device, dry=args.dry, attempt=attempt)
                if ret != 0:
                    failed[cfg] = ret

    # Write failed file
    with open(args.failed_file, 'w') as f:
        for cfg, code in failed.items():
            log_dir = get_log_dir_from_cfg(cfg)
            f.write(f"{cfg}\texit_code={code}\tlog_dir={log_dir}\n")

    print(f"Done. Failed runs: {len(failed)}. See {args.failed_file} and {RUN_LOG_DIR} for details.")


if __name__ == '__main__':
    main()
