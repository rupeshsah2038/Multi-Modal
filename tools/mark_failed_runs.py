#!/usr/bin/env python3
"""
Scan config folders and mark experiments whose outputs are missing.

Produces `failed_runs.txt` with one config path per line.
"""
import os
import glob
import yaml

CONFIG_DIRS = [
    "config/ultra-edge-base-256",
    "config/ultra-edge-base-384",
    "config/ultra-edge-hp-tuned-all",
]

OUTFILE = "failed_runs.txt"

def get_log_dir(cfg, cfg_path):
    log_cfg = cfg.get('logging', {}) or {}
    log_dir = log_cfg.get('log_dir')
    if log_dir:
        return log_dir
    # fallback: logs/<config_basename>
    name = os.path.splitext(os.path.basename(cfg_path))[0]
    return os.path.join('logs', name)

def main():
    failed = []
    checked = 0
    for d in CONFIG_DIRS:
        for cfg in sorted(glob.glob(os.path.join(d, '*.yaml'))):
            checked += 1
            try:
                with open(cfg, 'r') as f:
                    doc = yaml.safe_load(f)
            except Exception as e:
                failed.append((cfg, f'yaml_load_error: {e}'))
                continue
            data = doc.get('data', {}) or {}
            dtype = data.get('type')
            log_dir = get_log_dir(doc, cfg)
            results_path = os.path.join(log_dir, 'results.json')
            if not os.path.exists(results_path):
                failed.append((cfg, f'missing results.json at {results_path}'))

    with open(OUTFILE, 'w') as f:
        for cfg, reason in failed:
            f.write(f"{cfg}\t{reason}\n")

    print(f'Checked {checked} configs, failed: {len(failed)}. See {OUTFILE}')

if __name__ == '__main__':
    main()
