#!/usr/bin/env python3
"""Batch-run wrapper to swap vision/text models and run experiments.

Creates a per-run config under `logs/<run_name>/config.yaml` and calls
`python experiments/run.py <config>`.

Supported run names:
  - original: use base config as-is
  - swap_vision: swap teacher and student vision backbones
  - swap_text: swap teacher and student text backbones
  - swap_both: swap both vision and text backbones
  - mobile-edge: student with tiny-vit + mobile-bert (edge-friendly)
  - ultra-edge: student with tiny-vit + bert-tiny (ultra-compact)
  - edge-vision: student with tiny-vit vision backbone
  - edge-text: student with mobile-bert text backbone

Usage examples:
  # dry-run: create configs only
  python tools/batch_runs.py --base config/default.yaml --runs original,swap_vision

  # execute with overrides (safe defaults for quick smoke)
  python tools/batch_runs.py --base config/default.yaml --runs swap_vision \
      --execute --epochs 1 --batch-size 8

  # test edge-friendly student backbones
  python tools/batch_runs.py --base config/default.yaml --runs original,mobile-edge,ultra-edge \
      --execute --epochs 5 --batch-size 16 --device cuda:0

The script writes per-run logs to `logs/<run_name>/` (set as `logging.log_dir`).
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path
import yaml
import uuid


def load_yaml(path):
    with open(path, 'r') as fh:
        return yaml.safe_load(fh)


def write_yaml(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as fh:
        yaml.safe_dump(obj, fh)


def make_run_configs(base_cfg_path, runs, epochs=None, batch_size=None, device=None):
    base = load_yaml(base_cfg_path)
    created = []
    run_id = str(uuid.uuid4())[:8]
    for name in runs:
        cfg = dict(base)
        run_dir = Path('logs') / f'run_{run_id}' / name
        cfg.setdefault('logging', {})['log_dir'] = str(run_dir)
        if device is not None:
            cfg['device'] = device
        # optional overrides
        if epochs is not None:
            cfg.setdefault('training', {})['student_epochs'] = int(epochs)
        if batch_size is not None:
            cfg.setdefault('data', {})['batch_size'] = int(batch_size)

        # swap behaviors and edge-friendly presets
        if name == 'original':
            pass
        elif name == 'swap_vision':
            tvis = cfg['teacher'].get('vision')
            svis = cfg['student'].get('vision')
            cfg['teacher']['vision'], cfg['student']['vision'] = svis, tvis
        elif name == 'swap_text':
            ttxt = cfg['teacher'].get('text')
            stxt = cfg['student'].get('text')
            cfg['teacher']['text'], cfg['student']['text'] = stxt, ttxt
        elif name == 'swap_both':
            # swap both vision and text
            tvis = cfg['teacher'].get('vision')
            svis = cfg['student'].get('vision')
            ttxt = cfg['teacher'].get('text')
            stxt = cfg['student'].get('text')
            cfg['teacher']['vision'], cfg['student']['vision'] = svis, tvis
            cfg['teacher']['text'], cfg['student']['text'] = stxt, ttxt
        # edge-friendly student presets (teacher unchanged)
        elif name == 'mobile-edge':
            # compact vision + compact text for mobile/edge devices
            cfg['student']['vision'] = 'tiny-vit'
            cfg['student']['text'] = 'mobile-bert'
        elif name == 'ultra-edge':
            # ultra-compact for resource-constrained edge (smallest models)
            cfg['student']['vision'] = 'tiny-vit'
            cfg['student']['text'] = 'bert-tiny'
        elif name == 'edge-vision':
            # compact vision only; keep current text
            cfg['student']['vision'] = 'tiny-vit'
        elif name == 'edge-text':
            # compact text only; keep current vision
            cfg['student']['text'] = 'mobile-bert'
        else:
            raise ValueError(f'Unknown run name: {name}. Supported: original, swap_vision, swap_text, swap_both, mobile-edge, ultra-edge, edge-vision, edge-text')

        cfg_path = run_dir / 'config.yaml'
        write_yaml(cfg, cfg_path)
        created.append((name, cfg_path, run_dir))
    return created


def execute_run(cfg_path):
    cmd = [sys.executable, 'experiments/run.py', str(cfg_path)]
    print('Running:', ' '.join(cmd))
    return subprocess.run(cmd)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', required=True, help='Base config yaml')
    p.add_argument('--runs', required=True, help='Comma-separated run names: original, swap_vision, swap_text, swap_both, mobile-edge, ultra-edge, edge-vision, edge-text')
    p.add_argument('--execute', action='store_true', help='Actually run experiments')
    p.add_argument('--epochs', type=int, help='Override student_epochs for quick runs')
    p.add_argument('--batch-size', type=int, help='Override data.batch_size')
    p.add_argument('--device', type=str, help="Device string to set in config, e.g. 'cuda:3' or 'cpu'")
    args = p.parse_args()

    runs = [r.strip() for r in args.runs.split(',') if r.strip()]
    created = make_run_configs(args.base, runs, epochs=args.epochs, batch_size=args.batch_size, device=args.device)

    for name, cfg_path, run_dir in created:
        print(f'Created config for run "{name}": {cfg_path}')
        if args.execute:
            run_dir.mkdir(parents=True, exist_ok=True)
            res = execute_run(cfg_path)
            if res.returncode != 0:
                print(f'Run {name} exited with code {res.returncode}', file=sys.stderr)
                return res.returncode

    return 0


if __name__ == '__main__':
    sys.exit(main())
