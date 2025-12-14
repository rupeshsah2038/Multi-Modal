#!/usr/bin/env python3
"""
Run all ultra-edge2 configs sequentially.
"""
import os
import sys
import glob
import yaml
from pathlib import Path

# Ensure repo root in path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from trainer.engine import main as run_main


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def run_all():
    cfg_dir = Path(ROOT) / "config" / "ultra-edge2"
    yaml_files = sorted(cfg_dir.glob("*.yaml"))
    if not yaml_files:
        print(f"No configs found in {cfg_dir}")
        return
    print(f"Found {len(yaml_files)} ultra-edge2 configs. Running sequentially...")
    for yf in yaml_files:
        print(f"\n{'='*80}")
        print(f"Running {yf.name}")
        print(f"{'='*80}\n")
        cfg = load_config(yf)
        run_main(cfg)


if __name__ == "__main__":
    run_all()
