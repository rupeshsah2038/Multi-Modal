#!/usr/bin/env python3
"""
Run all ultra-edge2-tuned-hp configs sequentially.

Usage:
    python tools/run_ultra_edge2_tuned_hp.py
    python tools/run_ultra_edge2_tuned_hp.py --dataset medpix
    python tools/run_ultra_edge2_tuned_hp.py --dataset wound
    python tools/run_ultra_edge2_tuned_hp.py --filter mobilevit_xxs
"""
import os
import sys
import argparse
from pathlib import Path
import yaml

# Ensure repo root in path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from trainer.engine import main as run_main


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def run_all(dataset_filter=None, name_filter=None):
    cfg_dir = Path(ROOT) / "config" / "ultra-edge2-tuned-hp"
    yaml_files = sorted(cfg_dir.glob("*.yaml"))
    
    # Filter by dataset if specified
    if dataset_filter:
        yaml_files = [f for f in yaml_files if f.name.startswith(dataset_filter)]
    
    # Filter by name pattern if specified
    if name_filter:
        yaml_files = [f for f in yaml_files if name_filter in f.name]
    
    if not yaml_files:
        print(f"No configs found in {cfg_dir} matching filters")
        print(f"  Dataset filter: {dataset_filter}")
        print(f"  Name filter: {name_filter}")
        return
    
    print(f"\n{'='*80}")
    print(f"Ultra-Edge2 Tuned-HP Experiment Runner")
    print(f"{'='*80}")
    print(f"Config directory: {cfg_dir}")
    print(f"Found {len(yaml_files)} configs to run")
    if dataset_filter:
        print(f"Dataset filter: {dataset_filter}")
    if name_filter:
        print(f"Name filter: {name_filter}")
    print(f"{'='*80}\n")
    
    # List all configs
    print("Configs to run:")
    for i, yf in enumerate(yaml_files, 1):
        cfg = load_config(yf)
        vision = cfg['student']['vision']
        text = cfg['student']['text']
        print(f"  {i}. {yf.name}")
        print(f"     Student: {vision} + {text}")
    print()
    
    # Run each config
    for i, yf in enumerate(yaml_files, 1):
        print(f"\n{'='*80}")
        print(f"[{i}/{len(yaml_files)}] Running {yf.name}")
        print(f"{'='*80}\n")
        
        try:
            cfg = load_config(yf)
            run_main(cfg)
            print(f"\n✓ Successfully completed: {yf.name}")
        except KeyboardInterrupt:
            print(f"\n⚠ Interrupted by user")
            sys.exit(1)
        except Exception as e:
            print(f"\n✗ Failed: {yf.name}")
            print(f"Error: {e}")
            print("\nContinue with next config? (y/n): ", end='')
            response = input().strip().lower()
            if response != 'y':
                sys.exit(1)
    
    print(f"\n{'='*80}")
    print(f"All experiments completed!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run all ultra-edge2-tuned-hp experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        choices=["medpix", "wound"],
        help="Filter by dataset (medpix or wound)",
    )
    parser.add_argument(
        "--filter",
        help="Filter configs by name pattern (e.g., 'mobilevit_xxs', 'bert_tiny')",
    )
    
    args = parser.parse_args()
    run_all(dataset_filter=args.dataset, name_filter=args.filter)
