#!/usr/bin/env python3
"""
Run all fusion-explore experiments for both MedPix and Wound datasets.
Tests all 9 fusion modules with combined loss.
"""
import subprocess
import sys
from pathlib import Path

# All fusion modules to test
FUSION_MODULES = [
    "simple",
    "concat_mlp",
    "cross_attention",
    "gated",
    "transformer_concat",
    "modality_dropout",
    "film",
    "energy_aware_adaptive",
    "shomr",
]

DATASETS = ["medpix", "wound"]

def run_experiment(dataset, fusion_module):
    """Run a single fusion-explore experiment."""
    config_file = f"config/fusion-explore/{dataset}-{fusion_module}-combined.yaml"
    
    print(f"\n{'='*80}")
    print(f"Running: {dataset.upper()} - {fusion_module}")
    print(f"Config: {config_file}")
    print(f"{'='*80}\n")
    
    try:
        subprocess.run(
            ["python", "experiments/run.py", config_file],
            check=True
        )
        print(f"\n✓ Completed: {dataset}-{fusion_module}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Failed: {dataset}-{fusion_module}")
        print(f"Error: {e}")
        return False
    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user.")
        sys.exit(1)

def main():
    """Run all fusion-explore experiments."""
    config_dir = Path("config/fusion-explore")
    if not config_dir.exists():
        print(f"Error: Config directory {config_dir} not found!")
        sys.exit(1)
    
    total = len(DATASETS) * len(FUSION_MODULES)
    completed = 0
    failed = 0
    
    print(f"Starting fusion-explore experiments")
    print(f"Total experiments: {total}")
    print(f"Datasets: {', '.join(DATASETS)}")
    print(f"Fusion modules: {', '.join(FUSION_MODULES)}")
    
    failed_experiments = []
    
    for dataset in DATASETS:
        for fusion_module in FUSION_MODULES:
            success = run_experiment(dataset, fusion_module)
            if success:
                completed += 1
            else:
                failed += 1
                failed_experiments.append(f"{dataset}-{fusion_module}")
    
    # Summary
    print(f"\n{'='*80}")
    print("FUSION-EXPLORE EXPERIMENTS SUMMARY")
    print(f"{'='*80}")
    print(f"Total experiments: {total}")
    print(f"Completed: {completed}")
    print(f"Failed: {failed}")
    
    if failed_experiments:
        print(f"\nFailed experiments:")
        for exp in failed_experiments:
            print(f"  - {exp}")
    
    print(f"\nResults saved in: logs/fusion-explore/")
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
