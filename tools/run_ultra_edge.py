#!/usr/bin/env python3
"""
Run all ultra-edge experiments sequentially.
Executes cross-attention fusion with combined loss for mobile/edge student models.
"""

import subprocess
import sys
import argparse
from pathlib import Path
from datetime import datetime

# All ultra-edge configs
CONFIGS = [
    # Wound dataset - MobileViT-Small students
    "config/ultra-edge/wound-mobilevit_small-distilbert.yaml",
    "config/ultra-edge/wound-mobilevit_small-minilm.yaml",
    # Wound dataset - MobileViT-XXS students
    "config/ultra-edge/wound-mobilevit_xxs-distilbert.yaml",
    "config/ultra-edge/wound-mobilevit_xxs-minilm.yaml",
    # MedPix dataset - MobileViT-Small students
    "config/ultra-edge/medpix-mobilevit_small-distilbert.yaml",
    "config/ultra-edge/medpix-mobilevit_small-minilm.yaml",
    # MedPix dataset - MobileViT-XXS students
    "config/ultra-edge/medpix-mobilevit_xxs-distilbert.yaml",
    "config/ultra-edge/medpix-mobilevit_xxs-minilm.yaml",
]


def run_experiment(config_path, skip_on_failure=False):
    """Run a single experiment."""
    print(f"\n{'='*80}")
    print(f"Running: {config_path}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    try:
        result = subprocess.run(
            ["python", "experiments/run.py", config_path],
            check=True,
            capture_output=False
        )
        print(f"\n✓ Successfully completed: {config_path}\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Failed: {config_path}")
        print(f"Error code: {e.returncode}\n")
        if not skip_on_failure:
            response = input("Continue with remaining experiments? (y/n): ")
            if response.lower() != 'y':
                print("Stopping execution.")
                return False
        return True
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run all ultra-edge experiments (mobile student models)"
    )
    parser.add_argument(
        "--skip-failures",
        action="store_true",
        help="Skip failed experiments without prompting"
    )
    parser.add_argument(
        "--start-from",
        type=int,
        default=0,
        help="Start from experiment index (0-7)"
    )
    parser.add_argument(
        "--dataset",
        choices=["wound", "medpix", "all"],
        default="all",
        help="Run only specific dataset experiments"
    )
    parser.add_argument(
        "--student-vision",
        choices=["mobilevit-small", "mobilevit-xxs", "all"],
        default="all",
        help="Run only specific vision student experiments"
    )
    parser.add_argument(
        "--pause",
        action="store_true",
        help="Pause between experiments"
    )
    
    args = parser.parse_args()
    
    # Filter configs by dataset
    configs = CONFIGS
    if args.dataset == "wound":
        configs = [c for c in configs if "wound" in c]
    elif args.dataset == "medpix":
        configs = [c for c in configs if "medpix" in c]
    
    # Filter by student vision model
    if args.student_vision == "mobilevit-small":
        configs = [c for c in configs if "mobilevit_small" in c]
    elif args.student_vision == "mobilevit-xxs":
        configs = [c for c in configs if "mobilevit_xxs" in c]
    
    # Start from specific index
    if args.start_from > 0:
        configs = configs[args.start_from:]
        print(f"Starting from experiment {args.start_from}")
    
    print(f"\n{'='*80}")
    print(f"Ultra-Edge Experiment Runner")
    print(f"Teacher: ViT-Base + Bio-ClinicalBERT")
    print(f"Students: MobileViT variants + Lightweight text models")
    print(f"{'='*80}")
    print(f"Total experiments to run: {len(configs)}")
    print(f"Dataset filter: {args.dataset}")
    print(f"Vision student filter: {args.student_vision}")
    print(f"Skip failures: {args.skip_failures}")
    print(f"{'='*80}\n")
    
    # Track results
    successful = []
    failed = []
    start_time = datetime.now()
    
    for i, config in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] {config}")
        
        if args.pause and i > 1:
            input("Press Enter to continue to next experiment...")
        
        success = run_experiment(config, args.skip_failures)
        
        if success:
            successful.append(config)
        else:
            failed.append(config)
            if not args.skip_failures:
                break
    
    # Print summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"Total time: {duration}")
    print(f"Successful: {len(successful)}/{len(configs)}")
    print(f"Failed: {len(failed)}/{len(configs)}")
    
    if successful:
        print(f"\n✓ Successful experiments:")
        for config in successful:
            print(f"  - {config}")
    
    if failed:
        print(f"\n✗ Failed experiments:")
        for config in failed:
            print(f"  - {config}")
    
    print(f"\n{'='*80}\n")
    
    return 0 if len(failed) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
