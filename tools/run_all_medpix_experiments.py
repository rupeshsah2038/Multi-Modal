#!/usr/bin/env python3
"""
Script to run all MedPix configuration experiments sequentially.
Usage: python tools/run_all_medpix_experiments.py [--skip-failures]
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
import time

# Color codes for terminal output
class Colors:
    GREEN = '\033[0;32m'
    BLUE = '\033[0;34m'
    RED = '\033[0;31m'
    YELLOW = '\033[1;33m'
    CYAN = '\033[0;36m'
    NC = '\033[0m'  # No Color

def print_colored(message, color):
    """Print colored message."""
    print(f"{color}{message}{Colors.NC}")

def print_header(message):
    """Print a header with separator."""
    sep = "=" * 70
    print_colored(f"\n{sep}", Colors.BLUE)
    print_colored(f"{message:^70}", Colors.BLUE)
    print_colored(f"{sep}\n", Colors.BLUE)

def print_section(message):
    """Print a section separator."""
    sep = "-" * 70
    print_colored(sep, Colors.YELLOW)
    print_colored(message, Colors.YELLOW)
    print_colored(sep, Colors.YELLOW)

def run_experiment(config_path, experiment_num, total_experiments):
    """Run a single experiment."""
    config_name = config_path.name
    
    print_section(f"Experiment {experiment_num}/{total_experiments}: {config_name}")
    print(f"Config path: {config_path}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check if config exists
    if not config_path.exists():
        print_colored(f"✗ Config file not found: {config_name}", Colors.RED)
        return False, "not found"
    
    # Run the experiment
    try:
        result = subprocess.run(
            ["python", "experiments/run.py", str(config_path)],
            check=True,
            capture_output=False,  # Show output in real-time
            text=True
        )
        
        print()
        print_colored(f"✓ Experiment {experiment_num} completed successfully: {config_name}", Colors.GREEN)
        return True, None
        
    except subprocess.CalledProcessError as e:
        print()
        print_colored(f"✗ Experiment {experiment_num} failed: {config_name}", Colors.RED)
        return False, f"exit code {e.returncode}"
    except KeyboardInterrupt:
        print()
        print_colored("✗ Experiment interrupted by user", Colors.RED)
        return False, "interrupted"
    except Exception as e:
        print()
        print_colored(f"✗ Experiment failed with exception: {str(e)}", Colors.RED)
        return False, str(e)

def main():
    parser = argparse.ArgumentParser(
        description="Run all MedPix configuration experiments sequentially"
    )
    parser.add_argument(
        "--skip-failures",
        action="store_true",
        help="Continue running experiments even if some fail"
    )
    parser.add_argument(
        "--start-from",
        type=int,
        default=1,
        help="Start from experiment number N (1-indexed)"
    )
    parser.add_argument(
        "--pause",
        type=int,
        default=3,
        help="Seconds to pause between experiments (default: 3)"
    )
    args = parser.parse_args()
    
    # Configuration
    base_dir = Path("/home/rupesh_2421cs03/projects/Federated-KD/Medpix_modular")
    config_dir = base_dir / "config"
    
    # Get all MedPix config files
    configs = [
        "medpix-simple-combined.yaml",
        "medpix-concat_mlp-combined.yaml",
        "medpix-cross_attention-combined.yaml",
        "medpix-gated-combined.yaml",
        "medpix-transformer_concat-combined.yaml",
        "medpix-modality_dropout-combined.yaml",
        "medpix-film-combined.yaml",
        "medpix-energy_aware-combined.yaml",
        "medpix-shomr-combined.yaml",
    ]
    
    # Apply start-from filter
    if args.start_from > 1:
        configs = configs[args.start_from - 1:]
        print_colored(f"Starting from experiment {args.start_from}", Colors.CYAN)
        print()
    
    # Change to base directory
    os.chdir(base_dir)
    
    # Track results
    success_count = 0
    fail_count = 0
    failed_experiments = []
    
    # Print header
    print_header("MedPix Fusion Module Comparison Experiments")
    print(f"Total experiments to run: {len(configs)}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Run each experiment
    total_experiments = len(configs)
    start_offset = args.start_from - 1
    
    for i, config_file in enumerate(configs, start=1):
        config_path = config_dir / config_file
        experiment_num = i + start_offset
        
        success, error = run_experiment(config_path, experiment_num, 
                                       total_experiments + start_offset)
        
        print()
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        if success:
            success_count += 1
        else:
            fail_count += 1
            failed_experiments.append((config_file, error))
            
            if not args.skip_failures and error != "interrupted":
                print_colored("Experiment failed. Stop here or continue?", Colors.YELLOW)
                response = input("Continue with remaining experiments? (y/n): ").strip().lower()
                if response != 'y':
                    print_colored("Stopping experiments as requested.", Colors.YELLOW)
                    break
            elif error == "interrupted":
                print_colored("Stopping due to keyboard interrupt.", Colors.YELLOW)
                break
        
        # Pause between experiments
        if i < total_experiments:
            print(f"Waiting {args.pause} seconds before next experiment...")
            time.sleep(args.pause)
    
    # Print summary
    print_header("EXPERIMENT SUMMARY")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print(f"Total experiments: {total_experiments + start_offset}")
    print_colored(f"Successful: {success_count}", Colors.GREEN)
    print_colored(f"Failed: {fail_count}", Colors.RED)
    print()
    
    if failed_experiments:
        print_colored("Failed experiments:", Colors.RED)
        for config_name, error in failed_experiments:
            print(f"  - {config_name} ({error})")
        print()
    
    # List fusion modules
    print("Fusion modules tested:")
    modules = [
        "simple", "concat_mlp", "cross_attention", "gated",
        "transformer_concat", "modality_dropout", "film",
        "energy_aware_adaptive", "shomr"
    ]
    for idx, module in enumerate(modules, start=1):
        print(f"  {idx}. {module}")
    print()
    
    print("Results are saved in individual log directories:")
    print("  logs/medpix-vit-base-512-{fusion_module}-combined/")
    print()
    print_header("")
    
    # Exit with appropriate code
    sys.exit(1 if fail_count > 0 else 0)

if __name__ == "__main__":
    main()
