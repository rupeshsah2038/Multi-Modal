import os
import sys
import glob
import subprocess

CONFIG_DIRS = [
    "config/loss-explore-hp-wound",
    #"config/fusion-explore-hp",
    #"config/fusion-explore-hp-wound",
    #"config/loss-explore-hp"
    #"config/ultra-edge-hp-tuned-all",
    #"config/ultra-edge-base-384",
]

DEFAULT_GPU = "cuda:0"

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run all configs in specified folders with optional GPU override.")
    parser.add_argument('--gpu', type=str, default=DEFAULT_GPU, help='GPU id to override (e.g., cuda:1)')
    parser.add_argument('--dry', action='store_true', help='Print commands without running')
    args = parser.parse_args()

    for config_dir in CONFIG_DIRS:
        config_files = sorted(glob.glob(os.path.join(config_dir, '*.yaml')))
        for cfg in config_files:
            # Read config and override device
            with open(cfg, 'r') as f:
                lines = f.readlines()
            new_lines = []
            found_device = False
            for line in lines:
                if line.strip().startswith('device:'):
                    new_lines.append(f'device: "{args.gpu}"\n')
                    found_device = True
                else:
                    new_lines.append(line)
            if not found_device:
                new_lines.append(f'device: "{args.gpu}"\n')
            # Write to a temp config
            temp_cfg = cfg + ".tmp.yaml"
            with open(temp_cfg, 'w') as f:
                f.writelines(new_lines)
            cmd = [
                sys.executable, 'experiments/run.py', temp_cfg
            ]
            print(f"Running: {' '.join(cmd)}")
            if not args.dry:
                subprocess.run(cmd)
            os.remove(temp_cfg)
