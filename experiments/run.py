import argparse
import os
import sys
import yaml

# Ensure the repository root is on sys.path when this module is executed as a
# script (e.g. `python experiments/run.py`). When Python executes a script the
# script's containing directory is placed at sys.path[0] which means sibling
# packages (like `trainer`) are not importable. Add the repository root so
# absolute imports work reliably from the project root and when running the
# script directly.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from trainer.engine import main


def parse_args():
    parser = argparse.ArgumentParser(description="Run Multi-Modal Federated KD Experiment")
    parser.add_argument(
        "config",
        nargs="?",
        default=None,
        help="Path to YAML configuration file (default: config/default.yaml)",
    )
    parser.add_argument(
        "--config",
        dest="config_flag",
        default=None,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility across random, numpy, and torch",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override compute device (e.g. 'cuda:0', 'cpu')",
    )
    parser.add_argument(
        "--log_dir",
        "--log-dir",
        dest="log_dir",
        type=str,
        default=None,
        help="Override logging.log_dir",
    )
    parser.add_argument(
        "--teacher_checkpoint",
        "--teacher-checkpoint",
        type=str,
        default=None,
        help="Path to pre-trained teacher checkpoint (.pth file)",
    )
    parser.add_argument(
        "--force_retrain_teacher",
        "--force-retrain-teacher",
        action="store_true",
        default=False,
        help="Force retraining teacher even if checkpoint exists",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg_path = args.config_flag or args.config or "config/default.yaml"

    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f) or {}

    if args.seed is not None:
        cfg['seed'] = args.seed
    if args.device is not None:
        cfg['device'] = args.device
    if args.log_dir is not None:
        if 'logging' not in cfg or not isinstance(cfg['logging'], dict):
            cfg['logging'] = {}
        cfg['logging']['log_dir'] = args.log_dir
    if args.teacher_checkpoint is not None:
        if 'training' not in cfg or not isinstance(cfg['training'], dict):
            cfg['training'] = {}
        cfg['training']['teacher_checkpoint'] = args.teacher_checkpoint
    if args.force_retrain_teacher:
        if 'training' not in cfg or not isinstance(cfg['training'], dict):
            cfg['training'] = {}
        cfg['training']['force_retrain_teacher'] = True

    main(cfg)

