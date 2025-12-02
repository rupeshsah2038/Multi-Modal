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

if __name__ == "__main__":
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config/default.yaml"
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    main(cfg)
