import yaml
import sys
from trainer.engine import main

if __name__ == "__main__":
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config/default.yaml"
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    main(cfg)
