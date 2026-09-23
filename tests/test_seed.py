import os
import sys
import random
import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from trainer.engine import set_seed
from experiments.run import parse_args


def test_set_seed_reproducibility():
    set_seed(42)
    py_val1 = random.random()
    np_val1 = float(np.random.rand(1)[0])
    torch_val1 = float(torch.rand(1).item())

    # Generate next values to advance state
    _ = random.random()
    _ = np.random.rand(1)
    _ = torch.rand(1)

    # Re-seed with 42
    set_seed(42)
    py_val2 = random.random()
    np_val2 = float(np.random.rand(1)[0])
    torch_val2 = float(torch.rand(1).item())

    assert py_val1 == py_val2, f"random mismatch: {py_val1} vs {py_val2}"
    assert np_val1 == np_val2, f"numpy mismatch: {np_val1} vs {np_val2}"
    assert torch_val1 == torch_val2, f"torch mismatch: {torch_val1} vs {torch_val2}"
    print("[PASS] test_set_seed_reproducibility passed successfully.")


def test_cli_argument_parsing():
    import sys
    orig_argv = sys.argv

    try:
        # Test 1: positional config and --seed
        sys.argv = ["run.py", "config/test.yaml", "--seed", "123", "--device", "cuda:1", "--log_dir", "logs/test"]
        args = parse_args()
        assert args.config == "config/test.yaml"
        assert args.seed == 123
        assert args.device == "cuda:1"
        assert args.log_dir == "logs/test"

        # Test 2: --config flag with --seed
        sys.argv = ["run.py", "--config", "config/other.yaml", "--seed", "999"]
        args = parse_args()
        assert args.config_flag == "config/other.yaml"
        assert args.seed == 999

        # Test 3: default config when none given
        sys.argv = ["run.py", "--seed", "777"]
        args = parse_args()
        cfg_path = args.config_flag or args.config or "config/default.yaml"
        assert cfg_path == "config/default.yaml"
        assert args.seed == 777

        print("[PASS] test_cli_argument_parsing passed successfully.")
    finally:
        sys.argv = orig_argv


if __name__ == "__main__":
    test_set_seed_reproducibility()
    test_cli_argument_parsing()
    print("All seed tests passed!")
