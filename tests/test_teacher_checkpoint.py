import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.run import parse_args


def test_cli_teacher_flags():
    orig_argv = sys.argv
    try:
        sys.argv = [
            "run.py", "config/default.yaml",
            "--teacher-checkpoint", "some/path/teacher.pth",
            "--force-retrain-teacher"
        ]
        args = parse_args()
        assert args.teacher_checkpoint == "some/path/teacher.pth"
        assert args.force_retrain_teacher is True
        print("[PASS] test_cli_teacher_flags passed successfully.")
    finally:
        sys.argv = orig_argv


def test_checkpoint_paths_exist():
    for dataset in ["medpix", "wound"]:
        ckpt_path = os.path.join("logs/ultra-edge-hp-tuned-all/teacher-only", dataset, "teacher_final.pth")
        assert os.path.exists(ckpt_path), f"Checkpoint missing: {ckpt_path}"
        assert os.path.getsize(ckpt_path) > 100_000_000, f"Checkpoint unexpectedly small: {ckpt_path}"
        print(f"[PASS] Verified {dataset} checkpoint at {ckpt_path} ({os.path.getsize(ckpt_path):,} bytes).")


if __name__ == "__main__":
    test_cli_teacher_flags()
    test_checkpoint_paths_exist()
    print("All teacher checkpoint tests passed!")
