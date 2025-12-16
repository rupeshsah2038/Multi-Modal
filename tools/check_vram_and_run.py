#!/usr/bin/env python3
"""Check GPU VRAM availability and conditionally run a command.

This script is designed for a multi-GPU system (e.g., 5 GPUs: 0–4).
It checks which GPUs have at least a requested amount of *free* VRAM,
then, if one is available, runs a user-specified command on that GPU.

Usage examples (from repo root):

  # Just print which GPU (if any) has >= 20GB free
  python tools/check_vram_and_run.py --min-gb 20 --gpus 0,1,2,3,4

  # If a suitable GPU is found, run an experiment on it
  python tools/check_vram_and_run.py --min-gb 20 --gpus 0,1,2,3,4 \
      -- python experiments/run.py config/default.yaml

Notes:
- By default this uses `nvidia-smi` if available, and falls back to
  `torch.cuda.mem_get_info` otherwise.
- When running the command, it sets `CUDA_VISIBLE_DEVICES` to the
  chosen physical GPU index so that your configs can still use
  `device: "cuda:0"`.
"""

import argparse
import os
import shutil
import subprocess
from typing import Dict, List, Optional


def _parse_gpu_list(gpu_str: str) -> List[int]:
    return [int(x.strip()) for x in gpu_str.split(',') if x.strip() != ""]


def query_vram_nvidia_smi(device_indices: List[int]) -> Optional[Dict[int, float]]:
    """Query free VRAM (in GB) per GPU using nvidia-smi.

    Returns a dict {gpu_index: free_gb} or None if nvidia-smi is not available.
    """
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None

    try:
        # Query index + free memory (MiB) without units/header for easy parsing
        cmd = [
            exe,
            "--query-gpu=index,memory.free",
            "--format=csv,noheader,nounits",
        ]
        out = subprocess.check_output(cmd, encoding="utf-8")
    except Exception:
        return None

    free_by_idx: Dict[int, float] = {}
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(',')]
        if len(parts) != 2:
            continue
        try:
            idx = int(parts[0])
            free_mib = float(parts[1])
        except ValueError:
            continue
        if idx in device_indices:
            free_by_idx[idx] = free_mib / 1024.0  # MiB → GiB approx
    return free_by_idx


def query_vram_torch(device_indices: List[int]) -> Optional[Dict[int, float]]:
    """Fallback VRAM query using torch.cuda.mem_get_info, if available.

    Returns a dict {gpu_index: free_gb} or None if torch/cuda is unavailable.
    """
    try:
        import torch  # type: ignore
    except Exception:
        return None

    if not torch.cuda.is_available():
        return None

    free_by_idx: Dict[int, float] = {}
    for idx in device_indices:
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(idx)
        except Exception:
            # Invalid index or driver issue
            continue
        free_by_idx[idx] = free_bytes / (1024.0 ** 3)  # bytes → GiB
    return free_by_idx


def pick_gpu_with_min_vram(device_indices: List[int], min_gb: float) -> Optional[int]:
    """Return the GPU index with >= min_gb free VRAM (preferring the most free)."""
    free_by_idx = query_vram_nvidia_smi(device_indices)
    if free_by_idx is None:
        free_by_idx = query_vram_torch(device_indices)
    if not free_by_idx:
        print("[check_vram_and_run] Could not query GPU memory (no nvidia-smi or torch.cuda).")
        return None

    candidates = [(idx, free) for idx, free in free_by_idx.items() if free >= min_gb]
    if not candidates:
        print("[check_vram_and_run] No GPU found with >= %.1f GB free VRAM." % min_gb)
        for idx, free in sorted(free_by_idx.items()):
            print(f"  GPU {idx}: {free:.2f} GB free")
        return None

    # Pick GPU with the most free memory among those that satisfy the constraint
    candidates.sort(key=lambda x: x[1], reverse=True)
    chosen_idx, chosen_free = candidates[0]
    print(f"[check_vram_and_run] Selected GPU {chosen_idx} with {chosen_free:.2f} GB free (>= {min_gb} GB).")
    return chosen_idx


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check GPU VRAM and optionally run a command on a suitable GPU.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--min-gb",
        type=float,
        default=20.0,
        help="Minimum free VRAM (GB) required to run the command.",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0,1,2,3,4",
        help="Comma-separated list of GPU indices to consider (physical IDs).",
    )
    # Everything after "--" is treated as the command to run
    parser.add_argument(
        "cmd",
        nargs=argparse.REMAINDER,
        help="Command to run if a suitable GPU is found (prefix with --).",
    )

    args = parser.parse_args()
    gpu_list = _parse_gpu_list(args.gpus)
    if not gpu_list:
        print("[check_vram_and_run] No GPU indices provided via --gpus.")
        raise SystemExit(1)

    chosen_gpu = pick_gpu_with_min_vram(gpu_list, args.min_gb)
    if chosen_gpu is None:
        # Nothing to run
        raise SystemExit(1)

    if not args.cmd:
        # Just report success and exit
        print("[check_vram_and_run] Suitable GPU found, but no command provided. Nothing to run.")
        return

    # Drop leading "--" if present in the remainder
    cmd = args.cmd
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        print("[check_vram_and_run] Command list is empty after '--'. Nothing to run.")
        return

    # Prepare environment: bind the chosen physical GPU as CUDA_VISIBLE_DEVICES=idx
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(chosen_gpu)

    print("[check_vram_and_run] Running command on GPU %s: %s" % (chosen_gpu, " ".join(cmd)))
    try:
        subprocess.run(cmd, env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[check_vram_and_run] Command failed with return code {e.returncode}.")
        raise SystemExit(e.returncode)


if __name__ == "__main__":
    main()
