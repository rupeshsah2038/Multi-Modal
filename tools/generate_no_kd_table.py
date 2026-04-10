#!/usr/bin/env python

import argparse
import csv
import glob
import json
import os
from typing import Any, Dict, List, Optional

import yaml


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _read_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _get(d: Dict[str, Any], keys: List[str], default=None):
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _fmt(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, (int,)):
        return str(x)
    if isinstance(x, float):
        # keep compact but readable
        return f"{x:.4f}".rstrip("0").rstrip(".")
    return str(x)


def _to_markdown_table(rows: List[Dict[str, Any]], columns: List[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, sep]
    for r in rows:
        lines.append("| " + " | ".join(_fmt(r.get(c)) for c in columns) + " |")
    return "\n".join(lines) + "\n"


def build_rows(
    config_dir: str,
    no_kd_logs_root: str,
    kd_logs_root: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    cfg_paths = sorted(glob.glob(os.path.join(config_dir, "*.yaml")))
    for cfg_path in cfg_paths:
        base = os.path.splitext(os.path.basename(cfg_path))[0]

        no_kd_dir = os.path.join(no_kd_logs_root, base)
        kd_dir = os.path.join(kd_logs_root, base)

        no_kd_results_path = os.path.join(no_kd_dir, "results.json")
        kd_results_path = os.path.join(kd_dir, "results.json")

        if not os.path.exists(no_kd_results_path):
            # Not run yet
            continue
        if not os.path.exists(kd_results_path):
            # Teacher metrics source missing; still include student-only row.
            kd = None
        else:
            kd = _read_json(kd_results_path)

        no_kd = _read_json(no_kd_results_path)
        cfg = _read_yaml(cfg_path)

        dataset_type = _get(no_kd, ["config", "data", "type"], _get(cfg, ["data", "type"], ""))
        task1 = _get(no_kd, ["config", "data", "task1_label"], _get(cfg, ["data", "task1_label"], "task1"))
        task2 = _get(no_kd, ["config", "data", "task2_label"], _get(cfg, ["data", "task2_label"], "task2"))

        # Student test metrics (No-KD)
        s_test = _get(no_kd, ["metrics", "test"], {}) or {}

        # Teacher test metrics (from KD run)
        t_test = {}
        if kd is not None:
            t_test = _get(kd, ["metrics", "teacher", "test"], {}) or {}

        def s(key: str) -> Optional[float]:
            return s_test.get(key)

        def t(key: str) -> Optional[float]:
            return t_test.get(key)

        row: Dict[str, Any] = {
            "config": base,
            "dataset": dataset_type,
            "student_vision": _get(no_kd, ["config", "student", "vision"], _get(cfg, ["student", "vision"], "")),
            "student_text": _get(no_kd, ["config", "student", "text"], _get(cfg, ["student", "text"], "")),
            "fusion": _get(no_kd, ["config", "fusion", "type"], _get(cfg, ["fusion", "type"], "")),
            "task1": task1,
            "task2": task2,
            "teacher_task1_acc": t(f"teacher_test_{task1}_acc"),
            "student_task1_acc": s(f"test_{task1}_acc"),
            "teacher_task1_f1": t(f"teacher_test_{task1}_f1"),
            "student_task1_f1": s(f"test_{task1}_f1"),
            "teacher_task2_acc": t(f"teacher_test_{task2}_acc"),
            "student_task2_acc": s(f"test_{task2}_acc"),
            "teacher_task2_f1": t(f"teacher_test_{task2}_f1"),
            "student_task2_f1": s(f"test_{task2}_f1"),
            "teacher_infer_ms": t("teacher_test_infer_ms"),
            "student_infer_ms": s("test_infer_ms"),
            "no_kd_log_dir": _get(no_kd, ["config", "logging", "log_dir"], ""),
            "kd_log_dir": _get(kd, ["config", "logging", "log_dir"], "") if kd else "",
        }

        rows.append(row)

    # Deterministic ordering: dataset then config
    rows.sort(key=lambda r: (r.get("dataset", ""), r.get("config", "")))
    return rows


def write_csv(path: str, rows: List[Dict[str, Any]], columns: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for r in rows:
            w.writerow({c: _fmt(r.get(c)) for c in columns})


def write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate Teacher vs No-KD Student test-set table from logs and config/no-kd/."
    )
    p.add_argument("--config-dir", default="config/no-kd", help="Directory containing No-KD YAML configs")
    p.add_argument(
        "--no-kd-logs-root",
        default="logs/no-kd/ultra-edge-hp-tuned-all",
        help="Root folder containing No-KD run subfolders",
    )
    p.add_argument(
        "--kd-logs-root",
        default="logs/ultra-edge-hp-tuned-all",
        help="Root folder containing KD run subfolders (teacher metrics source)",
    )
    p.add_argument(
        "--out-md",
        default="docs/aggregated_results/no_kd_teacher_vs_student_test.md",
        help="Output markdown path",
    )
    p.add_argument(
        "--out-csv",
        default="docs/aggregated_results/no_kd_teacher_vs_student_test.csv",
        help="Output csv path",
    )
    return p.parse_args()


def main():
    args = parse_args()

    rows = build_rows(args.config_dir, args.no_kd_logs_root, args.kd_logs_root)

    columns = [
        "config",
        "dataset",
        "student_vision",
        "student_text",
        "fusion",
        "task1",
        "teacher_task1_acc",
        "student_task1_acc",
        "teacher_task1_f1",
        "student_task1_f1",
        "task2",
        "teacher_task2_acc",
        "student_task2_acc",
        "teacher_task2_f1",
        "student_task2_f1",
        "teacher_infer_ms",
        "student_infer_ms",
        "no_kd_log_dir",
        "kd_log_dir",
    ]

    md = "# Teacher vs No-KD Student (Test Set)\n\n"
    md += "This table pairs each No-KD student run with teacher test metrics from the corresponding KD run.\n\n"
    md += _to_markdown_table(rows, columns)

    write_text(args.out_md, md)
    write_csv(args.out_csv, rows, columns)

    print(f"Wrote {len(rows)} rows")
    print(f"- {args.out_md}")
    print(f"- {args.out_csv}")


if __name__ == "__main__":
    main()
