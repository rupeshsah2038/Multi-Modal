#!/usr/bin/env python

import argparse
import csv
import glob
import json
import os
from typing import Any, Dict, List, Tuple

import yaml


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _read_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _fmt(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        return f"{x:.4f}".rstrip("0").rstrip(".")
    return str(x)


def _to_markdown_table(rows: List[Dict[str, Any]], columns: List[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, sep]
    for r in rows:
        lines.append("| " + " | ".join(_fmt(r.get(c)) for c in columns) + " |")
    return "\n".join(lines) + "\n"


def _write_csv(path: str, rows: List[Dict[str, Any]], columns: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for r in rows:
            w.writerow({c: _fmt(r.get(c)) for c in columns})


def _write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def _get_no_kd_results_path(cfg_path: str, no_kd_logs_root: str) -> str:
    base = os.path.splitext(os.path.basename(cfg_path))[0]

    cfg = _read_yaml(cfg_path)
    log_dir = ((cfg.get("logging") or {}).get("log_dir"))
    if log_dir:
        return os.path.join(log_dir, "results.json")

    return os.path.join(no_kd_logs_root, base, "results.json")


def _extract_rows_for_one_run(results: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]]]:
    cfg = results.get("config", {}) or {}
    dataset = (cfg.get("data") or {}).get("type", "")

    student = cfg.get("student", {}) or {}
    model = f"{student.get('vision', '')}-{student.get('text', '')}".strip("-")

    data_cfg = cfg.get("data", {}) or {}
    task1 = data_cfg.get("task1_label", "modality")
    task2 = data_cfg.get("task2_label", "location")

    test = (results.get("metrics") or {}).get("test", {}) or {}

    infer_ms = test.get("test_infer_ms")

    def row_for(task: str) -> Dict[str, Any]:
        return {
            "Models": model,
            "Task": task,
            "Accuracy": test.get(f"test_{task}_acc"),
            "Precision": test.get(f"test_{task}_prec"),
            "F1": test.get(f"test_{task}_f1"),
            "Recall": test.get(f"test_{task}_rec"),
            # User asked for "AUM"; this corresponds to stored "auc".
            "AUM": test.get(f"test_{task}_auc"),
            "Infer ms": infer_ms,
        }

    rows = [row_for(task1), row_for(task2)]
    return dataset, rows


def parse_args():
    p = argparse.ArgumentParser(description="Generate simple No-KD tables (MedPix + Wound) from logs.")
    p.add_argument("--config-dir", default="config/no-kd", help="Directory containing No-KD YAML configs")
    p.add_argument(
        "--no-kd-logs-root",
        default="logs/no-kd/ultra-edge-hp-tuned-all",
        help="Root folder containing No-KD run subfolders (used if config has no logging.log_dir)",
    )
    p.add_argument(
        "--out-dir",
        default="docs/aggregated_results",
        help="Output directory for markdown/csv files",
    )
    return p.parse_args()


def main():
    args = parse_args()

    columns = ["Models", "Task", "Accuracy", "Precision", "F1", "Recall", "AUM", "Infer ms"]

    medpix_rows: List[Dict[str, Any]] = []
    wound_rows: List[Dict[str, Any]] = []

    for cfg_path in sorted(glob.glob(os.path.join(args.config_dir, "*.yaml"))):
        results_path = _get_no_kd_results_path(cfg_path, args.no_kd_logs_root)
        if not os.path.exists(results_path):
            continue

        results = _read_json(results_path)
        dataset, rows = _extract_rows_for_one_run(results)

        if dataset == "medpix":
            medpix_rows.extend(rows)
        elif dataset == "wound":
            wound_rows.extend(rows)

    # stable sort: model then task
    medpix_rows.sort(key=lambda r: (r.get("Models", ""), r.get("Task", "")))
    wound_rows.sort(key=lambda r: (r.get("Models", ""), r.get("Task", "")))

    os.makedirs(args.out_dir, exist_ok=True)

    medpix_md_path = os.path.join(args.out_dir, "no_kd_medpix_test_simple.md")
    medpix_csv_path = os.path.join(args.out_dir, "no_kd_medpix_test_simple.csv")

    wound_md_path = os.path.join(args.out_dir, "no_kd_wound_test_simple.md")
    wound_csv_path = os.path.join(args.out_dir, "no_kd_wound_test_simple.csv")

    medpix_md = "# No-KD (CE-only) — MedPix Test Set\n\n"
    medpix_md += _to_markdown_table(medpix_rows, columns)

    wound_md = "# No-KD (CE-only) — Wound Test Set\n\n"
    wound_md += _to_markdown_table(wound_rows, columns)

    _write_text(medpix_md_path, medpix_md)
    _write_csv(medpix_csv_path, medpix_rows, columns)

    _write_text(wound_md_path, wound_md)
    _write_csv(wound_csv_path, wound_rows, columns)

    print(f"MedPix rows: {len(medpix_rows)}")
    print(f"- {medpix_md_path}")
    print(f"- {medpix_csv_path}")
    print(f"Wound rows: {len(wound_rows)}")
    print(f"- {wound_md_path}")
    print(f"- {wound_csv_path}")


if __name__ == "__main__":
    main()
