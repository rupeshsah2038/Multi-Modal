#!/usr/bin/env python

import argparse
import csv
import glob
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import yaml


VISION_DISPLAY = {
    "deit-small": "DeiT-Small",
    "deit-tiny": "DeiT-Tiny",
    "mobilevit-small": "MobileViT-Small",
    "mobilevit-xx-small": "MobileViT-XX-Small",
    "mobilevit-x-small": "MobileViT-X-Small",
    "vit-base": "ViT-Base",
    "vit-large": "ViT-Large",
}

TEXT_DISPLAY = {
    "distilbert": "DistilBERT",
    "bert-mini": "BERT-Mini",
    "bert-tiny": "BERT-Tiny",
    "minilm": "MiniLM",
    "bio-clinical-bert": "Bio-ClinicalBERT",
}


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _read_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _fmt(x: Any, decimals: int = 4) -> str:
    if x is None:
        return ""
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        return f"{x:.{decimals}f}".rstrip("0").rstrip(".")
    return str(x)


def _model_configuration(cfg: Dict[str, Any], results: Dict[str, Any]) -> str:
    student_cfg = (cfg.get("student") or {})
    arch = str(student_cfg.get("arch", "standard")).lower()

    # If you're using a custom tiny student, treat it as a proposed model.
    if arch == "custom_tiny":
        return "Proposed Model"

    student = (results.get("config") or {}).get("student") or student_cfg
    vis = student.get("vision", "")
    txt = student.get("text", "")

    vis_disp = VISION_DISPLAY.get(vis, str(vis))
    txt_disp = TEXT_DISPLAY.get(txt, str(txt))

    if vis_disp and txt_disp:
        return f"{vis_disp} + {txt_disp}"
    return vis_disp or txt_disp or ""


def _student_params_m(results: Dict[str, Any]) -> Optional[float]:
    stu = ((results.get("models") or {}).get("student") or {})
    pm = stu.get("params_millions")
    if pm is None:
        return None
    try:
        return float(pm)
    except Exception:
        return None


def _infer_ms(results: Dict[str, Any]) -> Optional[float]:
    test = ((results.get("metrics") or {}).get("test") or {})
    val = test.get("test_infer_ms")
    if val is None:
        return None
    try:
        return float(val)
    except Exception:
        return None


def _extract_task_labels(results: Dict[str, Any]) -> Tuple[str, str, str]:
    cfg = (results.get("config") or {})
    data = (cfg.get("data") or {})
    dataset = data.get("type", "")
    task1 = data.get("task1_label", "modality")
    task2 = data.get("task2_label", "location")
    return dataset, str(task1), str(task2)


def _metric(test: Dict[str, Any], task: str, suffix: str) -> Optional[float]:
    key = f"test_{task}_{suffix}"
    v = test.get(key)
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def _task_display(task: str) -> str:
    # keep the exact naming you asked for
    return str(task).capitalize()


def _results_path_from_cfg(cfg_path: str, fallback_logs_root: str) -> str:
    cfg = _read_yaml(cfg_path)
    base = os.path.splitext(os.path.basename(cfg_path))[0]
    log_dir = ((cfg.get("logging") or {}).get("log_dir"))
    if log_dir:
        return os.path.join(log_dir, "results.json")
    return os.path.join(fallback_logs_root, base, "results.json")


def _write_csv(path: str, rows: List[Dict[str, Any]], columns: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for r in rows:
            w.writerow({c: _fmt(r.get(c), decimals=2 if c.startswith("Student Params") else 4) for c in columns})


def build_publication_rows(config_dir: str, logs_root: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    medpix: List[Dict[str, Any]] = []
    wound: List[Dict[str, Any]] = []

    cfg_paths = sorted(glob.glob(os.path.join(config_dir, "*.yaml")))
    for cfg_path in cfg_paths:
        results_path = _results_path_from_cfg(cfg_path, logs_root)
        if not os.path.exists(results_path):
            continue

        results = _read_json(results_path)
        cfg = _read_yaml(cfg_path)

        dataset, task1, task2 = _extract_task_labels(results)
        test = ((results.get("metrics") or {}).get("test") or {})

        model_conf = _model_configuration(cfg, results)
        params_m = _student_params_m(results)
        infer_ms = _infer_ms(results)

        def row(task: str) -> Dict[str, Any]:
            return {
                "Model Configuration": model_conf,
                "Student Params (M) ↓": params_m,
                "Task": _task_display(task),
                "Accuracy ↑": _metric(test, task, "acc"),
                "Precision ↑": _metric(test, task, "prec"),
                "F1-Score ↑": _metric(test, task, "f1"),
                "Recall ↑": _metric(test, task, "rec"),
                "AUC ↑": _metric(test, task, "auc"),
                "Inference (ms) ↓": infer_ms,
            }

        rows = [row(task1), row(task2)]

        # mimic multi-row grouping in CSV: blank out model + params on second task row
        rows[1]["Model Configuration"] = ""
        rows[1]["Student Params (M) ↓"] = ""

        # add a blank separator row after each model group (helps when opening in Excel)
        separator = {k: "" for k in rows[0].keys()}

        if dataset == "medpix":
            medpix.extend(rows)
            medpix.append(separator)
        elif dataset == "wound":
            wound.extend(rows)
            wound.append(separator)

    def sort_key(group_rows: List[Dict[str, Any]]):
        # sort by params desc; we only have params on first row per model group
        grouped: List[Tuple[float, str, List[Dict[str, Any]]]] = []
        i = 0
        while i < len(group_rows):
            r1 = group_rows[i]
            r2 = group_rows[i + 1] if i + 1 < len(group_rows) else {}
            # skip unexpected empty rows
            if not r1.get("Task"):
                i += 1
                continue
            pm = r1.get("Student Params (M) ↓")
            try:
                pm_f = float(pm)
            except Exception:
                pm_f = -1.0
            model_name = str(r1.get("Model Configuration", ""))
            grouped.append((pm_f, model_name, [r1, r2]))
            i += 3  # 2 rows + separator

        grouped.sort(key=lambda t: (-t[0], t[1]))

        out: List[Dict[str, Any]] = []
        for _, __, rs in grouped:
            out.extend(rs)
            out.append({k: "" for k in rs[0].keys()})
        return out

    medpix = sort_key(medpix)
    wound = sort_key(wound)
    return medpix, wound


def parse_args():
    p = argparse.ArgumentParser(description="Generate No-KD CSV tables in the same layout as the paper-style screenshot")
    p.add_argument("--config-dir", default="config/no-kd", help="Directory containing No-KD YAML configs")
    p.add_argument(
        "--no-kd-logs-root",
        default="logs/no-kd/ultra-edge-hp-tuned-all",
        help="Fallback log root if a config has no logging.log_dir",
    )
    p.add_argument(
        "--out-dir",
        default="docs/aggregated_results",
        help="Output directory",
    )
    return p.parse_args()


def main():
    args = parse_args()

    columns = [
        "Model Configuration",
        "Student Params (M) ↓",
        "Task",
        "Accuracy ↑",
        "Precision ↑",
        "F1-Score ↑",
        "Recall ↑",
        "AUC ↑",
        "Inference (ms) ↓",
    ]

    medpix_rows, wound_rows = build_publication_rows(args.config_dir, args.no_kd_logs_root)

    medpix_csv = os.path.join(args.out_dir, "no_kd_medpix_test_pub.csv")
    wound_csv = os.path.join(args.out_dir, "no_kd_wound_test_pub.csv")

    # Backward-compatible filenames (overwrite previous simple CSVs).
    medpix_csv_simple = os.path.join(args.out_dir, "no_kd_medpix_test_simple.csv")
    wound_csv_simple = os.path.join(args.out_dir, "no_kd_wound_test_simple.csv")

    _write_csv(medpix_csv, medpix_rows, columns)
    _write_csv(wound_csv, wound_rows, columns)

    _write_csv(medpix_csv_simple, medpix_rows, columns)
    _write_csv(wound_csv_simple, wound_rows, columns)

    print(f"Wrote:")
    print(f"- {medpix_csv} ({len(medpix_rows)} rows)")
    print(f"- {wound_csv} ({len(wound_rows)} rows)")
    print(f"- {medpix_csv_simple} ({len(medpix_rows)} rows)")
    print(f"- {wound_csv_simple} ({len(wound_rows)} rows)")


if __name__ == "__main__":
    main()
