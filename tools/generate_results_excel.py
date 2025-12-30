#!/usr/bin/env python3
"""Generate an Excel workbook summarising results.json files.

Scans `logs/ultra-edge-hp-tuned-all/*/results.json` and creates
an Excel workbook with one sheet per model (subfolder). Each sheet
contains rows for tasks with columns: Model Name, Teacher_Params,
Student_params, Task, Acc, Precision, F1, Recall, Infer_ms.
"""
import json
from pathlib import Path
from collections import defaultdict


LOG_ROOT = Path('logs/ultra-edge-hp-tuned-all')
OUT_XLSX = Path('logs/ultra-edge-hp-tuned-all/summary_results.xlsx')


def extract_params(cfg: dict) -> (str, str):
    teacher = cfg.get('teacher', {})
    student = cfg.get('student', {})
    # stringify selected teacher/student params
    t_keys = ['vision', 'text', 'fusion_layers', 'fusion_dim', 'fusion_heads', 'dropout']
    s_keys = ['vision', 'text', 'fusion_layers', 'fusion_dim', 'fusion_heads', 'dropout']
    teacher_str = ', '.join(f"{k}={teacher.get(k)}" for k in t_keys if k in teacher)
    student_str = ', '.join(f"{k}={student.get(k)}" for k in s_keys if k in student)
    return teacher_str, student_str


def process_results_file(data: dict, model_name: str):
    # data: loaded results.json content
    cfg = data.get('config') or data.get('args') or {}
    teacher_str, student_str = extract_params(cfg)

    # prefer test split
    root_metrics = data.get('metrics') or data.get('results') or {}
    test_metrics = None
    if isinstance(root_metrics, dict):
        for k in ('test', 'test_metrics', 'test_results'):
            if k in root_metrics:
                test_metrics = root_metrics[k]
                break
    if test_metrics is None:
        # if no explicit test dict, use root_metrics itself
        test_metrics = root_metrics if isinstance(root_metrics, dict) else {}

    def parse_task_metrics(test_metrics):
        tasks = defaultdict(dict)
        for key, value in (test_metrics or {}).items():
            if not key.startswith('test_'):
                continue
            parts = key.split('_', 2)
            if len(parts) != 3:
                continue
            _, task, metric = parts
            tasks[task][metric] = value
        return tasks

    tasks = parse_task_metrics(test_metrics)

    # capture any global infer_ms reported in the test metrics (e.g. 'test_infer_ms')
    global_infer_ms = None
    if isinstance(test_metrics, dict):
        for k, v in test_metrics.items():
            if isinstance(k, str) and 'infer' in k:
                global_infer_ms = v
                break

    # remove any parsed 'infer' pseudo-task (we'll use its value as global infer_ms)
    for infer_key in ('infer', 'inference', 'infer_ms', 'inference_ms'):
        tasks.pop(infer_key, None)

    teacher_params_m = data.get('models', {}).get('teacher', {}).get('params_millions')
    student_params_m = data.get('models', {}).get('student', {}).get('params_millions')

    base = {
        'Model Name': model_name,
        'Tchr_p_m': teacher_params_m,
        'std_p_m': student_params_m,
    }

    rows = []
    # for each parsed task, map metrics to standardized columns
    def metric_val(d, *keys):
        for k in keys:
            if k in d:
                return d[k]
        return None

    for task, metrics in tasks.items():
        r = dict(base)
        r['Task'] = task
        r['Accuracy'] = metric_val(metrics, 'acc', 'accuracy')
        r['F1'] = metric_val(metrics, 'f1')
        r['Precision'] = metric_val(metrics, 'prec', 'precision')
        r['Recall'] = metric_val(metrics, 'rec', 'recall')
        r['Auc'] = metric_val(metrics, 'auc')
        r['infer_ms'] = metric_val(metrics, 'infer_ms', 'inference_ms') or global_infer_ms
        rows.append(r)

    # (no average rows requested) keep per-task rows only

    # fallback to structured per_task lists
    if not rows:
        per = test_metrics.get('per_task') if isinstance(test_metrics, dict) else None
        if not per:
            per = test_metrics.get('per_task_metrics') if isinstance(test_metrics, dict) else None
        if isinstance(per, list) and per:
            for m in per:
                r = dict(base)
                r['Task'] = m.get('task') or m.get('name')
                r['Accuracy'] = m.get('accuracy') or m.get('acc')
                r['F1'] = m.get('f1') or m.get('f1_score')
                r['Precision'] = m.get('precision') or m.get('prec')
                r['Recall'] = m.get('recall') or m.get('rec')
                r['Auc'] = m.get('auc')
                r['infer_ms'] = m.get('infer_ms') or m.get('inference_ms') or global_infer_ms
                rows.append(r)

    return rows


def detect_dataset_from_cfg(cfg: dict, model_name: str) -> str:
    # Try common config locations for dataset type
    data_cfg = cfg.get('data') if isinstance(cfg, dict) else None
    if isinstance(data_cfg, dict):
        dt = data_cfg.get('type') or data_cfg.get('dataset')
        if dt:
            return str(dt)
    for key in ('dataset', 'data_type'):
        v = cfg.get(key)
        if v:
            return str(v)
    name = model_name.lower()
    if 'wound' in name:
        return 'wound'
    if 'medpix' in name or 'medpix' in str(cfg).lower():
        return 'medpix'
    return 'other'


def main():
    import pandas as pd

    all_dirs = [p for p in LOG_ROOT.iterdir() if p.is_dir()]
    dataset_rows = {}

    for d in sorted(all_dirs):
        results_file = d / 'results.json'
        if not results_file.exists():
            continue
        model_name = d.name
        try:
            with results_file.open('r') as f:
                data = json.load(f)
        except Exception:
            continue

        cfg = data.get('config') or data.get('args') or {}
        dataset = detect_dataset_from_cfg(cfg, model_name)
        rows = process_results_file(data, model_name)
        if not rows:
            continue
        dataset_rows.setdefault(dataset, []).extend(rows)

    if not dataset_rows:
        print('No results found.')
        return

    cols = ['Model Name', 'Tchr_p_m', 'std_p_m', 'Task', 'Accuracy', 'F1', 'Precision', 'Recall', 'Auc', 'infer_ms']

    with pd.ExcelWriter(OUT_XLSX, engine='openpyxl') as writer:
        for dataset, rows in dataset_rows.items():
            df = pd.DataFrame(rows)
            # ensure columns and order
            for c in cols:
                if c not in df.columns:
                    df[c] = None
            df = df[cols]
            sheet_name = dataset[:31]
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f'Wrote {OUT_XLSX}')


if __name__ == '__main__':
    main()
