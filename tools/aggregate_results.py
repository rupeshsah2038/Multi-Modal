#!/usr/bin/env python3
import json
import glob
import os
from pathlib import Path
try:
    import pandas as pd
except Exception:
    pd = None

patterns = [
    'logs/ultra-edge-base-*/**/results.json',
    'logs/ultra-edge-base-*/results.json',
    'logs/ultra-edge-hp-tuned-all/**/results.json',
    'logs/ultra-edge-hp-tuned-all/results.json'
]

files = []
for pat in patterns:
    files.extend(glob.glob(pat, recursive=True))

files = sorted(set(files))
rows = []
for f in files:
    try:
        with open(f, 'r') as fh:
            j = json.load(fh)
    except Exception as e:
        print(f"Skipping {f}: {e}")
        continue
    row = {}
    row['result_file'] = f
    cfg = j.get('config', {})
    row['data_type'] = cfg.get('data', {}).get('type') or j.get('data', {}).get('type') if j.get('data') else cfg.get('data', {}).get('type')
    row['data_root'] = cfg.get('data', {}).get('root') or j.get('data', {}).get('root') if j.get('data') else cfg.get('data', {}).get('root')

    teacher = cfg.get('teacher', {})
    student = cfg.get('student', {})
    row['teacher_vision'] = teacher.get('vision')
    row['teacher_text'] = teacher.get('text')
    row['teacher_fusion_layers'] = teacher.get('fusion_layers')
    row['teacher_fusion_dim'] = teacher.get('fusion_dim')

    row['student_vision'] = student.get('vision')
    row['student_text'] = student.get('text')
    row['student_fusion_layers'] = student.get('fusion_layers')
    row['student_fusion_dim'] = student.get('fusion_dim')

    models = j.get('models', {})
    tmod = models.get('teacher', {})
    smod = models.get('student', {})
    row['teacher_total_params'] = tmod.get('total_params')
    row['teacher_params_millions'] = tmod.get('params_millions')
    row['student_total_params'] = smod.get('total_params')
    row['student_params_millions'] = smod.get('params_millions')

    training = j.get('training', {})
    row['teacher_epochs'] = training.get('teacher_epochs')
    row['student_epochs'] = training.get('student_epochs')
    row['student_lr'] = training.get('student_lr')
    row['teacher_lr'] = training.get('teacher_lr')
    row['alpha'] = training.get('alpha')
    row['beta'] = training.get('beta')
    row['T'] = training.get('T')

    row['fusion_type'] = j.get('fusion', {}).get('type') or cfg.get('fusion', {}).get('type')
    row['loss_type'] = j.get('loss', {}).get('type') or cfg.get('loss', {}).get('type')

    metrics = j.get('metrics', {})
    dev = metrics.get('dev', {}) or {}
    test = metrics.get('test', {}) or {}
    train = metrics.get('train', {}) or {}
    teacher_metrics = metrics.get('teacher', {}) or {}
    teacher_dev = teacher_metrics.get('dev', {}) or {}
    teacher_test = teacher_metrics.get('test', {}) or {}

    # Flatten commonly used metrics (if present)
    for k, v in dev.items():
        row[f'dev_{k}'] = v
    for k, v in test.items():
        row[f'test_{k}'] = v
    for k, v in teacher_dev.items():
        row[f'teacher_dev_{k}'] = v
    for k, v in teacher_test.items():
        row[f'teacher_test_{k}'] = v
    # train final loss
    row['train_final_loss'] = train.get('final_loss')

    rows.append(row)

if not rows:
    print('No results.json files found for the specified patterns.')
    raise SystemExit(1)

out_xlsx = Path('logs') / 'aggregate_results.xlsx'
out_csv = Path('logs') / 'aggregate_results.csv'
out_xlsx.parent.mkdir(parents=True, exist_ok=True)

# Group rows by dataset (use data_root if available, else data_type)
from collections import defaultdict
# Expand each result into two rows: one for dev and one for test
expanded_rows = []
for r in rows:
    # common fields (exclude dev_/test_ metrics)
    common = {k: v for k, v in r.items() if not (k.startswith('dev_') or k.startswith('test_'))}
    for split in ('dev', 'test'):
        new = common.copy()
        new['split'] = split
        # copy split-specific metrics without the prefix
        for k, v in r.items():
            if k.startswith(f'{split}_'):
                new[k[len(split) + 1:]] = v
        expanded_rows.append(new)

groups = defaultdict(list)
for r in expanded_rows:
    key = r.get('data_root') or r.get('data_type') or 'unknown'
    groups[key].append(r)

def _slugify(name: str) -> str:
    s = ''.join([c if c.isalnum() else '_' for c in name])
    return s[:30]

if pd is not None:
    df_all = pd.DataFrame(expanded_rows)
    # write master sheet and per-dataset sheets
    with pd.ExcelWriter(out_xlsx, engine='openpyxl') as writer:
        df_all.to_excel(writer, sheet_name='ALL', index=False)
        for k, group_rows in groups.items():
            sheet = _slugify(k)
            df_group = pd.DataFrame(group_rows)
            # Excel sheet names must be <=31 chars
            writer.book.create_sheet if False else None
            try:
                df_group.to_excel(writer, sheet_name=sheet[:31], index=False)
            except Exception:
                # fallback to writing CSV for this group if sheet fails
                pass
    df_all.to_csv(out_csv, index=False)
    # per-dataset CSVs
    for k, group_rows in groups.items():
        slug = _slugify(k)
        out_group_csv = out_xlsx.parent / f'aggregate_{slug}.csv'
        pd.DataFrame(group_rows).to_csv(out_group_csv, index=False)
    print(f"Wrote {len(df_all)} rows to {out_xlsx} and {out_csv} and {len(groups)} per-dataset CSVs")
else:
    import csv
    # master CSV
    cols = []
    for r in rows:
        for k in r.keys():
            if k not in cols:
                cols.append(k)
    with open(out_csv, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        for r in expanded_rows:
            writer.writerow(r)
    # per-dataset CSVs
    for k, group_rows in groups.items():
        slug = _slugify(k)
        out_group_csv = out_xlsx.parent / f'aggregate_{slug}.csv'
        cols = []
        for r in group_rows:
            for c in r.keys():
                if c not in cols:
                    cols.append(c)
        with open(out_group_csv, 'w', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=cols)
            writer.writeheader()
            for r in group_rows:
                writer.writerow(r)
    print(f"Pandas not available. Wrote {len(expanded_rows)} rows to {out_csv} and {len(groups)} per-dataset CSVs")

# --- Produce compact summary (one row per model/task/split) ---
def _find_tasks_in_row(r):
    tasks = set()
    for k in r.keys():
        m = None
        if isinstance(k, str):
            import re
            m = re.match(r'(.+?)_(acc|f1|prec|rec|auc)$', k)
        if m:
            tasks.add(m.group(1))
    # fallback known tasks
    if not tasks:
        for t in ('modality', 'location', 'type', 'severity'):
            if f'{t}_acc' in r or f'{t}_f1' in r:
                tasks.add(t)
    if not tasks:
        tasks.add('unknown')
    return sorted(tasks)

def _get_metric(r, task, metric):
    # prefer exact key
    keys = [f'{task}_{metric}', f'{task}_{metric.lower()}']
    for k in keys:
        if k in r and r.get(k) is not None:
            return r.get(k)
    # common alternates
    if metric == 'prec':
        for alt in ('precision',):
            k = f'{task}_{alt}'
            if k in r and r.get(k) is not None:
                return r.get(k)
    if metric == 'rec':
        for alt in ('recall',):
            k = f'{task}_{alt}'
            if k in r and r.get(k) is not None:
                return r.get(k)
    return None

summary_rows = []
for r in expanded_rows:
    split = r.get('split') or ''
    tasks = _find_tasks_in_row(r)
    model_name = f"{r.get('student_vision','')}-{r.get('student_text','')}".strip('-')
    teacher_params = r.get('teacher_params_millions')
    student_params = r.get('student_params_millions')
    dataset = r.get('data_type') or r.get('data_root')
    infer = r.get('infer_ms') or r.get(f'{split}_infer_ms') or ''
    fusion_dim = r.get('teacher_fusion_dim') or r.get('student_fusion_dim') or ''
    for task in tasks:
        summary_rows.append({
            'model': model_name,
            'data': dataset,
            'fusion_dim': fusion_dim,
            'teacher_params_millions': teacher_params,
            'student_params_millions': student_params,
            'task': task,
            'split': split,
            'accuracy': _get_metric(r, task, 'acc'),
            'precision': _get_metric(r, task, 'prec'),
            'f1': _get_metric(r, task, 'f1'),
            'recall': _get_metric(r, task, 'rec'),
            'auc': _get_metric(r, task, 'auc'),
            'infer_ms': infer,
        })

out_sum_csv = out_xlsx.parent / 'summary_metrics.csv'
out_sum_xlsx = out_xlsx.parent / 'summary_metrics.xlsx'

# Group summary rows by dataset + fusion_dim
from collections import defaultdict as _dd
grouped = _dd(list)
for s in summary_rows:
    key = f"{s.get('data')}_fusion{s.get('fusion_dim')}"
    grouped[key].append(s)

if pd is not None:
    sdf = pd.DataFrame(summary_rows)
    sdf.to_csv(out_sum_csv, index=False)
    # write per-dataset+fusion sheets into summary workbook
    with pd.ExcelWriter(out_sum_xlsx, engine='openpyxl') as writer:
        # master sheet
        sdf.to_excel(writer, sheet_name='ALL', index=False)
        for k, grp in grouped.items():
            sheet = _slugify(k)
            try:
                pd.DataFrame(grp).to_excel(writer, sheet_name=sheet[:31], index=False)
            except Exception:
                # skip sheet on failure
                pass
    print(f'Wrote summary to {out_sum_csv} and {out_sum_xlsx} ({len(sdf)} rows, {len(grouped)} sheets)')
else:
    import csv
    cols = ['model','data','fusion_dim','teacher_params_millions','student_params_millions','task','split','accuracy','precision','f1','recall','auc','infer_ms']
    with open(out_sum_csv, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        for r in summary_rows:
            writer.writerow(r)
    # per-group CSVs
    for k, grp in grouped.items():
        slug = _slugify(k)
        out_group = out_xlsx.parent / f'summary_{slug}.csv'
        cols_g = list({c for row in grp for c in row.keys()})
        with open(out_group, 'w', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=cols_g)
            writer.writeheader()
            for r in grp:
                writer.writerow(r)
    print(f'Wrote summary CSV to {out_sum_csv} ({len(summary_rows)} rows) and {len(grouped)} per-group CSVs')
