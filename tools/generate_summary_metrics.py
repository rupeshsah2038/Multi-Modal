#!/usr/bin/env python3
import csv
from pathlib import Path
import re

try:
    import pandas as pd
except Exception:
    pd = None

in_csv = Path('logs') / 'aggregate_results.csv'
if not in_csv.exists():
    print('Missing logs/aggregate_results.csv; run tools/aggregate_results.py first')
    raise SystemExit(1)

if pd is not None:
    df = pd.read_csv(in_csv)
else:
    # naive CSV reader into list of dicts
    with open(in_csv, 'r', newline='') as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

# helper to fetch metric value from a row dict handling multiple column naming styles
def get_metric(row, split, task, metric):
    # try split-prefixed column first
    keys = [f'{split}_{task}_{metric}', f'{task}_{metric}', f'{split}_{task}_{metric.lower()}', f'{task}_{metric.lower()}']
    for k in keys:
        if k in row and pd.isna(row[k]) is False if pd is not None else row.get(k, '') != '':
            return row.get(k)
    # variants: prec vs precision, rec vs recall
    if metric == 'prec':
        for alt in ('precision',):
            for k in [f'{split}_{task}_{alt}', f'{task}_{alt}']:
                if k in row and (pd.isna(row[k]) is False if pd is not None else row.get(k, '') != ''):
                    return row.get(k)
    if metric == 'rec':
        for alt in ('recall',):
            for k in [f'{split}_{task}_{alt}', f'{task}_{alt}']:
                if k in row and (pd.isna(row[k]) is False if pd is not None else row.get(k, '') != ''):
                    return row.get(k)
    return None

out_rows = []
if pd is not None:
    it = df.to_dict(orient='records')
else:
    it = rows

for row in it:
    split = str(row.get('split') or '').strip()
    if not split:
        # try infer from columns
        split = 'dev'
    # determine tasks by checking available columns for this row
    candidates = set()
    # possible metric suffixes
    suffixes = ['acc', 'f1', 'prec', 'rec', 'auc']
    for col in row.keys():
        m = re.match(rf'^{split}_(.+)_acc$', col)
        if m:
            candidates.add(m.group(1))
        m2 = re.match(rf'^(.+)_acc$', col)
        if m2 and (f'{split}_' + m2.group(1) in row):
            candidates.add(m2.group(1))
    # fallback: look for known tasks
    if not candidates:
        for t in ('modality','location','type','severity'):
            if f'{split}_{t}_acc' in row or f'{t}_acc' in row:
                candidates.add(t)
    if not candidates:
        candidates = {'unknown'}

    model_name = f"{row.get('student_vision','')}-{row.get('student_text','')}".strip('-')
    teacher_params = row.get('teacher_params_millions')
    student_params = row.get('student_params_millions')
    dataset = row.get('data_type') or row.get('data_root')
    # inference time
    infer = row.get(f'{split}_infer_ms') or row.get('infer_ms') or ''

    for task in sorted(candidates):
        acc = get_metric(row, split, task, 'acc')
        prec = get_metric(row, split, task, 'prec')
        f1 = get_metric(row, split, task, 'f1')
        rec = get_metric(row, split, task, 'rec')
        auc = get_metric(row, split, task, 'auc')
        out_rows.append({
            'model': model_name,
            'data': dataset,
            'teacher_params_millions': teacher_params,
            'student_params_millions': student_params,
            'task': task,
            'split': split,
            'accuracy': acc,
            'precision': prec,
            'f1': f1,
            'recall': rec,
            'auc': auc,
            'infer_ms': infer
        })

out_csv = Path('logs') / 'summary_metrics.csv'
out_xlsx = Path('logs') / 'summary_metrics.xlsx'
if pd is not None:
    sdf = pd.DataFrame(out_rows)
    sdf.to_csv(out_csv, index=False)
    try:
        sdf.to_excel(out_xlsx, index=False)
        print(f'Wrote summary to {out_csv} and {out_xlsx} ({len(sdf)} rows)')
    except Exception:
        print(f'Wrote summary CSV to {out_csv} ({len(sdf)} rows); Excel write failed')
else:
    # write CSV only
    cols = ['model','data','teacher_params_millions','student_params_millions','task','split','accuracy','precision','f1','recall','auc','infer_ms']
    with open(out_csv, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        for r in out_rows:
            writer.writerow(r)
    print(f'Wrote summary CSV to {out_csv} ({len(out_rows)} rows)')
