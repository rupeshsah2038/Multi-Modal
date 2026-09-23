#!/usr/bin/env bash
import argparse
import csv
import json
import math
import os
import random
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.dataset import get_dataset, get_num_classes
from models.backbones import get_text_pretrained_name
from models.student import Student
from models.unimodal import VisionOnlyStudent, TextOnlyStudent
from utils.logger import MetricsLogger
from utils.results_logger import ResultsLogger


def set_seed(seed: int):
    if seed is None:
        return
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_classification_metrics(y_true, y_pred, y_prob):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    prec = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
    rec = float(recall_score(y_true, y_pred, average='macro', zero_division=0))

    auc = 0.0
    try:
        num_classes = y_prob.shape[1] if y_prob.ndim > 1 else 2
        if num_classes == 2:
            auc = float(roc_auc_score(y_true, y_prob[:, 1]))
        else:
            auc = float(roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro'))
    except Exception:
        auc = 0.0

    return {'acc': acc, 'f1': f1, 'prec': prec, 'rec': rec, 'auc': auc}


def evaluate_with_perturbation(model, loader, device, condition: str, seed: int = 42,
                               task1_label: str = 'modality', task2_label: str = 'location') -> Dict:
    model.eval()
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    all_t1_true, all_t1_pred, all_t1_prob = [], [], []
    all_t2_true, all_t2_pred, all_t2_prob = [], [], []

    start_time = time.time()

    with torch.no_grad():
        for batch in loader:
            pv = batch['pixel_values'].to(device)
            ids = batch['input_ids_student'].to(device)
            mask = batch['attention_mask_student'].to(device)
            y_t1 = batch['modality'].cpu().numpy()
            y_t2 = batch['location'].cpu().numpy()

            bs = pv.size(0)

            # Apply specific perturbation condition
            if condition == "clean":
                out = model(pv, ids, mask)

            elif condition == "mismatch_text":
                if bs > 1:
                    # Circular shift derangement: image i is paired with text (i + 1) % bs
                    perm = torch.roll(torch.arange(bs, device=device), shifts=1)
                    ids_m = ids[perm]
                    mask_m = mask[perm]
                else:
                    ids_m = ids
                    mask_m = mask
                out = model(pv, ids_m, mask_m)

            elif condition == "noise_0.1":
                noise = torch.randn(pv.size(), generator=rng, device=device) * 0.1
                out = model(pv + noise, ids, mask)

            elif condition == "noise_0.2":
                noise = torch.randn(pv.size(), generator=rng, device=device) * 0.2
                out = model(pv + noise, ids, mask)

            elif condition == "missing_30":
                # 30% random modality dropout per sample
                pv_mod = pv.clone()
                mask_mod = mask.clone()
                rand_img = torch.rand(bs, generator=rng, device=device)
                rand_txt = torch.rand(bs, generator=rng, device=device)
                drop_img = rand_img < 0.30
                drop_txt = rand_txt < 0.30
                pv_mod[drop_img] = 0.0
                mask_mod[drop_txt] = 0
                out = model(pv_mod, ids, mask_mod)

            elif condition == "missing_image":
                # 100% missing image (zeros)
                pv_mod = torch.zeros_like(pv)
                out = model(pv_mod, ids, mask)

            elif condition == "missing_text":
                # 100% missing text (attention mask = 0)
                mask_mod = torch.zeros_like(mask)
                out = model(pv, ids, mask_mod)

            else:
                raise ValueError(f"Unknown condition: {condition}")

            t1_logits = out['logits_modality']
            t2_logits = out['logits_location']

            all_t1_true.extend(y_t1)
            all_t2_true.extend(y_t2)
            all_t1_pred.extend(t1_logits.argmax(dim=-1).cpu().numpy())
            all_t2_pred.extend(t2_logits.argmax(dim=-1).cpu().numpy())
            all_t1_prob.extend(F.softmax(t1_logits, dim=-1).cpu().numpy())
            all_t2_prob.extend(F.softmax(t2_logits, dim=-1).cpu().numpy())

    infer_ms = (time.time() - start_time) / max(1, len(loader.dataset)) * 1000

    m1 = compute_classification_metrics(all_t1_true, all_t1_pred, all_t1_prob)
    m2 = compute_classification_metrics(all_t2_true, all_t2_pred, all_t2_prob)

    avg_acc = (m1['acc'] + m2['acc']) / 2.0
    avg_f1 = (m1['f1'] + m2['f1']) / 2.0

    return {
        'avg_acc': avg_acc,
        'avg_f1': avg_f1,
        f'{task1_label}_acc': m1['acc'],
        f'{task1_label}_f1': m1['f1'],
        f'{task1_label}_prec': m1['prec'],
        f'{task1_label}_rec': m1['rec'],
        f'{task1_label}_auc': m1['auc'],
        f'{task2_label}_acc': m2['acc'],
        f'{task2_label}_f1': m2['f1'],
        f'{task2_label}_prec': m2['prec'],
        f'{task2_label}_rec': m2['rec'],
        f'{task2_label}_auc': m2['auc'],
        'infer_ms': infer_ms,
    }


def build_test_loader(cfg, dataset_type, device):
    t_pretrained = get_text_pretrained_name(cfg['teacher']['text'])
    s_pretrained = get_text_pretrained_name(cfg['student']['text'])
    t_tok = AutoTokenizer.from_pretrained(t_pretrained)
    s_tok = AutoTokenizer.from_pretrained(s_pretrained)
    dataset_root = cfg['data']['root']

    if dataset_type == 'medpix':
        test_ds = get_dataset(
            dataset_type='medpix',
            data_jsonl_file=os.path.join(dataset_root, "splitted_dataset/data_test.jsonl"),
            desc_jsonl_file=os.path.join(dataset_root, "splitted_dataset/descriptions_test.jsonl"),
            image_dir=os.path.join(dataset_root, "images"),
            tokenizer_teacher=t_tok,
            tokenizer_student=s_tok,
        )
    else:
        test_ds = get_dataset(
            dataset_type='wound',
            csv_file=os.path.join(dataset_root, "metadata_test.csv"),
            image_dir=os.path.join(dataset_root, "images"),
            tokenizer_teacher=t_tok,
            tokenizer_student=s_tok,
            type_column=cfg['data'].get('type_column', 'type'),
            severity_column=cfg['data'].get('severity_column', 'severity'),
            description_column=cfg['data'].get('description_column', 'description'),
            filepath_column=cfg['data'].get('filepath_column', 'img_path'),
        )

    batch_size = int(cfg.get('data', {}).get('batch_size', 16))
    num_workers = int(cfg.get('data', {}).get('num_workers', 4))
    return DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def evaluate_multimodal_robustness(dataset_type: str, seeds: List[int], device: str,
                                   conditions: List[str]) -> Dict:
    print(f"\n========================================================")
    print(f"Evaluating Multimodal Robustness: {dataset_type.upper()}")
    print(f"Seeds: {seeds}")
    print(f"Conditions: {conditions}")
    print(f"========================================================")

    cfg_path = f"config/ultra-edge-hp-tuned-all/{dataset_type}-mobilevit_xx_small-bert-mini.yaml"
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    test_loader = build_test_loader(cfg, dataset_type, device)
    task1_label = cfg['data'].get('task1_label', 'modality' if dataset_type == 'medpix' else 'type')
    task2_label = cfg['data'].get('task2_label', 'location' if dataset_type == 'medpix' else 'severity')

    classes = get_num_classes(dataset_type, cfg['data']['root'])
    num_mod_classes = classes['modality']
    num_loc_classes = classes['location']

    results_by_cond = {cond: [] for cond in conditions}

    for seed in seeds:
        ckpt_path = f"logs/ultra-edge-hp-tuned-all/{dataset_type}-mobilevit_xx_small-bert-mini/seed_{seed}/student_best.pth"
        if not os.path.exists(ckpt_path):
            print(f"[Warning] Checkpoint not found: {ckpt_path}, skipping seed {seed}")
            continue

        print(f"\nLoading {dataset_type} student model for seed {seed} from {ckpt_path}...")
        student = Student(
            vision="mobilevit-xx-small",
            text="bert-mini",
            fusion_dim=256,
            fusion_type="cross_attention",
            fusion_heads=8,
            fusion_layers=1,
            dropout=0.292396,
            num_modality_classes=num_mod_classes,
            num_location_classes=num_loc_classes,
        ).to(device)

        student.load_state_dict(torch.load(ckpt_path, map_location=device))
        student.eval()

        for cond in conditions:
            metrics = evaluate_with_perturbation(
                student, test_loader, device, condition=cond, seed=seed,
                task1_label=task1_label, task2_label=task2_label
            )
            print(f"  [Seed {seed} | {cond:<20}] Avg Acc: {metrics['avg_acc']*100:.2f}%, Avg F1: {metrics['avg_f1']*100:.2f}% | {task1_label} F1: {metrics[f'{task1_label}_f1']*100:.2f}%, {task2_label} F1: {metrics[f'{task2_label}_f1']*100:.2f}%")
            results_by_cond[cond].append(metrics)

    # Compute Mean and Std across seeds
    summary = {}
    for cond, run_list in results_by_cond.items():
        if not run_list:
            continue
        keys = run_list[0].keys()
        cond_stats = {}
        for k in keys:
            vals = [r[k] for r in run_list]
            cond_stats[f"{k}_mean"] = float(np.mean(vals))
            cond_stats[f"{k}_std"] = float(np.std(vals))
        summary[cond] = cond_stats

    return {
        'task1_label': task1_label,
        'task2_label': task2_label,
        'per_seed': results_by_cond,
        'summary': summary,
    }


def train_unimodal_baseline(modality: str, dataset_type: str, seed: int, device: str,
                            epochs: int = 10, lr: float = 3.67e-4, force: bool = False) -> Dict:
    assert modality in ["image_only", "text_only"]
    out_dir = f"logs/ablations/unimodal/{dataset_type}/{modality}/seed_{seed}"
    ckpt_path = os.path.join(out_dir, "best_model.pth")
    results_path = os.path.join(out_dir, "results.json")

    cfg_path = f"config/ultra-edge-hp-tuned-all/{dataset_type}-mobilevit_xx_small-bert-mini.yaml"
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    task1_label = cfg['data'].get('task1_label', 'modality' if dataset_type == 'medpix' else 'type')
    task2_label = cfg['data'].get('task2_label', 'location' if dataset_type == 'medpix' else 'severity')

    if not force and os.path.exists(results_path) and os.path.exists(ckpt_path):
        print(f"[Skip] {dataset_type} {modality} seed {seed} already completed.")
        with open(results_path, 'r') as f:
            res = json.load(f)
        return res.get('metrics', {}).get('test', {})

    print(f"\n--------------------------------------------------------")
    print(f"Training Unimodal Baseline: {modality.upper()} | {dataset_type.upper()} | Seed {seed}")
    print(f"Directory: {out_dir}")
    print(f"--------------------------------------------------------")

    os.makedirs(out_dir, exist_ok=True)
    set_seed(seed)

    # DataLoaders
    t_pretrained = get_text_pretrained_name(cfg['teacher']['text'])
    s_pretrained = get_text_pretrained_name(cfg['student']['text'])
    t_tok = AutoTokenizer.from_pretrained(t_pretrained)
    s_tok = AutoTokenizer.from_pretrained(s_pretrained)
    dataset_root = cfg['data']['root']

    def make_ds(split):
        if dataset_type == 'medpix':
            return get_dataset(
                dataset_type='medpix',
                data_jsonl_file=os.path.join(dataset_root, f"splitted_dataset/data_{split}.jsonl"),
                desc_jsonl_file=os.path.join(dataset_root, f"splitted_dataset/descriptions_{split}.jsonl"),
                image_dir=os.path.join(dataset_root, "images"),
                tokenizer_teacher=t_tok,
                tokenizer_student=s_tok,
            )
        else:
            return get_dataset(
                dataset_type='wound',
                csv_file=os.path.join(dataset_root, f"metadata_{split}.csv"),
                image_dir=os.path.join(dataset_root, "images"),
                tokenizer_teacher=t_tok,
                tokenizer_student=s_tok,
                type_column=cfg['data'].get('type_column', 'type'),
                severity_column=cfg['data'].get('severity_column', 'severity'),
                description_column=cfg['data'].get('description_column', 'description'),
                filepath_column=cfg['data'].get('filepath_column', 'img_path'),
            )

    train_ds = make_ds("train")
    dev_ds = make_ds("dev")
    test_ds = make_ds("test")

    bs = int(cfg.get('data', {}).get('batch_size', 16))
    num_workers = int(cfg.get('data', {}).get('num_workers', 4))

    gen = torch.Generator()
    gen.manual_seed(seed)

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=num_workers, generator=gen)
    dev_loader = DataLoader(dev_ds, batch_size=bs, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=num_workers)

    classes = get_num_classes(dataset_type, dataset_root)
    num_mod_classes = classes['modality']
    num_loc_classes = classes['location']

    if modality == "image_only":
        model = VisionOnlyStudent(
            vision="mobilevit-xx-small",
            fusion_dim=256,
            dropout=0.292396,
            num_modality_classes=num_mod_classes,
            num_location_classes=num_loc_classes,
        ).to(device)
    else:
        model = TextOnlyStudent(
            text="bert-mini",
            fusion_dim=256,
            dropout=0.292396,
            num_modality_classes=num_mod_classes,
            num_location_classes=num_loc_classes,
        ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()

    best_dev_score = 0.0

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0
        for batch in train_loader:
            pv = batch['pixel_values'].to(device)
            ids = batch['input_ids_student'].to(device)
            mask = batch['attention_mask_student'].to(device)
            y_t1 = batch['modality'].to(device)
            y_t2 = batch['location'].to(device)

            out = model(pv, ids, mask)
            loss = ce(out['logits_modality'], y_t1) + ce(out['logits_location'], y_t2)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            steps += 1

        train_loss = total_loss / max(1, steps)

        # Dev evaluation
        dev_res = evaluate_with_perturbation(model, dev_loader, device, condition="clean", seed=seed,
                                             task1_label=task1_label, task2_label=task2_label)
        dev_score = dev_res['avg_f1']
        if dev_score > best_dev_score:
            best_dev_score = dev_score
            torch.save(model.state_dict(), ckpt_path)
            mark = "(*best*)"
        else:
            mark = ""

        print(f"  Epoch {ep:2d}/{epochs} | Train Loss: {train_loss:.4f} | Dev Avg F1: {dev_score*100:.2f}% {mark}")

    # Evaluate best model on test set
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

    test_res = evaluate_with_perturbation(model, test_loader, device, condition="clean", seed=seed,
                                          task1_label=task1_label, task2_label=task2_label)
    print(f"Test Result [{modality} | Seed {seed}]: Avg Acc: {test_res['avg_acc']*100:.2f}%, Avg F1: {test_res['avg_f1']*100:.2f}%")

    with open(results_path, 'w') as f:
        json.dump({'modality': modality, 'dataset': dataset_type, 'seed': seed,
                   'metrics': {'test': test_res}}, f, indent=2)

    return test_res


def format_stat(mean_val, std_val):
    return f"{mean_val * 100:.2f} ± {std_val * 100:.2f}"


def build_markdown_report(all_results: Dict, output_path: str):
    lines = []
    lines.append("# Multimodal Ablation & Robustness Study: MobileViT-xxs + BERT-mini\n")
    lines.append("Evaluated across both **MedPix** and **Wound** datasets averaged across seeds (42, 43, 44, 45, 46).\n")
    lines.append("- **Image-Only Baseline**: MobileViT-xxs trained from scratch.")
    lines.append("- **Text-Only Baseline**: BERT-mini trained from scratch.")
    lines.append("- **Proposed Multimodal Model**: MobileViT-xxs + BERT-mini (Cross-Attention).")
    lines.append("- **Perturbations on Trained Model**: Mismatch-Text, Noise ($\sigma=0.1, 0.2$), and Missing Modality (30% dropout).\n")

    for ds in ["medpix", "wound"]:
        ds_data = all_results.get(ds, {})
        t1 = ds_data.get('task1_label', 'Task1').capitalize()
        t2 = ds_data.get('task2_label', 'Task2').capitalize()

        lines.append(f"## Dataset: {ds.upper()}\n")
        lines.append(f"| Setting / Condition | Type | Overall Acc (%) | Overall Macro-F1 (%) | {t1} F1 (%) | {t2} F1 (%) |")
        lines.append(f"| :--- | :--- | :---: | :---: | :---: | :---: |")

        rows = ds_data.get('rows', [])
        for r in rows:
            name = r['name']
            stype = r['type']
            acc_str = format_stat(r['avg_acc_mean'], r['avg_acc_std'])
            f1_str = format_stat(r['avg_f1_mean'], r['avg_f1_std'])
            t1_str = format_stat(r[f'{ds_data["task1_label"]}_f1_mean'], r[f'{ds_data["task1_label"]}_f1_std'])
            t2_str = format_stat(r[f'{ds_data["task2_label"]}_f1_mean'], r[f'{ds_data["task2_label"]}_f1_std'])
            lines.append(f"| **{name}** | {stype} | {acc_str} | {f1_str} | {t1_str} | {t2_str} |")
        lines.append("\n")

    md_content = "\n".join(lines)
    with open(output_path, 'w') as f:
        f.write(md_content)
    print(f"\n[Report Saved] Summary report written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Ablation and Robustness Suite for MobileViT-xxs + BERT-mini")
    parser.add_argument("--eval-robustness", action="store_true", help="Evaluate trained multimodal model under perturbations")
    parser.add_argument("--train-unimodal", action="store_true", help="Train Image-only and Text-only unimodal baselines")
    parser.add_argument("--all", action="store_true", help="Run both unimodal baseline training and robustness evaluations")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46], help="Seeds to evaluate/train")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Compute device")
    parser.add_argument("--output-dir", type=str, default="logs/ablations", help="Output directory for reports")
    parser.add_argument("--force", action="store_true", help="Force retraining/re-evaluating even if results exist")
    args = parser.parse_args()

    if not (args.eval_robustness or args.train_unimodal or args.all):
        args.all = True

    os.makedirs(args.output_dir, exist_ok=True)
    datasets = ["medpix", "wound"]
    conditions = ["clean", "mismatch_text", "noise_0.1", "noise_0.2", "missing_30", "missing_image", "missing_text"]

    all_dataset_rows = {}

    for ds in datasets:
        all_dataset_rows[ds] = {'rows': []}

        # 1. Unimodal baselines
        unimodal_stats = {}
        for mod, name in [("image_only", "Image-Only (MobileViT-xxs)"), ("text_only", "Text-Only (BERT-mini)")]:
            mod_runs = []
            for s in args.seeds:
                res_path = f"logs/ablations/unimodal/{ds}/{mod}/seed_{s}/results.json"
                if os.path.exists(res_path) and not args.force:
                    with open(res_path, 'r') as f:
                        mod_runs.append(json.load(f).get('metrics', {}).get('test', {}))
                elif args.train_unimodal or args.all:
                    res = train_unimodal_baseline(mod, ds, s, args.device, force=args.force)
                    mod_runs.append(res)
            if len(mod_runs) == len(args.seeds) and len(mod_runs) > 0:
                keys = mod_runs[0].keys()
                stat_entry = {'name': name, 'type': 'Unimodal Baseline'}
                for k in keys:
                    vals = [r[k] for r in mod_runs]
                    stat_entry[f"{k}_mean"] = float(np.mean(vals))
                    stat_entry[f"{k}_std"] = float(np.std(vals))
                unimodal_stats[mod] = stat_entry

        # 2. Multimodal Robustness Evaluation
        robust_res = {}
        if args.eval_robustness or args.all:
            robust_res = evaluate_multimodal_robustness(ds, args.seeds, args.device, conditions)
            all_dataset_rows[ds]['task1_label'] = robust_res['task1_label']
            all_dataset_rows[ds]['task2_label'] = robust_res['task2_label']
        else:
            all_dataset_rows[ds]['task1_label'] = 'modality' if ds == 'medpix' else 'type'
            all_dataset_rows[ds]['task2_label'] = 'location' if ds == 'medpix' else 'severity'

        # Assemble orderly rows
        rows = []
        # Proposed Full Multimodal (Clean)
        if 'clean' in robust_res.get('summary', {}):
            s = robust_res['summary']['clean']
            s['name'] = "Proposed Full Multimodal (Clean)"
            s['type'] = "Proposed (Upper Bound)"
            rows.append(s)

        # Unimodal Baselines
        if "image_only" in unimodal_stats:
            rows.append(unimodal_stats["image_only"])
        if "text_only" in unimodal_stats:
            rows.append(unimodal_stats["text_only"])

        # Perturbation Ablations
        cond_display = [
            ("mismatch_text", "Mismatch-Text Pairing", "Cross-Modal Perturbation"),
            ("noise_0.1", "Noise (Image Gaussian $\sigma=0.1$)", "Robustness Perturbation"),
            ("noise_0.2", "Noise (Image Gaussian $\sigma=0.2$)", "Robustness Perturbation"),
            ("missing_30", "Missing Modality (30% Dropout)", "Missingness Perturbation"),
            ("missing_image", "Missing Image (Text Only Available)", "Degradation Evaluation"),
            ("missing_text", "Missing Text (Image Only Available)", "Degradation Evaluation"),
        ]

        for c_key, c_name, c_type in cond_display:
            if c_key in robust_res.get('summary', {}):
                s = robust_res['summary'][c_key]
                s['name'] = c_name
                s['type'] = c_type
                rows.append(s)

        all_dataset_rows[ds]['rows'] = rows

    # Write Markdown & JSON reports
    report_md = os.path.join(args.output_dir, "ablation_summary.md")
    report_json = os.path.join(args.output_dir, "ablation_summary.json")

    build_markdown_report(all_dataset_rows, report_md)
    with open(report_json, 'w') as f:
        json.dump(all_dataset_rows, f, indent=2)
    print(f"[Results JSON Saved] Raw metrics written to: {report_json}")


if __name__ == "__main__":
    main()
