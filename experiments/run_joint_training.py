#!/usr/bin/env python3
"""
Single-Phase Joint Training (Co-training / Online Knowledge Distillation)
for Multimodal Teacher and Student Models.

Enables rigorous side-by-side comparison between:
  1. Two-Phase Distillation (Teacher pre-trained, then frozen for student KD)
  2. Single-Phase Joint Training (Teacher and Student trained simultaneously in a single loop)

Includes comprehensive hardware profiling:
  - Peak allocated and reserved GPU VRAM (MB/GB)
  - Batch step latency (ms) and throughput (samples/sec)
  - Wall-clock epoch duration and total training time
"""

import argparse
import copy
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
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.dataset import get_dataset, get_num_classes
from models.backbones import get_text_pretrained_name
from models.teacher import Teacher
from models.student import Student
from losses.combined import MedKDCombinedLoss
from utils.logger import MetricsLogger
from utils.results_logger import ResultsLogger
from utils.metrics import evaluate_detailed, mcnemar_test


def set_seed(seed: int):
    if seed is None:
        return
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_parameters(model: nn.Module) -> Dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": total,
        "trainable_params": trainable,
        "params_millions": round(total / 1e6, 2),
    }


def get_model_size_mb(model: nn.Module) -> float:
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    total_bytes += sum(b.numel() * b.element_size() for b in model.buffers())
    return total_bytes / (1024 ** 2)


def build_dataloaders(cfg: Dict, dataset_type: str, seed: int = None, smoke_test: bool = False):
    t_pretrained = get_text_pretrained_name(cfg["teacher"]["text"])
    s_pretrained = get_text_pretrained_name(cfg["student"]["text"])
    t_tok = AutoTokenizer.from_pretrained(t_pretrained)
    s_tok = AutoTokenizer.from_pretrained(s_pretrained)
    dataset_root = cfg["data"]["root"]

    def make_ds(split):
        if dataset_type == "medpix":
            return get_dataset(
                dataset_type="medpix",
                data_jsonl_file=os.path.join(dataset_root, f"splitted_dataset/data_{split}.jsonl"),
                desc_jsonl_file=os.path.join(dataset_root, f"splitted_dataset/descriptions_{split}.jsonl"),
                image_dir=os.path.join(dataset_root, "images"),
                tokenizer_teacher=t_tok,
                tokenizer_student=s_tok,
            )
        else:
            return get_dataset(
                dataset_type="wound",
                csv_file=os.path.join(dataset_root, f"metadata_{split}.csv"),
                image_dir=os.path.join(dataset_root, "images"),
                tokenizer_teacher=t_tok,
                tokenizer_student=s_tok,
                type_column=cfg["data"].get("type_column", "type"),
                severity_column=cfg["data"].get("severity_column", "severity"),
                description_column=cfg["data"].get("description_column", "description"),
                filepath_column=cfg["data"].get("filepath_column", "img_path"),
            )

    train_ds = make_ds("train")
    dev_ds = make_ds("dev")
    test_ds = make_ds("test")

    batch_size = int(cfg.get("data", {}).get("batch_size", 16))
    num_workers = int(cfg.get("data", {}).get("num_workers", 2))

    generator = None
    worker_init_fn = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)
        worker_init_fn = seed_worker

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, generator=generator, worker_init_fn=worker_init_fn
    )
    dev_loader = DataLoader(
        dev_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, generator=generator, worker_init_fn=worker_init_fn
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, generator=generator, worker_init_fn=worker_init_fn
    )

    return train_loader, dev_loader, test_loader, train_ds, dev_ds, test_ds


def run_joint_training(
    dataset_type: str,
    seed: int,
    device_str: str,
    epochs: int = None,
    smoke_test: bool = False,
    mutual_distill: bool = False,
    output_base_dir: str = "logs/joint-training"
) -> Dict:
    print(f"\n{'='*75}")
    print(f"SINGLE-PHASE JOINT TRAINING: {dataset_type.upper()} | SEED {seed}")
    print(f"Device: {device_str} | Mode: {'Mutual Distillation' if mutual_distill else 'Online Distillation (Teacher -> Student)'}")
    print(f"{'='*75}")

    device = torch.device(device_str)
    set_seed(seed)

    cfg_path = f"config/ultra-edge-hp-tuned-all/{dataset_type}-mobilevit_xx_small-bert-mini.yaml"
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    task1_label = cfg["data"].get("task1_label", "modality" if dataset_type == "medpix" else "type")
    task2_label = cfg["data"].get("task2_label", "location" if dataset_type == "medpix" else "severity")

    exp_dir = os.path.join(output_base_dir, f"{dataset_type}-mobilevit_xx_small-bert-mini", f"seed_{seed}")
    os.makedirs(exp_dir, exist_ok=True)

    # 1. Dataloaders
    train_loader, dev_loader, test_loader, train_ds, dev_ds, test_ds = build_dataloaders(
        cfg, dataset_type, seed=seed, smoke_test=smoke_test
    )

    classes = get_num_classes(dataset_type, cfg["data"]["root"])
    num_mod_classes = classes["modality"]
    num_loc_classes = classes["location"]

    # 2. Setup Logger
    logger = MetricsLogger(exp_dir)
    if dataset_type == "medpix":
        mod_labels = [k for k, v in sorted(train_ds.modality_map.items(), key=lambda x: x[1])]
        loc_labels = [k for k, v in sorted(train_ds.location_map.items(), key=lambda x: x[1])]
    else:
        mod_labels = [v for k, v in sorted(train_ds.type_labels.items())]
        loc_labels = [v for k, v in sorted(train_ds.severity_labels.items())]

    logger.save_labels(mod_labels, task_name=task1_label)
    logger.save_labels(loc_labels, task_name=task2_label)

    # 3. Instantiate Models
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        mem_before_models = torch.cuda.memory_allocated() / (1024 ** 2)
    else:
        mem_before_models = 0.0

    print("Initializing Teacher Model (ViT-Base + Bio_ClinicalBERT)...")
    teacher = Teacher(
        vision=cfg["teacher"]["vision"],
        text=cfg["teacher"]["text"],
        fusion_type=cfg["fusion"]["type"],
        fusion_layers=cfg["teacher"]["fusion_layers"],
        fusion_dim=cfg["teacher"]["fusion_dim"],
        fusion_heads=cfg["teacher"]["fusion_heads"],
        dropout=cfg["teacher"]["dropout"],
        num_modality_classes=num_mod_classes,
        num_location_classes=num_loc_classes,
    ).to(device)

    print("Initializing Student Model (MobileViT-xxs + BERT-mini)...")
    student = Student(
        vision=cfg["student"]["vision"],
        text=cfg["student"]["text"],
        fusion_type=cfg["fusion"]["type"],
        fusion_layers=cfg["student"]["fusion_layers"],
        fusion_dim=cfg["student"]["fusion_dim"],
        fusion_heads=cfg["student"]["fusion_heads"],
        dropout=cfg["student"]["dropout"],
        num_modality_classes=num_mod_classes,
        num_location_classes=num_loc_classes,
    ).to(device)

    teacher_params = count_parameters(teacher)
    student_params = count_parameters(student)

    if torch.cuda.is_available() and device.type == "cuda":
        static_vram_mb = (torch.cuda.memory_allocated() / (1024 ** 2)) - mem_before_models
    else:
        static_vram_mb = 0.0

    print(f"Teacher Params: {teacher_params['params_millions']}M | Student Params: {student_params['params_millions']}M")
    print(f"Combined Static VRAM Footprint: {static_vram_mb:.2f} MB")

    # 4. Optimizers & Losses
    teacher_lr = float(cfg["training"].get("teacher_lr", 9.02e-5))
    student_lr = float(cfg["training"].get("student_lr", 3.67e-4))
    total_epochs = epochs or int(cfg["training"].get("student_epochs", 10))

    opt_teacher = torch.optim.AdamW(teacher.parameters(), lr=teacher_lr)
    opt_student = torch.optim.AdamW(student.parameters(), lr=student_lr)

    ce_loss = nn.CrossEntropyLoss()
    distill_fn = MedKDCombinedLoss(
        alpha=float(cfg["training"]["alpha"]),
        beta=float(cfg["training"]["beta"]),
        T=float(cfg["training"]["T"]),
    ).to(device)

    # Mutual distillation loss for teacher if enabled
    if mutual_distill:
        mutual_kl = nn.KLDivLoss(reduction="batchmean")

    best_dev_student = 0.0
    best_dev_teacher = 0.0
    best_student_path = os.path.join(exp_dir, "student_best.pth")
    best_teacher_path = os.path.join(exp_dir, "teacher_best.pth")

    # Hardware Profiling Trackers
    step_times_ms = []
    total_train_start_time = time.time()
    epoch_durations = []

    print(f"\n--- Starting Joint Optimization ({total_epochs} Epochs) ---")
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    start_event = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() and device.type == "cuda" else None
    end_event = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() and device.type == "cuda" else None

    for epoch in range(1, total_epochs + 1):
        teacher.train()
        student.train()

        total_loss_t = 0.0
        total_loss_s = 0.0
        steps = 0
        epoch_start_time = time.time()

        for batch_idx, batch in enumerate(train_loader):
            if smoke_test and batch_idx >= 2:
                break

            if start_event:
                start_event.record()
            batch_wall_start = time.time()

            pv = batch["pixel_values"].to(device)
            ids_t = batch["input_ids_teacher"].to(device)
            mask_t = batch["attention_mask_teacher"].to(device)
            ids_s = batch["input_ids_student"].to(device)
            mask_s = batch["attention_mask_student"].to(device)
            y_mod = batch["modality"].to(device)
            y_loc = batch["location"].to(device)

            opt_teacher.zero_grad()
            opt_student.zero_grad()

            # 1. Forward Teacher
            t_out = teacher(pv, ids_t, mask_t)
            loss_t_ce = ce_loss(t_out["logits_modality"], y_mod) + ce_loss(t_out["logits_location"], y_loc)

            # 2. Forward Student
            s_out = student(pv, ids_s, mask_s)

            # 3. Student Distillation Loss (Teacher output detached to preserve teacher training dynamics)
            t_out_detached = {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in t_out.items()}
            loss_s = distill_fn(s_out, t_out_detached, y_mod, y_loc)

            # Optional Mutual Distillation for Teacher
            if mutual_distill:
                T_temp = float(cfg["training"]["T"])
                kl_mod = mutual_kl(
                    F.log_softmax(t_out["logits_modality"] / T_temp, dim=-1),
                    F.softmax(s_out["logits_modality"].detach() / T_temp, dim=-1)
                ) * (T_temp ** 2)
                kl_loc = mutual_kl(
                    F.log_softmax(t_out["logits_location"] / T_temp, dim=-1),
                    F.softmax(s_out["logits_location"].detach() / T_temp, dim=-1)
                ) * (T_temp ** 2)
                loss_t = loss_t_ce + 0.1 * (kl_mod + kl_loc)
            else:
                loss_t = loss_t_ce

            # 4. Backward & Optimization
            loss_t.backward()
            loss_s.backward()

            opt_teacher.step()
            opt_student.step()

            total_loss_t += loss_t.item()
            total_loss_s += loss_s.item()
            steps += 1

            if end_event:
                end_event.record()
                torch.cuda.synchronize(device)
                step_ms = start_event.elapsed_time(end_event)
            else:
                step_ms = (time.time() - batch_wall_start) * 1000.0

            step_times_ms.append(step_ms)

        epoch_duration = time.time() - epoch_start_time
        epoch_durations.append(epoch_duration)
        avg_loss_t = total_loss_t / max(1, steps)
        avg_loss_s = total_loss_s / max(1, steps)

        # 5. Validation Evaluation on Dev Set
        dev_student = evaluate_detailed(
            student, dev_loader, device, logger=logger, split="dev", token_type="student",
            task1_label=task1_label, task2_label=task2_label
        )
        dev_teacher = evaluate_detailed(
            teacher, dev_loader, device, logger=logger, split="teacher_dev", token_type="teacher",
            task1_label=task1_label, task2_label=task2_label
        )

        s_score = (dev_student[f"dev_{task1_label}_f1"] + dev_student[f"dev_{task2_label}_f1"]) / 2.0
        t_score = (dev_teacher[f"teacher_dev_{task1_label}_f1"] + dev_teacher[f"teacher_dev_{task2_label}_f1"]) / 2.0

        mark_s = ""
        if s_score > best_dev_student:
            best_dev_student = s_score
            torch.save(student.state_dict(), best_student_path)
            mark_s = " (*best student*)"

        mark_t = ""
        if t_score > best_dev_teacher:
            best_dev_teacher = t_score
            torch.save(teacher.state_dict(), best_teacher_path)
            mark_t = " (*best teacher*)"

        print(
            f"Epoch {epoch:2d}/{total_epochs} | Time: {epoch_duration:.1f}s | "
            f"Teacher Loss: {avg_loss_t:.4f}, Dev F1: {t_score*100:.2f}%{mark_t} | "
            f"Student Loss: {avg_loss_s:.4f}, Dev F1: {s_score*100:.2f}%{mark_s}"
        )

        # Log epoch history
        all_metrics = copy.deepcopy(dev_student)
        all_metrics.update(dev_teacher)
        all_metrics["teacher_train_loss"] = avg_loss_t
        all_metrics["student_train_loss"] = avg_loss_s
        logger.log_epoch(epoch, avg_loss_s, all_metrics)

        if smoke_test:
            break

    total_train_duration = time.time() - total_train_start_time

    # 6. Capture Peak Hardware Stats
    if torch.cuda.is_available() and device.type == "cuda":
        peak_alloc_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        peak_res_mb = torch.cuda.max_memory_reserved() / (1024 ** 2)
    else:
        peak_alloc_mb, peak_res_mb = 0.0, 0.0

    avg_step_ms = float(np.mean(step_times_ms)) if step_times_ms else 0.0
    throughput = (int(cfg["data"]["batch_size"]) * 1000.0) / avg_step_ms if avg_step_ms > 0 else 0.0

    hardware_profile = {
        "static_vram_mb": static_vram_mb,
        "peak_allocated_vram_mb": peak_alloc_mb,
        "peak_allocated_vram_gb": peak_alloc_mb / 1024.0,
        "peak_reserved_vram_mb": peak_res_mb,
        "peak_reserved_vram_gb": peak_res_mb / 1024.0,
        "avg_step_time_ms": avg_step_ms,
        "throughput_samples_sec": throughput,
        "avg_epoch_duration_sec": float(np.mean(epoch_durations)) if epoch_durations else 0.0,
        "total_train_duration_sec": total_train_duration,
    }

    # 7. Final Test Evaluation (Reload Best Checkpoints)
    print("\n--- Final Test Evaluation ---")
    if os.path.exists(best_student_path):
        student.load_state_dict(torch.load(best_student_path, map_location=device))
    if os.path.exists(best_teacher_path):
        teacher.load_state_dict(torch.load(best_teacher_path, map_location=device))

    test_student, student_raw = evaluate_detailed(
        student, test_loader, device, logger=logger, split="test", token_type="student",
        task1_label=task1_label, task2_label=task2_label, return_raw=True
    )
    test_teacher, teacher_raw = evaluate_detailed(
        teacher, test_loader, device, logger=logger, split="teacher_test", token_type="teacher",
        task1_label=task1_label, task2_label=task2_label, return_raw=True
    )

    # Optional McNemar Test
    mcnemar_stats = {}
    try:
        m1 = mcnemar_test(student_raw["y_true_task1"], student_raw["y_pred_task1"], teacher_raw["y_pred_task1"])
        m2 = mcnemar_test(student_raw["y_true_task2"], student_raw["y_pred_task2"], teacher_raw["y_pred_task2"])
        mcnemar_stats[f"{task1_label}_p"] = m1.get("pvalue")
        mcnemar_stats[f"{task2_label}_p"] = m2.get("pvalue")
    except Exception as e:
        print(f"[Warning] McNemar test failed: {e}")

    logger.save_csv()
    logger.save_json()

    # Save comprehensive results.json
    results_payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "mode": "single_phase_joint_training",
        "dataset": dataset_type,
        "seed": seed,
        "hardware_profile": hardware_profile,
        "student_params": student_params,
        "teacher_params": teacher_params,
        "student_test_metrics": test_student,
        "teacher_test_metrics": test_teacher,
        "mcnemar": mcnemar_stats,
    }

    results_file = os.path.join(exp_dir, "results.json")
    with open(results_file, "w") as f:
        json.dump(results_payload, f, indent=2)

    print(f"\n[Saved] Joint training results written to: {results_file}")
    print(f"Student Test Result: Acc: {(test_student.get('test_modality_acc', 0) + test_student.get('test_location_acc', 0) + test_student.get('test_type_acc', 0) + test_student.get('test_severity_acc', 0)) / 2 * 100:.2f}%, "
          f"F1: {(test_student.get(f'test_{task1_label}_f1', 0) + test_student.get(f'test_{task2_label}_f1', 0)) / 2 * 100:.2f}%")
    print(f"Hardware Peak VRAM: {peak_alloc_mb / 1024:.2f} GB | Throughput: {throughput:.1f} samples/s | Total Time: {total_train_duration:.1f}s")

    return results_payload


def generate_comparison_report(datasets: List[str], seeds: List[int], output_dir: str = "logs/joint-training"):
    os.makedirs(output_dir, exist_ok=True)
    report_lines = []
    report_lines.append("# Comparative Analysis: Two-Phase Distillation vs. Single-Phase Joint Training\n")
    report_lines.append("A direct empirical comparison evaluating whether training Teacher and Student simultaneously in a **Single Unified Phase** performs better or worse than the canonical **Two-Phase Knowledge Distillation** pipeline.\n")

    report_lines.append("### Pipeline Overview:")
    report_lines.append("- **Two-Phase Distillation (Offline KD)**: Phase 1 trains Teacher to convergence ($E_T=3$). Phase 2 freezes Teacher and distills Student ($E_S=10$).")
    report_lines.append("- **Single-Phase Joint Training (Online KD)**: Teacher and Student start from pretrained backbones and co-evolve simultaneously ($E_{\\text{joint}}=10$), with Student distilling from dynamic teacher features in every batch.\n")

    summary_json_data = {}

    for ds in datasets:
        task1_name = "Modality" if ds == "medpix" else "Type"
        task2_name = "Location" if ds == "medpix" else "Severity"
        t1_key = task1_name.lower()
        t2_key = task2_name.lower()

        # Collect Two-Phase metrics
        two_phase_student_runs = []
        two_phase_teacher_runs = []
        # Collect Single-Phase metrics
        joint_student_runs = []
        joint_teacher_runs = []
        joint_hardware_runs = []

        for s in seeds:
            # Two-phase file
            two_p_file = f"logs/ultra-edge-hp-tuned-all/{ds}-mobilevit_xx_small-bert-mini/seed_{s}/results.json"
            if os.path.exists(two_p_file):
                with open(two_p_file, "r") as f:
                    d = json.load(f)
                    m = d.get("metrics", {})
                    # Student test
                    s_test = m.get("test", {})
                    if s_test:
                        acc = (s_test.get(f"test_{t1_key}_acc", 0) + s_test.get(f"test_{t2_key}_acc", 0)) / 2.0
                        f1 = (s_test.get(f"test_{t1_key}_f1", 0) + s_test.get(f"test_{t2_key}_f1", 0)) / 2.0
                        two_phase_student_runs.append({
                            "acc": acc, "f1": f1,
                            "t1_f1": s_test.get(f"test_{t1_key}_f1", 0),
                            "t2_f1": s_test.get(f"test_{t2_key}_f1", 0),
                        })
                    # Teacher test
                    t_test = m.get("teacher", {}).get("test", {})
                    if t_test:
                        acc = (t_test.get(f"teacher_test_{t1_key}_acc", 0) + t_test.get(f"teacher_test_{t2_key}_acc", 0)) / 2.0
                        f1 = (t_test.get(f"teacher_test_{t1_key}_f1", 0) + t_test.get(f"teacher_test_{t2_key}_f1", 0)) / 2.0
                        two_phase_teacher_runs.append({"acc": acc, "f1": f1})

            # Single-phase joint file
            joint_file = f"{output_dir}/{ds}-mobilevit_xx_small-bert-mini/seed_{s}/results.json"
            if os.path.exists(joint_file):
                with open(joint_file, "r") as f:
                    d = json.load(f)
                    s_test = d.get("student_test_metrics", {})
                    if s_test:
                        acc = (s_test.get(f"test_{t1_key}_acc", 0) + s_test.get(f"test_{t2_key}_acc", 0)) / 2.0
                        f1 = (s_test.get(f"test_{t1_key}_f1", 0) + s_test.get(f"test_{t2_key}_f1", 0)) / 2.0
                        joint_student_runs.append({
                            "acc": acc, "f1": f1,
                            "t1_f1": s_test.get(f"test_{t1_key}_f1", 0),
                            "t2_f1": s_test.get(f"test_{t2_key}_f1", 0),
                        })
                    t_test = d.get("teacher_test_metrics", {})
                    if t_test:
                        acc = (t_test.get(f"teacher_test_{t1_key}_acc", 0) + t_test.get(f"teacher_test_{t2_key}_acc", 0)) / 2.0
                        f1 = (t_test.get(f"teacher_test_{t1_key}_f1", 0) + t_test.get(f"teacher_test_{t2_key}_f1", 0)) / 2.0
                        joint_teacher_runs.append({"acc": acc, "f1": f1})
                    hw = d.get("hardware_profile", {})
                    if hw:
                        joint_hardware_runs.append(hw)

        def calc_stat(run_list, key):
            if not run_list:
                return "N/A"
            vals = [r[key] * 100 for r in run_list if key in r]
            if not vals:
                return "N/A"
            return f"{np.mean(vals):.2f} ± {np.std(vals):.2f}"

        report_lines.append(f"## Dataset: {ds.upper()}\n")
        report_lines.append("### 1. Classification Performance Comparison (Student & Teacher)")
        report_lines.append(f"| Model & Training Paradigm | Overall Acc (%) | Overall Macro-F1 (%) | {task1_name} F1 (%) | {task2_name} F1 (%) |")
        report_lines.append("| :--- | :---: | :---: | :---: | :---: |")

        # Two-Phase Teacher
        tp_t_acc = calc_stat(two_phase_teacher_runs, "acc")
        tp_t_f1 = calc_stat(two_phase_teacher_runs, "f1")
        report_lines.append(f"| **Teacher: Two-Phase (Pre-trained)** | {tp_t_acc} | {tp_t_f1} | — | — |")

        # Joint Teacher
        j_t_acc = calc_stat(joint_teacher_runs, "acc")
        j_t_f1 = calc_stat(joint_teacher_runs, "f1")
        report_lines.append(f"| **Teacher: Single-Phase (Co-trained)** | {j_t_acc} | {j_t_f1} | — | — |")

        # Two-Phase Student
        tp_s_acc = calc_stat(two_phase_student_runs, "acc")
        tp_s_f1 = calc_stat(two_phase_student_runs, "f1")
        tp_s_t1 = calc_stat(two_phase_student_runs, "t1_f1")
        tp_s_t2 = calc_stat(two_phase_student_runs, "t2_f1")
        report_lines.append(f"| **Student: Two-Phase Distillation (Baseline)** | **{tp_s_acc}** | **{tp_s_f1}** | **{tp_s_t1}** | **{tp_s_t2}** |")

        # Single-Phase Student
        j_s_acc = calc_stat(joint_student_runs, "acc")
        j_s_f1 = calc_stat(joint_student_runs, "f1")
        j_s_t1 = calc_stat(joint_student_runs, "t1_f1")
        j_s_t2 = calc_stat(joint_student_runs, "t2_f1")
        report_lines.append(f"| **Student: Single-Phase Joint Training (Proposed)** | {j_s_acc} | {j_s_f1} | {j_s_t1} | {j_s_t2} |")
        report_lines.append("\n")

        # Hardware profiling table
        report_lines.append("### 2. Hardware Resource & Efficiency Profiling")
        report_lines.append("| Paradigm | Peak Allocated VRAM | Peak Reserved VRAM | Step Latency | Throughput | Training Duration |")
        report_lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")

        # Two-phase reference stats (from our comprehensive profiling in docs/resource_usage.md)
        tp_vram_gb = "6.45 GB (Phase 1) / 2.78 GB (Phase 2)"
        tp_res_gb = "7.08 GB (Phase 1) / 3.11 GB (Phase 2)"
        tp_step = "~530 ms (Phase 1) / ~222 ms (Phase 2)"
        tp_thru = "~30.2 s/s (Phase 1) / ~71.8 s/s (Phase 2)"
        tp_time = "~395s total (3 ep Teacher + 10 ep Student)" if ds == "medpix" else "~260s total (3 ep Teacher + 10 ep Student)"

        report_lines.append(f"| **Two-Phase Distillation** | {tp_vram_gb} | {tp_res_gb} | {tp_step} | {tp_thru} | {tp_time} |")

        if joint_hardware_runs:
            j_vram_mean = np.mean([h["peak_allocated_vram_gb"] for h in joint_hardware_runs])
            j_res_mean = np.mean([h["peak_reserved_vram_gb"] for h in joint_hardware_runs])
            j_step_mean = np.mean([h["avg_step_time_ms"] for h in joint_hardware_runs])
            j_thru_mean = np.mean([h["throughput_samples_sec"] for h in joint_hardware_runs])
            j_time_mean = np.mean([h["total_train_duration_sec"] for h in joint_hardware_runs])

            report_lines.append(
                f"| **Single-Phase Joint Training** | **{j_vram_mean:.2f} GB** | **{j_res_mean:.2f} GB** | "
                f"**{j_step_mean:.1f} ms** | **{j_thru_mean:.1f} samples/s** | **{j_time_mean:.1f} s** |"
            )
        else:
            report_lines.append("| **Single-Phase Joint Training** | *Pending Run* | *Pending Run* | *Pending Run* | *Pending Run* | *Pending Run* |")

        report_lines.append("\n---\n")

    summary_md_path = os.path.join(output_dir, "comparison_summary.md")
    with open(summary_md_path, "w") as f:
        f.write("\n".join(report_lines))

    print(f"\n[Comparison Report Generated] Written to: {summary_md_path}")


def main():
    parser = argparse.ArgumentParser(description="Single-Phase Joint Training & Hardware Profiling Engine")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML (optional)")
    parser.add_argument("--dataset", type=str, choices=["medpix", "wound", "all"], default="medpix", help="Dataset to run")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42], help="List of random seeds")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Compute device")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of joint training epochs")
    parser.add_argument("--mutual", action="store_true", help="Enable mutual bidirectional distillation")
    parser.add_argument("--smoke-test", action="store_true", help="Fast test pass (2 batches, 1 epoch)")
    parser.add_argument("--compare-only", action="store_true", help="Only generate comparison report from existing runs")
    parser.add_argument("--output-dir", type=str, default="logs/joint-training", help="Base output directory")
    args = parser.parse_args()

    datasets = ["medpix", "wound"] if args.dataset == "all" else [args.dataset]

    if not args.compare_only:
        for ds in datasets:
            for s in args.seeds:
                run_joint_training(
                    dataset_type=ds,
                    seed=s,
                    device_str=args.device,
                    epochs=args.epochs,
                    smoke_test=args.smoke_test,
                    mutual_distill=args.mutual,
                    output_base_dir=args.output_dir,
                )

    # Generate or update side-by-side comparison tables
    generate_comparison_report(datasets=datasets, seeds=args.seeds, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
