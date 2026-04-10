import argparse
import os
import random
import sys
from copy import deepcopy

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Ensure repository root is importable when executing as a script.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.dataset import get_dataset, get_num_classes  # noqa: E402
from models.backbones import get_text_pretrained_name  # noqa: E402
from models.student import Student  # noqa: E402
from models.student_custom import StudentCustom  # noqa: E402
from utils.logger import MetricsLogger  # noqa: E402
from utils.metrics import evaluate_detailed  # noqa: E402
from utils.results_logger import ResultsLogger  # noqa: E402


def set_seed(seed: int):
    if seed is None:
        return
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    params_millions = total_params / 1e6
    return {
        "total_params": int(total_params),
        "params_millions": round(float(params_millions), 2),
    }


def train_student_ce_only(student, loader, device, epochs, lr):
    try:
        epochs = int(epochs)
    except Exception:
        raise TypeError(f"student epochs must be int-like, got {type(epochs)}: {epochs}")
    try:
        lr = float(lr)
    except Exception:
        raise TypeError(f"student lr must be float-like, got {type(lr)}: {lr}")

    student.train()
    opt = torch.optim.AdamW(student.parameters(), lr=lr)
    ce = torch.nn.CrossEntropyLoss()

    last_avg = 0.0
    for _ in range(epochs):
        total = 0.0
        nsteps = 0
        for batch in loader:
            pv = batch["pixel_values"].to(device)
            ids_s = batch["input_ids_student"].to(device)
            mask_s = batch["attention_mask_student"].to(device)
            y_mod = batch["modality"].to(device)
            y_loc = batch["location"].to(device)

            s_out = student(pv, ids_s, mask_s)
            loss = ce(s_out["logits_modality"], y_mod) + ce(s_out["logits_location"], y_loc)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += float(loss.item())
            nsteps += 1
        last_avg = total / max(1, nsteps)
    return student, last_avg


def _get_vocab_size(tokenizer):
    vs = getattr(tokenizer, "vocab_size", None)
    if vs is not None:
        return int(vs)
    try:
        return len(tokenizer.get_vocab())
    except Exception:
        raise ValueError("Unable to determine tokenizer vocab size")


def main(cfg: dict, cfg_path: str, overrides: dict):
    cfg = deepcopy(cfg)

    # Force a CE-only / No-KD run and record that intent in the saved config.
    cfg.setdefault("loss", {})
    cfg["loss"]["type"] = "ce_only"
    cfg.setdefault("training", {})
    cfg["training"]["teacher_epochs"] = 0

    if overrides.get("device") is not None:
        cfg["device"] = overrides["device"]

    if overrides.get("student_epochs") is not None:
        cfg["training"]["student_epochs"] = int(overrides["student_epochs"])

    if overrides.get("student_lr") is not None:
        cfg["training"]["student_lr"] = float(overrides["student_lr"])

    if overrides.get("log_dir") is not None:
        cfg.setdefault("logging", {})
        cfg["logging"]["log_dir"] = overrides["log_dir"]

    device = torch.device(cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))

    # Tokenizers: keep teacher tokenizer for dataset fields, even though teacher is unused.
    t_text_name = cfg.get("teacher", {}).get("text")
    t_pretrained = get_text_pretrained_name(t_text_name) if t_text_name else None
    if t_pretrained is None:
        raise KeyError(f"Teacher text backbone '{t_text_name}' has no known pretrained mapping")

    s_text_name = cfg.get("student", {}).get("text")
    s_tokenizer_name = cfg.get("student", {}).get("tokenizer") or s_text_name
    s_pretrained = get_text_pretrained_name(s_tokenizer_name) if s_tokenizer_name else None
    if s_pretrained is None:
        raise KeyError(
            "Student tokenizer backbone has no known pretrained mapping. "
            "Set student.tokenizer (or student.text) to a supported name."
        )

    teacher_tokenizer = AutoTokenizer.from_pretrained(t_pretrained)
    student_tokenizer = AutoTokenizer.from_pretrained(s_pretrained)

    dataset_type = cfg.get("data", {}).get("type", "medpix")
    dataset_root = cfg.get("data", {}).get("root")
    if not dataset_root:
        raise KeyError("data.root must be set")

    def make_dataset(split: str):
        if dataset_type == "medpix":
            return get_dataset(
                dataset_type="medpix",
                data_jsonl_file=os.path.join(dataset_root, f"splitted_dataset/data_{split}.jsonl"),
                desc_jsonl_file=os.path.join(dataset_root, f"splitted_dataset/descriptions_{split}.jsonl"),
                image_dir=os.path.join(dataset_root, "images"),
                tokenizer_teacher=teacher_tokenizer,
                tokenizer_student=student_tokenizer,
            )
        if dataset_type == "wound":
            return get_dataset(
                dataset_type="wound",
                csv_file=os.path.join(dataset_root, f"metadata_{split}.csv"),
                image_dir=os.path.join(dataset_root, "images"),
                tokenizer_teacher=teacher_tokenizer,
                tokenizer_student=student_tokenizer,
                type_column=cfg.get("data", {}).get("type_column", "type"),
                severity_column=cfg.get("data", {}).get("severity_column", "severity"),
                description_column=cfg.get("data", {}).get("description_column", "description"),
                filepath_column=cfg.get("data", {}).get("filepath_column", "img_path"),
            )
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    train_dataset = make_dataset("train")
    dev_dataset = make_dataset("dev")
    test_dataset = make_dataset("test")

    num_classes = get_num_classes(
        dataset_type=dataset_type,
        dataset_root=dataset_root,
        type_column=cfg.get("data", {}).get("type_column", "type"),
        severity_column=cfg.get("data", {}).get("severity_column", "severity"),
    )
    num_modality_classes = int(num_classes["modality"])
    num_location_classes = int(num_classes["location"])

    task1_label = cfg.get("data", {}).get("task1_label", "modality")
    task2_label = cfg.get("data", {}).get("task2_label", "location")

    # Logging
    log_dir = cfg.get("logging", {}).get("log_dir")
    if not log_dir:
        base = os.path.splitext(os.path.basename(cfg_path))[0]
        log_dir = os.path.join("logs", "no-kd", base)
        cfg.setdefault("logging", {})
        cfg["logging"]["log_dir"] = log_dir

    logger = MetricsLogger(log_dir)

    # Save class labels for confusion matrices
    if dataset_type == "medpix":
        modality_labels = [k for k, v in sorted(train_dataset.modality_map.items(), key=lambda x: x[1])]
        location_labels = [k for k, v in sorted(train_dataset.location_map.items(), key=lambda x: x[1])]
    elif dataset_type == "wound":
        modality_labels = [v for k, v in sorted(train_dataset.type_labels.items())]
        location_labels = [v for k, v in sorted(train_dataset.severity_labels.items())]
    else:
        modality_labels, location_labels = [], []

    logger.save_labels(modality_labels, task_name=task1_label)
    logger.save_labels(location_labels, task_name=task2_label)

    try:
        batch_size = int(cfg.get("data", {}).get("batch_size", 16))
    except Exception:
        raise TypeError(f"data.batch_size must be int-like, got {cfg.get('data', {}).get('batch_size')}")

    try:
        num_workers = int(cfg.get("data", {}).get("num_workers", 4))
    except Exception:
        raise TypeError(f"data.num_workers must be int-like, got {cfg.get('data', {}).get('num_workers')}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Model
    fusion_type = cfg.get("fusion", {}).get("type", "simple")
    fusion_params = cfg.get("fusion", {})

    student_arch = cfg.get("student", {}).get("arch", "standard")
    student_fusion_heads = cfg.get("student", {}).get("fusion_heads", 8)
    student_dropout = cfg.get("student", {}).get("dropout", 0.1)

    if str(student_arch).lower() == "custom_tiny":
        custom_cfg = cfg.get("student", {}).get("custom", {}) or {}
        vocab_size = _get_vocab_size(student_tokenizer)
        pad_token_id = getattr(student_tokenizer, "pad_token_id", 0) or 0
        student = StudentCustom(
            vocab_size=vocab_size,
            pad_token_id=pad_token_id,
            vision_d=int(custom_cfg.get("vision_d", 192)),
            vision_depth=int(custom_cfg.get("vision_depth", 4)),
            vision_heads=int(custom_cfg.get("vision_heads", 3)),
            vision_mlp_ratio=float(custom_cfg.get("vision_mlp_ratio", 4.0)),
            text_d=int(custom_cfg.get("text_d", 128)),
            text_depth=int(custom_cfg.get("text_depth", 4)),
            text_heads=int(custom_cfg.get("text_heads", 4)),
            text_mlp_ratio=float(custom_cfg.get("text_mlp_ratio", 4.0)),
            max_len=int(custom_cfg.get("max_len", 256)),
            fusion_dim=int(custom_cfg.get("fusion_dim", cfg.get("student", {}).get("fusion_dim", 256))),
            dropout=float(custom_cfg.get("dropout", student_dropout)),
            num_modality_classes=num_modality_classes,
            num_location_classes=num_location_classes,
        ).to(device)
    else:
        student = Student(
            vision=cfg.get("student", {}).get("vision"),
            text=cfg.get("student", {}).get("text"),
            fusion_type=fusion_type,
            fusion_layers=cfg.get("student", {}).get("fusion_layers"),
            fusion_dim=cfg.get("student", {}).get("fusion_dim"),
            fusion_heads=student_fusion_heads,
            dropout=student_dropout,
            num_modality_classes=num_modality_classes,
            num_location_classes=num_location_classes,
            fusion_params=fusion_params,
        ).to(device)

    student_params = count_parameters(student)
    print(f"Student parameters: {student_params['params_millions']:.2f}M ({student_params['total_params']:,})")

    # Train CE-only
    try:
        student_epochs = int(cfg.get("training", {}).get("student_epochs", 1))
    except Exception:
        raise TypeError(f"training.student_epochs must be int-like, got {cfg.get('training', {}).get('student_epochs')}")

    student_lr = cfg.get("training", {}).get("student_lr", 3e-4)

    best_dev_score = -1.0
    best_path = os.path.join(log_dir, "student_best.pth")

    for epoch in range(1, student_epochs + 1):
        student, train_loss = train_student_ce_only(student, train_loader, device, epochs=1, lr=student_lr)
        dev_metrics = evaluate_detailed(
            student,
            dev_loader,
            device,
            logger=logger,
            split="dev",
            token_type="student",
            task1_label=task1_label,
            task2_label=task2_label,
        )

        dev_score = (dev_metrics[f"dev_{task1_label}_f1"] + dev_metrics[f"dev_{task2_label}_f1"]) / 2
        if dev_score > best_dev_score:
            best_dev_score = float(dev_score)
            torch.save(student.state_dict(), best_path)
            print(f"  New best dev score: {best_dev_score:.4f}")

        all_metrics = {}
        all_metrics.update(dev_metrics)
        logger.log_epoch(epoch, train_loss, all_metrics)

    # Test best checkpoint
    test_metrics = {}
    if os.path.exists(best_path):
        student.load_state_dict(torch.load(best_path, map_location=device))
        test_metrics = evaluate_detailed(
            student,
            test_loader,
            device,
            logger=logger,
            split="test",
            token_type="student",
            task1_label=task1_label,
            task2_label=task2_label,
        )

    final_path = os.path.join(log_dir, "student_final.pth")
    torch.save(student.state_dict(), final_path)

    logger.save_csv()
    logger.save_json()

    results_logger = ResultsLogger(log_dir)
    try:
        serial_history = {k: list(v) for k, v in logger.history.items()}
    except Exception:
        serial_history = {}

    train_metrics = {"train": {"history": serial_history, "final_loss": train_loss}}
    results_logger.log_experiment(
        cfg,
        train_metrics,
        dev_metrics=dev_metrics,
        test_metrics=test_metrics,
        teacher_dev_metrics={},
        teacher_test_metrics={},
        teacher_params=None,
        student_params=student_params,
    )

    print(f"All outputs saved in {log_dir}")


def parse_args():
    p = argparse.ArgumentParser(description="Train Student with CE-only (No KD) on MedPix or Wound")
    p.add_argument("cfg", help="Path to a YAML config (e.g., config/ultra-edge-hp-tuned-all/wound-....yaml)")
    p.add_argument("--device", default=None, help="Override device, e.g. cuda:0 or cpu")
    p.add_argument("--log-dir", default=None, help="Override logging.log_dir")
    p.add_argument("--student-epochs", type=int, default=None, help="Override training.student_epochs")
    p.add_argument("--student-lr", type=float, default=None, help="Override training.student_lr")
    p.add_argument("--seed", type=int, default=None, help="Random seed")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)

    with open(args.cfg, "r") as f:
        cfg = yaml.safe_load(f)

    overrides = {
        "device": args.device,
        "log_dir": args.log_dir,
        "student_epochs": args.student_epochs,
        "student_lr": args.student_lr,
    }

    main(cfg, cfg_path=args.cfg, overrides=overrides)
