import torch
from torch.utils.data import DataLoader
from data.dataset import get_dataset, get_num_classes, MedPixDataset, WoundDataset
from models.teacher import Teacher
from models.student import Student
import importlib
import inspect
from utils.logger import MetricsLogger
from utils.results_logger import ResultsLogger
from utils.metrics import evaluate_detailed
import yaml
import os
from datetime import datetime
from transformers import AutoTokenizer
from models.backbones import get_text_pretrained_name

def count_parameters(model):
    """
    Count the total number of parameters in a model.
    
    Args:
        model: PyTorch model
    
    Returns:
        dict with total_params (int) and params_millions (float)
    """
    total_params = sum(p.numel() for p in model.parameters())
    params_millions = total_params / 1e6
    return {
        'total_params': total_params,
        'params_millions': round(params_millions, 2)
    }

def train_teacher(model, loader, device, epochs, lr):
    # defensive: ensure epochs and lr are numeric
    try:
        epochs = int(epochs)
    except Exception:
        raise TypeError(f"teacher epochs must be int-like, got {type(epochs)}: {epochs}")
    try:
        lr = float(lr)
    except Exception:
        raise TypeError(f"teacher lr must be float-like, got {type(lr)}: {lr}")

    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    ce = torch.nn.CrossEntropyLoss()
    for epoch in range(epochs):
        total = 0.0
        nsteps = 0
        for batch in loader:
            pv = batch['pixel_values'].to(device)
            ids = batch['input_ids_teacher'].to(device)
            mask = batch['attention_mask_teacher'].to(device)
            y_mod = batch['modality'].to(device)
            y_loc = batch['location'].to(device)
            out = model(pv, ids, mask)
            loss = ce(out['logits_modality'], y_mod) + ce(out['logits_location'], y_loc)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            nsteps += 1
        avg = total / max(1, nsteps)
        print(f"Teacher Epoch {epoch+1}/{epochs} – loss {avg:.4f}")
    return model

def train_student(student, teacher, loader, device, epochs, lr, distill_fn):
    teacher.eval()
    # defensive: ensure epochs and lr are numeric
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
    for epoch in range(epochs):
        total = 0.0
        nsteps = 0
        for batch in loader:
            pv = batch['pixel_values'].to(device)
            ids_s = batch['input_ids_student'].to(device)
            mask_s = batch['attention_mask_student'].to(device)
            ids_t = batch['input_ids_teacher'].to(device)
            mask_t = batch['attention_mask_teacher'].to(device)
            y_mod = batch['modality'].to(device)
            y_loc = batch['location'].to(device)
            with torch.no_grad():
                t_out = teacher(pv, ids_t, mask_t)
            s_out = student(pv, ids_s, mask_s)
            loss = distill_fn(s_out, t_out, y_mod, y_loc)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            nsteps += 1
        avg = total / max(1, nsteps)
        print(f"Student Epoch {epoch+1}/{epochs} – loss {avg:.4f}")
    # Return average loss for the final epoch to allow logging
    return student, avg

def main(cfg):
    # Allow overriding device from the config (e.g. 'cuda:3' or 'cpu').
    cfg_device = cfg.get('device', None)
    if cfg_device:
        # trust user-provided device string
        device = torch.device(cfg_device)
    else:
        device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

    # Load tokenizers that match configured text backbones to avoid
    # token-id vs embedding-size mismatches when swapping backbones.
    t_text_name = cfg.get('teacher', {}).get('text')
    s_text_name = cfg.get('student', {}).get('text')
    t_pretrained = get_text_pretrained_name(t_text_name) if t_text_name else None
    s_pretrained = get_text_pretrained_name(s_text_name) if s_text_name else None

    if t_pretrained is None:
        raise KeyError(f"Teacher text backbone '{t_text_name}' has no known pretrained mapping")
    if s_pretrained is None:
        raise KeyError(f"Student text backbone '{s_text_name}' has no known pretrained mapping")

    teacher_tokenizer = AutoTokenizer.from_pretrained(t_pretrained)
    student_tokenizer = AutoTokenizer.from_pretrained(s_pretrained)
    
    # Determine dataset type (default to 'medpix' for backward compatibility)
    dataset_type = cfg.get('data', {}).get('type', 'medpix')
    dataset_root = cfg['data']['root']
    
    def make_dataset(split):
        if dataset_type == 'medpix':
            return get_dataset(
                dataset_type='medpix',
                data_jsonl_file=os.path.join(dataset_root, f"splitted_dataset/data_{split}.jsonl"),
                desc_jsonl_file=os.path.join(dataset_root, f"splitted_dataset/descriptions_{split}.jsonl"),
                image_dir=os.path.join(dataset_root, "images"),
                tokenizer_teacher=teacher_tokenizer,
                tokenizer_student=student_tokenizer,
            )
        elif dataset_type == 'wound':
            return get_dataset(
                dataset_type='wound',
                csv_file=os.path.join(dataset_root, f"metadata_{split}.csv"),
                image_dir=os.path.join(dataset_root, "images"),
                tokenizer_teacher=teacher_tokenizer,
                tokenizer_student=student_tokenizer,
                type_column=cfg['data'].get('type_column', 'type'),
                severity_column=cfg['data'].get('severity_column', 'severity'),
                description_column=cfg['data'].get('description_column', 'description'),
                filepath_column=cfg['data'].get('filepath_column', 'file_path'),
            )
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    train_dataset = make_dataset("train")
    dev_dataset = make_dataset("dev")
    test_dataset = make_dataset("test")
    
    # Get number of classes for this dataset (dynamic for wound, static for medpix)
    num_classes = get_num_classes(
        dataset_type=dataset_type, 
        dataset_root=dataset_root,
        type_column=cfg['data'].get('type_column', 'type'),
        severity_column=cfg['data'].get('severity_column', 'severity'),
    )
    num_modality_classes = num_classes['modality']
    num_location_classes = num_classes['location']
    
    # Get task labels for metrics (defaults for backward compatibility)
    task1_label = cfg['data'].get('task1_label', 'modality')
    task2_label = cfg['data'].get('task2_label', 'location')

    # Instantiate logger before saving labels
    logger = MetricsLogger(cfg['logging']['log_dir'])

    # Save class labels for confusion matrices
    # For MedPix: modality_map and location_map are available on the dataset instance
    # For Wound: use type_labels and severity_labels from the train dataset
    if dataset_type == 'medpix':
        modality_labels = [k for k, v in sorted(train_dataset.modality_map.items(), key=lambda x: x[1])]
        location_labels = [k for k, v in sorted(train_dataset.location_map.items(), key=lambda x: x[1])]
    elif dataset_type == 'wound':
        modality_labels = [v for k, v in sorted(train_dataset.type_labels.items())]
        location_labels = [v for k, v in sorted(train_dataset.severity_labels.items())]
    else:
        modality_labels = []
        location_labels = []

    logger.save_labels(modality_labels, task_name=task1_label)
    logger.save_labels(location_labels, task_name=task2_label)
    
    # defensive parsing of common numeric config values
    try:
        batch_size = int(cfg['data'].get('batch_size', 16))
    except Exception:
        raise TypeError(f"data.batch_size must be int-like, got {cfg['data'].get('batch_size')}")
    try:
        num_workers = int(cfg['data'].get('num_workers', 4))
    except Exception:
        raise TypeError(f"data.num_workers must be int-like, got {cfg['data'].get('num_workers')}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # Get fusion type from config (default to 'simple' for backward compatibility)
    fusion_type = cfg.get('fusion', {}).get('type', 'simple')
    
    # Extract fusion_heads and dropout from config (with defaults for backward compatibility)
    teacher_fusion_heads = cfg.get('teacher', {}).get('fusion_heads', 8)
    teacher_dropout = cfg.get('teacher', {}).get('dropout', 0.1)
    student_fusion_heads = cfg.get('student', {}).get('fusion_heads', 8)
    student_dropout = cfg.get('student', {}).get('dropout', 0.1)
    
    # Get fusion-specific parameters (optional, used by specific fusion modules)
    fusion_params = cfg.get('fusion', {})
    
    teacher = Teacher(
        vision=cfg['teacher']['vision'],
        text=cfg['teacher']['text'],
        fusion_type=fusion_type,
        fusion_layers=cfg['teacher']['fusion_layers'],
        fusion_dim=cfg['teacher']['fusion_dim'],
        fusion_heads=teacher_fusion_heads,
        dropout=teacher_dropout,
        num_modality_classes=num_modality_classes,
        num_location_classes=num_location_classes,
        fusion_params=fusion_params,
    ).to(device)
    
    student = Student(
        vision=cfg['student']['vision'],
        text=cfg['student']['text'],
        fusion_type=fusion_type,
        fusion_layers=cfg['student']['fusion_layers'],
        fusion_dim=cfg['student']['fusion_dim'],
        fusion_heads=student_fusion_heads,
        dropout=student_dropout,
        num_modality_classes=num_modality_classes,
        num_location_classes=num_location_classes,
        fusion_params=fusion_params,
    ).to(device)
    
    # Count model parameters
    teacher_params = count_parameters(teacher)
    student_params = count_parameters(student)
    print(f"\nTeacher parameters: {teacher_params['params_millions']:.2f}M ({teacher_params['total_params']:,})")
    print(f"Student parameters: {student_params['params_millions']:.2f}M ({student_params['total_params']:,})")
    
    # Create a distillation/loss object from config
    def _make_loss_from_cfg(cfg):
        # mapping: config name -> (module, class)
        mapping = {
            'vanilla': ('losses.vanilla', 'DistillationLoss'),
            'combined': ('losses.combined', 'MedKDCombinedLoss'),
            'crd': ('losses.crd', 'CRDLoss'),
            'rkd': ('losses.rkd', 'RKDLoss'),
            'mmd': ('losses.mmd', 'MMDLoss'),
        }
        loss_type = cfg.get('loss', {}).get('type', 'vanilla')
        module_name, class_name = mapping.get(loss_type, mapping['vanilla'])
        try:
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)
        except Exception:
            # Fall back to vanilla DistillationLoss if anything goes wrong
            module = importlib.import_module('losses.vanilla')
            cls = getattr(module, 'DistillationLoss')

        training_cfg = cfg.get('training', {}) or {}
        loss_cfg = cfg.get('loss', {}) or {}

        kwargs = {}
        accepts_fusion_dim = False
        accepts_kwargs = False
        try:
            sig = inspect.signature(cls.__init__)
            params = sig.parameters
            accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
            accepts_fusion_dim = ('fusion_dim' in params) or accepts_kwargs
            for name, param in params.items():
                if name == 'self' or param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                    continue
                if name in training_cfg:
                    kwargs[name] = training_cfg[name]
        except Exception:
            # if introspection fails, be permissive and allow fusion_dim injection
            accepts_fusion_dim = True

        if accepts_fusion_dim:
            # Prefer explicit loss.fusion_dim, otherwise fall back to student/teacher fusion size
            loss_fusion_dim = loss_cfg.get('fusion_dim')
            if loss_fusion_dim is None:
                loss_fusion_dim = cfg.get('student', {}).get('fusion_dim') or cfg.get('teacher', {}).get('fusion_dim')
            if loss_fusion_dim is None:
                raise KeyError("fusion_dim must be set in loss, student, or teacher config")
            kwargs['fusion_dim'] = loss_fusion_dim

        return cls(**kwargs)

    distill_fn = _make_loss_from_cfg(cfg)
    
    # Train teacher
    print("\n=== Training Teacher ===")
    teacher = train_teacher(
        teacher, train_loader, device,
        epochs=cfg['training'].get('teacher_epochs', 1),
        lr=cfg['training'].get('teacher_lr', 1e-5)
    )
    
    torch.cuda.empty_cache()
    
    print("\n=== Distilling to Student ===")
    best_dev_score = 0.0
    try:
        student_epochs = int(cfg['training'].get('student_epochs', 1))
    except Exception:
        raise TypeError(f"training.student_epochs must be int-like, got {cfg['training'].get('student_epochs')}")

    for epoch in range(1, student_epochs + 1):
        student, train_loss = train_student(
            student, teacher, train_loader, device, epochs=1,
            lr=cfg['training'].get('student_lr', 3e-4), distill_fn=distill_fn
        )
        dev_metrics = evaluate_detailed(student, dev_loader, device, logger=logger, split="dev", token_type='student',
                                       task1_label=task1_label, task2_label=task2_label)
        dev_score = (dev_metrics[f'dev_{task1_label}_f1'] + dev_metrics[f'dev_{task2_label}_f1']) / 2
        if dev_score > best_dev_score:
            best_dev_score = dev_score
            best_path = os.path.join(cfg['logging']['log_dir'], "student_best.pth")
            torch.save(student.state_dict(), best_path)
            print(f"  New best dev score: {dev_score:.4f}")
        # Log epoch-level metrics including train loss and evaluation metrics
        all_metrics = {}
        all_metrics.update(dev_metrics)
        logger.log_epoch(epoch, train_loss, all_metrics)
    
    # Test
    test_metrics = {}
    if best_dev_score > 0:
        best_path = os.path.join(cfg['logging']['log_dir'], "student_best.pth")
        student.load_state_dict(torch.load(best_path, map_location=device))
        test_metrics = evaluate_detailed(student, test_loader, device, logger=logger, split="test", token_type='student',
                                        task1_label=task1_label, task2_label=task2_label)
    
    final_path = os.path.join(cfg['logging']['log_dir'], "student_final.pth")
    torch.save(student.state_dict(), final_path)
    logger.save_csv()
    logger.save_json()

    # Save complete experiment results with all metadata
    results_logger = ResultsLogger(cfg['logging']['log_dir'])
    # Serialize train history from MetricsLogger so results.json contains per-epoch info
    try:
        serial_history = {k: list(v) for k, v in logger.history.items()}
    except Exception:
        serial_history = {}
    train_metrics = {'train': {'history': serial_history}}
    if 'train_loss' in locals():
        train_metrics['train']['final_loss'] = train_loss
    results_logger.log_experiment(cfg, train_metrics, dev_metrics, test_metrics, 
                                   teacher_params=teacher_params, student_params=student_params)
    
    print(f"All outputs saved in {cfg['logging']['log_dir']}")
    
if __name__ == "__main__":
    import sys
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config/default.yaml"
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    main(cfg)
