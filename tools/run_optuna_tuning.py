#!/usr/bin/env python3
"""
Simplified Optuna hyperparameter tuning script.

Quick usage examples:

# Basic tuning (LR, loss weights, architecture dims):
python tools/run_optuna_tuning.py --config config/default.yaml --n-trials 30

# Full tuning (including fusion and loss types):
python tools/run_optuna_tuning.py --config config/default.yaml --n-trials 50 \
    --tune-fusion --tune-loss --tune-backbones

# Quick tuning with fewer epochs per trial:
python tools/run_optuna_tuning.py --config config/wound.yaml --n-trials 20 \
    --teacher-epochs 2 --student-epochs 5 --gpu cuda:1

# Resume existing study:
python tools/run_optuna_tuning.py --config config/default.yaml --n-trials 20 \
    --study-name medpix_tuning_20251215_120000
"""
import argparse
import yaml
import torch
import optuna
from optuna.trial import TrialState
import sys
import os
from pathlib import Path
from datetime import datetime
import json
import copy

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from torch.utils.data import DataLoader
from data.dataset import get_dataset, get_num_classes
from models.teacher import Teacher
from models.student import Student
from trainer.engine import train_teacher, train_student
from transformers import AutoTokenizer
from models.backbones import get_text_pretrained_name
from utils.metrics import evaluate_detailed
import importlib
import inspect


def create_loss_function(cfg):
    """Create distillation loss function from config."""
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
            if name == 'self' or param.kind in (inspect.Parameter.VAR_POSITIONAL, 
                                               inspect.Parameter.VAR_KEYWORD):
                continue
            if name in training_cfg:
                kwargs[name] = training_cfg[name]
    except Exception:
        accepts_fusion_dim = True

    if accepts_fusion_dim:
        loss_fusion_dim = loss_cfg.get('fusion_dim')
        if loss_fusion_dim is None:
            loss_fusion_dim = (cfg.get('student', {}).get('fusion_dim') or 
                             cfg.get('teacher', {}).get('fusion_dim'))
        if loss_fusion_dim is None:
            raise KeyError("fusion_dim must be set in loss, student, or teacher config")
        kwargs['fusion_dim'] = loss_fusion_dim

    return cls(**kwargs)


def objective(trial, base_config, args):
    """Optuna objective function."""
    cfg = copy.deepcopy(base_config)
    
    # === Core hyperparameters (always tuned) ===
    cfg['training']['teacher_lr'] = trial.suggest_float('teacher_lr', 5e-6, 1e-4, log=True)
    cfg['training']['student_lr'] = trial.suggest_float('student_lr', 1e-4, 5e-4, log=True)
    cfg['training']['alpha'] = trial.suggest_float('alpha', 0.5, 2.0)
    cfg['training']['beta'] = trial.suggest_float('beta', 50.0, 200.0)
    cfg['training']['T'] = trial.suggest_float('T', 2.0, 6.0)
    
    # Architecture dimensions
    cfg['teacher']['fusion_dim'] = trial.suggest_categorical('teacher_fusion_dim', [256, 384])
    cfg['student']['fusion_dim'] = trial.suggest_categorical('student_fusion_dim', [256, 384])
    cfg['teacher']['fusion_layers'] = trial.suggest_int('teacher_fusion_layers', 1, 3)
    cfg['student']['fusion_layers'] = trial.suggest_int('student_fusion_layers', 1, 2)
    cfg['teacher']['fusion_heads'] = trial.suggest_categorical('teacher_fusion_heads', [4, 8])
    cfg['student']['fusion_heads'] = trial.suggest_categorical('student_fusion_heads', [4, 8])
    cfg['teacher']['dropout'] = trial.suggest_float('teacher_dropout', 0.05, 0.3)
    cfg['student']['dropout'] = trial.suggest_float('student_dropout', 0.05, 0.3)
    
    # === Optional: Fusion module type ===
    if args.tune_fusion:
        fusion_types = ["simple", "concat_mlp", "cross_attention", "gated"]
        cfg['fusion']['type'] = trial.suggest_categorical('fusion_type', fusion_types)
    
    # === Optional: Loss type ===
    if args.tune_loss:
        loss_types = ["vanilla", "combined", "crd", "rkd"]
        cfg['loss']['type'] = trial.suggest_categorical('loss_type', loss_types)
    
    # === Optional: Student backbones ===
    if args.tune_backbones:
        vision_backbones = ["deit-tiny", "deit-small"]
        text_backbones = ["distilbert", "minilm"]
        cfg['student']['vision'] = trial.suggest_categorical('student_vision', vision_backbones)
        cfg['student']['text'] = trial.suggest_categorical('student_text', text_backbones)
    
    # Update log directory
    base_log_dir = cfg['logging']['log_dir']
    cfg['logging']['log_dir'] = f"{base_log_dir}_optuna_trial_{trial.number:04d}"
    
    try:
        # Device
        device = args.gpu or cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        cfg['device'] = device
        
        # Get tokenizers
        teacher_text_name = get_text_pretrained_name(cfg['teacher']['text'])
        student_text_name = get_text_pretrained_name(cfg['student']['text'])
        teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_text_name)
        student_tokenizer = AutoTokenizer.from_pretrained(student_text_name)
        
        # Create datasets
        dataset_type = cfg.get('data', {}).get('type', 'medpix')
        dataset_root = cfg['data']['root']
        
        if dataset_type == 'medpix':
            train_ds = get_dataset(
                dataset_type='medpix',
                data_jsonl_file=os.path.join(dataset_root, "splitted_dataset/data_train.jsonl"),
                desc_jsonl_file=os.path.join(dataset_root, "splitted_dataset/descriptions_train.jsonl"),
                image_dir=os.path.join(dataset_root, "images"),
                tokenizer_teacher=teacher_tokenizer,
                tokenizer_student=student_tokenizer,
            )
            dev_ds = get_dataset(
                dataset_type='medpix',
                data_jsonl_file=os.path.join(dataset_root, "splitted_dataset/data_dev.jsonl"),
                desc_jsonl_file=os.path.join(dataset_root, "splitted_dataset/descriptions_dev.jsonl"),
                image_dir=os.path.join(dataset_root, "images"),
                tokenizer_teacher=teacher_tokenizer,
                tokenizer_student=student_tokenizer,
            )
        elif dataset_type == 'wound':
            train_ds = get_dataset(
                dataset_type='wound',
                csv_file=os.path.join(dataset_root, "metadata_train.csv"),
                image_dir=os.path.join(dataset_root, "images"),
                tokenizer_teacher=teacher_tokenizer,
                tokenizer_student=student_tokenizer,
                type_column=cfg['data'].get('type_column', 'type'),
                severity_column=cfg['data'].get('severity_column', 'severity'),
                description_column=cfg['data'].get('description_column', 'description'),
                filepath_column=cfg['data'].get('filepath_column', 'file_path'),
            )
            dev_ds = get_dataset(
                dataset_type='wound',
                csv_file=os.path.join(dataset_root, "metadata_dev.csv"),
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
        
        batch_size = cfg['data'].get('batch_size', 16)
        num_workers = cfg['data'].get('num_workers', 4)
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                 num_workers=num_workers, pin_memory=True)
        dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False,
                               num_workers=num_workers, pin_memory=True)
        
        # Get class counts
        if dataset_type == 'medpix':
            num_modality_classes = 2
            num_location_classes = 5
        else:  # wound
            # Get dynamic class counts from wound dataset
            import pandas as pd
            train_csv = os.path.join(dataset_root, "metadata_train.csv")
            df = pd.read_csv(train_csv)
            type_col = cfg['data'].get('type_column', 'type')
            sev_col = cfg['data'].get('severity_column', 'severity')
            num_modality_classes = df[type_col].nunique()
            num_location_classes = df[sev_col].nunique()
        
        # Task labels
        task1_label = cfg['data'].get('task1_label', 'modality')
        task2_label = cfg['data'].get('task2_label', 'location')
        
        # Fusion configuration
        fusion_type = cfg.get('fusion', {}).get('type', 'cross_attention')
        fusion_params = cfg.get('fusion', {})
        
        # Create models
        teacher = Teacher(
            vision=cfg['teacher']['vision'],
            text=cfg['teacher']['text'],
            fusion_type=fusion_type,
            fusion_layers=cfg['teacher']['fusion_layers'],
            fusion_dim=cfg['teacher']['fusion_dim'],
            fusion_heads=cfg['teacher']['fusion_heads'],
            dropout=cfg['teacher']['dropout'],
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
            fusion_heads=cfg['student']['fusion_heads'],
            dropout=cfg['student']['dropout'],
            num_modality_classes=num_modality_classes,
            num_location_classes=num_location_classes,
            fusion_params=fusion_params,
        ).to(device)
        
        # Create loss
        distill_fn = create_loss_function(cfg)
        
        # Train teacher (reduced epochs for tuning)
        teacher_epochs = args.teacher_epochs or cfg['training'].get('teacher_epochs', 3)
        teacher = train_teacher(
            teacher, train_loader, device,
            epochs=teacher_epochs,
            lr=cfg['training']['teacher_lr']
        )
        
        torch.cuda.empty_cache()
        
        # Train student with early stopping
        student_epochs = args.student_epochs or cfg['training'].get('student_epochs', 10)
        best_dev_score = 0.0
        
        for epoch in range(1, student_epochs + 1):
            student, _ = train_student(
                student, teacher, train_loader, device, epochs=1,
                lr=cfg['training']['student_lr'], distill_fn=distill_fn
            )
            
            # Evaluate on dev
            dev_metrics = evaluate_detailed(
                student, dev_loader, device, logger=None,
                split="dev", token_type='student',
                task1_label=task1_label, task2_label=task2_label
            )
            
            # Compute dev score (average F1)
            dev_score = (dev_metrics[f'dev_{task1_label}_f1'] + 
                        dev_metrics[f'dev_{task2_label}_f1']) / 2
            
            if dev_score > best_dev_score:
                best_dev_score = dev_score
            
            # Report for pruning
            trial.report(dev_score, epoch)
            
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        # Cleanup
        del teacher, student, train_loader, dev_loader
        torch.cuda.empty_cache()
        
        return best_dev_score
        
    except optuna.TrialPruned:
        raise
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        import traceback
        traceback.print_exc()
        return 0.0


def main():
    parser = argparse.ArgumentParser(
        description='Hyperparameter tuning with Optuna',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Required
    parser.add_argument('--config', type=str, required=True,
                       help='Base configuration file')
    
    # Study configuration
    parser.add_argument('--study-name', type=str, default=None,
                       help='Study name (auto-generated if not provided)')
    parser.add_argument('--n-trials', type=int, default=30,
                       help='Number of trials (default: 30)')
    parser.add_argument('--timeout', type=int, default=None,
                       help='Timeout in seconds (default: None)')
    
    # Training configuration
    parser.add_argument('--teacher-epochs', type=int, default=None,
                       help='Teacher epochs per trial (default: use config, typically 3)')
    parser.add_argument('--student-epochs', type=int, default=None,
                       help='Student epochs per trial (default: use config, typically 10)')
    parser.add_argument('--gpu', type=str, default=None,
                       help='GPU device (e.g., cuda:0)')
    
    # What to tune
    parser.add_argument('--tune-fusion', action='store_true',
                       help='Include fusion module type in search space')
    parser.add_argument('--tune-loss', action='store_true',
                       help='Include loss type in search space')
    parser.add_argument('--tune-backbones', action='store_true',
                       help='Include student backbone selection in search space')
    
    # Advanced
    parser.add_argument('--sampler', type=str, default='tpe',
                       choices=['tpe', 'random'],
                       help='Sampler (default: tpe)')
    parser.add_argument('--pruner', type=str, default='median',
                       choices=['median', 'none'],
                       help='Pruner (default: median)')
    parser.add_argument('--storage', type=str, default=None,
                       help='Database URL (default: SQLite in logs/optuna/)')
    
    args = parser.parse_args()
    
    # Load base config
    with open(args.config, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Generate study name
    if args.study_name is None:
        dataset_type = base_config['data'].get('type', 'unknown')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.study_name = f"{dataset_type}_tuning_{timestamp}"
    
    # Setup storage
    if args.storage is None:
        storage_dir = Path("logs/optuna")
        storage_dir.mkdir(parents=True, exist_ok=True)
        args.storage = f"sqlite:///{storage_dir / args.study_name}.db"
    
    # Create sampler
    sampler = (optuna.samplers.TPESampler(seed=42) if args.sampler == 'tpe'
              else optuna.samplers.RandomSampler(seed=42))
    
    # Create pruner
    pruner = (optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)
             if args.pruner == 'median' else optuna.pruners.NopPruner())
    
    # Create study
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        direction='maximize',
        sampler=sampler,
        pruner=pruner,
    )
    
    print("="*80)
    print("OPTUNA HYPERPARAMETER TUNING")
    print("="*80)
    print(f"Study name: {args.study_name}")
    print(f"Base config: {args.config}")
    print(f"Number of trials: {args.n_trials}")
    print(f"Storage: {args.storage}")
    print(f"GPU: {args.gpu or 'auto'}")
    print(f"\nSearch space:")
    print(f"  • Learning rates, loss weights, architecture dims (always)")
    print(f"  • Fusion type: {args.tune_fusion}")
    print(f"  • Loss type: {args.tune_loss}")
    print(f"  • Student backbones: {args.tune_backbones}")
    print("="*80 + "\n")
    
    # Run optimization
    study.optimize(
        lambda trial: objective(trial, base_config, args),
        n_trials=args.n_trials,
        timeout=args.timeout,
        show_progress_bar=True,
    )
    
    # Results
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)
    print(f"Total trials: {len(study.trials)}")
    print(f"Complete: {len([t for t in study.trials if t.state == TrialState.COMPLETE])}")
    print(f"Pruned: {len([t for t in study.trials if t.state == TrialState.PRUNED])}")
    
    if study.best_trial:
        trial = study.best_trial
        print(f"\nBest trial: #{trial.number}")
        print(f"Best dev F1: {trial.value:.4f}")
        print("\nBest hyperparameters:")
        for key, value in sorted(trial.params.items()):
            print(f"  {key:25s} = {value}")
        
        # Save best config
        best_config = copy.deepcopy(base_config)
        best_config['training']['teacher_lr'] = trial.params['teacher_lr']
        best_config['training']['student_lr'] = trial.params['student_lr']
        best_config['training']['alpha'] = trial.params['alpha']
        best_config['training']['beta'] = trial.params['beta']
        best_config['training']['T'] = trial.params['T']
        best_config['teacher']['fusion_dim'] = trial.params['teacher_fusion_dim']
        best_config['student']['fusion_dim'] = trial.params['student_fusion_dim']
        best_config['teacher']['fusion_layers'] = trial.params['teacher_fusion_layers']
        best_config['student']['fusion_layers'] = trial.params['student_fusion_layers']
        best_config['teacher']['fusion_heads'] = trial.params['teacher_fusion_heads']
        best_config['student']['fusion_heads'] = trial.params['student_fusion_heads']
        best_config['teacher']['dropout'] = trial.params['teacher_dropout']
        best_config['student']['dropout'] = trial.params['student_dropout']
        
        if 'fusion_type' in trial.params:
            best_config['fusion']['type'] = trial.params['fusion_type']
        if 'loss_type' in trial.params:
            best_config['loss']['type'] = trial.params['loss_type']
        if 'student_vision' in trial.params:
            best_config['student']['vision'] = trial.params['student_vision']
        if 'student_text' in trial.params:
            best_config['student']['text'] = trial.params['student_text']
        
        # Save files
        output_dir = Path("logs/optuna") / args.study_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        config_path = output_dir / "best_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(best_config, f, default_flow_style=False, sort_keys=False)
        
        summary = {
            'study_name': args.study_name,
            'base_config': args.config,
            'n_trials': len(study.trials),
            'best_trial': trial.number,
            'best_value': trial.value,
            'best_params': trial.params,
            'timestamp': datetime.now().isoformat(),
        }
        
        summary_path = output_dir / "study_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nBest config saved: {config_path}")
        print(f"Study summary saved: {summary_path}")
        
        # Generate visualizations
        try:
            import optuna.visualization as vis
            
            fig_importance = vis.plot_param_importances(study)
            fig_importance.write_html(str(output_dir / "param_importance.html"))
            
            fig_history = vis.plot_optimization_history(study)
            fig_history.write_html(str(output_dir / "optimization_history.html"))
            
            print(f"Visualizations saved in: {output_dir}")
        except ImportError:
            print("\nTip: Install plotly for visualizations: pip install plotly")
    
    print("="*80)


if __name__ == "__main__":
    main()
