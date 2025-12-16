#!/usr/bin/env python3
"""
Hyperparameter tuning using Optuna for multimodal federated knowledge distillation.

Usage:
    python tools/optuna_tune.py --config config/default.yaml --n-trials 50 --study-name medpix-tuning
    python tools/optuna_tune.py --config config/wound.yaml --n-trials 100 --gpu cuda:0
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

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from trainer.engine import main as train_main
from utils.metrics import evaluate_detailed
import copy


def objective(trial, base_config, args):
    """
    Optuna objective function for hyperparameter optimization.
    
    Args:
        trial: Optuna trial object
        base_config: Base configuration dictionary
        args: Command line arguments
        
    Returns:
        float: Validation score (higher is better)
    """
    # Create a copy of base config for this trial
    cfg = copy.deepcopy(base_config)
    
    # ============================================================================
    # HYPERPARAMETER SEARCH SPACE
    # ============================================================================
    
    # Training hyperparameters
    cfg['training']['teacher_lr'] = trial.suggest_float('teacher_lr', 1e-6, 5e-4, log=True)
    cfg['training']['student_lr'] = trial.suggest_float('student_lr', 1e-5, 1e-3, log=True)
    cfg['training']['alpha'] = trial.suggest_float('alpha', 0.1, 2.0)
    cfg['training']['beta'] = trial.suggest_float('beta', 10.0, 500.0, log=True)
    cfg['training']['T'] = trial.suggest_float('T', 1.0, 10.0)
    
    # Model architecture hyperparameters
    fusion_dim_choices = [256, 384, 512, 768]
    cfg['teacher']['fusion_dim'] = trial.suggest_categorical('teacher_fusion_dim', fusion_dim_choices)
    cfg['student']['fusion_dim'] = trial.suggest_categorical('student_fusion_dim', fusion_dim_choices)
    
    cfg['teacher']['fusion_layers'] = trial.suggest_int('teacher_fusion_layers', 1, 4)
    cfg['student']['fusion_layers'] = trial.suggest_int('student_fusion_layers', 1, 3)
    
    cfg['teacher']['fusion_heads'] = trial.suggest_categorical('teacher_fusion_heads', [4, 8, 16])
    cfg['student']['fusion_heads'] = trial.suggest_categorical('student_fusion_heads', [4, 8, 16])
    
    cfg['teacher']['dropout'] = trial.suggest_float('teacher_dropout', 0.0, 0.5)
    cfg['student']['dropout'] = trial.suggest_float('student_dropout', 0.0, 0.5)
    
    # Batch size (if enabled)
    if args.tune_batch_size:
        cfg['data']['batch_size'] = trial.suggest_categorical('batch_size', [8, 16, 32, 64])
    
    # Fusion type (if enabled)
    if args.tune_fusion_type:
        fusion_types = [
            "simple",
            "concat_mlp",
            "cross_attention",
            "gated",
            "transformer_concat",
            "modality_dropout",
            "film",
        ]
        cfg['fusion']['type'] = trial.suggest_categorical('fusion_type', fusion_types)
        
        # Fusion-specific hyperparameters
        if cfg['fusion']['type'] == 'concat_mlp':
            cfg['fusion']['hidden_mult'] = trial.suggest_int('hidden_mult', 2, 8)
        elif cfg['fusion']['type'] == 'modality_dropout':
            cfg['fusion']['p_img'] = trial.suggest_float('p_img', 0.1, 0.5)
            cfg['fusion']['p_txt'] = trial.suggest_float('p_txt', 0.1, 0.5)
    
    # Loss type (if enabled)
    if args.tune_loss_type:
        loss_types = ["vanilla", "combined", "crd", "rkd", "mmd"]
        cfg['loss']['type'] = trial.suggest_categorical('loss_type', loss_types)
    
    # Student backbone selection (if enabled)
    if args.tune_student_backbones:
        vision_backbones = ["deit-tiny", "deit-small", "vit-base"]
        text_backbones = ["distilbert", "minilm"]
        cfg['student']['vision'] = trial.suggest_categorical('student_vision', vision_backbones)
        cfg['student']['text'] = trial.suggest_categorical('student_text', text_backbones)
    
    # Update log directory to include trial number
    base_log_dir = cfg['logging']['log_dir']
    cfg['logging']['log_dir'] = f"{base_log_dir}_trial_{trial.number:04d}"
    
    # ============================================================================
    # RUN TRAINING
    # ============================================================================
    
    try:
        # Set device
        if args.gpu:
            cfg['device'] = args.gpu
        
        # Run training and get validation metrics
        # We'll modify the training to return metrics
        from torch.utils.data import DataLoader
        from data.dataset import get_dataset, get_num_classes
        from models.teacher import Teacher
        from models.student import Student
        from trainer.engine import train_teacher, train_student, count_parameters
        from transformers import AutoTokenizer
        from models.backbones import get_text_pretrained_name
        import importlib
        import inspect
        
        # Device setup
        device = cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # Get tokenizers
        teacher_text_name = get_text_pretrained_name(cfg['teacher']['text'])
        student_text_name = get_text_pretrained_name(cfg['student']['text'])
        teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_text_name)
        student_tokenizer = AutoTokenizer.from_pretrained(student_text_name)
        
        # Create datasets
        train_ds = get_dataset(cfg, split='train', teacher_tokenizer=teacher_tokenizer, 
                              student_tokenizer=student_tokenizer)
        dev_ds = get_dataset(cfg, split='dev', teacher_tokenizer=teacher_tokenizer,
                            student_tokenizer=student_tokenizer)
        
        batch_size = cfg['data'].get('batch_size', 16)
        num_workers = cfg['data'].get('num_workers', 4)
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                                 num_workers=num_workers, pin_memory=True)
        dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False,
                               num_workers=num_workers, pin_memory=True)
        
        # Get class counts
        num_modality_classes, num_location_classes = get_num_classes(cfg)
        
        # Task labels
        task1_label = cfg['data'].get('task1_label', 'modality')
        task2_label = cfg['data'].get('task2_label', 'location')
        
        # Get fusion configuration
        fusion_type = cfg.get('fusion', {}).get('type', 'cross_attention')
        fusion_params = cfg.get('fusion', {})
        
        teacher_fusion_heads = cfg['teacher'].get('fusion_heads', 8)
        teacher_dropout = cfg['teacher'].get('dropout', 0.1)
        student_fusion_heads = cfg['student'].get('fusion_heads', 8)
        student_dropout = cfg['student'].get('dropout', 0.1)
        
        # Create models
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
        
        # Create loss function
        def _make_loss_from_cfg(cfg):
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
                    if name == 'self' or param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                        continue
                    if name in training_cfg:
                        kwargs[name] = training_cfg[name]
            except Exception:
                accepts_fusion_dim = True

            if accepts_fusion_dim:
                loss_fusion_dim = loss_cfg.get('fusion_dim')
                if loss_fusion_dim is None:
                    loss_fusion_dim = cfg.get('student', {}).get('fusion_dim') or cfg.get('teacher', {}).get('fusion_dim')
                if loss_fusion_dim is None:
                    raise KeyError("fusion_dim must be set in loss, student, or teacher config")
                kwargs['fusion_dim'] = loss_fusion_dim

            return cls(**kwargs)

        distill_fn = _make_loss_from_cfg(cfg)
        
        # Train teacher (use fewer epochs for tuning)
        teacher_epochs = args.teacher_epochs if args.teacher_epochs else cfg['training'].get('teacher_epochs', 3)
        teacher = train_teacher(
            teacher, train_loader, device,
            epochs=teacher_epochs,
            lr=cfg['training'].get('teacher_lr', 1e-5)
        )
        
        torch.cuda.empty_cache()
        
        # Train student (use fewer epochs for tuning)
        student_epochs = args.student_epochs if args.student_epochs else cfg['training'].get('student_epochs', 10)
        best_dev_score = 0.0
        
        for epoch in range(1, student_epochs + 1):
            student, train_loss = train_student(
                student, teacher, train_loader, device, epochs=1,
                lr=cfg['training'].get('student_lr', 3e-4), distill_fn=distill_fn
            )
            
            # Evaluate on dev set
            dev_metrics = evaluate_detailed(student, dev_loader, device, logger=None, 
                                          split="dev", token_type='student',
                                          task1_label=task1_label, task2_label=task2_label)
            
            # Compute dev score
            dev_score = (dev_metrics[f'dev_{task1_label}_f1'] + dev_metrics[f'dev_{task2_label}_f1']) / 2
            
            if dev_score > best_dev_score:
                best_dev_score = dev_score
            
            # Report intermediate values for pruning
            trial.report(dev_score, epoch)
            
            # Handle pruning based on intermediate values
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        # Clean up to save memory
        del teacher, student, train_loader, dev_loader
        torch.cuda.empty_cache()
        
        return best_dev_score
        
    except optuna.TrialPruned:
        raise
    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        # Return a poor score so Optuna doesn't select this configuration
        return 0.0


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter tuning with Optuna')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to base configuration file')
    parser.add_argument('--study-name', type=str, default=None,
                       help='Name of the Optuna study (default: auto-generated)')
    parser.add_argument('--n-trials', type=int, default=50,
                       help='Number of trials to run (default: 50)')
    parser.add_argument('--timeout', type=int, default=None,
                       help='Timeout in seconds for the study (default: None)')
    parser.add_argument('--gpu', type=str, default=None,
                       help='GPU device to use (e.g., cuda:0)')
    parser.add_argument('--storage', type=str, default=None,
                       help='Database URL for study storage (default: SQLite in logs/optuna/)')
    parser.add_argument('--direction', type=str, default='maximize',
                       choices=['maximize', 'minimize'],
                       help='Optimization direction (default: maximize)')
    
    # Training configuration for tuning
    parser.add_argument('--teacher-epochs', type=int, default=None,
                       help='Number of teacher epochs for each trial (default: use config value)')
    parser.add_argument('--student-epochs', type=int, default=None,
                       help='Number of student epochs for each trial (default: use config value)')
    
    # What to tune
    parser.add_argument('--tune-batch-size', action='store_true',
                       help='Include batch size in hyperparameter search')
    parser.add_argument('--tune-fusion-type', action='store_true',
                       help='Include fusion module type in hyperparameter search')
    parser.add_argument('--tune-loss-type', action='store_true',
                       help='Include loss type in hyperparameter search')
    parser.add_argument('--tune-student-backbones', action='store_true',
                       help='Include student backbone selection in hyperparameter search')
    
    # Pruning and sampling
    parser.add_argument('--sampler', type=str, default='tpe',
                       choices=['tpe', 'random', 'grid', 'cmaes'],
                       help='Optuna sampler to use (default: tpe)')
    parser.add_argument('--pruner', type=str, default='median',
                       choices=['median', 'none', 'percentile', 'hyperband'],
                       help='Optuna pruner to use (default: median)')
    
    args = parser.parse_args()
    
    # Load base configuration
    with open(args.config, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Generate study name if not provided
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
    if args.sampler == 'tpe':
        sampler = optuna.samplers.TPESampler(seed=42)
    elif args.sampler == 'random':
        sampler = optuna.samplers.RandomSampler(seed=42)
    elif args.sampler == 'grid':
        sampler = optuna.samplers.GridSampler()
    elif args.sampler == 'cmaes':
        sampler = optuna.samplers.CmaEsSampler(seed=42)
    else:
        sampler = None
    
    # Create pruner
    if args.pruner == 'median':
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=3,
            interval_steps=1,
        )
    elif args.pruner == 'percentile':
        pruner = optuna.pruners.PercentilePruner(
            percentile=25.0,
            n_startup_trials=5,
            n_warmup_steps=3,
        )
    elif args.pruner == 'hyperband':
        pruner = optuna.pruners.HyperbandPruner(
            min_resource=1,
            max_resource='auto',
            reduction_factor=3,
        )
    else:
        pruner = optuna.pruners.NopPruner()
    
    # Create study
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        direction=args.direction,
        sampler=sampler,
        pruner=pruner,
    )
    
    print(f"Starting Optuna study: {args.study_name}")
    print(f"Base config: {args.config}")
    print(f"Number of trials: {args.n_trials}")
    print(f"Storage: {args.storage}")
    print(f"Sampler: {args.sampler}")
    print(f"Pruner: {args.pruner}")
    print(f"GPU: {args.gpu or 'auto'}")
    print(f"\nTuning options:")
    print(f"  - Batch size: {args.tune_batch_size}")
    print(f"  - Fusion type: {args.tune_fusion_type}")
    print(f"  - Loss type: {args.tune_loss_type}")
    print(f"  - Student backbones: {args.tune_student_backbones}")
    print(f"\n{'='*80}\n")
    
    # Run optimization
    study.optimize(
        lambda trial: objective(trial, base_config, args),
        n_trials=args.n_trials,
        timeout=args.timeout,
        show_progress_bar=True,
    )
    
    # Print results
    print(f"\n{'='*80}")
    print("OPTIMIZATION COMPLETE")
    print(f"{'='*80}\n")
    
    print(f"Number of finished trials: {len(study.trials)}")
    print(f"Number of pruned trials: {len([t for t in study.trials if t.state == TrialState.PRUNED])}")
    print(f"Number of complete trials: {len([t for t in study.trials if t.state == TrialState.COMPLETE])}")
    
    print("\nBest trial:")
    trial = study.best_trial
    print(f"  Trial number: {trial.number}")
    print(f"  Value (dev F1): {trial.value:.4f}")
    print(f"\n  Hyperparameters:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    # Save best configuration
    best_config = copy.deepcopy(base_config)
    
    # Update with best hyperparameters
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
    
    if 'batch_size' in trial.params:
        best_config['data']['batch_size'] = trial.params['batch_size']
    if 'fusion_type' in trial.params:
        best_config['fusion']['type'] = trial.params['fusion_type']
    if 'loss_type' in trial.params:
        best_config['loss']['type'] = trial.params['loss_type']
    if 'student_vision' in trial.params:
        best_config['student']['vision'] = trial.params['student_vision']
    if 'student_text' in trial.params:
        best_config['student']['text'] = trial.params['student_text']
    
    # Save best config
    output_dir = Path("logs/optuna") / args.study_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    best_config_path = output_dir / "best_config.yaml"
    with open(best_config_path, 'w') as f:
        yaml.dump(best_config, f, default_flow_style=False, sort_keys=False)
    print(f"\nBest configuration saved to: {best_config_path}")
    
    # Save study summary
    summary = {
        'study_name': args.study_name,
        'n_trials': len(study.trials),
        'n_complete': len([t for t in study.trials if t.state == TrialState.COMPLETE]),
        'n_pruned': len([t for t in study.trials if t.state == TrialState.PRUNED]),
        'best_trial': trial.number,
        'best_value': trial.value,
        'best_params': trial.params,
        'base_config': args.config,
        'timestamp': datetime.now().isoformat(),
    }
    
    summary_path = output_dir / "study_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Study summary saved to: {summary_path}")
    
    # Visualization (if available)
    try:
        import optuna.visualization as vis
        
        # Parameter importance
        fig_importance = vis.plot_param_importances(study)
        fig_importance.write_html(str(output_dir / "param_importance.html"))
        
        # Optimization history
        fig_history = vis.plot_optimization_history(study)
        fig_history.write_html(str(output_dir / "optimization_history.html"))
        
        # Parallel coordinate plot
        fig_parallel = vis.plot_parallel_coordinate(study)
        fig_parallel.write_html(str(output_dir / "parallel_coordinate.html"))
        
        print(f"\nVisualization plots saved in: {output_dir}")
    except ImportError:
        print("\nNote: Install plotly for visualization plots:")
        print("  pip install plotly")


if __name__ == "__main__":
    main()
