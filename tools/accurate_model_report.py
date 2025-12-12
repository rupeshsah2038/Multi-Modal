#!/usr/bin/env python3
"""
Accurate model size report by loading actual checkpoints and instantiating models
"""

import sys
import json
import torch
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.teacher import Teacher
from models.student import Student


def count_parameters(model):
    """Calculates the total number of trainable and non-trainable parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def count_by_component(model):
    """Count parameters by component."""
    total = sum(p.numel() for p in model.parameters())
    vision = sum(p.numel() for n, p in model.named_parameters() if 'vision' in n)
    text = sum(p.numel() for n, p in model.named_parameters() if 'text' in n)
    proj = sum(p.numel() for n, p in model.named_parameters() if 'proj_' in n)
    fusion = sum(p.numel() for n, p in model.named_parameters() if 'fusion' in n)
    heads = sum(p.numel() for n, p in model.named_parameters() if 'head_' in n)
    
    return {
        'total': total,
        'trainable': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'vision': vision,
        'text': text,
        'projection': proj,
        'fusion': fusion,
        'heads': heads,
    }


def load_model_from_checkpoint(ckpt_path, model_class, config):
    """Load model from checkpoint properly."""
    try:
        # Instantiate model with config
        # Student and Teacher have slightly different parameter names
        if model_class.__name__ == 'Student':
            model = model_class(
                vision=config['vision'],
                text=config['text'],
                fusion_dim=config['fusion_dim'],
                num_modality_classes=config.get('num_modality_classes', 10),
                num_location_classes=config.get('num_location_classes', 3),
                fusion_layers=config.get('fusion_layers', 1),
            )
        else:  # Teacher
            model = model_class(
                vision=config['vision'],
                text=config['text'],
                fusion_dim=config['fusion_dim'],
                num_modality_classes=config.get('num_modality_classes', 10),
                num_location_classes=config.get('num_location_classes', 3),
                fusion_type=config.get('fusion_type', 'cross_attention'),
                fusion_layers=config.get('fusion_layers', 2),
            )
        
        # Load checkpoint
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        elif isinstance(checkpoint, dict):
            # Checkpoint is the state_dict itself
            model.load_state_dict(checkpoint)
        else:
            # Entire model was saved
            model = checkpoint
        
        return model
    except Exception as e:
        print(f"Error loading model from {ckpt_path}: {e}")
        return None


def analyze_checkpoint(ckpt_path, model_class, config):
    """Load checkpoint, instantiate model, and count parameters."""
    model = load_model_from_checkpoint(ckpt_path, model_class, config)
    if model is None:
        return None
    
    stats = count_by_component(model)
    del model  # Free memory
    torch.cuda.empty_cache()
    
    return stats


def format_size(num_params):
    """Format parameter count."""
    if num_params >= 1e9:
        return f"{num_params/1e9:.2f}B"
    elif num_params >= 1e6:
        return f"{num_params/1e6:.2f}M"
    elif num_params >= 1e3:
        return f"{num_params/1e3:.2f}K"
    else:
        return str(num_params)


def main():
    log_root = Path("logs")
    
    # Find all experiment directories with results.json
    result_files = sorted(log_root.rglob("results.json"))
    
    if not result_files:
        print("No experiment results found!")
        return
    
    print(f"Found {len(result_files)} experiments")
    print("Analyzing models by instantiating and loading checkpoints...\n")
    
    results = []
    
    for result_file in tqdm(result_files, desc="Processing"):
        experiment_dir = result_file.parent
        experiment_name = experiment_dir.name
        
        # Load experiment config from results.json
        try:
            with open(result_file, 'r') as f:
                exp_data = json.load(f)
            
            teacher_config = exp_data['config']['teacher']
            student_config = exp_data['config']['student']
            
            # Get class counts from config or data
            data_config = exp_data['config']['data']
            if data_config['type'] == 'wound':
                num_modality = 10  # wound types
                num_location = 3   # severity levels
            else:  # medpix
                num_modality = 2   # CT/MR
                num_location = 5   # body locations
            
            # Add to configs
            teacher_config['num_modality_classes'] = num_modality
            teacher_config['num_location_classes'] = num_location
            teacher_config['fusion_type'] = exp_data['config'].get('fusion_type', 'cross_attention')
            
            student_config['num_modality_classes'] = num_modality
            student_config['num_location_classes'] = num_location
            student_config['fusion_type'] = exp_data['config'].get('fusion_type', 'cross_attention')
            
        except Exception as e:
            print(f"Error reading config from {result_file}: {e}")
            continue
        
        # Analyze teacher
        teacher_path = experiment_dir / "teacher_best.pth"
        teacher_stats = None
        if teacher_path.exists():
            teacher_stats = analyze_checkpoint(teacher_path, Teacher, teacher_config)
        
        # Analyze student
        student_path = experiment_dir / "student_best.pth"
        student_stats = None
        if student_path.exists():
            student_stats = analyze_checkpoint(student_path, Student, student_config)
        
        if not student_stats:
            continue
        
        # Calculate compression
        if teacher_stats:
            compression = teacher_stats['total'] / student_stats['total']
            reduction = (1 - student_stats['total'] / teacher_stats['total']) * 100
        else:
            compression = 0
            reduction = 0
        
        results.append({
            'experiment': experiment_name,
            'teacher_total': teacher_stats['total'] if teacher_stats else 0,
            'teacher_trainable': teacher_stats['trainable'] if teacher_stats else 0,
            'teacher_size': format_size(teacher_stats['total']) if teacher_stats else 'N/A',
            'student_total': student_stats['total'],
            'student_trainable': student_stats['trainable'],
            'student_size': format_size(student_stats['total']),
            'student_vision': student_stats['vision'],
            'student_text': student_stats['text'],
            'student_proj': student_stats['projection'],
            'student_fusion': student_stats['fusion'],
            'student_heads': student_stats['heads'],
            'compression': f"{compression:.2f}x" if compression > 0 else 'N/A',
            'reduction': f"{reduction:.1f}%" if reduction > 0 else 'N/A',
        })
    
    if not results:
        print("No valid models analyzed!")
        return
    
    df = pd.DataFrame(results)
    
    # Display summary
    print("\n" + "="*100)
    print("ACCURATE MODEL SIZES (instantiated from checkpoints)")
    print("="*100)
    print()
    
    display_df = df[['experiment', 'teacher_size', 'student_size', 'compression', 'reduction']].copy()
    display_df.columns = ['Experiment', 'Teacher', 'Student', 'Compression', 'Reduction']
    print(display_df.to_string(index=False, max_colwidth=40))
    
    # Component breakdown for students
    print("\n" + "="*100)
    print("STUDENT PARAMETER BREAKDOWN")
    print("="*100)
    print()
    
    breakdown_df = df[['experiment', 'student_vision', 'student_text', 'student_proj', 
                       'student_fusion', 'student_heads', 'student_total']].copy()
    
    # Convert to millions for readability
    for col in ['student_vision', 'student_text', 'student_proj', 'student_fusion', 'student_heads', 'student_total']:
        breakdown_df[f'{col}_M'] = breakdown_df[col].apply(lambda x: f"{x/1e6:.2f}M")
    
    breakdown_display = breakdown_df[['experiment', 'student_vision_M', 'student_text_M', 
                                      'student_proj_M', 'student_fusion_M', 'student_heads_M', 'student_total_M']]
    breakdown_display.columns = ['Experiment', 'Vision', 'Text', 'Proj', 'Fusion', 'Heads', 'Total']
    print(breakdown_display.to_string(index=False, max_colwidth=40))
    
    # Save full report
    output_file = "logs/accurate_model_sizes.csv"
    df.to_csv(output_file, index=False)
    print(f"\n✓ Full report saved to: {output_file}")
    
    # Statistics
    print("\n" + "="*100)
    print("STATISTICS")
    print("="*100)
    print(f"\nTotal experiments: {len(df)}")
    
    if df['teacher_total'].max() > 0:
        print(f"\nTeacher range: {format_size(df['teacher_total'].min())} - {format_size(df['teacher_total'].max())}")
        print(f"Teacher mean: {format_size(df['teacher_total'].mean())}")
    
    print(f"\nStudent range: {format_size(df['student_total'].min())} - {format_size(df['student_total'].max())}")
    print(f"Student mean: {format_size(df['student_total'].mean())}")
    
    # Trainability check
    all_trainable = df['student_total'] == df['student_trainable']
    print(f"\nAll student parameters trainable: {all_trainable.all()}")
    
    print("\n" + "="*100)


if __name__ == "__main__":
    main()
