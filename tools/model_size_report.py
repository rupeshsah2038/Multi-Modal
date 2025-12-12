#!/usr/bin/env python3
"""
Generate model size report using fast state_dict counting method.
"""

import json
import torch
from pathlib import Path
import pandas as pd
from tqdm import tqdm


# Standard backbone parameter counts (from exact measurements)
BACKBONE_PARAMS = {
    'vit-large': 304_351_232,
    'vit-base': 86_389_248,
    'deit-base': 86_389_248,
    'deit-small': 21_813_504,
    'deit-tiny': 5_561_472,
    'bio-clinical-bert': 108_310_272,
    'distilbert': 66_362_880,
    'minilm': 22_713_216,
}


def estimate_teacher_params(config, num_modality_classes, num_location_classes):
    """Estimate teacher parameters from config."""
    vision_params = BACKBONE_PARAMS.get(config['vision'], 0)
    text_params = BACKBONE_PARAMS.get(config['text'], 0)
    fusion_dim = config['fusion_dim']
    fusion_layers = config.get('fusion_layers', 2)
    
    # Projection layers: 768 -> fusion_dim for both modalities
    proj_params = 2 * (768 * fusion_dim + fusion_dim)
    
    # Fusion module (transformer-based): ~8*fusion_dim^2 per layer
    fusion_params = fusion_layers * 8 * (fusion_dim ** 2)
    
    # Task heads
    head_params = (fusion_dim * num_modality_classes + num_modality_classes) + \
                  (fusion_dim * num_location_classes + num_location_classes)
    
    total = vision_params + text_params + proj_params + fusion_params + head_params
    
    return {
        'total': total,
        'vision': vision_params,
        'text': text_params,
        'projection': proj_params,
        'fusion': fusion_params,
        'heads': head_params,
    }


def count_by_component(path):
    """Count parameters by component from state_dict."""
    state_dict = torch.load(path, map_location='cpu')
    
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    elif isinstance(state_dict, dict) and 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    
    total = sum(p.numel() for p in state_dict.values())
    vision = sum(p.numel() for k, p in state_dict.items() if k.startswith('vision.'))
    text = sum(p.numel() for k, p in state_dict.items() if k.startswith('text.'))
    proj = sum(p.numel() for k, p in state_dict.items() if k.startswith('proj_'))
    fusion = sum(p.numel() for k, p in state_dict.items() if k.startswith('fusion.'))
    heads = sum(p.numel() for k, p in state_dict.items() if k.startswith('head_'))
    
    return {
        'total': total,
        'vision': vision,
        'text': text,
        'projection': proj,
        'fusion': fusion,
        'heads': heads,
    }


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
    
    # Find all student checkpoints with results.json
    result_files = sorted(log_root.rglob("results.json"))
    
    if not result_files:
        print("No experiment results found!")
        return
    
    print(f"Found {len(result_files)} experiments")
    print("Analyzing using fast state_dict counting...\n")
    
    results = []
    
    for result_file in tqdm(result_files, desc="Processing"):
        experiment_dir = result_file.parent
        experiment_name = experiment_dir.name
        student_path = experiment_dir / "student_best.pth"
        
        if not student_path.exists():
            continue
        
        # Load config from results.json
        try:
            with open(result_file, 'r') as f:
                exp_data = json.load(f)
            
            teacher_config = exp_data['config']['teacher']
            student_config = exp_data['config']['student']
            data_config = exp_data['config']['data']
            
            # Determine class counts
            if data_config['type'] == 'wound':
                num_modality = 10
                num_location = 3
            else:  # medpix
                num_modality = 2
                num_location = 5
        except Exception as e:
            print(f"Error reading config from {result_file}: {e}")
            continue
        
        # Analyze student from checkpoint
        try:
            student_stats = count_by_component(student_path)
        except Exception as e:
            print(f"Error processing {student_path}: {e}")
            continue
        
        # Estimate teacher parameters
        teacher_stats = estimate_teacher_params(teacher_config, num_modality, num_location)
        
        # Calculate compression
        compression = teacher_stats['total'] / student_stats['total']
        reduction = (1 - student_stats['total'] / teacher_stats['total']) * 100
        
        # CSV schema to match existing report
        results.append({
            'experiment': experiment_name,
            'teacher_vision': teacher_config['vision'],
            'teacher_text': teacher_config['text'],
            'teacher_params': teacher_stats['total'],
            'teacher_size': format_size(teacher_stats['total']),
            'student_vision': student_config['vision'],
            'student_text': student_config['text'],
            'student_params': student_stats['total'],
            'student_size': format_size(student_stats['total']),
            'compression_ratio': f"{compression:.2f}x",
            'reduction_pct': f"{reduction:.1f}%",
            # Extras (not in main CSV but useful for internal summaries)
            'teacher_proj': teacher_stats['projection'],
            'teacher_fusion': teacher_stats['fusion'],
            'teacher_heads': teacher_stats['heads'],
            'student_proj': student_stats['projection'],
            'student_fusion': student_stats['fusion'],
            'student_heads': student_stats['heads'],
        })
    
    if not results:
        print("No valid results!")
        return
    
    df = pd.DataFrame(results)
    
    # Display summary
    print("\n" + "="*100)
    print("MODEL SIZE REPORT")
    print("="*100)
    print()
    
    display_df = df[['experiment', 'teacher_size', 'student_size', 'compression_ratio', 'reduction_pct']].copy()
    display_df.columns = ['Experiment', 'Teacher', 'Student', 'Compression', 'Reduction']
    print(display_df.to_string(index=False, max_colwidth=40))
    
    # Teacher breakdown
    print("\n" + "="*100)
    print("TEACHER PARAMETER BREAKDOWN")
    print("="*100)
    print()
    
    teacher_breakdown = df[['experiment', 'teacher_proj', 'teacher_fusion', 'teacher_heads', 'teacher_params']].copy()
    for col in ['teacher_proj', 'teacher_fusion', 'teacher_heads', 'teacher_params']:
        teacher_breakdown[f'{col}_M'] = teacher_breakdown[col].apply(lambda x: f"{x/1e6:.2f}M")
    
    teacher_display = teacher_breakdown[['experiment', 'teacher_proj_M', 'teacher_fusion_M', 'teacher_heads_M', 'teacher_params_M']]
    teacher_display.columns = ['Experiment', 'Proj', 'Fusion', 'Heads', 'Total']
    print(teacher_display.to_string(index=False, max_colwidth=40))
    
    # Student breakdown
    print("\n" + "="*100)
    print("STUDENT PARAMETER BREAKDOWN")
    print("="*100)
    print()
    
    breakdown_df = df[['experiment', 'student_proj', 'student_fusion', 'student_heads', 'student_params']].copy()
    
    # Convert to millions
    for col in ['student_proj', 'student_fusion', 'student_heads', 'student_params']:
        breakdown_df[f'{col}_M'] = breakdown_df[col].apply(lambda x: f"{x/1e6:.2f}M")
    
    breakdown_display = breakdown_df[['experiment', 'student_proj_M', 'student_fusion_M', 'student_heads_M', 'student_params_M']]
    breakdown_display.columns = ['Experiment', 'Proj', 'Fusion', 'Heads', 'Total']
    print(breakdown_display.to_string(index=False, max_colwidth=40))
    
    # Prepare aggregate helper columns
    df['compression_value'] = df['compression_ratio'].str.replace('x', '').astype(float)
    df['reduction_value'] = df['reduction_pct'].str.rstrip('%').astype(float)

    # Save to CSV
    output_file = "logs/model_size_report.csv"
    # Write CSV with the schema matching existing reports
    main_cols = ['experiment','teacher_vision','teacher_text','teacher_params','teacher_size',
                 'student_vision','student_text','student_params','student_size',
                 'compression_ratio','reduction_pct']
    df[main_cols].to_csv(output_file, index=False)
    
    # Append aggregate mean row to main CSV
    aggregate_row = {
        'experiment': 'AGGREGATE_MEAN',
        'teacher_vision': '',
        'teacher_text': '',
        'teacher_params': int(df['teacher_params'].mean()),
        'teacher_size': format_size(int(df['teacher_params'].mean())),
        'student_vision': '',
        'student_text': '',
        'student_params': int(df['student_params'].mean()),
        'student_size': format_size(int(df['student_params'].mean())),
        'compression_ratio': f"{df['compression_value'].mean():.2f}x",
        'reduction_pct': f"{df['reduction_value'].mean():.1f}%",
    }
    pd.DataFrame([aggregate_row])[main_cols].to_csv(output_file, mode='a', header=False, index=False)
    print(f"\n✓ Report saved to: {output_file} (with AGGREGATE_MEAN row)")
    
    # Write a separate summary CSV with ranges and means
    summary_file = "logs/model_size_report_stats.csv"
    summary_rows = [
        {'metric': 'teacher_min', 'raw': int(df['teacher_params'].min()), 'formatted': format_size(int(df['teacher_params'].min()))},
        {'metric': 'teacher_max', 'raw': int(df['teacher_params'].max()), 'formatted': format_size(int(df['teacher_params'].max()))},
        {'metric': 'teacher_mean', 'raw': int(df['teacher_params'].mean()), 'formatted': format_size(int(df['teacher_params'].mean()))},
        {'metric': 'student_min', 'raw': int(df['student_params'].min()), 'formatted': format_size(int(df['student_params'].min()))},
        {'metric': 'student_max', 'raw': int(df['student_params'].max()), 'formatted': format_size(int(df['student_params'].max()))},
        {'metric': 'student_mean', 'raw': int(df['student_params'].mean()), 'formatted': format_size(int(df['student_params'].mean()))},
        {'metric': 'compression_min', 'raw': df['compression_value'].min(), 'formatted': f"{df['compression_value'].min():.2f}x"},
        {'metric': 'compression_max', 'raw': df['compression_value'].max(), 'formatted': f"{df['compression_value'].max():.2f}x"},
        {'metric': 'compression_mean', 'raw': df['compression_value'].mean(), 'formatted': f"{df['compression_value'].mean():.2f}x"},
        {'metric': 'reduction_min', 'raw': df['reduction_value'].min(), 'formatted': f"{df['reduction_value'].min():.1f}%"},
        {'metric': 'reduction_max', 'raw': df['reduction_value'].max(), 'formatted': f"{df['reduction_value'].max():.1f}%"},
        {'metric': 'reduction_mean', 'raw': df['reduction_value'].mean(), 'formatted': f"{df['reduction_value'].mean():.1f}%"},
    ]
    pd.DataFrame(summary_rows).to_csv(summary_file, index=False)
    print(f"✓ Summary saved to: {summary_file}")
    
    # Statistics
    print("\n" + "="*100)
    print("STATISTICS")
    print("="*100)
    print(f"\nTotal experiments: {len(df)}")
    
    print(f"\nTeacher sizes:")
    print(f"  Range: {format_size(df['teacher_params'].min())} - {format_size(df['teacher_params'].max())}")
    print(f"  Mean: {format_size(df['teacher_params'].mean())}")
    
    print(f"\nStudent sizes:")
    print(f"  Range: {format_size(df['student_params'].min())} - {format_size(df['student_params'].max())}")
    print(f"  Mean: {format_size(df['student_params'].mean())}")
    
    # Compression stats
    print(f"\nCompression ratios:")
    print(f"  Range: {df['compression_value'].min():.2f}x - {df['compression_value'].max():.2f}x")
    print(f"  Mean: {df['compression_value'].mean():.2f}x")
    
    print("\n" + "="*100)


if __name__ == "__main__":
    main()
