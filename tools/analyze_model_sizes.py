#!/usr/bin/env python3
"""
Analyze model parameter sizes from saved .pth files in logs/
"""

import torch
import os
from pathlib import Path
import pandas as pd
from collections import defaultdict


def count_parameters(model_dict):
    """Count total parameters from a model state dict."""
    total = 0
    for key, param in model_dict.items():
        if isinstance(param, torch.Tensor):
            total += param.numel()
    return total


def format_size(num_params):
    """Format parameter count in human-readable form."""
    if num_params >= 1e9:
        return f"{num_params/1e9:.2f}B"
    elif num_params >= 1e6:
        return f"{num_params/1e6:.2f}M"
    elif num_params >= 1e3:
        return f"{num_params/1e3:.2f}K"
    else:
        return str(num_params)


def analyze_log_directory(log_root="logs"):
    """Scan all log directories and analyze model sizes."""
    results = []
    processed_experiments = {}  # Cache parameters per experiment type
    
    log_path = Path(log_root)
    if not log_path.exists():
        print(f"Log directory {log_root} does not exist.")
        return None
    
    # Find all .pth files
    pth_files = list(log_path.rglob("*.pth"))
    
    if not pth_files:
        print(f"No .pth files found in {log_root}")
        return None
    
    print(f"Found {len(pth_files)} .pth files. Analyzing...\n")
    
    # Group files by experiment
    exp_files = defaultdict(list)
    for pth_file in pth_files:
        rel_path = pth_file.relative_to(log_path)
        experiment_name = rel_path.parts[0] if len(rel_path.parts) > 0 else "unknown"
        exp_files[experiment_name].append(pth_file)
    
    # Process each experiment
    for exp_idx, (experiment_name, files) in enumerate(sorted(exp_files.items()), 1):
        print(f"[{exp_idx}/{len(exp_files)}] Processing {experiment_name}...")
        
        # Look for teacher_best and student_best (or _final)
        teacher_file = None
        student_file = None
        
        for pth_file in files:
            stem = pth_file.stem
            if 'teacher' in stem and 'best' in stem:
                teacher_file = pth_file
            elif 'student' in stem and 'best' in stem:
                student_file = pth_file
            elif 'student' in stem and 'final' in stem and student_file is None:
                student_file = pth_file
        
        # Load and analyze
        for pth_file in [teacher_file, student_file]:
            if pth_file is None:
                continue
            
            try:
                # Load the model checkpoint (weights_only for speed)
                checkpoint = torch.load(pth_file, map_location='cpu', weights_only=False)
                
                # Get relative path for display
                rel_path = pth_file.relative_to(log_path)
                model_type = pth_file.stem  # e.g., 'teacher_best', 'student_final'
                
                # Count parameters
                if isinstance(checkpoint, dict):
                    # Handle both raw state_dict and wrapped checkpoints
                    if 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                    elif 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    else:
                        state_dict = checkpoint
                else:
                    state_dict = checkpoint
                
                num_params = count_parameters(state_dict)
                
                results.append({
                    'experiment': experiment_name,
                    'model_type': model_type,
                    'num_parameters': num_params,
                    'size_formatted': format_size(num_params),
                    'file_path': str(rel_path),
                    'file_size_mb': pth_file.stat().st_size / (1024 * 1024)
                })
                
            except Exception as e:
                print(f"  Warning: Could not load {pth_file.name}: {e}")
                continue
    
    if not results:
        print("No valid model checkpoints found.")
        return None
    
    # Create DataFrame
    df = pd.DataFrame(results)
    return df


def summarize_by_experiment(df):
    """Summarize parameter counts by experiment."""
    summary = []
    
    for exp in df['experiment'].unique():
        exp_df = df[df['experiment'] == exp]
        
        teacher_best = exp_df[exp_df['model_type'] == 'teacher_best']
        student_best = exp_df[exp_df['model_type'] == 'student_best']
        student_final = exp_df[exp_df['model_type'] == 'student_final']
        
        entry = {'experiment': exp}
        
        if not teacher_best.empty:
            entry['teacher_params'] = teacher_best.iloc[0]['num_parameters']
            entry['teacher_size'] = teacher_best.iloc[0]['size_formatted']
        else:
            entry['teacher_params'] = 0
            entry['teacher_size'] = 'N/A'
        
        if not student_best.empty:
            entry['student_params'] = student_best.iloc[0]['num_parameters']
            entry['student_size'] = student_best.iloc[0]['size_formatted']
        elif not student_final.empty:
            entry['student_params'] = student_final.iloc[0]['num_parameters']
            entry['student_size'] = student_final.iloc[0]['size_formatted']
        else:
            entry['student_params'] = 0
            entry['student_size'] = 'N/A'
        
        # Calculate compression ratio
        if entry['teacher_params'] > 0 and entry['student_params'] > 0:
            entry['compression_ratio'] = f"{entry['teacher_params'] / entry['student_params']:.2f}x"
            entry['reduction_pct'] = f"{(1 - entry['student_params'] / entry['teacher_params']) * 100:.1f}%"
        else:
            entry['compression_ratio'] = 'N/A'
            entry['reduction_pct'] = 'N/A'
        
        summary.append(entry)
    
    return pd.DataFrame(summary)


def main():
    print("="*80)
    print("Model Parameter Analysis")
    print("="*80)
    print()
    
    # Analyze all models
    df = analyze_log_directory()
    
    if df is None:
        return
    
    # Summary by experiment
    summary_df = summarize_by_experiment(df)
    
    # Display summary
    print("\n" + "="*80)
    print("SUMMARY BY EXPERIMENT")
    print("="*80)
    print()
    
    # Format for display
    display_df = summary_df[['experiment', 'teacher_size', 'student_size', 'compression_ratio', 'reduction_pct']]
    display_df.columns = ['Experiment', 'Teacher', 'Student', 'Compression', 'Reduction']
    
    print(display_df.to_string(index=False))
    
    # Save detailed results to CSV
    output_file = "logs/model_parameter_analysis.csv"
    df.to_csv(output_file, index=False)
    print(f"\n✓ Detailed results saved to: {output_file}")
    
    # Save summary to CSV
    summary_file = "logs/model_parameter_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"✓ Summary saved to: {summary_file}")
    
    # Additional statistics
    print("\n" + "="*80)
    print("STATISTICS")
    print("="*80)
    
    teacher_sizes = summary_df[summary_df['teacher_params'] > 0]['teacher_params']
    student_sizes = summary_df[summary_df['student_params'] > 0]['student_params']
    
    if not teacher_sizes.empty:
        print(f"\nTeacher models:")
        print(f"  Min: {format_size(teacher_sizes.min())}")
        print(f"  Max: {format_size(teacher_sizes.max())}")
        print(f"  Mean: {format_size(teacher_sizes.mean())}")
    
    if not student_sizes.empty:
        print(f"\nStudent models:")
        print(f"  Min: {format_size(student_sizes.min())}")
        print(f"  Max: {format_size(student_sizes.max())}")
        print(f"  Mean: {format_size(student_sizes.mean())}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
