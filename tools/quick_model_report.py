#!/usr/bin/env python3
"""
Quick model size report from results.json files (no .pth loading needed)
"""

import json
from pathlib import Path
import pandas as pd


# Exact parameter counts (from get_exact_model_params.py)
MODEL_PARAMS = {
    # Vision models
    'vit-large': 304_351_232,
    'vit-base': 86_389_248,
    'deit-base': 86_389_248,
    'deit-small': 21_813_504,
    'deit-tiny': 5_561_472,
    
    # Text models
    'bio-clinical-bert': 108_310_272,
    'distilbert': 66_362_880,
    'minilm': 22_713_216,
}


def estimate_total_params(vision_model, text_model, fusion_dim, fusion_layers, num_classes1=10, num_classes2=3):
    """Estimate total parameters including fusion and heads.
    
    Based on actual checkpoint analysis:
    - Vision/Text backbones: exact counts
    - Projection layers: 2 * (768 * fusion_dim + fusion_dim) for vit/bert hidden_size=768
    - Fusion module: depends on architecture (transformer: ~fusion_dim²*8 per layer)
    - Task heads: 2 * (fusion_dim * num_classes + num_classes)
    """
    vision_params = MODEL_PARAMS.get(vision_model, 0)
    text_params = MODEL_PARAMS.get(text_model, 0)
    
    # Projection layers (vision_hidden→fusion_dim, text_hidden→fusion_dim)
    # Assuming hidden_size=768 for vit-base/large and bert models
    hidden_size = 768 if 'base' in vision_model or 'distilbert' in text_model else 1024
    proj_params = 2 * (hidden_size * fusion_dim + fusion_dim)
    
    # Fusion module (transformer-based): self-attn + FFN per layer
    # Self-attn: Q,K,V,O projections + MLP(4x expansion)
    # Approximate: 8 * fusion_dim² per layer (Q,K,V,O + 2 FFN layers)
    fusion_params = fusion_layers * 8 * (fusion_dim ** 2)
    
    # Classification heads: 2 tasks with linear+bias
    head_params = (fusion_dim * num_classes1 + num_classes1) + (fusion_dim * num_classes2 + num_classes2)
    
    total = vision_params + text_params + proj_params + fusion_params + head_params
    return total


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


def analyze_from_results(log_root="logs"):
    """Extract model info from results.json files."""
    results = []
    
    log_path = Path(log_root)
    if not log_path.exists():
        print(f"Log directory {log_root} does not exist.")
        return None
    
    # Find all results.json files
    json_files = list(log_path.rglob("results.json"))
    
    if not json_files:
        print(f"No results.json files found in {log_root}")
        return None
    
    print(f"Found {len(json_files)} result files. Analyzing...\n")
    
    for json_file in sorted(json_files):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Extract experiment name
            rel_path = json_file.relative_to(log_path)
            experiment_name = rel_path.parts[0] if len(rel_path.parts) > 0 else "unknown"
            
            # Extract model config
            teacher_vision = data.get('models', {}).get('teacher', {}).get('vision', 'unknown')
            teacher_text = data.get('models', {}).get('teacher', {}).get('text', 'unknown')
            teacher_fusion_layers = data.get('models', {}).get('teacher', {}).get('fusion_layers', 2)
            
            student_vision = data.get('models', {}).get('student', {}).get('vision', 'unknown')
            student_text = data.get('models', {}).get('student', {}).get('text', 'unknown')
            student_fusion_layers = data.get('models', {}).get('student', {}).get('fusion_layers', 1)
            
            # Fusion dim from config
            teacher_fusion_dim = data.get('config', {}).get('teacher', {}).get('fusion_dim', 512)
            student_fusion_dim = data.get('config', {}).get('student', {}).get('fusion_dim', 512)
            
            # Estimate parameters
            teacher_params = estimate_total_params(
                teacher_vision, teacher_text, teacher_fusion_dim, teacher_fusion_layers
            )
            student_params = estimate_total_params(
                student_vision, student_text, student_fusion_dim, student_fusion_layers
            )
            
            # Calculate ratios
            if teacher_params > 0 and student_params > 0:
                compression = teacher_params / student_params
                reduction = (1 - student_params / teacher_params) * 100
            else:
                compression = 0
                reduction = 0
            
            results.append({
                'experiment': experiment_name,
                'teacher_vision': teacher_vision,
                'teacher_text': teacher_text,
                'teacher_params': teacher_params,
                'teacher_size': format_size(teacher_params),
                'student_vision': student_vision,
                'student_text': student_text,
                'student_params': student_params,
                'student_size': format_size(student_params),
                'compression_ratio': f"{compression:.2f}x" if compression > 0 else 'N/A',
                'reduction_pct': f"{reduction:.1f}%" if reduction > 0 else 'N/A',
            })
            
        except Exception as e:
            print(f"Warning: Could not parse {json_file.name}: {e}")
            continue
    
    if not results:
        print("No valid results found.")
        return None
    
    return pd.DataFrame(results)


def main():
    print("="*100)
    print("Model Parameter Report (from results.json)")
    print("="*100)
    print()
    
    df = analyze_from_results()
    
    if df is None:
        return
    
    # Display compact summary
    print("\n" + "="*100)
    print("MODEL SIZE SUMMARY")
    print("="*100)
    print()
    
    display_df = df[['experiment', 'teacher_vision', 'teacher_text', 'teacher_size',
                     'student_vision', 'student_text', 'student_size', 
                     'compression_ratio', 'reduction_pct']]
    
    display_df.columns = ['Experiment', 'T-Vision', 'T-Text', 'T-Size',
                          'S-Vision', 'S-Text', 'S-Size', 'Compression', 'Reduction']
    
    print(display_df.to_string(index=False, max_colwidth=30))
    
    # Save to CSV
    output_file = "logs/model_size_report.csv"
    df.to_csv(output_file, index=False)
    print(f"\n✓ Report saved to: {output_file}")
    
    # Statistics
    print("\n" + "="*100)
    print("STATISTICS")
    print("="*100)
    
    print(f"\nTotal experiments: {len(df)}")
    print(f"\nTeacher sizes:")
    print(f"  Range: {df['teacher_size'].min()} - {df['teacher_size'].max()}")
    print(f"  Mean: {format_size(df['teacher_params'].mean())}")
    
    print(f"\nStudent sizes:")
    print(f"  Range: {df['student_size'].min()} - {df['student_size'].max()}")
    print(f"  Mean: {format_size(df['student_params'].mean())}")
    
    print("\n" + "="*100)


if __name__ == "__main__":
    main()
