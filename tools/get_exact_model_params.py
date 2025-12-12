#!/usr/bin/env python3
"""
Get exact parameter counts by instantiating models (config only, no weights)
"""

import sys
import torch
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.backbones import get_vision_backbone, get_text_backbone


def count_parameters(model):
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_exact_counts():
    """Get exact parameter counts for all backbones."""
    
    # Vision models
    vision_models = [
        'vit-large',
        'vit-base', 
        'deit-base',
        'deit-small',
        'deit-tiny',
    ]
    
    # Text models
    text_models = [
        'bio-clinical-bert',
        'distilbert',
        'minilm',
    ]
    
    print("="*80)
    print("EXACT MODEL PARAMETER COUNTS")
    print("="*80)
    print()
    
    print("VISION MODELS:")
    print("-" * 80)
    vision_counts = {}
    for model_name in vision_models:
        try:
            model = get_vision_backbone(model_name)
            params = count_parameters(model)
            vision_counts[model_name] = params
            print(f"  {model_name:25s}: {params:>12,} ({params/1e6:.2f}M)")
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  {model_name:25s}: ERROR - {e}")
    
    print()
    print("TEXT MODELS:")
    print("-" * 80)
    text_counts = {}
    for model_name in text_models:
        try:
            model = get_text_backbone(model_name)
            params = count_parameters(model)
            text_counts[model_name] = params
            print(f"  {model_name:25s}: {params:>12,} ({params/1e6:.2f}M)")
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  {model_name:25s}: ERROR - {e}")
    
    print()
    print("="*80)
    print("PYTHON DICTIONARY FOR CODE:")
    print("="*80)
    print()
    print("MODEL_PARAMS = {")
    print("    # Vision models")
    for name, count in vision_counts.items():
        print(f"    '{name}': {count},")
    print()
    print("    # Text models")
    for name, count in text_counts.items():
        print(f"    '{name}': {count},")
    print("}")
    print()
    
    return vision_counts, text_counts


if __name__ == "__main__":
    get_exact_counts()
