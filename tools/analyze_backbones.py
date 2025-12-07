#!/usr/bin/env python3
"""
Analyze backbone models and generate CSV with specifications.
"""
import sys
import csv
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoConfig
from models.backbones import VISION_PRETRAINED, TEXT_PRETRAINED


def get_model_info(pretrained_name, model_type):
    """Get model configuration information."""
    try:
        config = AutoConfig.from_pretrained(pretrained_name)
        
        # Extract relevant attributes
        info = {
            'model_type': model_type,
            'pretrained_id': pretrained_name,
        }
        
        # Common attributes across models
        if hasattr(config, 'hidden_size'):
            info['hidden_size'] = config.hidden_size
        
        if hasattr(config, 'num_hidden_layers'):
            info['num_layers'] = config.num_hidden_layers
        
        if hasattr(config, 'num_attention_heads'):
            info['num_attention_heads'] = config.num_attention_heads
        
        if hasattr(config, 'intermediate_size'):
            info['intermediate_size'] = config.intermediate_size
        
        # Vision-specific
        if hasattr(config, 'image_size'):
            info['image_size'] = config.image_size
        
        if hasattr(config, 'patch_size'):
            info['patch_size'] = config.patch_size
        
        # Text-specific
        if hasattr(config, 'vocab_size'):
            info['vocab_size'] = config.vocab_size
        
        if hasattr(config, 'max_position_embeddings'):
            info['max_seq_length'] = config.max_position_embeddings
        
        # Estimate parameter count (approximate)
        params = estimate_parameters(config)
        info['estimated_params_M'] = round(params / 1e6, 2)
        
        return info
    except Exception as e:
        print(f"Error loading {pretrained_name}: {e}")
        return None


def estimate_parameters(config):
    """Estimate total parameters based on config."""
    hidden_size = getattr(config, 'hidden_size', 768)
    num_layers = getattr(config, 'num_hidden_layers', 12)
    intermediate_size = getattr(config, 'intermediate_size', 3072)
    num_heads = getattr(config, 'num_attention_heads', 12)
    vocab_size = getattr(config, 'vocab_size', 0)
    
    # Rough estimation
    # Embedding layer
    params = vocab_size * hidden_size if vocab_size > 0 else 0
    
    # Each transformer layer: attention + feedforward
    attn_params = 4 * hidden_size * hidden_size  # Q, K, V, O projections
    ffn_params = 2 * hidden_size * intermediate_size
    layer_params = attn_params + ffn_params
    params += num_layers * layer_params
    
    # Layer norms and other params (rough estimate)
    params += num_layers * 2 * hidden_size
    
    return params


def main():
    """Generate CSV file with backbone model specifications."""
    output_file = Path(__file__).parent.parent / 'logs' / 'backbone_models_comparison.csv'
    
    # Collect all model info
    models_data = []
    
    print("Analyzing vision backbones...")
    for name, pretrained in VISION_PRETRAINED.items():
        print(f"  - {name}: {pretrained}")
        info = get_model_info(pretrained, 'Vision')
        if info:
            info['friendly_name'] = name
            models_data.append(info)
    
    print("\nAnalyzing text backbones...")
    for name, pretrained in TEXT_PRETRAINED.items():
        print(f"  - {name}: {pretrained}")
        info = get_model_info(pretrained, 'Text')
        if info:
            info['friendly_name'] = name
            models_data.append(info)
    
    # Define CSV columns
    fieldnames = [
        'model_type',
        'friendly_name',
        'pretrained_id',
        'hidden_size',
        'num_layers',
        'num_attention_heads',
        'intermediate_size',
        'image_size',
        'patch_size',
        'vocab_size',
        'max_seq_length',
        'estimated_params_M',
    ]
    
    # Write to CSV
    print(f"\nWriting results to {output_file}...")
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for model_data in models_data:
            # Fill in missing fields with empty strings
            row = {field: model_data.get(field, '') for field in fieldnames}
            writer.writerow(row)
    
    print(f"✓ Successfully generated {output_file}")
    print(f"✓ Analyzed {len(models_data)} models")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    vision_models = [m for m in models_data if m['model_type'] == 'Vision']
    text_models = [m for m in models_data if m['model_type'] == 'Text']
    
    print(f"\nVision Models: {len(vision_models)}")
    if vision_models:
        hidden_sizes = [m.get('hidden_size') for m in vision_models if 'hidden_size' in m]
        params = [m.get('estimated_params_M') for m in vision_models if 'estimated_params_M' in m]
        if hidden_sizes:
            print(f"  Hidden sizes: {min(hidden_sizes)} - {max(hidden_sizes)}")
        if params:
            print(f"  Parameters: {min(params):.2f}M - {max(params):.2f}M")
    
    print(f"\nText Models: {len(text_models)}")
    if text_models:
        hidden_sizes = [m.get('hidden_size') for m in text_models if 'hidden_size' in m]
        params = [m.get('estimated_params_M') for m in text_models if 'estimated_params_M' in m]
        if hidden_sizes:
            print(f"  Hidden sizes: {min(hidden_sizes)} - {max(hidden_sizes)}")
        if params:
            print(f"  Parameters: {min(params):.2f}M - {max(params):.2f}M")
    
    # Recommended fusion dimensions
    print("\n" + "="*80)
    print("RECOMMENDED FUSION DIMENSIONS")
    print("="*80)
    print("\nBased on hidden sizes:")
    all_hidden = sorted(set([m.get('hidden_size') for m in models_data if 'hidden_size' in m]))
    for h in all_hidden:
        models = [m['friendly_name'] for m in models_data if m.get('hidden_size') == h]
        print(f"  Hidden {h} -> Fusion 512 or {h} (models: {', '.join(models[:3])})")


if __name__ == '__main__':
    main()
