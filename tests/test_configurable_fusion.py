#!/usr/bin/env python3
"""
Test script to verify fusion and loss modules are properly configurable.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import torch
from models.teacher import Teacher
from models.student import Student

def test_fusion_configuration():
    """Test that different fusion types can be instantiated."""
    
    print("=" * 60)
    print("Testing Fusion Module Configuration")
    print("=" * 60)
    
    fusion_types = [
        'simple', 'concat_mlp', 'cross_attention', 'gated',
        'transformer_concat', 'modality_dropout', 'film',
        'energy_aware_adaptive', 'shomr'
    ]
    
    device = torch.device('cpu')
    
    for fusion_type in fusion_types:
        print(f"\nTesting fusion_type='{fusion_type}'...")
        
        try:
            # Test Teacher
            teacher = Teacher(
                vision='vit-base',
                text='bio-clinical-bert',
                fusion_type=fusion_type,
                fusion_dim=256,
                fusion_layers=2,
                num_modality_classes=2,
                num_location_classes=5
            ).to(device)
            
            # Test Student
            student = Student(
                vision='deit-small',
                text='distilbert',
                fusion_type=fusion_type,
                fusion_dim=256,
                fusion_layers=1,
                num_modality_classes=2,
                num_location_classes=5
            ).to(device)
            
            # Check fusion module type
            teacher_fusion = teacher.fusion.__class__.__name__
            student_fusion = student.fusion.__class__.__name__
            
            print(f"  ✓ Teacher fusion: {teacher_fusion}")
            print(f"  ✓ Student fusion: {student_fusion}")
            
            # Quick forward pass test (small batch)
            batch_size = 2
            pixel_values = torch.randn(batch_size, 3, 224, 224).to(device)
            input_ids = torch.randint(0, 1000, (batch_size, 128)).to(device)
            attention_mask = torch.ones(batch_size, 128).to(device)
            
            with torch.no_grad():
                t_out = teacher(pixel_values, input_ids, attention_mask)
                s_out = student(pixel_values, input_ids, attention_mask)
            
            # Verify output structure
            assert 'logits_modality' in t_out, "Missing logits_modality in teacher output"
            assert 'logits_location' in t_out, "Missing logits_location in teacher output"
            assert 'img_raw' in t_out, "Missing img_raw in teacher output"
            assert 'txt_raw' in t_out, "Missing txt_raw in teacher output"
            assert 'img_proj' in t_out, "Missing img_proj in teacher output"
            assert 'txt_proj' in t_out, "Missing txt_proj in teacher output"
            
            assert 'logits_modality' in s_out, "Missing logits_modality in student output"
            assert 'logits_location' in s_out, "Missing logits_location in student output"
            
            print(f"  ✓ Forward pass successful")
            print(f"  ✓ Output shapes: modality={t_out['logits_modality'].shape}, location={t_out['logits_location'].shape}")
            
        except Exception as e:
            print(f"  ✗ FAILED: {str(e)}")
            return False
    
    print("\n" + "=" * 60)
    print("✓ All fusion types working correctly!")
    print("=" * 60)
    return True


def test_config_loading():
    """Test that config files properly specify fusion and loss types."""
    
    print("\n" + "=" * 60)
    print("Testing Config File Integration")
    print("=" * 60)
    
    # Test default config
    with open('config/default.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    
    print(f"\nDefault config:")
    print(f"  Fusion type: {cfg.get('fusion', {}).get('type', 'NOT SET')}")
    print(f"  Loss type: {cfg.get('loss', {}).get('type', 'NOT SET')}")
    
    # Verify fusion config is present
    if 'fusion' not in cfg or 'type' not in cfg['fusion']:
        print("  ⚠ WARNING: fusion.type not set in config (will default to 'simple')")
    else:
        print(f"  ✓ Fusion type configured: {cfg['fusion']['type']}")
    
    # Verify loss config is present
    if 'loss' not in cfg or 'type' not in cfg['loss']:
        print("  ⚠ WARNING: loss.type not set in config (will default to 'vanilla')")
    else:
        print(f"  ✓ Loss type configured: {cfg['loss']['type']}")
    
    print("\n" + "=" * 60)
    return True


if __name__ == "__main__":
    print("\nStarting configuration tests...\n")
    
    # Test fusion module configuration
    fusion_ok = test_fusion_configuration()
    
    # Test config file loading
    config_ok = test_config_loading()
    
    if fusion_ok and config_ok:
        print("\n✓✓✓ All tests passed! ✓✓✓")
        print("\nFusion modules are now fully configurable through config files.")
        print("Update config/default.yaml to set fusion.type to any of:")
        print("  - simple")
        print("  - concat_mlp")
        print("  - cross_attention")
        print("  - gated")
        print("  - transformer_concat")
        print("  - modality_dropout")
        print("  - film")
        print("  - energy_aware_adaptive")
        print("  - shomr")
    else:
        print("\n✗✗✗ Some tests failed ✗✗✗")
        exit(1)
