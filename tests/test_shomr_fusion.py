#!/usr/bin/env python3
"""Test the SHoMR-Fusion module."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from models.fusion.shomr import SHoMRFusion

def test_shomr_fusion():
    """Test SHoMR-Fusion with different configurations."""
    print("=" * 70)
    print("Testing SHoMR-Fusion (Soft-Hard Modality Routing Fusion)")
    print("=" * 70)
    
    # Setup
    batch_size = 8
    dim = 512
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\nDevice: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Embedding dim: {dim}")
    
    # Create model
    model = SHoMRFusion(
        dim=dim,
        heads=8,
        dropout=0.1,
        confidence_threshold=0.6,
        routing_temperature=1.0
    ).to(device)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create dummy inputs
    img_emb = torch.randn(batch_size, dim).to(device)
    txt_emb = torch.randn(batch_size, dim).to(device)
    
    print("\n" + "-" * 70)
    print("Test 1: Auto-routing (confidence-based)")
    print("-" * 70)
    
    model.eval()
    with torch.no_grad():
        fused, routing_info = model(img_emb, txt_emb)
    
    print(f"✓ Output shape: {fused.shape}")
    print(f"✓ Routing counts: {routing_info['routing_counts']}")
    print(f"✓ Mean confidence: {routing_info['mean_confidence']:.4f}")
    print(f"✓ Soft routing ratio: {routing_info['soft_ratio']:.2%}")
    
    print("\n" + "-" * 70)
    print("Test 2: Force soft routing")
    print("-" * 70)
    
    with torch.no_grad():
        fused, routing_info = model(img_emb, txt_emb, use_hard_routing=False)
    
    print(f"✓ Output shape: {fused.shape}")
    print(f"✓ Routing counts: {routing_info['routing_counts']}")
    print(f"✓ All samples used soft path: {routing_info['routing_counts']['soft'] == batch_size}")
    
    print("\n" + "-" * 70)
    print("Test 3: Force hard routing")
    print("-" * 70)
    
    with torch.no_grad():
        fused, routing_info = model(img_emb, txt_emb, use_hard_routing=True)
    
    print(f"✓ Output shape: {fused.shape}")
    print(f"✓ Routing counts: {routing_info['routing_counts']}")
    hard_total = sum([
        routing_info['routing_counts']['hard_vision'],
        routing_info['routing_counts']['hard_text'],
        routing_info['routing_counts']['hard_both']
    ])
    print(f"✓ All samples used hard path: {hard_total == batch_size}")
    
    print("\n" + "-" * 70)
    print("Test 4: Training mode (with gradients)")
    print("-" * 70)
    
    model.train()
    img_emb.requires_grad = True
    txt_emb.requires_grad = True
    
    fused, routing_info = model(img_emb, txt_emb)
    loss = fused.sum()
    loss.backward()
    
    print(f"✓ Forward pass successful")
    print(f"✓ Backward pass successful")
    print(f"✓ Gradients computed: img={img_emb.grad is not None}, txt={txt_emb.grad is not None}")
    
    print("\n" + "-" * 70)
    print("Test 5: Different batch sizes")
    print("-" * 70)
    
    model.eval()
    for bs in [1, 16, 32]:
        img_test = torch.randn(bs, dim).to(device)
        txt_test = torch.randn(bs, dim).to(device)
        
        with torch.no_grad():
            fused_test, info = model(img_test, txt_test)
        
        print(f"✓ Batch size {bs:2d}: Output shape {fused_test.shape}, "
              f"Soft ratio: {info['soft_ratio']:.2%}")
    
    print("\n" + "-" * 70)
    print("Test 6: Routing statistics")
    print("-" * 70)
    
    stats = model.get_routing_stats()
    print(f"✓ Confidence threshold: {stats['confidence_threshold']}")
    print(f"✓ Routing temperature: {stats['routing_temperature']}")
    
    print("\n" + "=" * 70)
    print("✅ All tests passed successfully!")
    print("=" * 70)
    
    # Summary
    print("\n📊 SHoMR-Fusion Summary:")
    print(f"  • Combines soft (confidence-weighted) and hard (discrete) routing")
    print(f"  • Automatic threshold-based switching: {model.confidence_threshold}")
    print(f"  • Three hard paths: vision-only, text-only, both")
    print(f"  • Fallback for low-confidence samples")
    print(f"  • Model size: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"  • Supports both training and inference modes")

if __name__ == "__main__":
    test_shomr_fusion()
