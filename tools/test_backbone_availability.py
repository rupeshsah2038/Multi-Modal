#!/usr/bin/env python3
"""Test if all backbone models defined in backbones.py are available."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.backbones import VISION_PRETRAINED, TEXT_PRETRAINED, get_vision_backbone, get_text_backbone

def test_vision_backbones():
    """Test all vision backbones."""
    print("=" * 70)
    print("TESTING VISION BACKBONES")
    print("=" * 70)
    
    results = []
    for name, pretrained_id in VISION_PRETRAINED.items():
        print(f"\nTesting: {name} ({pretrained_id})")
        try:
            model = get_vision_backbone(name)
            print(f"  ✅ SUCCESS - Loaded {type(model).__name__}")
            results.append((name, "✅ Available", type(model).__name__))
        except Exception as e:
            error_msg = str(e)[:100]
            print(f"  ❌ FAILED - {error_msg}")
            results.append((name, "❌ Failed", error_msg))
    
    return results

def test_text_backbones():
    """Test all text backbones."""
    print("\n" + "=" * 70)
    print("TESTING TEXT BACKBONES")
    print("=" * 70)
    
    results = []
    for name, pretrained_id in TEXT_PRETRAINED.items():
        print(f"\nTesting: {name} ({pretrained_id})")
        try:
            model = get_text_backbone(name)
            print(f"  ✅ SUCCESS - Loaded {type(model).__name__}")
            results.append((name, "✅ Available", type(model).__name__))
        except Exception as e:
            error_msg = str(e)[:100]
            print(f"  ❌ FAILED - {error_msg}")
            results.append((name, "❌ Failed", error_msg))
    
    return results

def print_summary(vision_results, text_results):
    """Print summary table."""
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print("\nVISION BACKBONES:")
    print("-" * 70)
    for name, status, info in vision_results:
        print(f"{name:25s} {status:15s} {info}")
    
    print("\nTEXT BACKBONES:")
    print("-" * 70)
    for name, status, info in text_results:
        print(f"{name:25s} {status:15s} {info}")
    
    # Count successes
    vision_success = sum(1 for _, status, _ in vision_results if "✅" in status)
    text_success = sum(1 for _, status, _ in text_results if "✅" in status)
    total_vision = len(vision_results)
    total_text = len(text_results)
    
    print("\n" + "=" * 70)
    print(f"Vision: {vision_success}/{total_vision} available")
    print(f"Text:   {text_success}/{total_text} available")
    print(f"Total:  {vision_success + text_success}/{total_vision + total_text} available")
    print("=" * 70)

if __name__ == "__main__":
    print("\nBackbone Model Availability Test")
    print("Note: This will attempt to download models from Hugging Face")
    print("if they are not already cached.\n")
    
    vision_results = test_vision_backbones()
    text_results = test_text_backbones()
    print_summary(vision_results, text_results)
