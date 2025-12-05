#!/usr/bin/env python3
"""
Display tabulated backbone configurations from config/default.yaml
Useful for selecting student vision/text backbones for different deployment targets.
"""
import yaml

def format_table(headers, rows, col_widths=None):
    """Simple ASCII table formatter."""
    if col_widths is None:
        col_widths = [max(len(str(h)), max(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    
    separator = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    header_row = "|" + "|".join(f" {h:^{col_widths[i]}} " for i, h in enumerate(headers)) + "|"
    
    print(separator)
    print(header_row)
    print(separator)
    
    for row in rows:
        row_str = "|" + "|".join(f" {str(row[i]):<{col_widths[i]}} " for i in range(len(headers))) + "|"
        print(row_str)
    
    print(separator)

def show_backbones(config_path="config/default.yaml"):
    """Load and display vision/text backbone tables."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    
    recommended = cfg.get("student", {}).get("recommended", {})
    
    print("\n" + "=" * 120)
    print("VISION BACKBONES".center(120))
    print("=" * 120)
    if "vision_table" in recommended:
        headers = ["Model", "HuggingFace ID", "Params (M)", "Latency (ms)", "Tier"]
        rows = [
            [v["name"], v["hf_id"], v["params_m"], v["latency_ms"], v["tier"]]
            for v in recommended["vision_table"]
        ]
        format_table(headers, rows, col_widths=[28, 50, 12, 13, 18])
    
    print("\n" + "=" * 120)
    print("TEXT BACKBONES".center(120))
    print("=" * 120)
    if "text_table" in recommended:
        headers = ["Model", "HuggingFace ID", "Params (M)", "Latency (ms)", "Tier"]
        rows = [
            [t["name"], t["hf_id"], t["params_m"], t["latency_ms"], t["tier"]]
            for t in recommended["text_table"]
        ]
        format_table(headers, rows, col_widths=[28, 50, 12, 13, 18])
    
    print("\n" + "=" * 120)
    print("DEPLOYMENT TIERS".center(120))
    print("=" * 120)
    tiers = {
        "standard": "Full-capacity models (86-110M params); baseline performance.",
        "standard-domain": "Domain-specific large models; optimized for medical text.",
        "standard-pretrain": "Large models with ImageNet-21K pretraining; strong features.",
        "edge": "Compact models (5-25M params); 70-80% of standard accuracy; 2-5x faster.",
        "ultra-edge": "Ultra-compact models (<6M params); 40-60% of standard accuracy; 8-10x faster.",
    }
    for tier, desc in tiers.items():
        print(f"  {tier:20s}: {desc}")
    
    print("\n" + "=" * 120)
    print("EXAMPLE RUN COMMANDS".center(120))
    print("=" * 120)
    print("# Original baseline (standard-tier)")
    print("python tools/batch_runs.py --base config/default.yaml --runs original --execute --epochs 10")
    print("\n# Edge deployment (5.7M vision + 25M text)")
    print("python tools/batch_runs.py --base config/default.yaml --runs edge-vision,edge-text --execute --epochs 10")
    print("\n# Ultra-edge deployment (5.7M vision + 4.4M text)")
    print("python tools/batch_runs.py --base config/default.yaml --runs ultra-edge --execute --epochs 10")
    print("\n# Compare all variants")
    print("python tools/batch_runs.py --base config/default.yaml --runs original,edge-vision,edge-text,ultra-edge --execute --epochs 5")
    print()

if __name__ == "__main__":
    show_backbones()
