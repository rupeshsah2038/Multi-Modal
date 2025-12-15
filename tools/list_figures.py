#!/usr/bin/env python3
"""
Quick preview and summary of generated research figures.
"""

import sys
from pathlib import Path
from typing import List

def list_figures(figures_dir: Path) -> List[Path]:
    """List all PNG figures in the directory."""
    return sorted(figures_dir.glob("*.png"))

def display_summary(figures_dir: Path):
    """Display a summary of available figures."""
    figures = list_figures(figures_dir)
    
    print("=" * 70)
    print("Research Article Figures Summary")
    print("=" * 70)
    print(f"\nLocation: {figures_dir.absolute()}")
    print(f"Total figures: {len(figures)} (each available in PNG and PDF)")
    print("\nAvailable figures:\n")
    
    categories = {
        'Fusion Comparison': [],
        'Loss Comparison': [],
        'Ultra-Edge Analysis': [],
        'Training Curves': [],
        'Cross-Dataset': [],
        'Model Size': []
    }
    
    for fig in figures:
        name = fig.stem
        if 'fusion_comparison' in name:
            categories['Fusion Comparison'].append(name)
        elif 'loss_comparison' in name:
            categories['Loss Comparison'].append(name)
        elif 'ultra_edge' in name:
            categories['Ultra-Edge Analysis'].append(name)
        elif 'training_curves' in name:
            categories['Training Curves'].append(name)
        elif 'cross_dataset' in name:
            categories['Cross-Dataset'].append(name)
        elif 'model_size' in name:
            categories['Model Size'].append(name)
    
    for category, figs in categories.items():
        if figs:
            print(f"  {category}:")
            for fig in figs:
                print(f"    - {fig}")
            print()
    
    print("=" * 70)
    print("\nUsage:")
    print("  View all:        cd figures/research_article && xdg-open *.png")
    print("  View specific:   xdg-open figures/research_article/fusion_comparison_medpix.png")
    print("  Regenerate all:  python tools/plot_research_figures.py")
    print("=" * 70)

def main():
    figures_dir = Path("figures/research_article")
    
    if not figures_dir.exists():
        print(f"Error: Figures directory not found: {figures_dir}")
        print("Run: python tools/plot_research_figures.py")
        sys.exit(1)
    
    display_summary(figures_dir)

if __name__ == "__main__":
    main()
