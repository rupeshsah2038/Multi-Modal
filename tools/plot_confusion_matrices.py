#!/usr/bin/env python3
"""
Generate confusion matrix visualizations and training metrics plots from experimental results.

This script processes:
- Confusion matrices stored as .npy files
- Training metrics from metrics.csv files
- Generates publication-quality visualizations for research articles
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse

# Publication-quality plot settings
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 13,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# Color schemes
CMAP_CM = 'Blues'
COLORS = {
    'train_loss': '#2E86AB',
    'modality_acc': '#A23B72',
    'location_acc': '#F18F01',
    'type_acc': '#06A77D',
    'severity_acc': '#C73E1D',
    'modality_f1': '#9D5C63',
    'location_f1': '#D6A99A',
    'type_f1': '#5BA3A3',
    'severity_f1': '#E76F51',
}


class ConfusionMatrixVisualizer:
    """Visualize confusion matrices from experimental results."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_confusion_matrix(self, cm: np.ndarray, title: str, 
                             class_names: Optional[List[str]] = None,
                             filename: str = None, normalize: bool = False):
        """Plot a single confusion matrix with optional normalization."""
        
        if normalize:
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm_display = cm_normalized
            fmt = '.2%'
            vmax = 1.0
        else:
            cm_display = cm
            fmt = 'd'
            vmax = None
        
        fig, ax = plt.subplots(figsize=(6, 5))
        
        sns.heatmap(cm_display, annot=True, fmt=fmt, cmap=CMAP_CM,
                   cbar=True, square=True, ax=ax, vmax=vmax,
                   xticklabels=class_names if class_names else 'auto',
                   yticklabels=class_names if class_names else 'auto',
                   cbar_kws={'label': 'Proportion' if normalize else 'Count'})
        
        ax.set_ylabel('True Label', fontweight='bold')
        ax.set_xlabel('Predicted Label', fontweight='bold')
        ax.set_title(title, pad=15)
        
        # Add statistics
        accuracy = np.trace(cm) / np.sum(cm)
        textstr = f'Accuracy: {accuracy:.3f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        
        if filename:
            plt.savefig(self.output_dir / f"{filename}.png")
            plt.savefig(self.output_dir / f"{filename}.pdf")
        
        plt.close()
    
    def plot_experiment_confusion_matrices(self, exp_dir: Path, exp_name: str):
        """Plot all confusion matrices for a single experiment."""
        
        # Find all confusion matrix files
        cm_files = list(exp_dir.glob("cm_*.npy"))
        
        if not cm_files:
            return 0
        
        # Group by task and split
        cms_by_task = {}
        for cm_file in cm_files:
            # Parse filename: cm_{task}_{split}.npy
            parts = cm_file.stem.split('_')
            task = parts[1]
            split = parts[2] if len(parts) > 2 else 'test'
            
            if task not in cms_by_task:
                cms_by_task[task] = {}
            
            cm = np.load(cm_file)
            cms_by_task[task][split] = cm
        
        # Generate plots for each task
        count = 0
        for task, splits in cms_by_task.items():
            # Plot both splits side by side if available
            if 'dev' in splits and 'test' in splits:
                self._plot_dual_cm(splits['dev'], splits['test'], 
                                  task, exp_name)
                count += 1
            
            # Plot individual matrices with normalization
            for split, cm in splits.items():
                title = f"{exp_name}\n{task.capitalize()} - {split.upper()}"
                filename = f"{exp_name}_{task}_{split}_normalized"
                self.plot_confusion_matrix(cm, title, filename=filename, normalize=True)
                count += 1
        
        return count
    
    def _plot_dual_cm(self, cm_dev: np.ndarray, cm_test: np.ndarray, 
                     task: str, exp_name: str):
        """Plot dev and test confusion matrices side by side."""
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Normalize both
        cm_dev_norm = cm_dev.astype('float') / cm_dev.sum(axis=1)[:, np.newaxis]
        cm_test_norm = cm_test.astype('float') / cm_test.sum(axis=1)[:, np.newaxis]
        
        # Dev confusion matrix
        ax = axes[0]
        sns.heatmap(cm_dev_norm, annot=True, fmt='.2%', cmap=CMAP_CM,
                   cbar=True, square=True, ax=ax, vmax=1.0,
                   cbar_kws={'label': 'Proportion'})
        ax.set_ylabel('True Label', fontweight='bold')
        ax.set_xlabel('Predicted Label', fontweight='bold')
        ax.set_title(f'Dev Set - {task.capitalize()}')
        
        dev_acc = np.trace(cm_dev) / np.sum(cm_dev)
        ax.text(0.02, 0.98, f'Acc: {dev_acc:.3f}', transform=ax.transAxes,
               fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Test confusion matrix
        ax = axes[1]
        sns.heatmap(cm_test_norm, annot=True, fmt='.2%', cmap=CMAP_CM,
                   cbar=True, square=True, ax=ax, vmax=1.0,
                   cbar_kws={'label': 'Proportion'})
        ax.set_ylabel('True Label', fontweight='bold')
        ax.set_xlabel('Predicted Label', fontweight='bold')
        ax.set_title(f'Test Set - {task.capitalize()}')
        
        test_acc = np.trace(cm_test) / np.sum(cm_test)
        ax.text(0.02, 0.98, f'Acc: {test_acc:.3f}', transform=ax.transAxes,
               fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        fig.suptitle(f'{exp_name} - {task.capitalize()} Task', 
                    fontsize=13, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{exp_name}_{task}_comparison.png")
        plt.savefig(self.output_dir / f"{exp_name}_{task}_comparison.pdf")
        plt.close()


class MetricsVisualizer:
    """Visualize training metrics from CSV files."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_training_metrics(self, metrics_file: Path, exp_name: str):
        """Plot training metrics (loss, accuracy, F1) over epochs."""
        
        if not metrics_file.exists():
            return False
        
        df = pd.read_csv(metrics_file)
        
        if 'epoch' not in df.columns:
            return False
        
        # Determine dataset type from column names
        is_medpix = 'dev_modality_acc' in df.columns
        is_wound = 'dev_type_acc' in df.columns
        
        if is_medpix:
            self._plot_medpix_metrics(df, exp_name)
        elif is_wound:
            self._plot_wound_metrics(df, exp_name)
        else:
            return False
        
        return True
    
    def _plot_medpix_metrics(self, df: pd.DataFrame, exp_name: str):
        """Plot MedPix-specific metrics."""
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Training Loss
        ax = axes[0, 0]
        ax.plot(df['epoch'], df['train_loss'], marker='o', linewidth=2,
               color=COLORS['train_loss'], label='Train Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss Over Epochs')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Plot 2: Accuracy Comparison
        ax = axes[0, 1]
        ax.plot(df['epoch'], df['dev_modality_acc'], marker='o', linewidth=2,
               color=COLORS['modality_acc'], label='Modality Acc')
        ax.plot(df['epoch'], df['dev_location_acc'], marker='s', linewidth=2,
               color=COLORS['location_acc'], label='Location Acc')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Validation Accuracy Over Epochs')
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Plot 3: F1 Score Comparison
        ax = axes[1, 0]
        ax.plot(df['epoch'], df['dev_modality_f1'], marker='o', linewidth=2,
               color=COLORS['modality_f1'], label='Modality F1')
        ax.plot(df['epoch'], df['dev_location_f1'], marker='s', linewidth=2,
               color=COLORS['location_f1'], label='Location F1')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('F1 Score')
        ax.set_title('Validation F1 Score Over Epochs')
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Plot 4: Inference Time
        ax = axes[1, 1]
        if 'dev_infer_ms' in df.columns:
            ax.plot(df['epoch'], df['dev_infer_ms'], marker='o', linewidth=2,
                   color=COLORS['train_loss'], label='Inference Time')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Time (ms)')
            ax.set_title('Inference Time Over Epochs')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Add mean line
            mean_time = df['dev_infer_ms'].mean()
            ax.axhline(mean_time, color='red', linestyle='--', alpha=0.7,
                      label=f'Mean: {mean_time:.2f}ms')
            ax.legend()
        
        fig.suptitle(f'{exp_name} - Training Metrics (MedPix)', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{exp_name}_metrics.png")
        plt.savefig(self.output_dir / f"{exp_name}_metrics.pdf")
        plt.close()
    
    def _plot_wound_metrics(self, df: pd.DataFrame, exp_name: str):
        """Plot Wound-specific metrics."""
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Training Loss
        ax = axes[0, 0]
        ax.plot(df['epoch'], df['train_loss'], marker='o', linewidth=2,
               color=COLORS['train_loss'], label='Train Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss Over Epochs')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Plot 2: Accuracy Comparison
        ax = axes[0, 1]
        ax.plot(df['epoch'], df['dev_type_acc'], marker='o', linewidth=2,
               color=COLORS['type_acc'], label='Type Acc')
        ax.plot(df['epoch'], df['dev_severity_acc'], marker='s', linewidth=2,
               color=COLORS['severity_acc'], label='Severity Acc')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Validation Accuracy Over Epochs')
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Plot 3: F1 Score Comparison
        ax = axes[1, 0]
        ax.plot(df['epoch'], df['dev_type_f1'], marker='o', linewidth=2,
               color=COLORS['type_f1'], label='Type F1')
        ax.plot(df['epoch'], df['dev_severity_f1'], marker='s', linewidth=2,
               color=COLORS['severity_f1'], label='Severity F1')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('F1 Score')
        ax.set_title('Validation F1 Score Over Epochs')
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Plot 4: Inference Time
        ax = axes[1, 1]
        if 'dev_infer_ms' in df.columns:
            ax.plot(df['epoch'], df['dev_infer_ms'], marker='o', linewidth=2,
                   color=COLORS['train_loss'], label='Inference Time')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Time (ms)')
            ax.set_title('Inference Time Over Epochs')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Add mean line
            mean_time = df['dev_infer_ms'].mean()
            ax.axhline(mean_time, color='red', linestyle='--', alpha=0.7,
                      label=f'Mean: {mean_time:.2f}ms')
            ax.legend()
        
        fig.suptitle(f'{exp_name} - Training Metrics (Wound)', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{exp_name}_metrics.png")
        plt.savefig(self.output_dir / f"{exp_name}_metrics.pdf")
        plt.close()
    
    def plot_comparison_across_experiments(self, experiments: Dict[str, pd.DataFrame],
                                          metric_name: str, title: str, filename: str):
        """Compare a specific metric across multiple experiments."""
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = sns.color_palette("husl", len(experiments))
        
        for idx, (exp_name, df) in enumerate(experiments.items()):
            if metric_name in df.columns:
                ax.plot(df['epoch'], df[metric_name], marker='o', linewidth=2,
                       color=colors[idx], label=exp_name, alpha=0.8)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric_name.replace('_', ' ').title())
        ax.set_title(title)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{filename}.png")
        plt.savefig(self.output_dir / f"{filename}.pdf")
        plt.close()


class ExperimentBatchVisualizer:
    """Visualize confusion matrices and metrics for batches of experiments."""
    
    def __init__(self, logs_dir: Path, output_base_dir: Path):
        self.logs_dir = logs_dir
        self.output_base_dir = output_base_dir
    
    def process_experiment_group(self, group_name: str, 
                                output_subdir: Optional[str] = None):
        """Process all experiments in a group (fusion-explore, loss-explore, etc.)."""
        
        group_dir = self.logs_dir / group_name
        if not group_dir.exists():
            print(f"  ⚠ Group directory not found: {group_dir}")
            return
        
        output_dir = self.output_base_dir / (output_subdir or group_name)
        
        cm_viz = ConfusionMatrixVisualizer(output_dir / "confusion_matrices")
        metrics_viz = MetricsVisualizer(output_dir / "training_metrics")
        
        exp_dirs = [d for d in group_dir.iterdir() if d.is_dir()]
        
        if not exp_dirs:
            print(f"  ⚠ No experiment directories found in {group_name}")
            return
        
        print(f"\n  Processing {len(exp_dirs)} experiments in {group_name}...")
        
        cm_count = 0
        metrics_count = 0
        
        for exp_dir in sorted(exp_dirs):
            exp_name = exp_dir.name
            
            # Process confusion matrices
            count = cm_viz.plot_experiment_confusion_matrices(exp_dir, exp_name)
            if count > 0:
                cm_count += count
            
            # Process training metrics
            metrics_file = exp_dir / "metrics.csv"
            if metrics_viz.plot_training_metrics(metrics_file, exp_name):
                metrics_count += 1
        
        print(f"    ✓ Generated {cm_count} confusion matrix plots")
        print(f"    ✓ Generated {metrics_count} training metric plots")
        
        return cm_count, metrics_count


def main():
    parser = argparse.ArgumentParser(
        description='Generate confusion matrix and training metrics visualizations'
    )
    parser.add_argument('--logs-dir', type=str, default='logs',
                       help='Path to logs directory')
    parser.add_argument('--output-dir', type=str, default='figures/detailed_analysis',
                       help='Output directory for generated plots')
    parser.add_argument('--groups', nargs='+',
                       choices=['fusion-explore', 'loss-explore', 'ultra-edge', 
                               'ultra-edge2', 'all'],
                       default=['all'],
                       help='Which experiment groups to process')
    parser.add_argument('--experiments', nargs='+',
                       help='Process specific experiment directories only')
    
    args = parser.parse_args()
    
    logs_dir = Path(args.logs_dir)
    output_dir = Path(args.output_dir)
    
    if not logs_dir.exists():
        print(f"Error: Logs directory not found: {logs_dir}")
        return
    
    print("=" * 70)
    print("Confusion Matrix & Training Metrics Visualizer")
    print("=" * 70)
    
    batch_viz = ExperimentBatchVisualizer(logs_dir, output_dir)
    
    groups_to_process = ['fusion-explore', 'loss-explore', 'ultra-edge', 'ultra-edge2'] \
                       if 'all' in args.groups else args.groups
    
    total_cm = 0
    total_metrics = 0
    
    for group in groups_to_process:
        print(f"\n[{group.upper()}]")
        result = batch_viz.process_experiment_group(group)
        if result:
            cm_count, metrics_count = result
            total_cm += cm_count
            total_metrics += metrics_count
    
    print("\n" + "=" * 70)
    print(f"✓ Total confusion matrix plots: {total_cm}")
    print(f"✓ Total training metric plots: {total_metrics}")
    print(f"✓ All plots saved to: {output_dir.absolute()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
