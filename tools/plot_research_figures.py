#!/usr/bin/env python3
"""
Generate publication-quality plots for research article from experimental results.

This script processes results.json files from various experiments and generates:
1. Fusion strategy comparison plots
2. Loss function comparison plots  
3. Ultra-edge performance plots
4. Model size vs accuracy trade-offs
5. Training curves
6. Cross-dataset comparisons
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import argparse

# Publication-quality plot settings
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# Color palette
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'accent1': '#F18F01',
    'accent2': '#C73E1D',
    'neutral': '#6C757D',
    'success': '#06A77D',
}

FUSION_COLORS = sns.color_palette("Set2", 10)
LOSS_COLORS = sns.color_palette("Dark2", 5)
DATASET_COLORS = {'medpix': COLORS['primary'], 'wound': COLORS['secondary']}


class ExperimentLoader:
    """Load and parse experimental results from logs directory."""
    
    def __init__(self, logs_dir: Path):
        self.logs_dir = logs_dir
        
    def load_results(self, experiment_path: Path) -> Dict:
        """Load results.json from experiment directory."""
        results_file = experiment_path / "results.json"
        if results_file.exists():
            with open(results_file, 'r') as f:
                return json.load(f)
        return None
    
    def load_fusion_explore(self) -> pd.DataFrame:
        """Load fusion exploration experiments."""
        fusion_dir = self.logs_dir / "fusion-explore"
        records = []
        
        for exp_dir in fusion_dir.iterdir():
            if exp_dir.is_dir():
                results = self.load_results(exp_dir)
                if results:
                    exp_name = exp_dir.name
                    parts = exp_name.split('-')
                    dataset = parts[0]
                    fusion_type = '-'.join(parts[1:-1])
                    loss_type = parts[-1]
                    
                    test_metrics = results['metrics']['test']
                    
                    record = {
                        'experiment': exp_name,
                        'dataset': dataset,
                        'fusion': fusion_type,
                        'loss': loss_type,
                    }
                    
                    # Add task-specific metrics
                    if dataset == 'medpix':
                        record.update({
                            'modality_acc': test_metrics.get('test_modality_acc', test_metrics.get('modality_accuracy', 0)),
                            'modality_f1': test_metrics.get('test_modality_f1', test_metrics.get('modality_f1', 0)),
                            'location_acc': test_metrics.get('test_location_acc', test_metrics.get('location_accuracy', 0)),
                            'location_f1': test_metrics.get('test_location_f1', test_metrics.get('location_f1', 0)),
                        })
                        record['avg_acc'] = (record['modality_acc'] + record['location_acc']) / 2
                        record['avg_f1'] = (record['modality_f1'] + record['location_f1']) / 2
                    else:  # wound
                        record.update({
                            'type_acc': test_metrics.get('test_type_acc', test_metrics.get('type_accuracy', 0)),
                            'type_f1': test_metrics.get('test_type_f1', test_metrics.get('type_f1', 0)),
                            'severity_acc': test_metrics.get('test_severity_acc', test_metrics.get('severity_accuracy', 0)),
                            'severity_f1': test_metrics.get('test_severity_f1', test_metrics.get('severity_f1', 0)),
                        })
                        record['avg_acc'] = (record['type_acc'] + record['severity_acc']) / 2
                        record['avg_f1'] = (record['type_f1'] + record['severity_f1']) / 2
                    
                    # Inference time
                    record['inference_ms'] = test_metrics.get('test_infer_ms', test_metrics.get('inference_time_ms', np.nan))
                    
                    records.append(record)
        
        return pd.DataFrame(records)
    
    def load_loss_explore(self) -> pd.DataFrame:
        """Load loss function exploration experiments."""
        loss_dir = self.logs_dir / "loss-explore"
        records = []
        
        for exp_dir in loss_dir.iterdir():
            if exp_dir.is_dir():
                results = self.load_results(exp_dir)
                if results:
                    exp_name = exp_dir.name
                    parts = exp_name.split('-')
                    dataset = parts[0]
                    fusion_type = parts[1]
                    loss_type = parts[-1]
                    
                    test_metrics = results['metrics']['test']
                    
                    record = {
                        'experiment': exp_name,
                        'dataset': dataset,
                        'fusion': fusion_type,
                        'loss': loss_type,
                    }
                    
                    if dataset == 'medpix':
                        record.update({
                            'modality_acc': test_metrics.get('test_modality_acc', test_metrics.get('modality_accuracy', 0)),
                            'modality_f1': test_metrics.get('test_modality_f1', test_metrics.get('modality_f1', 0)),
                            'location_acc': test_metrics.get('test_location_acc', test_metrics.get('location_accuracy', 0)),
                            'location_f1': test_metrics.get('test_location_f1', test_metrics.get('location_f1', 0)),
                        })
                        record['avg_acc'] = (record['modality_acc'] + record['location_acc']) / 2
                        record['avg_f1'] = (record['modality_f1'] + record['location_f1']) / 2
                    else:
                        record.update({
                            'type_acc': test_metrics.get('test_type_acc', test_metrics.get('type_accuracy', 0)),
                            'type_f1': test_metrics.get('test_type_f1', test_metrics.get('type_f1', 0)),
                            'severity_acc': test_metrics.get('test_severity_acc', test_metrics.get('severity_accuracy', 0)),
                            'severity_f1': test_metrics.get('test_severity_f1', test_metrics.get('severity_f1', 0)),
                        })
                        record['avg_acc'] = (record['type_acc'] + record['severity_acc']) / 2
                        record['avg_f1'] = (record['type_f1'] + record['severity_f1']) / 2
                    
                    record['inference_ms'] = test_metrics.get('test_infer_ms', test_metrics.get('inference_time_ms', np.nan))
                    
                    records.append(record)
        
        return pd.DataFrame(records)
    
    def load_ultra_edge(self, variant: str = 'ultra-edge') -> pd.DataFrame:
        """Load ultra-edge experiments (256-dim or 384-dim)."""
        ultra_dir = self.logs_dir / variant
        records = []
        
        for exp_dir in ultra_dir.iterdir():
            if exp_dir.is_dir():
                results = self.load_results(exp_dir)
                if results:
                    exp_name = exp_dir.name
                    parts = exp_name.split('-')
                    dataset = parts[0]
                    vision = parts[1]
                    text = parts[2]
                    
                    # Extract fusion_dim from config
                    fusion_dim = results['config']['student'].get('fusion_dim', 256)
                    
                    test_metrics = results['metrics']['test']
                    
                    # Get model parameters if available
                    student_params = results.get('models', {}).get('student', {}).get('params_millions', np.nan)
                    
                    record = {
                        'experiment': exp_name,
                        'dataset': dataset,
                        'vision': vision,
                        'text': text,
                        'student': f"{vision}/{text}",
                        'fusion_dim': fusion_dim,
                        'params_millions': student_params,
                    }
                    
                    if dataset == 'medpix':
                        record.update({
                            'modality_acc': test_metrics.get('test_modality_acc', test_metrics.get('modality_accuracy', 0)),
                            'modality_f1': test_metrics.get('test_modality_f1', test_metrics.get('modality_f1', 0)),
                            'location_acc': test_metrics.get('test_location_acc', test_metrics.get('location_accuracy', 0)),
                            'location_f1': test_metrics.get('test_location_f1', test_metrics.get('location_f1', 0)),
                        })
                        record['avg_acc'] = (record['modality_acc'] + record['location_acc']) / 2
                        record['avg_f1'] = (record['modality_f1'] + record['location_f1']) / 2
                    else:
                        record.update({
                            'type_acc': test_metrics.get('test_type_acc', test_metrics.get('type_accuracy', 0)),
                            'type_f1': test_metrics.get('test_type_f1', test_metrics.get('type_f1', 0)),
                            'severity_acc': test_metrics.get('test_severity_acc', test_metrics.get('severity_accuracy', 0)),
                            'severity_f1': test_metrics.get('test_severity_f1', test_metrics.get('severity_f1', 0)),
                        })
                        record['avg_acc'] = (record['type_acc'] + record['severity_acc']) / 2
                        record['avg_f1'] = (record['type_f1'] + record['severity_f1']) / 2
                    
                    record['inference_ms'] = test_metrics.get('test_infer_ms', test_metrics.get('inference_time_ms', np.nan))
                    
                    records.append(record)
        
        return pd.DataFrame(records)
    
    def load_training_curves(self, experiment_path: Path) -> Dict:
        """Extract training history for learning curves."""
        results = self.load_results(experiment_path)
        if results and 'metrics' in results and 'train' in results['metrics']:
            return results['metrics']['train']['history']
        return None


class ResearchPlotter:
    """Generate publication-quality plots for research article."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def plot_fusion_comparison(self, df: pd.DataFrame, dataset: str):
        """Plot fusion strategy comparison for a specific dataset."""
        df_dataset = df[df['dataset'] == dataset].copy()
        
        if len(df_dataset) == 0:
            print(f"No data for {dataset} fusion comparison")
            return
        
        # Sort by average F1
        df_dataset = df_dataset.sort_values('avg_f1', ascending=False)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: F1 scores for both tasks
        ax = axes[0]
        x = np.arange(len(df_dataset))
        width = 0.35
        
        if dataset == 'medpix':
            bars1 = ax.bar(x - width/2, df_dataset['modality_f1'], width, 
                          label='Modality F1', color=FUSION_COLORS[0])
            bars2 = ax.bar(x + width/2, df_dataset['location_f1'], width,
                          label='Location F1', color=FUSION_COLORS[1])
        else:
            bars1 = ax.bar(x - width/2, df_dataset['type_f1'], width,
                          label='Type F1', color=FUSION_COLORS[0])
            bars2 = ax.bar(x + width/2, df_dataset['severity_f1'], width,
                          label='Severity F1', color=FUSION_COLORS[1])
        
        ax.set_xlabel('Fusion Strategy')
        ax.set_ylabel('F1 Score')
        ax.set_title(f'{dataset.capitalize()} Dataset: Fusion Strategy Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(df_dataset['fusion'], rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1.0])
        
        # Plot 2: Accuracy vs Inference Time
        ax = axes[1]
        scatter = ax.scatter(df_dataset['inference_ms'], df_dataset['avg_acc'], 
                           s=100, c=range(len(df_dataset)), cmap='viridis', alpha=0.7)
        
        for idx, row in df_dataset.iterrows():
            ax.annotate(row['fusion'], (row['inference_ms'], row['avg_acc']),
                       fontsize=8, alpha=0.7, xytext=(5, 5), textcoords='offset points')
        
        ax.set_xlabel('Inference Time (ms)')
        ax.set_ylabel('Average Accuracy')
        ax.set_title('Accuracy vs Inference Time Trade-off')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'fusion_comparison_{dataset}.png')
        plt.savefig(self.output_dir / f'fusion_comparison_{dataset}.pdf')
        plt.close()
        
        print(f"✓ Generated fusion comparison plot for {dataset}")
    
    def plot_loss_comparison(self, df: pd.DataFrame, dataset: str):
        """Plot loss function comparison for a specific dataset."""
        df_dataset = df[df['dataset'] == dataset].copy()
        
        if len(df_dataset) == 0:
            print(f"No data for {dataset} loss comparison")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Group by loss type and calculate mean metrics
        loss_groups = df_dataset.groupby('loss').agg({
            'avg_f1': 'mean',
            'avg_acc': 'mean',
        }).reset_index()
        
        loss_groups = loss_groups.sort_values('avg_f1', ascending=False)
        
        x = np.arange(len(loss_groups))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, loss_groups['avg_acc'], width,
                      label='Average Accuracy', color=LOSS_COLORS[0])
        bars2 = ax.bar(x + width/2, loss_groups['avg_f1'], width,
                      label='Average F1', color=LOSS_COLORS[1])
        
        ax.set_xlabel('Loss Function')
        ax.set_ylabel('Score')
        ax.set_title(f'{dataset.capitalize()} Dataset: Loss Function Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(loss_groups['loss'], rotation=0)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1.0])
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'loss_comparison_{dataset}.png')
        plt.savefig(self.output_dir / f'loss_comparison_{dataset}.pdf')
        plt.close()
        
        print(f"✓ Generated loss comparison plot for {dataset}")
    
    def plot_ultra_edge_comparison(self, df_256: pd.DataFrame, df_384: pd.DataFrame = None):
        """Compare ultra-edge configurations (256 vs 384 dim)."""
        
        for dataset in ['medpix', 'wound']:
            df_256_ds = df_256[df_256['dataset'] == dataset].copy()
            
            if len(df_256_ds) == 0:
                continue
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Plot 1: Model size vs accuracy
            ax = axes[0]
            
            # Plot 256-dim configurations
            scatter1 = ax.scatter(df_256_ds['params_millions'], df_256_ds['avg_f1'],
                                s=150, marker='o', label='256-dim', 
                                color=COLORS['primary'], alpha=0.7, edgecolors='black')
            
            for idx, row in df_256_ds.iterrows():
                ax.annotate(row['student'], (row['params_millions'], row['avg_f1']),
                           fontsize=8, alpha=0.7, xytext=(5, 5), textcoords='offset points')
            
            # Add 384-dim if available
            if df_384 is not None:
                df_384_ds = df_384[df_384['dataset'] == dataset].copy()
                if len(df_384_ds) > 0:
                    scatter2 = ax.scatter(df_384_ds['params_millions'], df_384_ds['avg_f1'],
                                        s=150, marker='^', label='384-dim',
                                        color=COLORS['secondary'], alpha=0.7, edgecolors='black')
            
            ax.set_xlabel('Model Parameters (Millions)')
            ax.set_ylabel('Average F1 Score')
            ax.set_title(f'{dataset.capitalize()}: Model Size vs Performance')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Plot 2: Inference time vs accuracy
            ax = axes[1]
            
            scatter1 = ax.scatter(df_256_ds['inference_ms'], df_256_ds['avg_f1'],
                                s=150, marker='o', label='256-dim',
                                color=COLORS['primary'], alpha=0.7, edgecolors='black')
            
            for idx, row in df_256_ds.iterrows():
                ax.annotate(row['student'], (row['inference_ms'], row['avg_f1']),
                           fontsize=8, alpha=0.7, xytext=(5, 5), textcoords='offset points')
            
            if df_384 is not None:
                df_384_ds = df_384[df_384['dataset'] == dataset].copy()
                if len(df_384_ds) > 0:
                    scatter2 = ax.scatter(df_384_ds['inference_ms'], df_384_ds['avg_f1'],
                                        s=150, marker='^', label='384-dim',
                                        color=COLORS['secondary'], alpha=0.7, edgecolors='black')
            
            ax.set_xlabel('Inference Time (ms)')
            ax.set_ylabel('Average F1 Score')
            ax.set_title(f'{dataset.capitalize()}: Latency vs Performance')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / f'ultra_edge_{dataset}.png')
            plt.savefig(self.output_dir / f'ultra_edge_{dataset}.pdf')
            plt.close()
            
            print(f"✓ Generated ultra-edge comparison for {dataset}")
    
    def plot_training_curves(self, history: Dict, experiment_name: str):
        """Plot training curves (loss and accuracy over epochs)."""
        if not history:
            return
        
        epochs = history.get('epoch', [])
        if not epochs:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Loss curves
        ax = axes[0]
        if 'train_loss' in history:
            ax.plot(epochs, history['train_loss'], label='Train Loss', 
                   color=COLORS['primary'], linewidth=2, marker='o')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(f'{experiment_name}: Training Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Accuracy curves
        ax = axes[1]
        
        # Find all accuracy metrics in history
        acc_keys = [k for k in history.keys() if 'acc' in k.lower() and 'dev' in k]
        colors = sns.color_palette("husl", len(acc_keys))
        
        for idx, key in enumerate(acc_keys):
            label = key.replace('dev_', '').replace('_acc', '').capitalize()
            ax.plot(epochs, history[key], label=label, 
                   color=colors[idx], linewidth=2, marker='o')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'{experiment_name}: Validation Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.0])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'training_curves_{experiment_name}.png')
        plt.savefig(self.output_dir / f'training_curves_{experiment_name}.pdf')
        plt.close()
        
        print(f"✓ Generated training curves for {experiment_name}")
    
    def plot_cross_dataset_comparison(self, df_fusion: pd.DataFrame, df_loss: pd.DataFrame):
        """Compare performance across MedPix and Wound datasets."""
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Fusion strategies across datasets
        ax = axes[0, 0]
        
        # Get top 5 fusion strategies by average performance
        fusion_summary = df_fusion.groupby('fusion').agg({
            'avg_f1': 'mean'
        }).reset_index().sort_values('avg_f1', ascending=False).head(5)
        
        top_fusions = fusion_summary['fusion'].tolist()
        df_top = df_fusion[df_fusion['fusion'].isin(top_fusions)]
        
        medpix_data = df_top[df_top['dataset'] == 'medpix'].groupby('fusion')['avg_f1'].mean()
        wound_data = df_top[df_top['dataset'] == 'wound'].groupby('fusion')['avg_f1'].mean()
        
        x = np.arange(len(top_fusions))
        width = 0.35
        
        ax.bar(x - width/2, [medpix_data.get(f, 0) for f in top_fusions], width,
               label='MedPix', color=DATASET_COLORS['medpix'])
        ax.bar(x + width/2, [wound_data.get(f, 0) for f in top_fusions], width,
               label='Wound', color=DATASET_COLORS['wound'])
        
        ax.set_ylabel('Average F1 Score')
        ax.set_title('Top Fusion Strategies: Cross-Dataset Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(top_fusions, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Plot 2: Loss functions across datasets
        ax = axes[0, 1]
        
        loss_medpix = df_loss[df_loss['dataset'] == 'medpix'].groupby('loss')['avg_f1'].mean()
        loss_wound = df_loss[df_loss['dataset'] == 'wound'].groupby('loss')['avg_f1'].mean()
        
        loss_types = sorted(set(loss_medpix.index) | set(loss_wound.index))
        x = np.arange(len(loss_types))
        
        ax.bar(x - width/2, [loss_medpix.get(l, 0) for l in loss_types], width,
               label='MedPix', color=DATASET_COLORS['medpix'])
        ax.bar(x + width/2, [loss_wound.get(l, 0) for l in loss_types], width,
               label='Wound', color=DATASET_COLORS['wound'])
        
        ax.set_ylabel('Average F1 Score')
        ax.set_title('Loss Functions: Cross-Dataset Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(loss_types, rotation=0)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Plot 3: Dataset complexity comparison
        ax = axes[1, 0]
        
        categories = ['Avg Accuracy', 'Avg F1', 'Inference Time (ms)']
        
        medpix_avg_acc = df_fusion[df_fusion['dataset'] == 'medpix']['avg_acc'].mean()
        medpix_avg_f1 = df_fusion[df_fusion['dataset'] == 'medpix']['avg_f1'].mean()
        medpix_avg_time = df_fusion[df_fusion['dataset'] == 'medpix']['inference_ms'].mean()
        
        wound_avg_acc = df_fusion[df_fusion['dataset'] == 'wound']['avg_acc'].mean()
        wound_avg_f1 = df_fusion[df_fusion['dataset'] == 'wound']['avg_f1'].mean()
        wound_avg_time = df_fusion[df_fusion['dataset'] == 'wound']['inference_ms'].mean()
        
        # Normalize inference time to 0-1 scale for comparison
        max_time = max(medpix_avg_time, wound_avg_time)
        
        medpix_values = [medpix_avg_acc, medpix_avg_f1, medpix_avg_time / max_time]
        wound_values = [wound_avg_acc, wound_avg_f1, wound_avg_time / max_time]
        
        x = np.arange(len(categories))
        ax.bar(x - width/2, medpix_values, width, label='MedPix', 
               color=DATASET_COLORS['medpix'])
        ax.bar(x + width/2, wound_values, width, label='Wound',
               color=DATASET_COLORS['wound'])
        
        ax.set_ylabel('Normalized Score')
        ax.set_title('Dataset Characteristics Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Plot 4: Performance consistency
        ax = axes[1, 1]
        
        # Calculate standard deviation as measure of consistency
        fusion_std = df_fusion.groupby('dataset')['avg_f1'].std()
        loss_std = df_loss.groupby('dataset')['avg_f1'].std()
        
        datasets = ['MedPix', 'Wound']
        fusion_stds = [fusion_std.get('medpix', 0), fusion_std.get('wound', 0)]
        loss_stds = [loss_std.get('medpix', 0), loss_std.get('wound', 0)]
        
        x = np.arange(len(datasets))
        ax.bar(x - width/2, fusion_stds, width, label='Fusion Strategies',
               color=FUSION_COLORS[0])
        ax.bar(x + width/2, loss_stds, width, label='Loss Functions',
               color=LOSS_COLORS[0])
        
        ax.set_ylabel('Standard Deviation (F1)')
        ax.set_title('Performance Variance Across Configurations')
        ax.set_xticks(x)
        ax.set_xticklabels(datasets)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'cross_dataset_comparison.png')
        plt.savefig(self.output_dir / 'cross_dataset_comparison.pdf')
        plt.close()
        
        print("✓ Generated cross-dataset comparison plot")
    
    def plot_model_size_analysis(self, csv_path: Path):
        """Plot model size compression analysis."""
        if not csv_path.exists():
            print(f"Model size CSV not found: {csv_path}")
            return
        
        df = pd.read_csv(csv_path)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Compression ratio distribution
        ax = axes[0]
        
        # Extract numeric compression ratio
        df['compression'] = df['compression_ratio'].str.replace('x', '').astype(float)
        
        ax.hist(df['compression'], bins=15, color=COLORS['primary'], 
               alpha=0.7, edgecolor='black')
        ax.axvline(df['compression'].mean(), color='red', linestyle='--',
                  linewidth=2, label=f'Mean: {df["compression"].mean():.2f}x')
        
        ax.set_xlabel('Compression Ratio')
        ax.set_ylabel('Frequency')
        ax.set_title('Model Compression Distribution')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Plot 2: Parameter reduction percentage
        ax = axes[1]
        
        df['reduction'] = df['reduction_pct'].str.replace('%', '').astype(float)
        
        ax.hist(df['reduction'], bins=15, color=COLORS['secondary'],
               alpha=0.7, edgecolor='black')
        ax.axvline(df['reduction'].mean(), color='red', linestyle='--',
                  linewidth=2, label=f'Mean: {df["reduction"].mean():.1f}%')
        
        ax.set_xlabel('Parameter Reduction (%)')
        ax.set_ylabel('Frequency')
        ax.set_title('Parameter Reduction Distribution')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'model_size_analysis.png')
        plt.savefig(self.output_dir / 'model_size_analysis.pdf')
        plt.close()
        
        print("✓ Generated model size analysis plot")


def main():
    parser = argparse.ArgumentParser(
        description='Generate publication-quality plots from experimental results'
    )
    parser.add_argument('--logs-dir', type=str, default='logs',
                       help='Path to logs directory')
    parser.add_argument('--output-dir', type=str, default='figures',
                       help='Output directory for generated plots')
    parser.add_argument('--plots', nargs='+', 
                       choices=['fusion', 'loss', 'ultra-edge', 'training', 
                               'cross-dataset', 'model-size', 'all'],
                       default=['all'],
                       help='Which plots to generate')
    
    args = parser.parse_args()
    
    logs_dir = Path(args.logs_dir)
    output_dir = Path(args.output_dir)
    
    if not logs_dir.exists():
        print(f"Error: Logs directory not found: {logs_dir}")
        return
    
    print("=" * 60)
    print("Research Article Figure Generator")
    print("=" * 60)
    
    loader = ExperimentLoader(logs_dir)
    plotter = ResearchPlotter(output_dir)
    
    plot_all = 'all' in args.plots
    
    # 1. Fusion comparison plots
    if plot_all or 'fusion' in args.plots:
        print("\n[1/6] Generating fusion comparison plots...")
        df_fusion = loader.load_fusion_explore()
        if len(df_fusion) > 0:
            plotter.plot_fusion_comparison(df_fusion, 'medpix')
            plotter.plot_fusion_comparison(df_fusion, 'wound')
        else:
            print("  ⚠ No fusion exploration data found")
    
    # 2. Loss comparison plots
    if plot_all or 'loss' in args.plots:
        print("\n[2/6] Generating loss comparison plots...")
        df_loss = loader.load_loss_explore()
        if len(df_loss) > 0:
            plotter.plot_loss_comparison(df_loss, 'medpix')
            plotter.plot_loss_comparison(df_loss, 'wound')
        else:
            print("  ⚠ No loss exploration data found")
    
    # 3. Ultra-edge comparison plots
    if plot_all or 'ultra-edge' in args.plots:
        print("\n[3/6] Generating ultra-edge comparison plots...")
        df_ultra_256 = loader.load_ultra_edge('ultra-edge')
        df_ultra_384 = loader.load_ultra_edge('ultra-edge2')
        
        if len(df_ultra_256) > 0:
            plotter.plot_ultra_edge_comparison(df_ultra_256, df_ultra_384 if len(df_ultra_384) > 0 else None)
        else:
            print("  ⚠ No ultra-edge data found")
    
    # 4. Training curves
    if plot_all or 'training' in args.plots:
        print("\n[4/6] Generating training curves...")
        # Generate for a few representative experiments
        sample_experiments = [
            logs_dir / "ultra-edge" / "medpix-deit_small-minilm",
            logs_dir / "ultra-edge" / "wound-deit_small-minilm",
        ]
        
        for exp_path in sample_experiments:
            if exp_path.exists():
                history = loader.load_training_curves(exp_path)
                if history:
                    plotter.plot_training_curves(history, exp_path.name)
    
    # 5. Cross-dataset comparison
    if plot_all or 'cross-dataset' in args.plots:
        print("\n[5/6] Generating cross-dataset comparison...")
        df_fusion = loader.load_fusion_explore()
        df_loss = loader.load_loss_explore()
        
        if len(df_fusion) > 0 and len(df_loss) > 0:
            plotter.plot_cross_dataset_comparison(df_fusion, df_loss)
        else:
            print("  ⚠ Insufficient data for cross-dataset comparison")
    
    # 6. Model size analysis
    if plot_all or 'model-size' in args.plots:
        print("\n[6/6] Generating model size analysis...")
        model_size_csv = logs_dir / "model_size_report.csv"
        plotter.plot_model_size_analysis(model_size_csv)
    
    print("\n" + "=" * 60)
    print(f"✓ All plots saved to: {output_dir.absolute()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
