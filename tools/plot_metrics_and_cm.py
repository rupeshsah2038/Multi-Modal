
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

FIGURES_DIR = os.path.join('figures', 'new-plots')
os.makedirs(FIGURES_DIR, exist_ok=True)

# Utility to plot confusion matrix
def plot_confusion_matrix(cm, labels, title, save_path=None):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()

# Utility to plot loss/accuracy curves
def plot_metric_curve(metrics_df, metric, save_path=None, model_name=None):
    plt.figure()
    plt.plot(metrics_df['epoch'], metrics_df[metric], marker='o')
    plt.xlabel('Epoch')
    plt.ylabel(metric.capitalize())
    title = f'{model_name}: {metric.capitalize()} over Epochs' if model_name else f'{metric.capitalize()} over Epochs'
    plt.title(title)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def main(logs_root='logs', pattern='ultra-edge-hp-tuned-*'):
    base_dirs = glob.glob(os.path.join(logs_root, pattern))
    for base_dir in base_dirs:
        base_name = os.path.basename(base_dir.rstrip('/'))
        base_figures_dir = os.path.join(FIGURES_DIR, base_name)
        os.makedirs(base_figures_dir, exist_ok=True)
        model_folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
        for model_folder in model_folders:
            folder_path = os.path.join(base_dir, model_folder)
            print(f'Processing: {folder_path}')
            # Determine dataset type from folder name
            if model_folder.startswith('wound-'):
                cm_types = ['type', 'severity']
                label_types = ['type', 'severity']
            else:
                cm_types = ['modality', 'location']
                label_types = ['modality', 'location']
            # Plot confusion matrices for each task
            for cm_type, label_type in zip(cm_types, label_types):
                for split in ['dev', 'test']:
                    cm_file = os.path.join(folder_path, f'cm_{cm_type}_{split}.npy')
                    labels_file = os.path.join(folder_path, f'labels_{label_type}.npy')
                    if os.path.exists(cm_file) and os.path.exists(labels_file):
                        cm = np.load(cm_file)
                        labels = np.load(labels_file)
                        plot_confusion_matrix(
                            cm,
                            labels,
                            title=f'{model_folder}: {cm_type} {split}',
                            save_path=os.path.join(base_figures_dir, f'{model_folder}_cm_{cm_type}_{split}.png')
                        )
            # Plot metrics (loss, accuracy, etc)
            metrics_file = os.path.join(folder_path, 'metrics.csv')
            metrics_json = os.path.join(folder_path, 'metrics.json')
            metrics_df = None
            if os.path.exists(metrics_file):
                metrics_df = pd.read_csv(metrics_file)
            elif os.path.exists(metrics_json):
                import json
                with open(metrics_json, 'r') as f:
                    metrics_data = json.load(f)
                metrics_df = pd.DataFrame(metrics_data)
            if metrics_df is not None:
                # Plot all metrics except epoch and infer_ms if numeric
                for metric in metrics_df.columns:
                    if metric.lower() in ['epoch', 'infer_ms', 'dev_infer_ms']:
                        continue
                    if np.issubdtype(metrics_df[metric].dtype, np.number):
                        plot_metric_curve(
                            metrics_df,
                            metric,
                            save_path=os.path.join(base_figures_dir, f'{model_folder}_plot_{metric}.png'),
                            model_name=model_folder
                        )
                # Plot epochs vs average of all f1 columns
                f1_cols = [col for col in metrics_df.columns if 'f1' in col.lower() and np.issubdtype(metrics_df[col].dtype, np.number)]
                if f1_cols:
                    avg_f1 = metrics_df[f1_cols].mean(axis=1)
                    plt.figure()
                    plt.plot(metrics_df['epoch'], avg_f1, marker='o')
                    plt.xlabel('Epoch')
                    plt.ylabel('Average F1 Score')
                    plt.title(f'{model_folder}: Average F1 Score over Epochs')
                    plt.savefig(os.path.join(base_figures_dir, f'{model_folder}_plot_avg_f1.png'), bbox_inches='tight')
                    plt.close()

if __name__ == '__main__':
    main()
