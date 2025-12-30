#!/usr/bin/env python3
"""
Plot confusion matrices and training metrics from logs/ultra-edge-hp-tuned-all.

Usage:
  python scripts/plot_results_from_logs.py

This script will:
 - iterate over directories in `logs/ultra-edge-hp-tuned-all`
 - for selected models, load `metrics.csv` and `cm_*_test.npy` + `labels_*.npy`
 - save confusion matrix heatmaps and epoch plots (loss + per-task accuracies)

"""
import os
import argparse
import numpy as np
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


DEFAULT_LOGS_DIR = Path("logs/ultra-edge-hp-tuned-all")
OUT_DIR = Path("plots")

# Models to include (substrings matched against folder name after dataset-)
MODELS = [
    "mobilevit_xx_small-bert-tiny",
    "mobilevit_xx_small-bert-mini",
    "deit_tiny-bert-mini",
    "deit_small-bert-tiny",
    "deit_tiny-minilm",
    "deit_small-bert-mini",
    "deit_small-minilm",
    "mobilevit_small-distilbert",
    "deit_tiny-distilbert",
    "deit_small-distilbert",
]


def find_model_match(model_name: str, allowed_models):
    for m in allowed_models:
        if m == model_name:
            return True
    # fallback: check substring
    for m in allowed_models:
        if m in model_name:
            return True
    return False


# Appearance settings — increase font sizes for readability
FONT_SIZE = 15
TITLE_SIZE = 15
TICK_SIZE = 15
LEGEND_SIZE = 15
# Improve seaborn/matplotlib defaults for clearer plots
sns.set_theme(context='notebook', style='white', font_scale=1.2)
plt.rcParams.update({
    'font.size': FONT_SIZE,
    'axes.titlesize': TITLE_SIZE,
    'axes.labelsize': FONT_SIZE,
    'xtick.labelsize': TICK_SIZE,
    'ytick.labelsize': TICK_SIZE,
    'legend.fontsize': LEGEND_SIZE,
    'figure.titlesize': TITLE_SIZE,
})


def plot_confusion(cm: np.ndarray, labels: list, out_path: Path, title: str):
    figsize = (max(5, len(labels) * 0.6), max(5, len(labels) * 0.6))
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        annot_kws={"size": max(10, FONT_SIZE - 2)},
        cbar_kws={"shrink": 0.6},
    )
    plt.xlabel('Predicted', fontsize=FONT_SIZE)
    plt.ylabel('True', fontsize=FONT_SIZE)
    plt.title(title, fontsize=TITLE_SIZE)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_metrics(metrics_csv: Path, tasks: list, out_dir: Path):
    # Read CSV with standard csv module to avoid external deps
    with metrics_csv.open() as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    if not rows:
        return

    # Collect columns
    cols = rows[0].keys()
    if 'epoch' in cols:
        epochs = [int(r['epoch']) for r in rows]
    else:
        epochs = list(range(1, len(rows) + 1))

    # Loss series (train_loss if available)
    losses = None
    if 'train_loss' in cols:
        losses = [float(r['train_loss']) if r['train_loss'] != '' else float('nan') for r in rows]

    # Collect all available dev_*_acc columns and plot them together with loss
    acc_series = {}
    for task in tasks:
        col = f'dev_{task}_acc'
        if col in cols:
            acc_series[task] = [float(r[col]) if r[col] != '' else float('nan') for r in rows]

    if acc_series:
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.set_xlabel('Epoch')

        if losses is not None:
            color_loss = 'tab:blue'
            ax1.set_ylabel('Train Loss', color=color_loss, fontsize=FONT_SIZE)
            ax1.plot(epochs, losses, marker='o', color=color_loss, label='Train Loss', markersize=6)
            ax1.tick_params(axis='y', labelcolor=color_loss)
        else:
            ax1.set_ylabel('Value', fontsize=FONT_SIZE)

        ax2 = ax1.twinx()
        ax2.set_ylabel('Dev Accuracy', fontsize=FONT_SIZE)
        colors = ['tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
        lines = []
        labels = []
        for i, (task, vals) in enumerate(acc_series.items()):
            c = colors[i % len(colors)]
            ln, = ax2.plot(epochs, vals, marker='o', color=c, label=f'dev_{task}_acc', markersize=6)
            lines.append(ln)
            labels.append(f'dev_{task}_acc')

        ax2.set_ylim(0, 1)
        ax2.tick_params(axis='y', labelsize=TICK_SIZE)
        ax1.tick_params(axis='x', labelsize=TICK_SIZE)

        # Combine legends from loss and accuracy lines
        l1, lab1 = ax1.get_legend_handles_labels()
        all_lines = l1 + lines
        all_labels = lab1 + labels
        if all_lines:
            ax1.legend(all_lines, all_labels, loc='best', fontsize=LEGEND_SIZE)

        plt.title('Epochs vs Loss and Dev Accuracies', fontsize=TITLE_SIZE)
        fig.tight_layout()
        plt.grid(True, alpha=0.2)
        plt.savefig(out_dir / f'loss_and_accuracies.png', dpi=300)
        plt.close()


def process_folder(folder: Path, out_root: Path, allowed_models):
    name = folder.name
    if '-' not in name:
        return
    dataset, model_name = name.split('-', 1)
    if not find_model_match(model_name, allowed_models):
        return

    out_dir = out_root / model_name / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine tasks and file names
    if dataset.lower() == 'medpix':
        tasks = ['modality', 'location']
    elif dataset.lower() == 'wound':
        tasks = ['type', 'severity']
    else:
        tasks = []

    # Plot confusion matrices using cm_*_test.npy if present
    for task in tasks:
        cm_file = folder / f'cm_{task}_test.npy'
        labels_file = folder / f'labels_{task}.npy'
        if cm_file.exists() and labels_file.exists():
            cm = np.load(cm_file)
            labels = np.load(labels_file, allow_pickle=True)
            # ensure labels is list of strings
            labels = [str(x) for x in labels.tolist()]
            # For MedPix location labels, replace long names with short abbreviations
            if dataset.lower() == 'medpix' and task == 'location':
                short_map = {
                    'Reproductive and Urinary System': 'RUS',
                    'Spine and Muscles': 'SaM',
                }
                labels = [short_map.get(l, l) for l in labels]
            plot_confusion(cm, labels, out_dir / f'confusion_{task}_test.png', f'{model_name} - {dataset} - {task}')

    # Plot metrics
    metrics_csv = folder / 'metrics.csv'
    if metrics_csv.exists():
        plot_metrics(metrics_csv, tasks, out_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logs', type=str, default=str(DEFAULT_LOGS_DIR), help='Base logs directory')
    parser.add_argument('--out', type=str, default=str(OUT_DIR), help='Output plots directory')
    parser.add_argument('--models', type=str, nargs='*', help='Optional list of model substrings to include')
    args = parser.parse_args()

    base = Path(args.logs)
    out_root = Path(args.out)
    if args.models and len(args.models) > 0:
        allowed = args.models
    else:
        allowed = MODELS

    if not base.exists():
        print(f'Logs directory not found: {base}')
        return

    for child in sorted(base.iterdir()):
        if child.is_dir():
            process_folder(child, out_root, allowed)

    print('Plots written to', out_root)


if __name__ == '__main__':
    main()
