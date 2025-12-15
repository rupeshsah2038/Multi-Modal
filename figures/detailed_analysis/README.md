# Detailed Analysis Visualizations

This directory contains detailed confusion matrices and training metrics plots for all experimental configurations.

## Overview

Generated using `tools/plot_confusion_matrices.py`, this collection provides comprehensive visualizations of:
- **264 confusion matrix plots** (normalized and comparison views)
- **44 training metrics plots** (loss, accuracy, F1 scores over epochs)

## Directory Structure

```
detailed_analysis/
├── fusion-explore/           # Fusion strategy experiments
│   ├── confusion_matrices/   # CM plots for each experiment
│   └── training_metrics/     # Training curves
├── loss-explore/             # Loss function experiments
│   ├── confusion_matrices/
│   └── training_metrics/
├── ultra-edge/               # Ultra-edge 256-dim experiments
│   ├── confusion_matrices/
│   └── training_metrics/
└── ultra-edge2/              # Ultra-edge 384-dim experiments
    ├── confusion_matrices/
    └── training_metrics/
```

## Confusion Matrix Visualizations

### File Types

For each experiment and task, multiple confusion matrix visualizations are generated:

1. **Normalized Individual Matrices** (`*_{task}_{split}_normalized.{png,pdf}`)
   - Displays proportions (percentages) instead of raw counts
   - Separate plots for dev and test splits
   - Includes accuracy metric in annotation box
   - Example: `medpix-cross_attention-combined_modality_test_normalized.png`

2. **Side-by-Side Comparisons** (`*_{task}_comparison.{png,pdf}`)
   - Dev and test confusion matrices displayed together
   - Enables direct comparison of model performance across splits
   - Both matrices normalized for consistency
   - Example: `wound-shomr-combined_type_comparison.png`

### Tasks by Dataset

**MedPix Dataset:**
- `modality`: CT vs MR classification (2 classes)
- `location`: Body location classification (5 classes)

**Wound Dataset:**
- `type`: Wound type classification (dynamic classes)
- `severity`: Wound severity classification (dynamic classes)

### Reading Confusion Matrices

- **Rows**: True labels (ground truth)
- **Columns**: Predicted labels (model output)
- **Diagonal**: Correct predictions (darker = better)
- **Off-diagonal**: Misclassifications
- **Percentages**: Proportion of samples per true class
- **Annotation Box**: Overall accuracy for that split

### Key Insights from Confusion Matrices

1. **Class Balance**: Check if all rows sum to similar totals
2. **Confusion Patterns**: Identify which classes are confused with each other
3. **Model Confidence**: High diagonal values = good classification
4. **Generalization**: Compare dev vs test matrices for overfitting signs

## Training Metrics Visualizations

### Plot Structure

Each training metrics visualization (`*_metrics.{png,pdf}`) contains 4 panels:

**Top Left: Training Loss**
- Shows convergence behavior
- Decreasing trend indicates learning
- Plateau indicates convergence

**Top Right: Validation Accuracy**
- Both tasks plotted together
- Shows how well the model performs on unseen data during training
- Compare performance between the two classification tasks

**Bottom Left: Validation F1 Scores**
- F1 score for both tasks
- Better metric for imbalanced datasets
- Combines precision and recall

**Bottom Right: Inference Time**
- Inference latency measured in milliseconds
- Mean inference time shown with dashed red line
- Helps identify computational efficiency

### Dataset-Specific Metrics

**MedPix:**
- Modality accuracy/F1 (cyan/pink lines)
- Location accuracy/F1 (orange/beige lines)

**Wound:**
- Type accuracy/F1 (teal/light blue lines)
- Severity accuracy/F1 (red/coral lines)

### Interpreting Training Curves

**Good Signs:**
- ✓ Smooth, monotonically decreasing loss
- ✓ Increasing or stable accuracy/F1
- ✓ Small gap between tasks (balanced learning)
- ✓ Consistent inference time

**Warning Signs:**
- ⚠ Oscillating loss (may need lower learning rate)
- ⚠ Diverging accuracy curves (one task dominating)
- ⚠ Increasing inference time (potential memory issues)
- ⚠ Plateauing too early (may need more epochs or higher LR)

## Experiment Groups

### 1. Fusion-Explore (18 experiments)
Compares different multimodal fusion strategies:
- `cross_attention`: Cross-attention between modalities
- `concat_mlp`: Simple concatenation + MLP
- `transformer_concat`: Transformer-based fusion
- `shomr`: Stochastic higher-order moment regularization
- `film`: Feature-wise linear modulation
- `gated`: Gated fusion mechanism
- `energy_aware_adaptive`: Energy-aware adaptive fusion
- `modality_dropout`: Dropout-based fusion
- `simple`: Simple fusion baseline

### 2. Loss-Explore (10 experiments)
Compares knowledge distillation loss functions:
- `combined`: Task loss + KL divergence + feature matching
- `vanilla`: Standard KD with soft targets
- `crd`: Contrastive Representation Distillation
- `rkd`: Relational Knowledge Distillation
- `mmd`: Maximum Mean Discrepancy

### 3. Ultra-Edge (8 experiments)
Student models with 256-dim fusion:
- Vision backbones: `deit-small`, `deit-tiny`
- Text backbones: `distilbert`, `minilm`
- 4 configurations per dataset

### 4. Ultra-Edge2 (8 experiments)
Student models with 384-dim fusion:
- Same backbone combinations as ultra-edge
- Larger fusion dimension for potential accuracy gains
- Trade-off: slightly more parameters

## Usage Examples

### View Specific Experiment

```bash
# View confusion matrices for a specific experiment
xdg-open figures/detailed_analysis/fusion-explore/confusion_matrices/medpix-cross_attention-combined_modality_comparison.png

# View training metrics
xdg-open figures/detailed_analysis/fusion-explore/training_metrics/medpix-cross_attention-combined_metrics.png
```

### Regenerate All Visualizations

```bash
# All groups
python tools/plot_confusion_matrices.py

# Specific groups
python tools/plot_confusion_matrices.py --groups fusion-explore loss-explore

# Custom output directory
python tools/plot_confusion_matrices.py --output-dir figures/custom_analysis
```

### Find Specific Patterns

```bash
# Find all normalized test confusion matrices
find figures/detailed_analysis -name "*_test_normalized.png"

# Find all training metrics plots
find figures/detailed_analysis -name "*_metrics.png"

# Find confusion matrices for a specific task
find figures/detailed_analysis -name "*_modality_*.png"
```

## Integration with Research

### For Papers

**Confusion Matrices:**
- Use comparison plots to show model performance across splits
- Highlight specific misclassification patterns
- Demonstrate generalization (dev vs test agreement)

**Training Curves:**
- Show convergence behavior and training stability
- Compare learning dynamics across configurations
- Illustrate multi-task learning balance

### LaTeX Example

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=0.8\columnwidth]{figures/detailed_analysis/fusion-explore/confusion_matrices/medpix-cross_attention-combined_modality_comparison.pdf}
  \caption{Confusion matrices for modality classification using cross-attention fusion on MedPix dataset, comparing validation (left) and test (right) performance.}
  \label{fig:cm_modality}
\end{figure}

\begin{figure}[t]
  \centering
  \includegraphics[width=\columnwidth]{figures/detailed_analysis/ultra-edge/training_metrics/medpix-deit_small-minilm_metrics.pdf}
  \caption{Training dynamics for ultra-edge student model showing loss convergence, accuracy progression, F1 scores, and inference time stability over 10 epochs.}
  \label{fig:training_curves}
\end{figure}
```

## File Formats

All visualizations are provided in:
- **PNG** (300 DPI): For presentations and quick viewing
- **PDF** (vector): For publication and LaTeX documents

## Statistics

- **Total PNG files**: 308
- **Total PDF files**: 308
- **Experiments visualized**: 44
- **Total disk space**: ~150 MB

## Customization

To modify visualization styles, edit `tools/plot_confusion_matrices.py`:

- **Color schemes**: Modify `CMAP_CM` and `COLORS` dictionaries
- **Figure sizes**: Adjust `figsize` parameters in plot functions
- **Annotations**: Change font sizes and positioning in heatmap calls
- **Metrics**: Add or remove metrics from training plots

## Notes

- All confusion matrices are normalized by true class (rows sum to 100%)
- Training metrics use consistent color coding across experiments
- Inference time plots show mean with dashed red line for reference
- Comparison plots use the same scale for easy side-by-side evaluation
- File naming convention: `{experiment_name}_{task}_{visualization_type}.{ext}`
