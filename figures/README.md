# Research Article Figures

This directory contains publication-quality figures generated from experimental results for the multi-modal knowledge distillation research paper.

## Generated Figures

### 1. Fusion Strategy Comparison
**Files**: `fusion_comparison_medpix.{png,pdf}`, `fusion_comparison_wound.{png,pdf}`

Comparison of different fusion strategies (cross-attention, concat_mlp, transformer_concat, etc.) for both datasets:
- **Left panel**: F1 scores for both classification tasks (modality/location for MedPix, type/severity for Wound)
- **Right panel**: Accuracy vs inference time trade-off scatter plot

**Key Insights**:
- Shows which fusion strategies achieve the best task performance
- Identifies latency-performance trade-offs
- Helps select optimal fusion architecture for deployment

### 2. Loss Function Comparison
**Files**: `loss_comparison_medpix.{png,pdf}`, `loss_comparison_wound.{png,pdf}`

Performance comparison across different knowledge distillation loss functions (combined, vanilla, CRD, RKD, MMD):
- Bar charts showing average accuracy and F1 scores grouped by loss type
- Value labels on bars for precise comparisons

**Key Insights**:
- Identifies most effective distillation loss for each dataset
- Combined loss typically performs best by incorporating task and distillation objectives
- Some losses work better for specific dataset characteristics

### 3. Ultra-Edge Model Comparison
**Files**: `ultra_edge_medpix.{png,pdf}`, `ultra_edge_wound.{png,pdf}`

Analysis of ultra-edge student models with different backbone combinations:
- **Left panel**: Model size (parameters) vs F1 score
- **Right panel**: Inference time vs F1 score
- Compares 256-dim and 384-dim fusion configurations (circles vs triangles)
- Annotated with student model names (deit-small/minilm, deit-tiny/distilbert, etc.)

**Key Insights**:
- Demonstrates model compression effectiveness
- Shows pareto frontier of size/latency vs accuracy
- deit-small/minilm consistently offers best overall trade-off
- 384-dim fusion adds minimal overhead but can improve accuracy

### 4. Training Curves
**Files**: `training_curves_medpix-deit_small-minilm.{png,pdf}`, `training_curves_wound-deit_small-minilm.{png,pdf}`

Learning curves for representative student model configurations:
- **Left panel**: Training loss over epochs
- **Right panel**: Validation accuracy for both tasks over epochs

**Key Insights**:
- Demonstrates stable convergence
- Shows learning dynamics for both classification tasks
- Validates that 10 epochs is sufficient for student training
- No signs of overfitting

### 5. Cross-Dataset Comparison
**Files**: `cross_dataset_comparison.{png,pdf}`

Four-panel comprehensive comparison between MedPix and Wound datasets:
- **Top-left**: Top 5 fusion strategies performance on both datasets
- **Top-right**: Loss function effectiveness across datasets
- **Bottom-left**: Dataset characteristics comparison (accuracy, F1, normalized inference time)
- **Bottom-right**: Performance variance/consistency across configurations

**Key Insights**:
- Some fusion strategies generalize well across datasets (e.g., cross-attention)
- Combined loss consistently performs best on both datasets
- Wound dataset generally achieves higher accuracy (simpler tasks)
- MedPix shows higher variance across configurations (more challenging)

### 6. Model Size Analysis
**Files**: `model_size_analysis.{png,pdf}`

Distribution analysis of model compression:
- **Left panel**: Compression ratio distribution with mean line
- **Right panel**: Parameter reduction percentage with mean line

**Key Insights**:
- Typical compression ratios: 1.9x to 4.3x
- Parameter reduction: 47% to 77%
- Most experiments achieve 2-3x compression
- Ultra-edge configurations achieve highest compression (4.3x for deit-tiny/minilm)

## File Formats

All figures are provided in two formats:
- **PNG** (300 DPI): For presentations, web, and quick viewing
- **PDF** (vector): For publication, printing, and LaTeX documents

## Regenerating Figures

To regenerate all figures from scratch:

```bash
python tools/plot_research_figures.py --output-dir figures/research_article
```

To generate specific figure types:

```bash
# Only fusion comparison plots
python tools/plot_research_figures.py --plots fusion --output-dir figures/research_article

# Multiple specific plot types
python tools/plot_research_figures.py --plots fusion loss ultra-edge --output-dir figures/research_article

# Available plot types: fusion, loss, ultra-edge, training, cross-dataset, model-size, all
```

## Customization

The plotting script (`tools/plot_research_figures.py`) uses publication-quality settings:
- **DPI**: 300 (high resolution)
- **Font**: Serif family for professional appearance
- **Colors**: Carefully selected color palettes (Set2, Dark2, custom)
- **Style**: Clean with grid lines, value labels where helpful
- **Size**: Optimized for two-column academic papers

To customize:
1. Edit `plt.rcParams` in the script for global style changes
2. Modify `COLORS` dictionary for custom color schemes
3. Adjust figure sizes in individual plot functions

## Integration with Paper

### Suggested Figure Placement:

1. **Section 3 (Methodology)**:
   - Model size analysis → illustrate compression achieved
   - Training curves → show training stability

2. **Section 4 (Fusion Strategies)**:
   - Fusion comparison plots → demonstrate fusion strategy effectiveness

3. **Section 5 (Distillation Losses)**:
   - Loss comparison plots → compare distillation approaches

4. **Section 6 (Ultra-Edge Deployment)**:
   - Ultra-edge comparison plots → show deployment configurations

5. **Section 7 (Discussion)**:
   - Cross-dataset comparison → generalization analysis

### LaTeX Integration:

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=\columnwidth]{figures/research_article/fusion_comparison_medpix.pdf}
  \caption{Fusion strategy comparison on MedPix-2-0 dataset showing F1 scores for modality and location classification (left) and accuracy vs inference time trade-off (right).}
  \label{fig:fusion_medpix}
\end{figure}
```

## Dependencies

The plotting script requires:
- matplotlib >= 3.5.0
- seaborn >= 0.11.0
- pandas >= 1.3.0
- numpy >= 1.21.0

Install via:
```bash
pip install matplotlib seaborn pandas numpy
```

## Notes

- All plots use consistent color schemes and styling for cohesive presentation
- Annotations and labels are optimized for readability at publication size
- Grid lines and value labels enhance data interpretation
- Both datasets are analyzed with parallel visualizations for easy comparison
