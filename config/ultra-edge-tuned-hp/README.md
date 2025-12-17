# Ultra-Edge Configurations with Tuned Hyperparameters

This directory contains ultra-edge student configurations optimized with **Optuna hyperparameter tuning**.

## Overview

These configurations use the **best hyperparameters** discovered through automated optimization (Study: `medpix_tuning_20251216_203104`), which achieved a **92.94% validation F1 score** — a **6-9% improvement** over baseline.

## Tuned Hyperparameters

All configurations in this directory use the following optimized hyperparameters:

### Teacher Configuration
- **Learning Rate:** `4.786827188881545e-05` (↑ 4.8× from baseline)
- **Fusion Dim:** `384` (changed from 256/512)
- **Fusion Layers:** `2` (reduced from 3)
- **Fusion Heads:** `8` (unchanged)
- **Dropout:** `0.185` (↑ 1.85× from baseline)

### Student Configuration
- **Learning Rate:** `0.00010451580210282471` (↓ 2.9× from baseline)
- **Fusion Dim:** `384` (changed from 256/512)
- **Fusion Layers:** `2` (increased from 1)
- **Fusion Heads:** `4` (reduced from 8)
- **Dropout:** `0.238` (↑ 2.4× from baseline)

### Loss Weights
- **Alpha (CE weight):** `0.518` (reduced from 1.0)
- **Beta (KD weight):** `112.4` (increased from 100.0)
- **Temperature (T):** `3.19` (increased from 2.0)

## Available Configurations

### MedPix Dataset
- `medpix-mobilevit_small-distilbert.yaml` — deit-small + distilbert
- `medpix-mobilevit_small-minilm.yaml` — deit-small + minilm
- `medpix-mobilevit_xxs-distilbert.yaml` — deit-tiny + distilbert
- `medpix-mobilevit_xxs-minilm.yaml` — deit-tiny + minilm

### Wound Dataset
- `wound-mobilevit_small-distilbert.yaml` — deit-small + distilbert
- `wound-mobilevit_small-minilm.yaml` — deit-small + minilm
- `wound-mobilevit_xxs-distilbert.yaml` — deit-tiny + distilbert
- `wound-mobilevit_xxs-minilm.yaml` — deit-tiny + minilm

## Usage

Train with tuned hyperparameters:

```bash
# MedPix - deit-small + distilbert (recommended)
python experiments/run.py config/ultra-edge-tuned-hp/medpix-mobilevit_small-distilbert.yaml

# MedPix - deit-small + minilm (best efficiency)
python experiments/run.py config/ultra-edge-tuned-hp/medpix-mobilevit_small-minilm.yaml

# Wound - deit-small + distilbert
python experiments/run.py config/ultra-edge-tuned-hp/wound-mobilevit_small-distilbert.yaml

# Wound - deit-tiny + minilm (smallest model)
python experiments/run.py config/ultra-edge-tuned-hp/wound-mobilevit_xxs-minilm.yaml
```

## Key Improvements

Compared to baseline `config/ultra-edge/`:

1. **384-dim fusion** for both teacher and student (sweet spot between 256 and 512)
2. **Higher teacher LR** enables faster convergence for larger model
3. **Lower student LR** provides more stable distillation
4. **Increased dropout** (0.185-0.238) improves generalization
5. **Student gets 2 fusion layers** instead of 1 for better capacity
6. **Balanced loss weights** with reduced alpha, moderate beta, higher temperature

## Expected Performance

Based on validation results from the tuning study:

- **Validation F1:** 92.94% (average of modality + location F1)
- **Improvement:** +6-9% over baseline configurations
- **Training time:** Similar to baseline (~10-15 min per epoch on V100)

## Differences from Baseline

| Parameter | Baseline (ultra-edge) | Tuned (ultra-edge-tuned-hp) | Change |
|-----------|----------------------|----------------------------|--------|
| Teacher LR | 1e-5 | 4.79e-5 | ↑ 4.8× |
| Student LR | 3e-4 | 1.05e-4 | ↓ 2.9× |
| Alpha | 1.0 | 0.518 | ↓ 48% |
| Beta | 100.0 | 112.4 | ↑ 12% |
| T | 2.0 | 3.19 | ↑ 60% |
| Teacher fusion_dim | 256 | 384 | Changed |
| Student fusion_dim | 256 | 384 | Changed |
| Teacher fusion_layers | 2 | 2 | Same |
| Student fusion_layers | 1 | 2 | ↑ 1 layer |
| Teacher fusion_heads | 8 | 8 | Same |
| Student fusion_heads | 8 | 4 | ↓ 50% |
| Teacher dropout | 0.1 | 0.185 | ↑ 85% |
| Student dropout | 0.1 | 0.238 | ↑ 138% |

## Notes

- These hyperparameters were optimized on **MedPix-2-0** but should transfer well to Wound-1-0
- For dataset-specific fine-tuning, run Optuna again with the target dataset:
  ```bash
  python tools/run_optuna_tuning.py --config config/wound.yaml --n-trials 30
  ```
- Results may vary slightly due to random initialization and data ordering
- For reproducible results, set a fixed random seed in the training script

## Reference

For full details on the hyperparameter tuning process, methodology, and results, see:
- **Tuning Summary:** `docs/HYPERPARAMETER_TUNING_SUMMARY.md`
- **Tuning Guide:** `docs/OPTUNA_TUNING.md`
- **Study Results:** `logs/optuna/medpix_tuning_20251216_203104/study_summary.json`
