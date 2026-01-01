# Configurable Fusion and Loss Modules Guide

## Overview

Both **fusion modules** and **loss functions** are now fully configurable through YAML config files. No hardcoding required!

## Configuration

### Setting Fusion Type

In your config file (e.g., `config/default.yaml`):

```yaml
fusion:
  type: "cross_attention"  # Options: see list below
```

### Setting Loss Type

```yaml
loss:
  type: "combined"  # Options: vanilla, combined, crd, rkd, mmd
```

## Available Fusion Modules

This guide assumes `simple` / `SimpleFusion` is not available.

Configure any of these 8 fusion strategies:

| Fusion Type | Config Value | Description |
|------------|--------------|-------------|
| **ConcatMLPFusion** | `concat_mlp` | Concatenate modalities + MLP |
| **CrossAttentionFusion** | `cross_attention` | Explicit cross-attention mechanism |
| **GatedFusion** | `gated` | Gated fusion with learned gates |
| **TransformerConcatFusion** | `transformer_concat` | Transformer encoder for fusion |
| **ModalityDropoutFusion** | `modality_dropout` | Stochastic modality dropout |
| **FiLMFusion** | `film` | Feature-wise Linear Modulation |
| **EnergyAwareAdaptiveFusion** | `energy_aware_adaptive` | Energy-aware adaptive fusion |
| **SHoMRFusion** | `shomr` | Soft-Hard Modality Routing fusion |

### Example Configs

**Cross-Attention Fusion (Default)**:
```yaml
fusion:
  type: "cross_attention"

teacher:
  vision: "vit-base"
  text: "bio-clinical-bert"
  fusion_dim: 512
  fusion_layers: 2

student:
  vision: "deit-small"
  text: "distilbert"
  fusion_dim: 512
  fusion_layers: 1
```

**Cross-Attention Fusion**:
```yaml
fusion:
  type: "cross_attention"

teacher:
  fusion_dim: 256  # Dimension for cross-attention
  fusion_layers: 2
```

**Energy-Aware Adaptive Fusion**:
```yaml
fusion:
  type: "energy_aware_adaptive"

teacher:
  fusion_dim: 384
  fusion_layers: 3
```

## Available Loss Functions

Configure any of these 5 distillation loss functions:

| Loss Type | Config Value | Description |
|-----------|--------------|-------------|
| **DistillationLoss** | `vanilla` | KL divergence + feature matching (default) |
| **MedKDCombinedLoss** | `combined` | Task loss + KL + MSE + CRD |
| **CRDLoss** | `crd` | Contrastive Representation Distillation |
| **RKDLoss** | `rkd` | Relational Knowledge Distillation |
| **MMDLoss** | `mmd` | Maximum Mean Discrepancy |

### Loss Hyperparameters

```yaml
loss:
  type: "combined"

training:
  alpha: 1.0    # KL divergence weight
  beta: 100.0   # Feature matching weight
  gamma: 10.0   # CRD weight (for combined loss)
  T: 4.0        # Temperature for soft targets
```

## Running Experiments

### Example 1: Test Different Fusion Strategies

```bash
# Create config with cross-attention fusion
cat > config/test_cross_attention.yaml << EOF
data:
  type: "medpix"
  root: "datasets/MedPix-2-0"
  batch_size: 16

teacher:
  vision: "vit-base"
  text: "bio-clinical-bert"
  fusion_dim: 256
  fusion_layers: 2

student:
  vision: "deit-small"
  text: "distilbert"
  fusion_dim: 256
  fusion_layers: 1

fusion:
  type: "cross_attention"

loss:
  type: "combined"

training:
  teacher_epochs: 3
  student_epochs: 10
  teacher_lr: 1e-5
  student_lr: 3e-4

logging:
  log_dir: "logs/test_cross_attention"
EOF

# Run experiment
python experiments/run.py config/test_cross_attention.yaml
```

### Example 2: Batch Fusion Comparison

```bash
# Test multiple fusion strategies
for fusion in simple concat_mlp cross_attention gated film; do
  python experiments/run.py config/default.yaml \
    --fusion-type $fusion \
    --log-dir logs/fusion_compare/$fusion
done
```

### Example 3: Loss Function Comparison

```bash
# Test all loss functions
for loss in vanilla combined crd rkd mmd; do
  python experiments/run.py config/default.yaml \
    --loss-type $loss \
    --log-dir logs/loss_compare/$loss
done
```

## Verification

### Check Current Configuration

```python
import yaml

with open('config/default.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

print(f"Fusion type: {cfg['fusion']['type']}")
print(f"Loss type: {cfg['loss']['type']}")
```

### Verify Model Instantiation

```python
from models.teacher import Teacher
from models.student import Student

# Create models with specific fusion
teacher = Teacher(
    vision='vit-base',
    text='bio-clinical-bert',
    fusion_type='cross_attention',  # ← Configurable!
    fusion_dim=256,
    fusion_layers=2
)

student = Student(
    vision='deit-small',
    text='distilbert',
    fusion_type='cross_attention',  # ← Configurable!
    fusion_dim=256,
    fusion_layers=1
)

print(f"Teacher fusion: {teacher.fusion.__class__.__name__}")
print(f"Student fusion: {student.fusion.__class__.__name__}")
```

## Migration from Old Code

### Before (Hardcoded)

```python
# OLD: Hardcoded to SimpleFusion
from .fusion.simple import SimpleFusion

class Teacher(nn.Module):
    def __init__(self, ...):
        self.fusion = SimpleFusion(fusion_dim, fusion_heads, fusion_layers)
```

### After (Configurable)

```python
# NEW: Configurable from config file
from .fusion import (SimpleFusion, CrossAttentionFusion, ...)

class Teacher(nn.Module):
    def __init__(self, fusion_type='simple', ...):
        self.fusion = self._create_fusion(fusion_type, ...)
    
    def _create_fusion(self, fusion_type, ...):
        # Factory method selects appropriate fusion module
        ...
```

## Troubleshooting

### Issue: Fusion type not recognized

**Error**: Getting SimpleFusion even though config specifies different type

**Solution**: Make sure you're using the latest code with configurable fusion:
```bash
git pull origin main
# Check that models/teacher.py has _create_fusion method
grep "_create_fusion" models/teacher.py
```

### Issue: Fusion module constructor error

**Error**: `__init__() got unexpected keyword argument`

**Solution**: The fusion factory methods handle different constructor signatures automatically. Check that you're passing the fusion type correctly in the config.

### Issue: Config not being loaded

**Error**: Fusion type defaults to 'simple'

**Solution**: Add fusion section to your config file:
```yaml
fusion:
  type: "cross_attention"  # Your desired fusion type
```

## Testing

Run the test suite to verify all fusion modules work:

```bash
python tests/test_configurable_fusion.py
```

Expected output:
```
✓ simple: Teacher=SimpleFusion, Student=SimpleFusion
✓ concat_mlp: Teacher=ConcatMLPFusion, Student=ConcatMLPFusion
✓ cross_attention: Teacher=CrossAttentionFusion, Student=CrossAttentionFusion
...
✓✓✓ All tests passed! ✓✓✓
```

## Notes

- **Backward Compatibility**: If `fusion.type` is not specified in config, defaults to `"simple"`
- **Case Insensitive**: Fusion types are normalized (underscores and hyphens removed, lowercase)
- **Both Models Share Fusion**: Teacher and student use the same fusion type from config
- **Loss Modules**: Already were configurable, now fusion matches this pattern
- **Config Validation**: Engine validates fusion type and falls back to SimpleFusion if invalid

## References

- **Fusion Modules**: `models/fusion/`
- **Loss Modules**: `losses/`
- **Model Definitions**: `models/teacher.py`, `models/student.py`
- **Engine**: `trainer/engine.py`
- **Config Examples**: `config/`
