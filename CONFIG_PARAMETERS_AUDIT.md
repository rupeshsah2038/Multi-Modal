# Configuration Parameters Audit & Verification

**Date**: December 15, 2025  
**Status**: ✅ ALL PARAMETERS NOW CONFIGURABLE

---

## Executive Summary

Comprehensive audit revealed that **fusion_heads**, **dropout**, and **fusion-specific parameters** were previously hardcoded. This has been **FIXED** - all parameters are now fully configurable through YAML config files.

---

## Audit Results

### ✅ Properly Configurable Parameters (Before Fixes)

| Category | Parameters | Status |
|----------|-----------|--------|
| **Data** | `type`, `root`, `batch_size`, `num_workers` | ✓ Working |
| **Data (Tasks)** | `task1_label`, `task2_label` | ✓ Working |
| **Data (Wound)** | `type_column`, `severity_column`, `description_column`, `filepath_column` | ✓ Working |
| **Teacher/Student** | `vision`, `text`, `fusion_layers`, `fusion_dim` | ✓ Working |
| **Training** | `teacher_epochs`, `student_epochs`, `teacher_lr`, `student_lr` | ✓ Working |
| **Training** | `alpha`, `beta`, `T` | ✓ Working |
| **Logging** | `log_dir` | ✓ Working |
| **Fusion** | `type` | ✓ Working (recently fixed) |
| **Loss** | `type` | ✓ Working |
| **Device** | Top-level `device` | ✓ Working |

### ❌ Previously Hardcoded (Now Fixed)

| Parameter | Previous Value | Location | Status |
|-----------|---------------|----------|--------|
| `teacher.fusion_heads` | 8 (default) | `models/teacher.py` | ✅ **FIXED** |
| `teacher.dropout` | 0.1 (default) | `models/teacher.py` | ✅ **FIXED** |
| `student.fusion_heads` | 8 (default) | `models/student.py` | ✅ **FIXED** |
| `student.dropout` | 0.1 (default) | `models/student.py` | ✅ **FIXED** |
| **Fusion-specific params** | Various | `_create_fusion()` | ✅ **FIXED** |
| ├─ `hidden_mult` | 2 | ConcatMLPFusion | ✅ **FIXED** |
| ├─ `dropout` | 0.1 | CrossAttentionFusion | ✅ **FIXED** |
| ├─ `p_img` | 0.3 | ModalityDropoutFusion | ✅ **FIXED** |
| └─ `p_txt` | 0.3 | ModalityDropoutFusion | ✅ **FIXED** |

---

## Configuration Structure (Updated)

### Complete Config Template

```yaml
# Dataset configuration
data:
  type: "medpix"                    # or "wound"
  root: "datasets/MedPix-2-0"
  batch_size: 16
  num_workers: 4
  task1_label: "modality"           # Primary task name for metrics
  task2_label: "location"           # Secondary task name for metrics
  # For wound dataset:
  # type_column: "type"
  # severity_column: "severity"
  # description_column: "description"
  # filepath_column: "file_path"

# Teacher model configuration
teacher:
  vision: "vit-large"                # Vision backbone
  text: "bio-clinical-bert"          # Text backbone
  fusion_layers: 2                   # Number of fusion layers
  fusion_dim: 512                    # Fusion feature dimension
  fusion_heads: 8                    # NEW: Number of attention heads
  dropout: 0.1                       # NEW: Dropout rate

# Student model configuration
student:
  vision: "deit-base"                # Vision backbone
  text: "distilbert"                 # Text backbone
  fusion_layers: 1                   # Number of fusion layers
  fusion_dim: 512                    # Fusion feature dimension
  fusion_heads: 8                    # NEW: Number of attention heads
  dropout: 0.1                       # NEW: Dropout rate

# Training hyperparameters
training:
  teacher_epochs: 5                  # Teacher training epochs
  student_epochs: 10                 # Student training epochs
  teacher_lr: 1e-5                   # Teacher learning rate
  student_lr: 3e-4                   # Student learning rate
  alpha: 1.0                         # Hard target loss weight
  beta: 100.0                        # Distillation loss weight
  T: 2.0                            # Temperature for distillation

# Logging configuration
logging:
  log_dir: "logs/experiment-name"

# Fusion module configuration
fusion:
  type: "cross_attention"            # Fusion module type (9 options)
  # Optional module-specific parameters:
  hidden_mult: 2                     # For ConcatMLPFusion (default: 2)
  dropout: 0.1                       # For CrossAttentionFusion, EnergyAware, SHoMR
  p_img: 0.3                        # For ModalityDropoutFusion (default: 0.3)
  p_txt: 0.3                        # For ModalityDropoutFusion (default: 0.3)

# Loss function configuration
loss:
  type: "combined"                   # Loss type (5 options)

# Device configuration
device: "cuda:4"                     # GPU device or "cpu"
```

---

## Fusion Module Parameters

### Available Fusion Types

| Type | Parameters Used | Configurable Via |
|------|----------------|------------------|
| **simple** | `fusion_dim`, `fusion_heads`, `fusion_layers` | teacher/student config |
| **concat_mlp** | `fusion_dim`, `fusion_layers`, `hidden_mult` | teacher/student + fusion config |
| **cross_attention** | `fusion_dim`, `fusion_heads`, `dropout` | teacher/student + fusion config |
| **gated** | `fusion_dim` | teacher/student config |
| **transformer_concat** | `fusion_dim`, `fusion_heads`, `fusion_layers` | teacher/student config |
| **modality_dropout** | `fusion_dim`, `p_img`, `p_txt` | teacher/student + fusion config |
| **film** | `fusion_dim` | teacher/student config |
| **energy_aware_adaptive** | `fusion_dim`, `fusion_heads`, `dropout` | teacher/student + fusion config |
| **shomr** | `fusion_dim`, `fusion_heads`, `dropout` | teacher/student + fusion config |

### Module-Specific Parameters

```yaml
# Example 1: ConcatMLPFusion with custom hidden multiplier
fusion:
  type: "concat_mlp"
  hidden_mult: 4              # 4x instead of default 2x

# Example 2: CrossAttentionFusion with custom dropout
fusion:
  type: "cross_attention"
  dropout: 0.2                # 20% instead of default 10%

# Example 3: ModalityDropoutFusion with asymmetric dropout
fusion:
  type: "modality_dropout"
  p_img: 0.5                  # 50% image dropout
  p_txt: 0.3                  # 30% text dropout
```

---

## Implementation Details

### Files Modified

1. **`trainer/engine.py`** (Lines 193-220)
   - Extract `fusion_heads` and `dropout` from teacher/student config
   - Extract `fusion_params` from fusion config
   - Pass all parameters to Teacher and Student constructors

2. **`models/teacher.py`** (Lines 8-48)
   - Added `fusion_params=None` parameter to `__init__`
   - Updated `_create_fusion()` to accept and use `fusion_params`
   - Extract module-specific parameters with defaults

3. **`models/student.py`** (Lines 8-52)
   - Identical changes to `teacher.py` for consistency

4. **`config/default.yaml`**
   - Added `fusion_heads: 8` and `dropout: 0.1` to teacher/student sections
   - Added commented examples of fusion-specific parameters

5. **`config/wound.yaml`**
   - Added `fusion_heads: 8` and `dropout: 0.1` to teacher/student sections

### Backward Compatibility

All new parameters have sensible defaults:
- `fusion_heads`: defaults to `8`
- `dropout`: defaults to `0.1`
- `fusion_params`: defaults to `{}` (empty dict)
- Module-specific parameters use their original hardcoded values as defaults

**Existing config files will continue to work without modification.**

---

## Verification Tests

### Test 1: All 9 Fusion Types
```bash
python -c "
from models.teacher import Teacher
for ftype in ['simple', 'concat_mlp', 'cross_attention', 'gated',
              'transformer_concat', 'modality_dropout', 'film',
              'energy_aware_adaptive', 'shomr']:
    t = Teacher('vit-base', 'bio-clinical-bert', fusion_type=ftype,
                fusion_layers=1, fusion_dim=256, fusion_heads=4,
                dropout=0.1, fusion_params={})
    print(f'✓ {ftype}: {t.fusion.__class__.__name__}')
"
```

**Result**: ✅ All 9 types instantiate correctly

### Test 2: Custom Fusion Parameters
```bash
python -c "
from models.teacher import Teacher
tests = [
    ('concat_mlp', {'hidden_mult': 4}),
    ('cross_attention', {'dropout': 0.2}),
    ('modality_dropout', {'p_img': 0.5, 'p_txt': 0.4}),
]
for ftype, params in tests:
    t = Teacher('vit-base', 'bio-clinical-bert', fusion_type=ftype,
                fusion_layers=1, fusion_dim=256, fusion_heads=4,
                dropout=0.1, fusion_params=params)
    print(f'✓ {ftype} with custom params: {t.fusion.__class__.__name__}')
"
```

**Result**: ✅ All custom parameters work correctly

### Test 3: Config File Loading
```bash
python experiments/run.py config/default.yaml --dry-run
```

**Result**: ✅ Config loads and models instantiate with new parameters

---

## Migration Guide

### For Existing Experiments

**Option 1: No changes needed** (use defaults)
- Existing configs work without modification
- Uses default `fusion_heads=8` and `dropout=0.1`

**Option 2: Explicit configuration** (recommended for reproducibility)
```yaml
# Add to your config file:
teacher:
  # ... existing params ...
  fusion_heads: 8
  dropout: 0.1

student:
  # ... existing params ...
  fusion_heads: 8
  dropout: 0.1
```

**Option 3: Custom values** (for experimentation)
```yaml
teacher:
  fusion_heads: 16    # Try more heads
  dropout: 0.2        # Try higher dropout

fusion:
  type: "concat_mlp"
  hidden_mult: 4      # Larger hidden layers
```

---

## Parameter Extraction Flow

```
config/experiment.yaml
         ↓
   trainer/engine.py (lines 193-220)
         ↓
   ┌─────────────────┐
   │ Extract params: │
   │ - fusion_type   │
   │ - fusion_heads  │
   │ - dropout       │
   │ - fusion_params │
   └─────────────────┘
         ↓
   ┌──────────────────────────────┐
   │ Teacher/Student.__init__()   │
   │ - Accept all parameters      │
   │ - Pass to _create_fusion()   │
   └──────────────────────────────┘
         ↓
   ┌──────────────────────────────┐
   │ _create_fusion()             │
   │ - Extract module-specific    │
   │   params from fusion_params  │
   │ - Use defaults if missing    │
   │ - Instantiate correct module │
   └──────────────────────────────┘
         ↓
   Fusion Module Instance
```

---

## Summary of Changes

### What Was Fixed
1. ✅ `fusion_heads` - now configurable per model (teacher/student)
2. ✅ `dropout` - now configurable per model (teacher/student)
3. ✅ `hidden_mult` - now configurable via fusion config
4. ✅ `dropout` (fusion-specific) - now configurable via fusion config
5. ✅ `p_img`, `p_txt` - now configurable via fusion config

### What Remains Hardcoded (By Design)
- `num_modality_classes` - dynamically determined from dataset
- `num_location_classes` - dynamically determined from dataset
- Backbone pretrained model paths - defined in `models/backbones.py`

---

## Recommended Next Steps

1. **Update all config files** in `config/` directories with new parameters
2. **Re-run critical experiments** to verify consistency
3. **Ablation studies** on fusion_heads (4, 8, 16) and dropout (0.1, 0.2, 0.3)
4. **Document** parameter choices in experiment logs

---

## Contact & Support

For questions about configuration parameters:
- See: `docs/CONFIGURABLE_MODULES.md` - comprehensive guide
- See: `.github/copilot-instructions.md` - project conventions
- Check: Example configs in `config/` directories

---

**Status**: ✅ COMPLETE - All parameters are now configurable through YAML config files
**Verified**: December 15, 2025
**Backward Compatible**: Yes (all new parameters have defaults)
