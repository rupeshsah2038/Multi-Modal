# Wound Dataset Integration

This codebase now supports both **MedPix-2-0** and **Wound-1-0** datasets with minimal configuration changes.

## Dataset Structure

### Wound-1-0
```
datasets/Wound-1-0/
├── images/              # All wound images
├── metadata.csv         # Original metadata with columns: file_path, type, severity, description
├── metadata_train.csv   # Training split (generated)
├── metadata_dev.csv     # Validation split (generated)
└── metadata_test.csv    # Test split (generated)
```

### MedPix-2-0 (existing)
```
datasets/MedPix-2-0/
├── images/
└── splitted_dataset/
    ├── data_train.jsonl
    ├── data_dev.jsonl
    ├── data_test.jsonl
    ├── descriptions_train.jsonl
    ├── descriptions_dev.jsonl
    └── descriptions_test.jsonl
```

## Quick Start

### 1. Split Wound Dataset
First, split your `metadata.csv` into train/dev/test:

```bash
python tools/split_wound_dataset.py \
  --input datasets/Wound-1-0/metadata.csv \
  --output datasets/Wound-1-0 \
  --train 0.7 --dev 0.15 --test 0.15
```

### 2. Run Training

**For Wound dataset:**
```bash
python experiments/run.py config/wound.yaml
```

**For MedPix dataset (backward compatible):**
```bash
python experiments/run.py config/default.yaml
```

## Configuration

### Wound Dataset Config (`config/wound.yaml`)
```yaml
data:
  type: 'wound'           # Set to 'wound' for Wound-1-0 dataset
  root: "datasets/Wound-1-0"
  batch_size: 16
  num_workers: 4
  # Column names (customize if your CSV has different column names)
  type_column: "type"
  severity_column: "severity"
  description_column: "description"
  filepath_column: "file_path"

teacher:
  vision: "vit-base"
  text: "bio-clinical-bert"
  fusion_layers: 2
  fusion_dim: 512

student:
  vision: "deit-base"
  text: "distilbert"
  fusion_layers: 1
  fusion_dim: 512

training:
  teacher_epochs: 5
  student_epochs: 10
  teacher_lr: 1e-5
  student_lr: 3e-4
  alpha: 1.0
  beta: 100.0
  T: 2.0

logging:
  log_dir: "logs/wound_experiment"

fusion:
  type: "simple"

loss:
  type: "vanilla"

device: "cuda:4"
```

### MedPix Dataset Config (unchanged)
Set `data.type: "medpix"` or omit (defaults to medpix for backward compatibility).

## Implementation Details

### Architecture Changes
- **Dynamic class counts**: Models now accept `num_modality_classes` and `num_location_classes` parameters
- **Unified interface**: Both datasets use the same keys (`modality`, `location`) for compatibility
- **Factory pattern**: `get_dataset()` function creates appropriate dataset based on type

### Task Mapping
**Wound dataset:**
- `type` (wound type) → `modality` classification head
- `severity` (wound severity) → `location` classification head

**MedPix dataset:**
- `modality` (CT/MR) → `modality` classification head (2 classes)
- `location` (body region) → `location` classification head (5 classes)

### Files Modified/Created
1. **data/dataset.py** - Added factory functions and WoundDataset class
2. **data/wound_dataset.py** - Standalone wound dataset (legacy, now integrated into dataset.py)
3. **models/teacher.py** - Added dynamic class count parameters
4. **models/student.py** - Added dynamic class count parameters
5. **trainer/engine.py** - Dataset-agnostic loader with dynamic class detection
6. **tools/split_wound_dataset.py** - Utility to split wound metadata
7. **config/wound.yaml** - Example config for wound dataset
8. **config/default.yaml** - Updated with dataset type field

## Backward Compatibility
✅ All existing MedPix experiments work without changes
✅ Default behavior (when `data.type` is omitted) is MedPix
✅ All tests pass with both datasets
