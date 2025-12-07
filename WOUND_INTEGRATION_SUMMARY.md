# Wound Dataset Integration - Summary

## ✅ Completed Changes

Successfully extended the codebase to support **Wound-1-0** dataset while maintaining full backward compatibility with **MedPix-2-0**.

### Core Modifications

#### 1. **Data Layer** (`data/dataset.py`)
- Added `WoundDataset` class for Wound-1-0 support
- Implemented factory pattern with `get_dataset()` function
- Added `get_num_classes()` for dynamic class count detection
- Unified interface: both datasets return same keys (`modality`, `location`)

#### 2. **Model Layer** 
- **`models/teacher.py`**: Added `num_modality_classes` and `num_location_classes` parameters
- **`models/student.py`**: Added `num_modality_classes` and `num_location_classes` parameters
- Models now dynamically adapt to dataset class counts

#### 3. **Training Engine** (`trainer/engine.py`)
- Dataset-agnostic loader with automatic dataset type detection
- Dynamic class count detection from config or CSV inspection
- Passes class counts to Teacher/Student models at instantiation

#### 4. **Configuration**
- **`config/default.yaml`**: Added `data.type: "medpix"` for explicit dataset selection
- **`config/wound.yaml`**: New config template for Wound-1-0 dataset
- **`config/test-1epoch.yaml`**: Updated for quick testing

#### 5. **Utilities**
- **`tools/split_wound_dataset.py`**: CLI tool to split metadata.csv into train/dev/test
- Stratified splitting by wound type for balanced distribution

#### 6. **Documentation**
- **`docs/WOUND_DATASET.md`**: Complete integration guide
- **`.github/copilot-instructions.md`**: Updated with dataset switching instructions

### New Files Created
```
data/wound_dataset.py                     # Standalone (legacy, merged into dataset.py)
tools/split_wound_dataset.py              # Dataset splitting utility
config/wound.yaml                         # Wound dataset config template
docs/WOUND_DATASET.md                     # Integration documentation
tests/test_wound_integration.py           # End-to-end integration test
```

### Modified Files
```
data/dataset.py                           # Added WoundDataset + factory functions
models/teacher.py                         # Dynamic class count parameters
models/student.py                         # Dynamic class count parameters
trainer/engine.py                         # Dataset-agnostic loading logic
config/default.yaml                       # Added data.type field
.github/copilot-instructions.md           # Updated documentation
```

## 🎯 Key Features

### 1. **Unified Interface**
Both datasets use the same tensor keys for compatibility:
- **MedPix**: CT/MR → `modality`, Body Location → `location`
- **Wound**: Type → `modality`, Severity → `location`

### 2. **Dynamic Class Detection**
- **MedPix**: Fixed 2 modality classes, 5 location classes
- **Wound**: Dynamically detected from CSV (e.g., 3 types, 3 severities)

### 3. **Backward Compatibility**
- ✅ All existing MedPix experiments work without changes
- ✅ Default behavior (no `data.type`) assumes MedPix
- ✅ All previous configs remain valid

### 4. **Flexible Configuration**
Wound dataset supports customizable column names:
```yaml
data:
  type: "wound"
  type_column: "type"           # Customize if needed
  severity_column: "severity"
  description_column: "description"
  filepath_column: "file_path"
```

## 📊 Test Results

### Unit Tests
✅ All loss tests passed (vanilla, combined, CRD, RKD, MMD)
✅ Logging test passed
✅ Wound dataset loading test passed
✅ Wound training integration test passed

### Integration Tests
✅ **MedPix**: 1-epoch run successful (97% modality acc, 79.5% location acc)
✅ **Wound**: End-to-end training successful with mock data

## 🚀 Usage Examples

### MedPix Dataset (existing)
```bash
# Default - no changes needed
python experiments/run.py config/default.yaml
```

### Wound Dataset (new)
```bash
# 1. Split the dataset first
python tools/split_wound_dataset.py \
  --input Wound-1-0/metadata.csv \
  --output Wound-1-0

# 2. Run training
python experiments/run.py config/wound.yaml
```

### Custom Wound Configuration
```yaml
data:
  type: "wound"
  root: "Wound-1-0"
  batch_size: 16
  num_workers: 4

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
```

## 🔧 Dataset Structure Requirements

### Wound-1-0
```
Wound-1-0/
├── images/                    # All wound images
├── metadata.csv              # Original CSV (optional)
├── metadata_train.csv        # Required: training split
├── metadata_dev.csv          # Required: validation split
└── metadata_test.csv         # Required: test split
```

**CSV columns**: `file_path`, `type`, `severity`, `description`

### MedPix-2-0 (unchanged)
```
datasets/MedPix-2-0/
├── images/
└── splitted_dataset/
    ├── data_{split}.jsonl
    └── descriptions_{split}.jsonl
```

## 📝 Notes

1. **No hardcoded values**: All class counts and dimensions are config-driven
2. **Lazy projections**: Loss modules create projections at runtime for backbone flexibility
3. **Tokenizer alignment**: Automatically matches tokenizers to configured text backbones
4. **Stratified splitting**: Tool splits by type for balanced class distribution
5. **Extensible design**: Easy to add more datasets following the same pattern

## ✨ Next Steps

To use with your actual Wound-1-0 dataset:
1. Place dataset in `datasets/Wound-1-0/` directory
2. Run splitter: `python tools/split_wound_dataset.py --input datasets/Wound-1-0/metadata.csv --output datasets/Wound-1-0`
3. Adjust `config/wound.yaml` if needed (batch size, epochs, backbones)
4. Run: `python experiments/run.py config/wound.yaml`
