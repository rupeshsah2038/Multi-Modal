# Configurable Metrics Labels

## Overview
The metrics system now supports configurable task labels, allowing each dataset to use meaningful names for their specific tasks instead of hardcoded "modality" and "location" labels.

## Configuration

### Adding Task Labels to Config Files

Add `task1_label` and `task2_label` to your config's `data` section:

```yaml
data:
  type: "wound"
  root: "datasets/Wound-1-0"
  batch_size: 16
  num_workers: 4
  # Task labels for metrics
  task1_label: "type"        # Primary task label
  task2_label: "severity"    # Secondary task label
```

### Default Values

If not specified, the system defaults to:
- `task1_label: "modality"` (for backward compatibility with MedPix)
- `task2_label: "location"` (for backward compatibility with MedPix)

## Dataset-Specific Labels

### MedPix-2-0
```yaml
task1_label: "modality"  # CT vs MR classification
task2_label: "location"  # Body location classification
```

**Metrics output:**
- `dev_modality_acc`, `test_modality_acc`
- `dev_location_acc`, `test_location_acc`
- `dev_modality_f1`, `dev_location_f1`, etc.

### Wound-1-0
```yaml
task1_label: "type"      # Wound type classification (10 classes)
task2_label: "severity"  # Severity classification (3 classes)
```

**Metrics output:**
- `dev_type_acc`, `test_type_acc`
- `dev_severity_acc`, `test_severity_acc`
- `dev_type_f1`, `dev_severity_f1`, etc.

## Affected Files

### Configuration Files
All config files have been updated:
- `config/default.yaml` - MedPix config with modality/location labels
- `config/wound.yaml` - Wound config with type/severity labels
- `config/test-1epoch.yaml` - MedPix test config
- `config/test-wound-1epoch.yaml` - Wound test config

### Code Changes

#### `utils/metrics.py`
- `evaluate_detailed()` now accepts `task1_label` and `task2_label` parameters
- Metric dictionary keys use configurable labels instead of hardcoded strings
- Confusion matrix filenames use configurable labels

#### `trainer/engine.py`
- Reads `task1_label` and `task2_label` from config (lines ~158-159)
- Passes labels to all `evaluate_detailed()` calls
- Computes dev_score using configurable label names

## CSV Output Examples

### MedPix CSV Header
```csv
epoch,train_loss,dev_modality_acc,dev_location_acc,dev_modality_f1,dev_location_f1,...
```

### Wound CSV Header
```csv
epoch,train_loss,dev_type_acc,dev_severity_acc,dev_type_f1,dev_severity_f1,...
```

## Confusion Matrix Filenames

Confusion matrices are saved with configurable labels:

**MedPix:**
- `cm_modality_dev.npy`, `cm_modality_test.npy`
- `cm_location_dev.npy`, `cm_location_test.npy`

**Wound:**
- `cm_type_dev.npy`, `cm_type_test.npy`
- `cm_severity_dev.npy`, `cm_severity_test.npy`

## Backward Compatibility

All existing configs and code that don't specify task labels will continue to work with default values (`modality` and `location`). This ensures no breaking changes for existing experiments.

## Testing

### Wound Dataset Test
```bash
conda activate fedenv
python experiments/run.py config/test-wound-1epoch.yaml
```

Expected output includes:
```
dev_type_acc: 0.4681
dev_severity_acc: 0.8638
dev_type_f1: 0.3365
dev_severity_f1: 0.7265
```

### MedPix Dataset Test
```bash
conda activate fedenv
python experiments/run.py config/test-1epoch.yaml
```

Expected output includes:
```
dev_modality_acc: 0.9442
dev_location_acc: 0.5431
dev_modality_f1: 0.9428
dev_location_f1: 0.4361
```

## Benefits

1. **Clarity**: Metrics immediately show which dataset and tasks are being evaluated
2. **Flexibility**: Easy to add new datasets with custom task names
3. **Maintainability**: No hardcoded task names in evaluation code
4. **Compatibility**: Existing configs work without modification
5. **Documentation**: CSV outputs are self-documenting with meaningful column names
