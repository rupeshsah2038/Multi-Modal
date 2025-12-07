# Quick Reference: Wound-1-0 Dataset Integration

## Setup Checklist

### 1. Prepare Your Dataset Structure
```
datasets/Wound-1-0/
├── images/              # All your wound images
└── metadata.csv         # CSV with: file_path, type, severity, description
```

### 2. Split Dataset (One-Time Setup)
```bash
python tools/split_wound_dataset.py \
  --input datasets/Wound-1-0/metadata.csv \
  --output datasets/Wound-1-0 \
  --train 0.7 --dev 0.15 --test 0.15
```

This creates:
- `metadata_train.csv` (70% of data)
- `metadata_dev.csv` (15% of data)  
- `metadata_test.csv` (15% of data)

### 3. Verify Setup (Optional but Recommended)
```bash
python tools/verify_wound_dataset.py datasets/Wound-1-0
```

### 4. Run Training
```bash
conda activate fedenv
python experiments/run.py config/wound.yaml
```

## Configuration Examples

### Basic Wound Config
```yaml
data:
  type: "wound"              # Required: specify dataset type
  root: "datasets/Wound-1-0"         # Path to dataset root
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

training:
  teacher_epochs: 5
  student_epochs: 10
  teacher_lr: 1e-5
  student_lr: 3e-4

logging:
  log_dir: "logs/wound_experiment"

device: "cuda:4"
```

### Custom Column Names
If your CSV has different column names:
```yaml
data:
  type: "wound"
  root: "datasets/Wound-1-0"
  type_column: "wound_type"        # Default: "type"
  severity_column: "severity_level" # Default: "severity"
  description_column: "notes"       # Default: "description"
  filepath_column: "image_path"     # Default: "file_path"
```

## Key Differences from MedPix

| Feature | MedPix-2-0 | Wound-1-0 |
|---------|------------|-----------|
| Config `data.type` | `"medpix"` or omit | `"wound"` (required) |
| File format | JSONL | CSV |
| Task 1 (modality) | CT vs MR (2 classes) | Wound type (dynamic) |
| Task 2 (location) | Body region (5 classes) | Severity (dynamic) |
| Split files | `data_{split}.jsonl` | `metadata_{split}.csv` |

## Troubleshooting

### Error: "Missing split file"
**Solution**: Run the splitter first:
```bash
python tools/split_wound_dataset.py --input datasets/Wound-1-0/metadata.csv --output datasets/Wound-1-0
```

### Error: "Image not found"
**Check**:
1. Images are in `datasets/Wound-1-0/images/`
2. `file_path` column matches actual filenames
3. File extensions are correct (.jpg, .png, etc.)

### Error: "Column not found"
**Solution**: Specify custom column names in config:
```yaml
data:
  type: "wound"
  type_column: "your_type_column_name"
  severity_column: "your_severity_column_name"
  # etc.
```

### Low accuracy / Poor results
**Tips**:
1. Check class balance with verify script
2. Ensure descriptions are meaningful (not empty/null)
3. Increase `teacher_epochs` and `student_epochs`
4. Try different backbones (see config files)
5. Adjust `batch_size` based on GPU memory

## Batch Processing Multiple Configs

```bash
# Run experiments with different backbones
python tools/batch_runs.py \
  --base config/wound.yaml \
  --runs original,swap_vision,swap_text,swap_both \
  --execute --epochs 10 --batch-size 16 --device cuda:3
```

## Switching Back to MedPix

Simply change config or use default:
```bash
python experiments/run.py config/default.yaml
```

Or explicitly set in config:
```yaml
data:
  type: "medpix"  # Switch back to MedPix
  root: "datasets/MedPix-2-0"
```

## Quick Test (1 epoch)

Verify everything works before long training:
```yaml
training:
  teacher_epochs: 1
  student_epochs: 1
```

```bash
python experiments/run.py config/wound.yaml
```

## Outputs

All results saved to `logging.log_dir`:
- `student_best.pth` - Best model checkpoint
- `student_final.pth` - Final model checkpoint
- `metrics.csv` - Training metrics
- `metrics.json` - Detailed metrics
- `results.json` - Complete experiment results
- `cm_*.npy` - Confusion matrices

## Need Help?

1. Run verification: `python tools/verify_wound_dataset.py datasets/Wound-1-0`
2. Check logs in `logging.log_dir`
3. Review `docs/WOUND_DATASET.md` for details
4. Check `WOUND_INTEGRATION_SUMMARY.md` for implementation details
