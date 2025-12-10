# Medpix_kd_modular

Lightweight repository for two-stage multimodal model training + knowledge distillation (Teacher→Student) supporting **multiple medical imaging datasets**.

This repo implements a vision+text Teacher network and a smaller Student distilled from it. The training pipeline, experiment orchestration, and modular loss implementations allow quick iteration on model/backbone choices, fusion strategies, and distillation methods.

**Supported Datasets:**
- **MedPix-2-0**: Medical imaging dataset (CT/MR modality classification, body location classification)
- **Wound-1-0**: Wound image dataset (wound type classification, severity classification)

## Quick start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run an experiment

**For MedPix dataset:**
```bash
python experiments/run.py config/default.yaml
```

**For Wound dataset:**
```bash
# First time: split the dataset
python tools/split_wound_dataset.py --input datasets/Wound-1-0/metadata.csv --output datasets/Wound-1-0

# Then run training
python experiments/run.py config/wound.yaml
```

Outputs are saved to the directory specified in `config.logging.log_dir` (default: `logs/`). Each run generates:
- `metrics.csv` and `metrics.json` — per-epoch and per-split metrics
- `results.json` — complete experiment metadata (config, hyperparameters, dev/test metrics)
- `student_best.pth`, `student_final.pth` — model checkpoints
- Confusion matrices (`.npy`) for each split and task

## Supported Backbones

This project supports configurable vision and text backbones. Below are the commonly used/backed mappings available in `models/backbones.py` (friendly name → Hugging Face identifier).

### Vision backbones

| Friendly name | HuggingFace ID |
|---|---|
| `vit-large` (teacher) | `google/vit-large-patch16-224` |
| `vit-base` | `google/vit-base-patch16-224` |
| `deit-base` | `facebook/deit-base-distilled-patch16-224` |
| `deit-tiny` | `facebook/deit-tiny-patch16-224` |
| `deit-tiny-distilled` / `mobile-vit` | `facebook/deit-tiny-distilled-patch16-224` |
| `tiny-vit` | `google/vit-base-patch16-224-in21k` |
| `mobilevit-xx-small` | `apple/mobilevit-xx-small` |
| `mobilevit-small` | `apple/mobilevit-small` |
| `mobilevit-medium` | `apple/mobilevit-medium` |
| `efficientvit-b0` | `mit-han-lab/efficientvit-b0` |
| `mobilenet-v2` | `google/mobilenet_v2_1.0_224` |

Notes: use the friendly name in `config/default.yaml` (e.g., `teacher.vision: "vit-large"` or `student.vision: "deit-tiny"`).

### Text backbones

| Friendly name | HuggingFace ID |
|---|---|
| `bio-clinical-bert` (teacher) | `emilyalsentzer/Bio_ClinicalBERT` |
| `distilbert` | `distilbert-base-uncased` |
| `mobile-bert` / `mobilebert` | `google/mobilebert-uncased` |
| `bert-tiny` | `prajjwal1/bert-tiny` |
| `bert-mini` | `prajjwal1/bert-mini` |
| `minilm` | `nreimers/MiniLM-L6-H384-uncased` |

Notes: The dataset is tokenized twice (teacher tokenizer and student tokenizer). Ensure the chosen text backbones are valid HF models — mismatched tokenizer/model pairs can raise token-id range errors.

### Example backbone selections

- Baseline teacher: `teacher.vision: "vit-large"`, `teacher.text: "bio-clinical-bert"`
- Standard student: `student.vision: "deit-base"`, `student.text: "distilbert"`
- Edge student: `student.vision: "deit-tiny"`, `student.text: "mobile-bert"`
- Ultra-edge student: `student.vision: "deit-tiny"`, `student.text: "bert-tiny"`

You can test multiple presets with `tools/batch_runs.py` by using run names such as `original`, `mobile-edge`, `ultra-edge`, `edge-vision`, and `edge-text`.


### 3. Run batch experiments

**Loss exploration (compare different loss functions):**
```bash
python tools/run_loss_explore.py
# or
./tools/run_loss_explore.sh
```
Tests all 5 loss functions (vanilla, combined, crd, rkd, mmd) with cross-attention fusion on both datasets.

**Fusion exploration (compare different fusion modules):**
```bash
python tools/run_fusion_explore.py
# or
./tools/run_fusion_explore.sh
```
Tests all 9 fusion modules (simple, concat_mlp, cross_attention, gated, transformer_concat, modality_dropout, film, energy_aware_adaptive, shomr) with combined loss on both datasets.

**Ultra-edge experiments (lightweight student models):**
```bash
python tools/run_ultra_edge.py
# or
./tools/run_ultra_edge.sh
```
Tests lightweight student configurations (deit-small/deit-tiny with distilbert/minilm) on both datasets.

**Custom backbone swaps:**
```bash
python tools/batch_runs.py --base config/default.yaml \
  --runs original,swap_vision,swap_text,swap_both \
  --execute --epochs 5 --batch-size 8 --device cuda:3
```

All batch experiments:
- Execute runs sequentially
- Save outputs to separate log directories
- Provide progress updates and final summary

## Key files & architecture

### Training & orchestration
- **`experiments/run.py`**: Entry point; loads config and calls trainer
- **`trainer/engine.py`**: Main orchestration — loads data, builds models, trains teacher, distills to student, evaluates, logs results

### Data & models
- **`data/dataset.py`**: Unified dataset factory supporting MedPixDataset and WoundDataset
- **`data/wound_dataset.py`**: Standalone Wound dataset implementation (legacy)
- **`models/backbones.py`**: Vision (ViT, DeiT) and text (Bio_ClinicalBERT, DistilBERT) backbone loaders
- **`models/teacher.py`**, **`models/student.py`**: Teacher and student models with dual tokenization, fusion, dynamic class counts
- **`models/fusion/`**: Fusion modules (simple, concat_mlp, cross_attention, gated, transformer_concat, modality_dropout, film, energy_aware_adaptive, shomr)
- **`models/heads.py`**: Classification heads for modality and location tasks

### Losses & metrics
- **`losses/`**: Distillation loss implementations:
  - `vanilla.py` — CE + KL + MSE (baseline)
  - `combined.py` — CE + KL + MSE + CRD
  - `crd.py` — Contrastive Representation Distillation
  - `rkd.py` — Relational Knowledge Distillation
  - `mmd.py` — Maximum Mean Discrepancy alignment
  - ✓ All losses support lazy projection layers (adapt to any backbone dimensions)
- **`utils/metrics.py`**: Evaluation functions (accuracy, F1, AUC, per-split metrics)
- **`utils/logger.py`**: Metrics CSV/JSON export and confusion matrix saving
- **`utils/results_logger.py`**: Full experiment metadata persistence (`results.json`)

### Tooling
- **`tools/batch_runs.py`**: Batch experiment runner supporting vision/text backbone swaps
- **`tools/run_loss_explore.py` / `.sh`**: Run all loss function experiments (vanilla, combined, crd, rkd, mmd)
- **`tools/run_fusion_explore.py` / `.sh`**: Run all fusion module experiments (9 fusion types)
- **`tools/run_ultra_edge.py` / `.sh`**: Run ultra-edge experiments (lightweight student models)
- **`tools/split_wound_dataset.py`**: Split Wound dataset CSV into train/dev/test splits
- **`tools/verify_wound_dataset.py`**: Verify Wound dataset structure before training
- **`tools/test_backbone_availability.py`**: Check availability of backbone models

## Configuration

### Structure (config/default.yaml)
```yaml
data:
  type: "medpix"              # Dataset type: 'medpix' or 'wound'
  root: "datasets/MedPix-2-0" # Dataset root
  batch_size: 16              # Batch size for training/eval
  num_workers: 4              # DataLoader workers

teacher:
  vision: "vit-large"         # Vision backbone: vit-large, deit-base, deit-small, etc.
  text: "bio-clinical-bert"   # Text backbone: bio-clinical-bert, distilbert, etc.
  fusion_layers: 2            # Fusion module layers
  fusion_dim: 512             # Fusion dimension (required)

student:
  vision: "deit-base"         # Student vision (typically smaller)
  text: "distilbert"          # Student text (typically smaller)
  fusion_layers: 1
  fusion_dim: 512             # Fusion dimension (required)

training:
  teacher_epochs: 1           # Number of teacher pre-training epochs
  student_epochs: 1           # Number of distillation epochs
  teacher_lr: 1e-5
  student_lr: 3e-4
  alpha: 1.0                  # KL divergence weight
  beta: 100.0                 # Feature MSE weight
  T: 2.0                      # Temperature for KL
  # gamma: null               # Optional: used by some losses (combined, etc.)

logging:
  log_dir: "logs"             # Output directory

fusion:
  type: "simple"              # Fusion type: simple, concat_mlp, cross_attention, gated, transformer_concat, modality_dropout, film, energy_aware_adaptive, shomr

loss:
  type: "rkd"                 # Loss type: vanilla, combined, crd, rkd, mmd

device: "cuda:3"              # Optional: GPU device; defaults to cuda:4 if omitted
```

### Loss types & supported hyperparameters
| Loss | Config key | Additional params |
|------|------------|-------------------|
| `vanilla` | `loss.type: vanilla` | `alpha`, `beta`, `T` |
| `combined` | `loss.type: combined` | `alpha`, `beta`, `gamma`, `T` |
| `crd` | `loss.type: crd` | `temperature`, `base_temperature` |
| `rkd` | `loss.type: rkd` | `w_dist`, `w_angle` |
| `mmd` | `loss.type: mmd` | (none) |

The trainer automatically forwards matching keys from `cfg['training']` to the loss constructor.

## Data layout (expected)

### MedPix-2-0 Dataset
The dataset root (default: `datasets/MedPix-2-0`) must contain:

```
datasets/MedPix-2-0/
├── splitted_dataset/
│   ├── data_train.jsonl
│   ├── data_dev.jsonl
│   ├── data_test.jsonl
│   ├── descriptions_train.jsonl
│   ├── descriptions_dev.jsonl
│   └── descriptions_test.jsonl
└── images/
    ├── image_001.png
    ├── image_002.png
    └── ...
```

Configuration:
```yaml
data:
  type: "medpix"              # or omit (defaults to medpix)
  root: "datasets/MedPix-2-0"
```

### Wound-1-0 Dataset
The dataset root (default: `datasets/Wound-1-0`) must contain:

```
datasets/Wound-1-0/
├── images/                   # All wound images
├── metadata.csv             # Original metadata (optional after split)
├── metadata_train.csv       # Required: training split
├── metadata_dev.csv         # Required: validation split
└── metadata_test.csv        # Required: test split
```

CSV columns: `img_path`, `type`, `severity`, `description`

Configuration:
```yaml
data:
  type: "wound"                      # Required for Wound dataset
  root: "datasets/Wound-1-0"
  # Optional: customize column names if your CSV differs
  filepath_column: "img_path"        # Default: "file_path"
  type_column: "type"                # Default: "type"
  severity_column: "severity"        # Default: "severity"
  description_column: "description"  # Default: "description"
```

**First time setup for Wound dataset:**
```bash
python tools/split_wound_dataset.py \
  --input datasets/Wound-1-0/metadata.csv \
  --output datasets/Wound-1-0 \
  --train 0.7 --dev 0.15 --test 0.15
```

If images are missing, the dataset raises `FileNotFoundError`. Verify `config.data.root` is correct.

## Important implementation details

### Dual tokenization
- Each dataset example is tokenized twice: once with the **teacher tokenizer** (Bio_ClinicalBERT by default), once with the **student tokenizer** (DistilBERT by default)
- This ensures each model gets appropriately-sized token sequences
- Keep both tokenizers in sync if you change text backbones

### Model output keys
All models (teacher & student) return:
```python
{
  "logits_modality": tensor,       # Shape (B, num_modality_classes)
  "logits_location": tensor,       # Shape (B, num_location_classes)
  "img_raw": tensor,               # Vision backbone last hidden (B, D_vis)
  "txt_raw": tensor,               # Text backbone last hidden (B, D_txt)
  "img_proj": tensor,              # Projected vision (B, fusion_dim)
  "txt_proj": tensor,              # Projected text (B, fusion_dim)
}
```

**Note:** Class counts are dynamic:
- **MedPix**: 2 modality classes (CT/MR), 5 location classes (body regions)
- **Wound**: Dynamic based on CSV (e.g., 10 wound types, 3 severity levels)

Standardized model outputs (important):
- **`img_raw` / `txt_raw`**: Always refer to the *backbone* raw features (the output of the vision/text encoder before any linear projection). Both the Teacher and Student now provide these raw backbone features.
- **`img_proj` / `txt_proj`**: Always refer to the features after the model-specific linear projection into the shared `fusion_dim` used by fusion modules and many losses.

Rationale and compatibility:
- This standardization ensures losses and fusion modules have a stable interface: losses that need access to raw backbone representations (for e.g. CRD or RKD teacher comparisons) should read `*_raw` from the teacher, while comparisons against the student's projected features should use `*_proj` from the student.
- Historically the Student returned `img_raw` equal to its projected features in earlier commits; that ambiguity has been removed. If you maintain external code that relied on the old behavior, update it to use `img_proj` when you meant the Student's projected/fusion features.

Example usage in a loss implementation:
```python
# teacher: use backbone raw features
t_img_raw = t_out['img_raw']
# student: use projected features for fusion/heads
s_img_proj = s_out['img_proj']
```

### Loss interfaces & backbone flexibility
- **All losses now support lazy projection layers**: projections are created at first forward pass based on actual tensor shapes, so teacher/student backbone swaps work without code changes
- Supported loss calls:
  ```python
  loss = loss_fn(s_out, t_out, y_mod, y_loc)
  # where s_out, t_out are the model output dicts above
  ```

### Fusion modules
- All fusion modules (simple, concat_mlp, cross_attention, gated, transformer_concat) receive pre-projected features (both at the same `fusion_dim`)
- They work seamlessly with any backbone combination since projections happen before fusion

### Device handling
- Trainer respects `cfg['device']` if provided; otherwise defaults to `cuda:4`
- Multi-GPU support: set `device: cuda:0` (or any GPU index) in the config

### Metrics & logging
- Per-epoch metrics (train loss, dev F1, dev accuracy, etc.) are saved to `metrics.csv` and `metrics.json`
- Full experiment metadata (config, hyperparameters, final dev/test metrics) is saved to `results.json`
- Confusion matrices (`.npy`) are saved for modality and location tasks on each split

## Experiment Results

Comprehensive experiment results are available in the `docs/` directory:

- **`docs/ULTRA_EDGE_RESULTS.md`**: Ultra-edge model comparison (lightweight students)
  - Tests deit-small/deit-tiny with distilbert/minilm
  - Covers both MedPix and Wound datasets
  - Provides accuracy vs latency trade-off analysis
  
- **`docs/LOSS_EXPLORE_RESULTS.md`**: Loss function comparison
  - Tests vanilla, combined, crd, rkd, mmd losses
  - Fixed cross-attention fusion with vit-base + distilbert student
  - Shows per-loss performance on both datasets

- **`docs/CONFIGURABLE_METRICS.md`**: Custom task label configuration guide
- **`docs/LOSS_FUNCTIONS_COMPARISON.md`**: Detailed loss function documentation

### Key Findings

**Best configurations by use case:**

- **MedPix - Best accuracy:** `combined` loss with `cross_attention` fusion
- **MedPix - Best ultra-edge:** `deit_tiny-minilm` student (good F1, fast inference)
- **Wound - Best accuracy:** `vanilla` loss with `cross_attention` fusion  
- **Wound - Best ultra-edge:** `deit_small-minilm` student (dominates accuracy/latency)

See individual result documents for detailed metrics, critical observations, and recommendations.

## Common use cases

### Quick experiment with smaller batch size
```yaml
data:
  batch_size: 8
training:
  student_epochs: 3
```
Then run:
```bash
python experiments/run.py config/default.yaml
```

### Run on a specific GPU
```yaml
device: "cuda:2"
```

### Swap vision and text backbones
```yaml
teacher:
  vision: "deit-base"         # Smaller model as teacher
  text: "distilbert"
student:
  vision: "vit-large"         # Larger model as student
  text: "bio-clinical-bert"
```
Or use the batch runner to test multiple configurations automatically.

### Use a different loss function
```yaml
loss:
  type: "mmd"
training:
  # MMD doesn't use alpha/beta but still respects T if present
  T: 2.0
```

### Switch fusion strategy
```yaml
fusion:
  type: "cross_attention"
```

## Debugging & troubleshooting

### ModuleNotFoundError: No module named 'trainer'
- Ensure you run from the repository root: `python experiments/run.py config/default.yaml`
- The launcher adds the repo root to `sys.path` automatically

### FileNotFoundError: images missing
- Verify `config.data.root` points to the correct dataset folder
- Check that `{root}/images/` contains the expected image files

### CUDA out of memory
- Reduce `data.batch_size` (e.g., 8 or 4)
- Reduce `training.student_epochs` or `training.teacher_epochs` for quicker iteration
- Use a GPU with more memory or target a specific GPU with `device: cuda:X`

### Slow data loading
- Reduce `data.num_workers` if you see I/O bottlenecks
- Increase `data.num_workers` if CPU is underutilized

### Metrics files empty or missing
- Ensure `config.logging.log_dir` exists or is writable
- Check that `training.student_epochs >= 1` (at least one epoch needed to log metrics)

## Example workflows

### Baseline run (1 epoch for quick test)
```bash
python experiments/run.py config/default.yaml
```

### Multi-run comparison (backbone swaps)
```bash
python tools/batch_runs.py --base config/default.yaml \
  --runs original,swap_vision,swap_text \
  --execute --epochs 3 --batch-size 8 --device cuda:3
```
Results saved to `logs/run_original/`, `logs/run_swap_vision/`, etc.

### Longer training run
Edit `config/default.yaml`:
```yaml
training:
  teacher_epochs: 5
  student_epochs: 10
data:
  batch_size: 32
```
Then run:
```bash
python experiments/run.py config/default.yaml
```

### Compare loss functions
Edit `config/default.yaml`:
```yaml
loss:
  type: "combined"  # Try vanilla, combined, crd, rkd, mmd
```

## References & implementation notes

- Teacher/Student models use Hugging Face transformers for backbones (ViT, DeiT, BERT variants)
- Dual tokenization ensures both models handle text appropriately for their architecture
- Lazy projection layers in all losses enable flexible backbone swapping without code changes
- Batch runner supports configuration generation and sequential experiment execution
- Results are persisted in multiple formats (CSV, JSON, `.npy` matrices) for analysis

## Tips for fast iteration

1. **Start small**: Use 1 epoch and `batch_size=8` for quick validation
2. **Check metrics early**: Inspect `logs/metrics.json` after first run to confirm training is working
3. **Inspect confusion matrices**: Review `.npy` files to understand per-class performance
4. **Use batch runner for ablations**: Test multiple backbone/loss/fusion combinations systematically
5. **Monitor GPU memory**: Use `nvidia-smi` to ensure no other processes are consuming VRAM

## Additional Resources

- **Experiment Results**:
  - `docs/ULTRA_EDGE_RESULTS.md` — Ultra-edge model comparison
  - `docs/LOSS_EXPLORE_RESULTS.md` — Loss function comparison
  - `docs/CONFIGURABLE_METRICS.md` — Custom metrics configuration
  - `docs/LOSS_FUNCTIONS_COMPARISON.md` — Loss function details
- **Dataset Guides**:
  - `QUICK_START_WOUND.md` — Wound dataset quick start
  - `docs/WOUND_DATASET.md` — Detailed Wound integration guide
  - `WOUND_INTEGRATION_SUMMARY.md` — Implementation details

## Project Structure

```
Medpix_modular/
├── config/                    # Configuration files
│   ├── default.yaml          # MedPix config
│   ├── wound.yaml            # Wound config
│   ├── test-*.yaml           # Quick test configs
│   ├── loss-explore/         # Loss comparison configs
│   ├── fusion-explore/       # Fusion comparison configs
│   └── ultra-edge/           # Ultra-edge model configs
├── data/                      # Dataset implementations
│   ├── dataset.py            # Unified dataset factory
│   └── wound_dataset.py      # Standalone Wound dataset
├── datasets/                  # Data storage
│   ├── MedPix-2-0/           # MedPix dataset
│   └── Wound-1-0/            # Wound dataset
├── docs/                      # Documentation
│   ├── ULTRA_EDGE_RESULTS.md
│   ├── LOSS_EXPLORE_RESULTS.md
│   ├── CONFIGURABLE_METRICS.md
│   ├── LOSS_FUNCTIONS_COMPARISON.md
│   └── WOUND_DATASET.md
├── experiments/               # Experiment runners
├── losses/                    # Loss implementations
├── models/                    # Model architectures
│   ├── backbones.py          # Vision/text backbones
│   ├── teacher.py            # Teacher model
│   ├── student.py            # Student model
│   ├── heads.py              # Classification heads
│   └── fusion/               # Fusion modules (9 types)
├── tools/                     # Utility scripts
│   ├── batch_runs.py         # Batch experiment runner
│   ├── run_loss_explore.py/.sh
│   ├── run_fusion_explore.py/.sh
│   ├── run_ultra_edge.py/.sh
│   ├── split_wound_dataset.py
│   └── verify_wound_dataset.py
├── trainer/                   # Training engine
├── utils/                     # Metrics and logging
└── logs/                      # Experiment outputs
    ├── loss-explore/         # Loss comparison results
    ├── fusion-explore/       # Fusion comparison results
    └── ultra-edge/           # Ultra-edge results
```
