# Medpix_kd_modular

Lightweight repository for two-stage multimodal model training + knowledge distillation (Teacher→Student) on the MedPix dataset.

This repo implements a vision+text Teacher network and a smaller Student distilled from it. The training pipeline, experiment orchestration, and modular loss implementations allow quick iteration on model/backbone choices, fusion strategies, and distillation methods.

## Quick start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run an experiment
```bash
python experiments/run.py config/default.yaml
```

Outputs are saved to the directory specified in `config.logging.log_dir` (default: `logs/`). Each run generates:
- `metrics.csv` and `metrics.json` — per-epoch and per-split metrics
- `results.json` — complete experiment metadata (config, hyperparameters, dev/test metrics)
- `student_best.pth`, `student_final.pth` — model checkpoints
- Confusion matrices (`.npy`) for each split and task

### 3. Run multiple experiments with different backbone swaps (batch mode)
```bash
python tools/batch_runs.py --base config/default.yaml \
  --runs original,swap_vision,swap_text,swap_both \
  --execute --epochs 5 --batch-size 8 --device cuda:3
```

This will:
- Create per-run configs under `logs/run_<name>/config.yaml`
- Execute each run sequentially
- Save outputs to separate `logs/run_<name>/` directories

## Key files & architecture

### Training & orchestration
- **`experiments/run.py`**: Entry point; loads config and calls trainer
- **`trainer/engine.py`**: Main orchestration — loads data, builds models, trains teacher, distills to student, evaluates, logs results

### Data & models
- **`data/dataset.py`**: MedPixDataset — produces tokenized inputs for both teacher and student
- **`models/backbones.py`**: Vision (ViT, DeiT) and text (Bio_ClinicalBERT, DistilBERT) backbone loaders
- **`models/teacher.py`**, **`models/student.py`**: Teacher and student models with dual tokenization, fusion, and output heads
- **`models/fusion/`**: Fusion modules (simple, concat_mlp, cross_attention, gated, transformer_concat)
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

## Configuration

### Structure (config/default.yaml)
```yaml
data:
  root: "MedPix-2-0"          # Dataset root (contains splitted_dataset/, images/)
  batch_size: 16              # Batch size for training/eval
  num_workers: 4              # DataLoader workers

teacher:
  vision: "vit-large"         # Vision backbone: vit-large, deit-base, deit-small, etc.
  text: "bio-clinical-bert"   # Text backbone: bio-clinical-bert, distilbert, etc.
  fusion_layers: 2            # Fusion module layers

student:
  vision: "deit-base"         # Student vision (typically smaller)
  text: "distilbert"          # Student text (typically smaller)
  fusion_layers: 1

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
  type: "simple"              # Fusion type: simple, concat_mlp, cross_attention, gated, transformer_concat

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

The dataset root (default: `MedPix-2-0`) must contain:

```
MedPix-2-0/
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

If images are missing, `MedPixDataset` raises `FileNotFoundError`. Verify `config.data.root` is correct.

## Important implementation details

### Dual tokenization
- Each dataset example is tokenized twice: once with the **teacher tokenizer** (Bio_ClinicalBERT by default), once with the **student tokenizer** (DistilBERT by default)
- This ensures each model gets appropriately-sized token sequences
- Keep both tokenizers in sync if you change text backbones

### Model output keys
All models (teacher & student) return:
```python
{
  "logits_modality": tensor,       # Shape (B, 2)
  "logits_location": tensor,       # Shape (B, 5)
  "img_raw": tensor,               # Vision backbone last hidden (B, D_vis)
  "txt_raw": tensor,               # Text backbone last hidden (B, D_txt)
  "img_proj": tensor,              # Projected vision (B, 512)
  "txt_proj": tensor,              # Projected text (B, 512)
}
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

## Common use cases

### Quick experiment with smaller batch size
Edit `config/default.yaml` and change:
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
