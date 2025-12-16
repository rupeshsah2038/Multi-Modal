# Hyperparameter Tuning with Optuna

## Quick Start

### Basic Usage (Recommended)

Tune learning rates, loss weights, and architecture dimensions:

```bash
# MedPix dataset
python tools/run_optuna_tuning.py --config config/default.yaml --n-trials 30 --gpu cuda:0

# Wound dataset
python tools/run_optuna_tuning.py --config config/wound.yaml --n-trials 30 --gpu cuda:1
```

### Fast Tuning (Quick Iterations)

Reduce epochs per trial for faster exploration:

```bash
python tools/run_optuna_tuning.py --config config/default.yaml --n-trials 20 \
  --teacher-epochs 2 --student-epochs 5 --gpu cuda:0
```

### Full Search Space

Include fusion modules, loss functions, and student backbones:

```bash
python tools/run_optuna_tuning.py --config config/default.yaml --n-trials 50 \
  --tune-fusion --tune-loss --tune-backbones --gpu cuda:0
```

### Resume Existing Study

```bash
python tools/run_optuna_tuning.py --config config/default.yaml --n-trials 20 \
  --study-name medpix_tuning_20251215_120000
```

## Search Space

### Always Tuned
- **Learning rates**: `teacher_lr`, `student_lr` (log scale)
- **Loss weights**: `alpha`, `beta` (CE/KL distillation weights)
- **Temperature**: `T` (distillation temperature)
- **Architecture**: `fusion_dim`, `fusion_layers`, `fusion_heads`
- **Regularization**: `dropout`

### Optional (with flags)
- **Fusion type** (`--tune-fusion`): simple, concat_mlp, cross_attention, gated
- **Loss type** (`--tune-loss`): vanilla, combined, crd, rkd
- **Student backbones** (`--tune-backbones`): vision (deit-tiny/small), text (distilbert/minilm)

## Output

Results are saved in `logs/optuna/<study_name>/`:

- `best_config.yaml` — Best hyperparameters as a runnable config
- `study_summary.json` — Study statistics and best trial info
- `param_importance.html` — Hyperparameter importance plot (if plotly installed)
- `optimization_history.html` — Optimization progress over trials

### Using Best Config

```bash
# Train with optimized hyperparameters
python experiments/run.py logs/optuna/medpix_tuning_20251215_120000/best_config.yaml
```

## Pruning

The script uses **median pruning** to stop unpromising trials early:
- Trials are evaluated after each student epoch
- If dev F1 is below the median of recent trials, the trial is pruned
- This saves significant compute time

## Tips

### For Quick Exploration (30 min - 1 hour)
```bash
python tools/run_optuna_tuning.py --config config/default.yaml \
  --n-trials 15 --teacher-epochs 2 --student-epochs 5
```

### For Production (several hours)
```bash
python tools/run_optuna_tuning.py --config config/default.yaml \
  --n-trials 50 --tune-fusion --tune-loss
```

### For Ultra-Edge Students
Start with a base ultra-edge config and tune only the key hyperparameters:
```bash
python tools/run_optuna_tuning.py --config config/ultra-edge/medpix-deit_small-minilm.yaml \
  --n-trials 30 --gpu cuda:0
```

## Advanced Options

### Custom Database Storage
```bash
# Use PostgreSQL or MySQL for distributed tuning
python tools/run_optuna_tuning.py --config config/default.yaml \
  --storage postgresql://user:pass@host/dbname --n-trials 100
```

### Random Sampling (Baseline)
```bash
python tools/run_optuna_tuning.py --config config/default.yaml \
  --sampler random --n-trials 30
```

### No Pruning (Run All Trials to Completion)
```bash
python tools/run_optuna_tuning.py --config config/default.yaml \
  --pruner none --n-trials 20
```

## Monitoring Progress

While tuning is running:

```bash
# Check study database
sqlite3 logs/optuna/medpix_tuning_20251215_120000.db \
  "SELECT number, value, state FROM trials ORDER BY number DESC LIMIT 10;"

# Monitor GPU usage
watch -n 2 nvidia-smi

# Check trial logs
ls -lh logs/default_optuna_trial_*/
```

## Example Output

```
================================================================================
OPTUNA HYPERPARAMETER TUNING
================================================================================
Study name: medpix_tuning_20251215_143022
Base config: config/default.yaml
Number of trials: 30
Storage: sqlite:///logs/optuna/medpix_tuning_20251215_143022.db
GPU: cuda:0

Search space:
  • Learning rates, loss weights, architecture dims (always)
  • Fusion type: False
  • Loss type: False
  • Student backbones: False
================================================================================

[I 2025-12-15 14:30:25,123] Trial 0 finished with value: 0.8723
[I 2025-12-15 14:35:42,456] Trial 1 pruned.
[I 2025-12-15 14:37:18,789] Trial 2 finished with value: 0.8891
...

================================================================================
OPTIMIZATION COMPLETE
================================================================================
Total trials: 30
Complete: 24
Pruned: 6

Best trial: #14
Best dev F1: 0.9124

Best hyperparameters:
  alpha                     = 1.24
  beta                      = 127.5
  student_dropout           = 0.18
  student_fusion_dim        = 384
  student_fusion_heads      = 8
  student_fusion_layers     = 2
  student_lr                = 0.00032
  T                         = 3.8
  teacher_dropout           = 0.12
  teacher_fusion_dim        = 512
  teacher_fusion_heads      = 8
  teacher_fusion_layers     = 2
  teacher_lr                = 3.2e-05

Best config saved: logs/optuna/medpix_tuning_20251215_143022/best_config.yaml
Study summary saved: logs/optuna/medpix_tuning_20251215_143022/study_summary.json
================================================================================
```

## Comparison with Manual Tuning

| Method | Time | Coverage | Best F1 |
|--------|------|----------|---------|
| Manual grid search | Days | Limited | 0.89 |
| Random search | Hours | Good | 0.90 |
| Optuna (this script) | 2-4 hours | Excellent | 0.91+ |

Optuna intelligently explores the search space using Tree-structured Parzen Estimators (TPE) and prunes unpromising trials early.
