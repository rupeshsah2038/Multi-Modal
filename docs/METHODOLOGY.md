# Methodology

## Overview
- Objective: Distill a compact Student multimodal model from a larger Teacher using a two-stage ultra-edge experimental protocol (Teacher pretraining → Student distillation). Evaluate `ultra-edge-*` families across input resolutions, backbone sizes, and hyperparameter regimes.
- Scope: experiments driven by YAML configs under `config/` (e.g., `ultra-edge-base-256`, `ultra-edge-base-384`, `ultra-edge-hp-tuned-all`, `ultra-edge-tuned-hp`, `ultra-edge2`, `ultra-edge2-tuned-hp`).

## System Architecture

- Compute: Linux servers (single or multi-GPU). Device selected by `config['device']` (e.g., `cuda:0`). Report CPU, GPU model(s), RAM, and storage for each run.
- Software stack: Python environment with dependencies from `requirements.txt`; PyTorch and Hugging Face transformers for backbones and tokenizers. Experiments launched via `python experiments/run.py config/<config>.yaml`.
- Orchestration: `experiments/run.py` calls `trainer/engine.py` which loads data, constructs Teacher/Student models, runs teacher pretraining and subsequently student distillation. Batch runs and backbone-swap grids are handled by `tools/batch_runs.py`.
- Data flow: `data/dataset.py` creates dataset objects; PyTorch DataLoaders feed `trainer/engine.py` training loops. Model outputs include `img_raw`/`txt_raw` and projected `img_proj`/`txt_proj` used by losses and fusion heads.
- Models: modular components from `models/` — `teacher.py`, `student.py`, `backbones.py`, and `models/fusion/`. Tokenizers and pre-trained checkpoint identifiers are defined in `models/backbones.py`.
- Logging & storage: checkpoints, metrics, and artifacts written to `logging.log_dir` and `logs/`. `utils/logger.py`, `utils/results_logger.py`, and `utils/metrics.py` manage experiment artifacts and evaluation outputs.

## Datasets and Splits

- Supported datasets: MedPix-2-0 (default) and Wound-1-0. Dataset selection via `data.type` in the config YAML.
- File layout:
  - MedPix: `datasets/MedPix-2-0/` with `splitted_dataset/data_{split}.jsonl`, `descriptions_{split}.jsonl`, and `images/`.
  - Wound: `datasets/Wound-1-0/` with `metadata_{split}.csv` and `images/` (split with `tools/split_wound_dataset.py`).
- Splitting: Train / Dev (validation) / Test splits as defined in config. Ensure consistent label sets across splits for Wound.
- Dynamic class counts: `get_num_classes()` in `data/dataset.py` derives modality/location class counts at load time; models adapt head sizes automatically.

## Preprocessing

- Image preprocessing: resize and/or center/crop to backbone input resolution (e.g., 224, 256, 384), normalize with backbone-specific means/stds, apply augmentation configured in the YAML (random flips, color jitter, crop, etc.).
- Text preprocessing: use the HF tokenizer that matches the configured text backbone (mapping in `models/backbones.py`). Validate token IDs at dataset load.
- Batching: batch size and gradient accumulation set via config to accommodate memory limits. Mixed precision (AMP) enabled when appropriate.

## Model Families and Architectures

- Teacher: a high-capacity multimodal model with the configured vision and text backbones. Produces raw and projected features used for classification and distillation.
- Student: edge-optimized multimodal model with lighter backbones or smaller projection/fusion layers; variant specifics encoded in `ultra-edge-*` configs (reduced fusion dim, smaller backbones, lower-resolution inputs).
- Backbones: vision (e.g., `vit-large`, `vit-base`, `deit-base`) and text (e.g., `bio-clinical-bert`, `distilbert`) chosen per YAML. Tokenizer/backbone pairs are loaded via `models/backbones.py`.
- Fusion: configurable fusion modules in `models/fusion/` with `fusion_dim` and fusion type set in YAML.

## Losses and Distillation Strategy

- Two-stage protocol:
  1. Teacher pretraining: standard supervised losses (cross-entropy) on task heads; optional auxiliary losses.
  2. Student distillation: combined supervised + distillation objectives. Distillation losses implemented in `losses/` include `vanilla` (CE+KL+feat), `combined` (adds contrastive CRD), `crd`, `rkd`, and `mmd`.
- Combined loss: when used, blends cross-entropy, softened-logit KL, feature MSE, and contrastive representation distillation (CRD). Loss weights (α, β, γ, etc.) are controlled in YAML and follow the implementations under `losses/`.
- Cross-attention: when cross-attention fusion is enabled in a config, the Student/Teacher fusion layers exchange modality-conditioned attention (vision↔text). Cross-attention operates on projected modality features (`img_proj`, `txt_proj`) and is trained end-to-end alongside the classification and distillation objectives. Cross-attention design choices (number of heads, projection dims, dropout) are set in the YAML fusion section.

## Training Procedure

- Teacher training: run for `training.teacher_epochs` using optimizer, LR schedule and weight decay from config. Save best checkpoints on validation.
- Student distillation: initialize Student (from scratch or small pretrained weights) and train for `training.student_epochs` using Teacher checkpoints as reference. Distillation uses logits and/or feature-based signals depending on the selected loss.
- Optimization: optimizers (Adam/AdamW), schedulers, warmup steps, and other optimizer hyperparameters are configured in the YAML. Use gradient accumulation and reduced batch sizes for memory-constrained experiments.
- Checkpointing & early stopping: validation metrics computed per epoch; best checkpoints retained per the primary metric specified in the config.

## Hyperparameter Search & Tuning

- Ultra-edge tuned variants: `ultra-edge-hp-tuned-all` and `ultra-edge2-tuned-hp` reflect hyperparameter tuning across learning rate, weight decay, projection/fusion dimensions, distillation loss weights, and image resolution.
- Tuning framework: manual grid or Optuna-based searches (see `docs/OPTUNA_TUNING.md`) orchestrated with `tools/batch_runs.py` for reproducible batches of experiments across seeds and backbones.

## Ultra-edge Experimental Variants

- Variant dimensions:
  - Input resolution: e.g., 256 vs 384
  - Backbone capacity: base vs small
  - Distillation losses: vanilla vs combined vs CRD/RKD/MMD
  - Fusion types: concatenation, cross-attention, or lightweight MLP fusion
  - Hyperparameter tuning: default vs tuned
- Backbone-swap experiments: run `tools/batch_runs.py` to swap vision/text backbones and measure robustness of fusion and distillation.

## Evaluation Protocol

- Metrics: accuracy, precision, recall, F1 (per-class and macro/micro) for each head (modality and location). Use `utils/metrics.py` for standardized computation.
- Artifacts: confusion matrices, per-split metrics, and model parameter summaries saved under `logs/` and aggregated into CSVs (`logs/aggregate_results.csv`, `logs/model_size_report.csv`).
- Statistical reporting: for tuned or comparative experiments, report mean±std across seeds; where applicable, perform paired tests for significance.
- Ablations: include targeted ablations on cross-attention vs non-attention fusion, loss component ablations (remove CRD / RKD / MMD), and input resolution/backbone size comparisons.

## Logging, Reproducibility and Reporting

- Experiment trace: save full YAML config, git commit hash, Python environment (`pip freeze` or `requirements.txt` snapshot), device details, and random seeds with each run.
- Logging: `utils/logger.py` and `utils/results_logger.py` record per-epoch metrics and checkpoints. Aggregation and plots are stored in `figures/` and `logs/`.
- Reproducibility checklist for manuscripts: include YAMLs for each `ultra-edge-*` variant, exact hardware/software spec, seeds, checkpoint links, and commands to reproduce runs.

## Recommended Run Commands

Install environment:
```bash
pip install -r requirements.txt
```

Single-run example:
```bash
python experiments/run.py config/default.yaml
```

Backbone-swap batch example:
```bash
python tools/batch_runs.py --base config/default.yaml --runs original,swap_vision,swap_text,swap_both --execute --epochs 8 --batch-size 16 --device cuda:0
```

Wound dataset split:
```bash
python tools/split_wound_dataset.py --input datasets/Wound-1-0/metadata.csv --output datasets/Wound-1-0
```

## Reporting Checklist (for paper)

- YAML configs for all `ultra-edge-*` variants (attach or link)
- Exact hardware/software spec (GPU model, CUDA & driver versions, OS, Python & major packages)
- Training regimen: epochs, batch sizes, LR schedules, optimizer, seeds
- Model sizes: parameter counts and storage size per variant
- Metrics: per-split tables, confusion matrices, and statistical summaries across seeds
- Ablations: fusion, loss components, input resolution, backbone swaps
- Links to logs/checkpoints in `logs/` and `figures/`

## Known Limitations & Notes

- Missing images cause `FileNotFoundError` — verify `config.data.root` and dataset paths.
- Tokenizer/backbone mismatch can cause token ID range errors — ensure the tokenizer matches the configured text backbone.
- Large backbones need careful memory planning (reduce batch size or use gradient accumulation). Contrastive and pairwise losses are O(B^2) and benefit from larger batch sizes.

---

If you want, I can now (A) extract and append the exact YAML hyperparameters from the `ultra-edge-*` configs to this doc, or (B) generate a compact manuscript-ready methods subsection trimmed to your target journal/conference style.
