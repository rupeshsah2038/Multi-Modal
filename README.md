# Medpix_kd_modular

Light-weight repository for two-stage multimodal model training + KD (Teacher→Student) on the MedPix dataset.

This repo implements a vision+text Teacher network and a smaller Student distilled from it. The experiment entrypoint and training loop are intentionally compact so you can iterate quickly on model/backbone choices, fusion modules and distillation losses.

## Quickstart (run an experiment)

1. Install dependencies:

```fish
pip install -r requirements.txt
```

2. Run the default experiment (uses `config/default.yaml`):

```fish
python experiments/run.py config/default.yaml
```

Outputs (check `config/default.yaml.logging.log_dir`) will include model checkpoints (`student_best.pth`, `student_final.pth`), `metrics.csv`, `metrics.json` and confusion matrix `.npy` files.

## Key files & architecture

- `experiments/run.py`: tiny wrapper that loads a config and calls `trainer/engine.main(cfg)`.
- `trainer/engine.py`: main orchestration — builds tokenizers, datasets, teacher + student models, picks loss (see `config['loss']['type']`), runs teacher training and distillation loop.
- `data/dataset.py`: dataset class — expects JSONL splits + images, produces both teacher/student tokenized inputs.
- `models/`: model backbones, fusion modules, and final classification heads.
- `losses/`: distillation and auxiliary loss implementations (vanilla, combined, crd, rkd, mmd).
- `utils/metrics.py`, `utils/logger.py`: evaluation, per-split metrics, and saved logs.

## Config / important options

- `config/default.yaml` contains dataset location, batch size, model choices and training hyperparameters.
- New: `config['loss']['type']` lets you select which loss implementation to use at runtime. Supported keys: `vanilla`, `combined`, `crd`, `rkd`, `mmd`.
- `trainer/engine` will introspect the selected loss class and forward supported keys from `cfg['training']` (e.g., alpha, beta, gamma, T) to the loss constructor.

## Data layout (expected)

Following items are expected under `config.data.root` (default: `Medpix-2-0`):

- `splitted_dataset/data_{split}.jsonl` (train/dev/test item metadata)
- `splitted_dataset/descriptions_{split}.jsonl` (descriptions mapping)
- `images/{image_name}.png` (preprocessed image files)

If images are missing, `MedPixDataset` will raise `FileNotFoundError` — verify `config.data.root` points to the correct folder.

## Project-specific gotchas & debugging

- Two tokenizers are used simultaneously in each dataset example: `input_ids_teacher` / `attention_mask_teacher` and `input_ids_student` / `attention_mask_student`. Keep both tokenizers in sync when changing text preprocessing.
- Model output naming: most losses expect teacher outputs to include `img_raw`/`txt_raw` and student outputs `img_proj`/`txt_proj`. There is a naming mismatch in `losses/vanilla.py` which uses `img_emb`/`txt_emb` — switching `loss.type` to `vanilla` may need either adjusting the loss to read `img_raw`/`img_proj` consistently or updating models to include the keys `img_emb`/`txt_emb`.
- Transformer / Hugging Face downloads may fail in offline environments; prefer pre-populating the HF cache if needed.

## Fast iteration & tips

- For quick experiments: lower `data.batch_size`, `training.teacher_epochs` and `training.student_epochs` in `config/default.yaml`.
- Use `utils/metrics.evaluate_detailed()` for per-split metrics and inspect confusion matrices saved into the log directory.

## Next steps you might ask me to do

- Fix the `losses/vanilla.py` naming mismatch so it matches other loss implementations.
- Add unit tests / a small synthetic data runner to validate each loss and model forward pass without requiring full dataset or GPUs.

If you want, I can implement one of those next — tell me which and I’ll follow up with code + tests.
