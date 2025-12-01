# Copilot / AI agent guidance — medpix_kd_modular

Short, actionable guide for editing this repository. Focus on concrete code paths, conventions, and reproducible dev workflows.

## Big picture
- Two-stage multimodal training: a large Teacher model (vision + text) is trained first, then a smaller Student is distilled from the teacher.
- Main orchestration lives in `trainer/engine.py` (also callable via `experiments/run.py`). Configs live in `config/default.yaml`.
- Data flows: `data/dataset.py` -> DataLoader -> training loops in `trainer/engine.py`.

## Entry points & how to run
- Install deps: `pip install -r requirements.txt` (Hugging Face models are downloaded at first run).
- Run a train+distill experiment (uses config/default.yaml by default):

```bash
python experiments/run.py config/default.yaml
```

- Use `config/default.yaml` to change backbones, batch size, epochs, and `logging.log_dir`.

## Key files to modify or inspect
- Orchestration: `trainer/engine.py` (training loops, saving checkpoints)
- Experiment wrapper: `experiments/run.py`
- Data: `data/dataset.py` (expects `splitted_dataset/data_{split}.jsonl` and `descriptions_{split}.jsonl` + `images/` folder)
- Models: `models/teacher.py`, `models/student.py`, `models/backbones.py`, `models/fusion/*`
- Losses: `losses/` (vanilla, combined, crd, rkd, mmd)
- Evaluation + logging: `utils/metrics.py`, `utils/logger.py`

## Project-specific conventions & gotchas
- Two tokenizers are used simultaneously: `input_ids_teacher` / `attention_mask_teacher` and `input_ids_student` / `attention_mask_student` (see `data/dataset.py`). When changing tokenizers, update both.
- Feature naming: most code expects teacher outputs with `img_raw` / `txt_raw` and student outputs with `img_proj` / `txt_proj`. Many loss implementations (e.g. `losses/combined.py`) follow this convention.
- Warning: `losses/vanilla.py` references `img_emb` / `txt_emb` — this name differs from the rest of the project and is a likely source of runtime bugs. Prefer the `combined.py` convention when adding or adapting losses.
- You can now pick the distillation/loss implementation from `config['loss']['type']` (supported: `vanilla`, `combined`, `crd`, `rkd`, `mmd`). The engine will load the matching class at runtime and pass relevant `training` keys (e.g. `alpha`, `beta`, `gamma`, `T`) if they match the loss constructor.

- Warning: `losses/vanilla.py` references `img_emb` / `txt_emb` — this name differs from the rest of the project and is a likely source of runtime bugs. Prefer the `combined.py` convention when using a loss that expects `img_raw`/`txt_raw` vs `img_proj`/`txt_proj`.

## Debugging & common errors
- Missing images: `MedPixDataset` raises `FileNotFoundError` if expected image is missing — verify `config.data.root` points to the correct dataset and image filenames.
- Transformer / HF model download errors: ensure network access for initial download or pre-download models into cache.
- Mismatched tensor keys: verify your `forward()` returns the required dict keys (see `models/*` and `losses/*`) when implementing new modules.

## Testing & iteration tips
- Smaller, faster experiments: reduce `data.batch_size` and `training.teacher_epochs` / `student_epochs` in `config/default.yaml` for quicker iterations.
- Local validation: use `utils/metrics.evaluate_detailed()` (printed metrics and saved confusion matrices) to inspect model behavior per split.

If any of these areas are unclear or you want me to expand examples (e.g. a minimal unit test or CI job), tell me which part to iterate on next.
