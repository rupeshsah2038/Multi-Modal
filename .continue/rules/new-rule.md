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
conda activate fedenv
python experiments/run.py config/default.yaml
```

- Run multiple backbone-swap experiments in batch:

```bash
conda activate fedenv
python tools/batch_runs.py --base config/default.yaml --runs original,swap_vision,swap_text,swap_both \
  --execute --epochs 8 --batch-size 16 --device cuda:0
```

- Use `config/default.yaml` to change backbones, batch size, epochs, device, and `logging.log_dir`.

## Key files to modify or inspect
- Orchestration: `trainer/engine.py` (training loops, saving checkpoints, device config, epoch-level logging)
- Experiment wrapper: `experiments/run.py`
- Batch runner: `tools/batch_runs.py` (generates per-run configs and optionally executes backbone swap experiments)
- Data: `data/dataset.py` (expects `splitted_dataset/data_{split}.jsonl` and `descriptions_{split}.jsonl` + `images/` folder; includes defensive token-id validation)
- Models: `models/teacher.py`, `models/student.py`, `models/backbones.py` (includes `TEXT_PRETRAINED` and `VISION_PRETRAINED` mappings), `models/fusion/*`
- Losses: `losses/` (vanilla, combined, crd, rkd, mmd — all use lazy projections)
- Evaluation + logging: `utils/metrics.py`, `utils/logger.py` (MetricsLogger), `utils/results_logger.py` (ResultsLogger)

## Project-specific conventions & gotchas
- **Tokenizers match backbones:** Tokenizers are now loaded from the pretrained HF identifiers matching the configured text backbones. See `models/backbones.py` for mappings (`TEXT_PRETRAINED` dict). When swapping a text backbone in config, the tokenizer automatically changes to match (e.g., swapping from `bio-clinical-bert` to `distilbert` updates both model and tokenizer). Both `input_ids_teacher` / `input_ids_student` are validated to ensure token ids fit within the respective tokenizer vocab sizes (see `data/dataset.py`).
- **Feature naming (tensor keys):** Teacher outputs `img_raw` / `txt_raw` (backbone features) and `img_proj` / `txt_proj` (projected features). Student outputs preserve the same convention: `img_raw` / `txt_raw` are backbone outputs, `img_proj` / `txt_proj` are fused/projected features. All losses now accept a dict-based interface and create teacher→student projection layers lazily (at runtime) to support backbone shape mismatches (e.g., when vision or text backbones differ).
- **Loss module refactor:** All losses (`vanilla`, `combined`, `crd`, `rkd`, `mmd`) now use lazy projection creation (`nn.Linear(in_dim, out_dim)` created on first call with actual tensor shapes). This allows safe backbone swaps without hard-coding projection dimensions. The engine will instantiate the loss from `config['loss']['type']` and forward relevant training hyperparameters if they match the loss constructor signature.
- **Metrics & results logging:** `trainer/engine.py` now logs epoch-level metrics via `MetricsLogger` (outputs `metrics.csv`, `metrics.json`, confusion matrices). A new `ResultsLogger` (in `utils/results_logger.py`) writes a comprehensive `results.json` file with experiment config, train/dev/test metrics, and the full per-epoch training history. Both are saved to `cfg['logging']['log_dir']`.

## Debugging & common errors
- Missing images: `MedPixDataset` raises `FileNotFoundError` if expected image is missing — verify `config.data.root` points to the correct dataset and image filenames.
- Token-id out of range: `MedPixDataset` validates `input_ids.max() < tokenizer.vocab_size` and raises a clear error if mismatched. This usually means tokenizer and model are not aligned — check that configured text backbones map to the correct pretrained identifiers in `models/backbones.py`.
- Transformer / HF model download errors: ensure network access for initial download or pre-download models into cache.
- Mismatched tensor keys: verify your `forward()` returns the required dict keys (see `models/*` and `losses/*`) when implementing new modules.
- Device/GPU out-of-memory: Use `config['device']` to select a less-busy GPU (e.g., `cuda:3` instead of default `cuda:4`), reduce `batch_size`, or enable gradient accumulation.

## Testing & iteration tips
- Smaller, faster experiments: reduce `data.batch_size` and `training.teacher_epochs` / `student_epochs` in `config/default.yaml` for quicker iterations.
- Local validation: use `utils/metrics.evaluate_detailed()` (printed metrics and saved confusion matrices) to inspect model behavior per split.

## Supported backbones and swaps
Vision backbones (see `models/backbones.py` `VISION_PRETRAINED`):
- `vit-large` → `google/vit-large-patch16-224`
- `vit-base` → `google/vit-base-patch16-224`
- `deit-base` → `facebook/deit-base-distilled-patch16-224`

Text backbones (see `models/backbones.py` `TEXT_PRETRAINED`):
- `bio-clinical-bert` → `emilyalsentzer/Bio_ClinicalBERT`
- `distilbert` → `distilbert-base-uncased`

To run backbone-swap experiments, use `tools/batch_runs.py`:
```bash
python tools/batch_runs.py --base config/default.yaml --runs original,swap_vision,swap_text,swap_both \
  --execute --epochs 8 --batch-size 16 --device cuda:3
```

This generates per-run configs in `logs/run_<name>/config.yaml` and saves outputs (metrics, results, models) per run.
