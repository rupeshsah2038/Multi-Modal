# Copilot / AI agent guidance — Multi-Modal

Short, actionable guide for editing this repository. Focus on concrete code paths, conventions, and reproducible dev workflows.

## Big picture
- Two-stage multimodal training: a large Teacher model (vision + text) is trained first, then a smaller Student is distilled from the teacher.
- Supports multiple datasets: **MedPix-2-0** (medical imaging) and **Wound-1-0** (wound classification) with unified interface.
- Main orchestration lives in `trainer/engine.py` (also callable via `experiments/run.py`). Configs live in `config/`.
- Data flows: `data/dataset.py` -> DataLoader -> training loops in `trainer/engine.py`.

## Entry points & how to run
- Install deps: `pip install -r requirements.txt` (Hugging Face models are downloaded at first run).
- Run a train+distill experiment:

```bash
conda activate fedenv
# MedPix dataset
python experiments/run.py config/default.yaml

# Wound dataset
python experiments/run.py config/wound.yaml
```

- Run multiple backbone-swap experiments in batch:

```bash
conda activate fedenv
python tools/batch_runs.py --base config/default.yaml --runs original,swap_vision,swap_text,swap_both \
  --execute --epochs 8 --batch-size 16 --device cuda:0
```

- Use config files to change datasets, backbones, batch size, epochs, device, and `logging.log_dir`.

## Datasets

### MedPix-2-0 (default)
- Structure: `datasets/MedPix-2-0/` with `splitted_dataset/data_{split}.jsonl`, `descriptions_{split}.jsonl`, `images/`
- Tasks: CT/MR modality (2 classes), body location (5 classes)
- Config: `data.type: "medpix"` (or omit for backward compatibility)

### Wound-1-0
- Structure: `datasets/Wound-1-0/` with `metadata_{split}.csv` (columns: file_path, type, severity, description), `images/`
- Tasks: wound type (dynamic classes), severity (dynamic classes)
- Config: `data.type: "wound"`
- Split tool: `python tools/split_wound_dataset.py --input datasets/Wound-1-0/metadata.csv --output datasets/Wound-1-0`

## Key files to modify or inspect
- Orchestration: `trainer/engine.py` (training loops, dataset loading, dynamic class counts, device config)
- Experiment wrapper: `experiments/run.py`
- Batch runner: `tools/batch_runs.py` (generates per-run configs and optionally executes backbone swap experiments)
- Data: `data/dataset.py` (unified dataset factory with MedPixDataset and WoundDataset classes)
- Dataset splitter: `tools/split_wound_dataset.py` (splits wound metadata.csv into train/dev/test)
- Models: `models/teacher.py`, `models/student.py` (now accept `num_modality_classes`, `num_location_classes`), `models/backbones.py`
- Losses: `losses/` (vanilla, combined, crd, rkd, mmd — all use lazy projections)
- Evaluation + logging: `utils/metrics.py`, `utils/logger.py` (MetricsLogger), `utils/results_logger.py` (ResultsLogger)

## Project-specific conventions & gotchas
- **Dataset switching:** Set `data.type` to `"medpix"` or `"wound"` in config. Dataset type is auto-detected and class counts are dynamically determined.
- **Dynamic class counts:** Models and heads adapt to dataset — no hardcoded class numbers. Wound dataset inspects CSV to determine number of type/severity classes.
- **Unified task mapping:** Both datasets use `modality`/`location` keys. For Wound: type→modality, severity→location.
- **Tokenizers match backbones:** Tokenizers are loaded from pretrained HF identifiers matching configured text backbones (see `models/backbones.py`).
- **Feature naming:** Teacher/Student output `img_raw`/`txt_raw` (backbone features) and `img_proj`/`txt_proj` (projected features).
- **Loss module refactor:** All losses use lazy projection creation to support backbone swaps. Engine instantiates loss from `config['loss']['type']`.
- **No hardcoded defaults:** All parameters (fusion_dim, class counts, etc.) are explicitly configured via YAML.

## Debugging & common errors
- Missing images: Datasets raise `FileNotFoundError` if expected image is missing — verify `config.data.root` and image paths.
- Token-id out of range: Dataset validates token IDs fit vocab size. Ensure tokenizer matches text backbone in `models/backbones.py`.
- Wrong dataset type: Set `data.type` correctly (`medpix` or `wound`) in config.
- Missing split files: For Wound dataset, run `tools/split_wound_dataset.py` first to create metadata_{train,dev,test}.csv.
- Class count mismatch: For Wound dataset, ensure all splits use consistent type/severity labels.
- Device/GPU out-of-memory: Use `config['device']` to select GPU, reduce `batch_size`, or enable gradient accumulation.

## Testing & iteration tips
- Smaller experiments: reduce `data.batch_size` and `training.teacher_epochs`/`student_epochs` for quicker iterations.
- Quick test: use `config/test-1epoch.yaml` for 1-epoch mock runs.
- Local validation: `utils/metrics.evaluate_detailed()` prints metrics and saves confusion matrices per split.
- Dataset inspection: Use `get_num_classes()` in `data/dataset.py` to verify class counts before training.

## Supported backbones and swaps
Vision backbones (see `models/backbones.py` `VISION_PRETRAINED`):
- `vit-large` → `google/vit-large-patch16-224`
- `vit-base` → `google/vit-base-patch16-224`
- `deit-base` → `facebook/deit-base-distilled-patch16-224`

Text backbones (see `models/backbones.py` `TEXT_PRETRAINED`):
- `bio-clinical-bert` → `emilyalsentzer/Bio_ClinicalBERT`
- `distilbert` → `distilbert-base-uncased`

To run backbone-swap experiments:
```bash
python tools/batch_runs.py --base config/default.yaml --runs original,swap_vision,swap_text,swap_both \
  --execute --epochs 8 --batch-size 16 --device cuda:3
```
