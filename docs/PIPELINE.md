# Training & Distillation Pipeline

This document summarizes the end-to-end pipeline used in this repository: from config loading, through data and model setup, to teacher training, student distillation, and logging.

---

## 1. Entry Points & Configuration

- Main script: `experiments/run.py`
- Usage:
  - MedPix: `python experiments/run.py config/default.yaml`
  - Wound: `python experiments/run.py config/wound.yaml`
- Steps:
  1. Load YAML config into a Python dict.
  2. Call `trainer.engine.main(cfg)` with the config.

---

## 2. Tokenizers & Backbones

**Files:**
- `trainer/engine.py`
- `models/backbones.py`

**Flow:**
1. Read backbone names from `cfg['teacher']` and `cfg['student']`.
2. Map text backbones to Hugging Face IDs using `get_text_pretrained_name`.
3. Instantiate tokenizers:
   - `teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_text_pretrained)`
   - `student_tokenizer = AutoTokenizer.from_pretrained(student_text_pretrained)`
4. These tokenizers are passed into the dataset factory so each batch contains:
   - `input_ids_teacher`, `attention_mask_teacher`
   - `input_ids_student`, `attention_mask_student`

---

## 3. Dataset Factory & Splits

**Files:**
- `data/dataset.py`
- `data/wound_dataset.py`

**Dataset selection:**
- `cfg['data'].get('type', 'medpix')` → one of:
  - `"medpix"` (default)
  - `"wound"`

**MedPix-2-0:**
- Input:
  - JSONL: `splitted_dataset/data_{split}.jsonl`
  - Text: `splitted_dataset/descriptions_{split}.jsonl`
  - Images: `images/`
- Unified labels:
  - `modality` (task 1)
  - `location` (task 2)

**Wound-1-0:**
- Input:
  - CSV: `metadata_{split}.csv`
  - Images: `images/`
- Columns:
  - `type`, `severity`, `description`, `file_path` (or configured equivalents)
- Mapped to unified labels:
  - `modality` ← `type`
  - `location` ← `severity`

**Splits:**
- `train_dataset = make_dataset("train")`
- `dev_dataset   = make_dataset("dev")`
- `test_dataset  = make_dataset("test")`

---

## 4. Dynamic Class Discovery & Dataloaders

**Files:**
- `data/dataset.py`
- `trainer/engine.py`

1. Call `get_num_classes(...)`:
   - Returns `{"modality": num_mod, "location": num_loc}` for the selected dataset.
2. Use these counts to configure classification heads:
   - `num_modality_classes`
   - `num_location_classes`
3. Wrap datasets into PyTorch `DataLoader`s:
   - `train_loader` (shuffled)
   - `dev_loader`   (no shuffle)
   - `test_loader`  (no shuffle)
4. Batch size and workers come from `cfg['data']` (e.g. `batch_size`, `num_workers`).

---

## 5. Model Construction (Teacher & Student)

**Files:**
- `models/teacher.py`
- `models/student.py`
- `models/backbones.py`
- `models/fusion/*`

**Teacher:**
- Vision backbone (e.g., ViT large).
- Text backbone (e.g., Bio_ClinicalBERT).
- Fusion module (from `cfg['fusion']['type']`, implemented under `models/fusion/`).
- Two task heads:
  - Modality / Type
  - Location / Severity

**Student:**
- Smaller vision backbone (e.g., ViT base).
- Smaller text backbone (e.g., DistilBERT).
- Same fusion interface and heads (matching class counts).

Both models expose a consistent interface returning:
- `img_raw`, `txt_raw` (backbone features)
- `img_proj`, `txt_proj` (projected / fused features)
- `logits_modality`, `logits_location` (task outputs)

---

## 6. Loss / Distillation Module

**Files:**
- `losses/vanilla.py`
- `losses/combined.py`
- `losses/crd.py`, `losses/rkd.py`, `losses/mmd.py`
- `trainer/engine.py` (`_make_loss_from_cfg`)

**Config-driven selection:**
- `cfg['loss']['type']` ∈ {`"vanilla"`, `"combined"`, `"crd"`, `"rkd"`, `"mmd"`}.
- Dynamically import the right class and instantiate it with:
  - Selected training hyperparameters (`alpha`, `beta`, `T`, etc.).
  - `fusion_dim` (from `loss`, `student`, or `teacher` section).

**Distillation call signature:**
- `loss = distill_fn(student_outputs, teacher_outputs, y_modality, y_location)`

Depending on loss type, this may combine:
- CE on student logits vs. labels.
- KL divergence between student and teacher logits.
- Feature-level losses over raw / projected features.
- Contrastive / relational / distribution matching terms (CRD, RKD, MMD).

---

## 7. Training Teacher (Stage 1)

**Function:** `train_teacher(model, loader, device, epochs, lr)` in `trainer/engine.py`.

Loop:
1. Put teacher in `train()` mode.
2. For each epoch:
   - For each batch:
     - Move `pixel_values`, teacher tokens, and labels to device.
     - Forward: `out = teacher(pv, ids_teacher, mask_teacher)`.
     - Compute loss: CE over `logits_modality` and `logits_location`.
     - Backprop + `AdamW` step.
3. Print average loss per epoch.
4. Return the trained teacher model.

No distillation happens here; this is pure supervised training.

---

## 8. Distilling Student (Stage 2)

**Function:** `train_student(student, teacher, loader, device, epochs, lr, distill_fn)`.

Setup:
- Freeze teacher weights (`teacher.eval()` and `torch.no_grad()` for teacher forward).
- Train student with KD + supervised components.

Per student epoch (called from `main`):
1. Iterate once over `train_loader`:
   - Forward teacher: `t_out = teacher(pv, ids_teacher, mask_teacher)` (no grad).
   - Forward student: `s_out = student(pv, ids_student, mask_student)`.
   - Compute KD loss: `loss = distill_fn(s_out, t_out, y_mod, y_loc)`.
   - Backprop and update student parameters.
2. Return student and average training loss for that epoch.

In `main`, this 1-epoch loop is repeated `student_epochs` times, with dev evaluation after each.

---

## 9. Evaluation, Model Selection & Testing

**Files:**
- `utils/metrics.py` (`evaluate_detailed`)
- `utils/logger.py` (`MetricsLogger`)
- `utils/results_logger.py` (`ResultsLogger`)
- `trainer/engine.py`

### Dev Evaluation & Best Checkpoint

After each student epoch:
1. Run `evaluate_detailed(student, dev_loader, device, ...)`.
2. Compute a scalar dev score:
   - Mean of task1 F1 and task2 F1 (e.g., modality F1 + location F1 / 2).
3. If dev score improves:
   - Save `student_best.pth` under `cfg['logging']['log_dir']`.
4. Log per-epoch metrics (dev metrics + train loss) via `MetricsLogger.log_epoch`.

### Final Test

After training completes:
1. Reload `student_best.pth` (if it exists).
2. Evaluate on `test_loader` with `evaluate_detailed`.
3. Save final student weights as `student_final.pth`.

---

## 10. Logging & Results Artifacts

**Per-run log directory:**
- Example: `logs/fusion-explore/medpix-simple-combined/`

**Artifacts:**
- Checkpoints:
  - `student_best.pth`
  - `student_final.pth`
- Metrics:
  - `metrics.csv` (per-epoch)
  - `metrics.json`
- Aggregated experiment results:
  - `results.json` via `ResultsLogger.log_experiment`:
    - Config snapshot
    - Train/dev history (from `MetricsLogger`)
    - Final dev/test metrics
    - Model / dataset / fusion / loss metadata

These structured results are what the ULTRA_EDGE, LOSS_EXPLORE, and FUSION_EXPLORE reports read from.

---

## 11. Modality-Aware Sketch

Below is a conceptual sketch of how image and text flows through teacher and student during distillation.

```text
          ┌─────────────────────────────────────────────────────────┐
          │                       INPUTS                            │
          │   Image (CT/MR or wound photo)  +  Text (description)  │
          └───────────────┬────────────────────────────────────────┘
                          │
        ┌─────────────────┴─────────────────┐
        │                                   │
        ▼                                   ▼
┌────────────────┐                 ┌────────────────┐
│ Teacher Branch │                 │ Student Branch │
└────────────────┘                 └────────────────┘
        │                                   │
        │  (frozen / trained with CE)       │  (trained with KD + CE)
        │                                   │
   ┌────▼─────┐                       ┌────▼─────┐
   │ Vision T │  image → img_raw_T    │ Vision S │  image → img_raw_S
   └──────────┘                       └──────────┘
   ┌──────────┐                       ┌──────────┐
   │ Text T   │  text  → txt_raw_T    │ Text S   │  text  → txt_raw_S
   └──────────┘                       └──────────┘
        │                                   │
        │ (backbone features)               │
        ▼                                   ▼
   ┌──────────────┐                   ┌──────────────┐
   │ Fusion T     │  (e.g. simple,    │ Fusion S     │  same type as T
   │ (cfg.fusion) │   cross_attn, …)  │ (cfg.fusion) │
   └────┬─────────┘                   └────┬─────────┘
        │ img_proj_T, txt_proj_T          │ img_proj_S, txt_proj_S
        ▼                                 ▼
   ┌──────────────┐                   ┌──────────────┐
   │ Heads T      │                   │ Heads S      │
   │ (2 tasks)    │                   │ (2 tasks)    │
   └────┬────┬────┘                   └────┬────┬────┘
        │    │                              │    │
        │    │ logits_T_modality            │    │ logits_S_modality
        │    └ logits_T_location            │    └ logits_S_location
        │                                   │
        └───────────────┬───────────────────┘
                        │
                        ▼
              ┌──────────────────────────────┐
              │       LOSS / DISTILLATION    │
              │  (from cfg['loss']['type'])  │
              ├──────────────────────────────┤
              │ • CE(student_logits, labels) │
              │ • KL(student vs teacher)     │
              │ • Feature losses (img/txt)   │
              │ • Optional CRD / RKD / MMD  │
              └──────────────────────────────┘
                        │
                        ▼
                 Student parameter update
```

**Stage ordering:**
1. Train teacher with CE only (left branch).
2. Freeze teacher, train student with KD + CE (both branches).
3. Track best dev F1 and evaluate the best student on test.
