# Methods

We conducted a two-stage multimodal distillation protocol to produce compact edge-ready Student models from higher-capacity Teachers. Experiments use the `ultra-edge-*` configuration families (see `config/`) and follow a reproducible, config-driven pipeline implemented in `experiments/run.py` and `trainer/engine.py`.

Datasets. We evaluate on MedPix-2-0 (default) and Wound-1-0. Data splits (train/dev/test) are prepared as repository artifacts; Wound splits can be created with `tools/split_wound_dataset.py`. Class counts are inferred at load time so model heads adapt automatically.

Preprocessing. Images are resized/cropped and normalized to the pretrained vision-backbone statistics; input resolution is set per-config (e.g., 256 or 384 px). Text is tokenized using the Hugging Face tokenizer matching the configured text backbone. Typical augmentations: random horizontal flip and color jitter; batch size and gradient-accumulation are tuned to fit device memory.

Architectures. The Teacher uses high-capacity vision and text backbones; the Student uses smaller backbones and reduced fusion/projection dimensions defined by `ultra-edge-*` configs. Models are implemented modularly under `models/` (`teacher.py`, `student.py`, `backbones.py`) and produce both raw features (`img_raw`, `txt_raw`) and projected features (`img_proj`, `txt_proj`). Fusion layers are configurable; cross-attention fusion (vision↔text) uses projected modality features and is parameterized by attention heads, projection dimensions and dropout in the YAML.

Losses and distillation. Training follows two stages: (1) Teacher pretraining with supervised cross-entropy, (2) Student distillation combining supervised loss with one or more distillation signals. We use the repository loss suite: vanilla (CE + temperature-scaled KL + feature MSE), combined (adds contrastive CRD to vanilla), CRD, RKD and MMD (see `losses/`). When configured, the Combined loss blends cross-entropy, softened-logit KL, feature matching (MSE) and contrastive representation distillation; weights (α, β, γ) and temperatures are set in the YAML.

Cross-attention. For experiments labeled with cross-attention fusion, the fusion module applies modality-conditioned cross-attention on `img_proj` and `txt_proj` so that visual features attend to textual features and vice versa. Cross-attention parameters (number of heads, per-head dim, dropout) are configured per-experiment and trained end-to-end together with classification and distillation objectives.

Training protocol. Teachers are trained for `training.teacher_epochs`; Students for `training.student_epochs`. Optimization uses Adam/AdamW with LR schedules, warmup, and weight decay as specified in configs. Mixed precision (AMP) and gradient accumulation are used when enabled. Best checkpoints are selected on validation metrics and saved under the configured `logging.log_dir`.

Hyperparameter tuning. `ultra-edge-hp-tuned-*` variants reflect systematic tuning across learning rate, weight decay, fusion/projection dimensions, distillation weights and input resolution. Tuning is performed via manual grid or Optuna-driven sweeps coordinated by `tools/batch_runs.py` and logged to `logs/optuna/` when used.

Evaluation. We report accuracy, precision, recall and F1 (per-class and aggregate) for each prediction head; confusion matrices and model-size summaries are produced for every run. For tuned experiments we report mean±SD across seeds and perform paired comparisons when applicable.

Reproducibility. Each run records the full YAML config, random seeds, git commit, environment packages and device details. Commands to reproduce typical runs:

```bash
pip install -r requirements.txt
python experiments/run.py config/default.yaml
```

For backbone-swap batches:

```bash
python tools/batch_runs.py --base config/default.yaml --runs original,swap_vision,swap_text,swap_both --execute --epochs 8 --batch-size 16 --device cuda:0
```

Notes. Confirm dataset root paths to avoid `FileNotFoundError`. Ensure the tokenizer matches the configured text backbone to prevent token-id range errors. Contrastive and pairwise losses incur O(B^2) complexity and therefore favour larger effective batch sizes or approximations.

The repository includes the full implementation and configuration YAMLs under `config/` for all `ultra-edge-*` variants; include those files in supplementary materials for exact reproducibility.
