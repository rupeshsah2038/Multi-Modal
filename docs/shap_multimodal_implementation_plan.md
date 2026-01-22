# SHAP Multimodal Explainer — Implementation Plan

Goal
- Add a reproducible SHAP-based explainer for Image+Text models in this repo that uses saved checkpoints (no retraining).

Scope
- Target models: Teacher/Student multimodal models that accept image tensors and tokenized text (the repo's `Teacher`/`Student`).
- Explainer: gradient-based SHAP (`shap.GradientExplainer`) operating on image tensors and text embeddings.
- Outputs: per-sample image heatmaps (PNG), token importance plots (PNG), and a JSON summary with per-modality SHAP values.

Prerequisites
- Saved checkpoints available under `logs/<run>/student_best.pth` or `student_final.pth` (and teacher if desired).
- Matching tokenizers accessible via `models.backbones.get_text_pretrained_name()` and `transformers.AutoTokenizer`.
- Add `shap` to `requirements.txt` and (optionally) `plotly` for interactive plots.

Design overview
1. Use a small background dataset (N=20–100) sampled from `dev` or `test` to initialize SHAP explainer baseline.
2. Expose numeric inputs for both modalities:
   - Images: normalized `pixel_values` tensors (float) with `requires_grad=True`.
   - Text: obtain token embeddings from the model's embedding layer (or via a small wrapper) and set `requires_grad=True`; pass embeddings into the model forward so SHAP can compute gradients w.r.t embeddings.
3. Wrap model into a `predict_fn((images, embeddings)) -> probs` returning class probabilities (numpy) for target task(s).
4. Create `shap.GradientExplainer(predict_fn, background)` and compute `shap_values = explainer.shap_values(samples)`.
5. Postprocess SHAP outputs:
   - Image: aggregate per-channel SHAP to 2D heatmap, optionally upsample to original image size, overlay on image and save.
   - Text: aggregate SHAP per token (sum over embedding dims) and map to token strings, save importance bar chart and JSON mapping.

Files to add
- `tools/shap_multimodal.py` — CLI script implementing the pipeline (arguments: `--run-log`, `--checkpoint`, `--device`, `--n-background`, `--n-samples`, `--out-dir`, `--task`).
- `docs/shap_multimodal_implementation_plan.md` — this file (saved).
- `README.md` snippet referencing the new tool and usage examples.

CLI usage (example)
```
python tools/shap_multimodal.py \
  --run logs/ultra-edge-hp-tuned-all/wound-mobilevit_xx_small-bert-mini \
  --checkpoint student_best.pth \
  --device cuda:0 \
  --n-background 40 \
  --n-samples 5 \
  --out plots/shap_run1
```

Implementation steps (detailed)
1. Add dependency
   - Add `shap` to `requirements.txt`.
2. Implement loader & wrapper
   - Load tokenizers and instantiate model class with same config used at training.
   - Load checkpoint state dict and `model.eval()`.
   - Provide a function `get_background(run_dir, n)` sampling `n` examples returning `(images_tensor, token_embeddings, token_strings)`.
3. Expose text embeddings
   - Option A (preferred): call model's embedding layer directly (e.g., `model.text.get_input_embeddings()(input_ids)`) to obtain embeddings and forward through a modified model forward that accepts embeddings.
   - Option B: register a forward hook on embedding layer to capture inputs and use them for SHAP (more brittle).
4. SHAP explainer
   - Build `predict_fn` that accepts concatenated modality inputs and returns softmax probabilities.
   - Create `shap.GradientExplainer(predict_fn, background)` on GPU if supported.
   - Explain target samples and collect `shap_values`.
5. Visualization
   - Image heatmap: sum absolute SHAP across channels, normalize, and overlay on original image using `matplotlib`/`PIL`.
   - Token importance: sum absolute SHAP across embedding dims per token; plot bar chart with token labels.
   - Save JSON with arrays for reproducibility.
6. Tests & smoke run
   - Add a smoke test script that loads a small background (n=10) and explains a single sample, asserts outputs created.
7. Docs & examples
   - Add usage snippet to `README.md` and document caveats (background selection, runtime cost, embedding exposure).

Acceptance criteria
- `tools/shap_multimodal.py` runs on a saved checkpoint without retraining and produces PNG + JSON outputs for at least one sample.
- README includes a usage example and notes on limitations.
- A small smoke test is available and passes locally.

Caveats & notes
- Explaining raw token ids isn't possible — must use embeddings for gradient explainers.
- KernelExplainer is possible but much slower; we prefer `GradientExplainer` for performance and scalability.
- SHAP results are sensitive to background selection — document that clearly.
- Consider adding an option to explain teacher vs student outputs separately.

Estimated effort
- Prototype script + README: 3–5 hours
- Visualization polish and tests: additional 2–4 hours
- Full integration + CI smoke test: additional 1–2 hours

Contact
- If you want I can implement the prototype script now (creates the tool, updates `requirements.txt`, and a smoke test). Choose: `GradientExplainer` (recommended) or `KernelExplainer` (slow, model-agnostic).
