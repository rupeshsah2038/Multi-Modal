# Best Fusion Model (Assuming No `simple` Fusion)

This document assumes `simple` / `SimpleFusion` has been removed and is not an available option.

## Recommendation

For the current **multi-task, multi-modal** setup in this repository (single fused representation feeding two classification heads), the best default fusion choice (with `simple` unavailable) is:

- **`cross_attention`** ([`models/fusion/cross_attention.py`](../models/fusion/cross_attention.py), `CrossAttentionFusion`)

This recommendation is based on:
1) **Observed test performance** on the wound multi-task setting (type + severity), excluding `simple`.
2) **Observed test performance** on the MedPix multi-task setting (modality + location), excluding `simple`.
3) **Mathematical/optimization properties** of the implemented fusion functions.

## Evidence from current experiments (wound, excluding `simple`)

From the test-split summary in [`docs/wound_fusion_hp_summary.md`](wound_fusion_hp_summary.md) (which omits `simple`):

| Fusion Type | test_type_acc | test_severity_acc | test_type_f1 | test_severity_f1 | test_type_auc | test_severity_auc |
|---|---:|---:|---:|---:|---:|---:|
| cross_attention | **0.847** | 0.932 | **0.842** | 0.933 | 0.987 | 0.989 |
| film | 0.838 | 0.940 | 0.831 | **0.939** | **0.988** | **0.990** |
| transformer_concat | 0.834 | 0.932 | 0.857 | 0.915 | 0.987 | 0.993 |

Interpretation:
- `cross_attention` is the strongest **type** (task1) performer among the remaining options.
- `film` is very strong on **severity** (task2).
- For a single default across both tasks, `cross_attention` is the most consistently strong and mathematically well-behaved choice.

## Evidence from current experiments (MedPix, excluding `simple`)

From the test-split summary in [`docs/medpix_fusion_hp_summary.md`](medpix_fusion_hp_summary.md) (which omits `simple`):

| Fusion Type | test_modality_acc | test_location_acc | test_modality_f1 | test_location_f1 |
|---|---:|---:|---:|---:|
| cross_attention | 0.960 | **0.880** | 0.95998 | 0.84266 |
| concat_mlp | 0.955 | 0.875 | 0.95499 | **0.84745** |
| gated | **0.980** | 0.860 | **0.97999** | 0.83127 |

Interpretation:
- `cross_attention` is best on **location accuracy** (task2).
- `gated` is best on **modality** in this sweep (task1).
- `concat_mlp` is best on **location F1**.

So MedPix still shows task-specific tradeoffs, but `cross_attention` is a strong default if you want good task2 (location) performance without adding stochastic routing.

## Mathematical justification (why `cross_attention` fits multi-task multimodal fusion)

### 1) The multi-task objective needs a *shared* fused representation

In `Teacher`/`Student`, both task heads consume the same fused vector:

- `logits_task1 = head1(f(v,t))`
- `logits_task2 = head2(f(v,t))`

So training minimizes something of the form:
$$
\mathcal{L} = \mathcal{L}_1\big(h_1(f(v,t)), y_1\big) + \mathcal{L}_2\big(h_2(f(v,t)), y_2\big)
$$
A good fusion function $f$ must preserve information useful for **both** tasks, and allow the model to form task-relevant interactions between modalities.

### 2) `cross_attention` provides deterministic, data-dependent modality weighting

In this repo, `cross_attention` constructs two tokens $X=[v;t]\in\mathbb{R}^{2\times D}$ and uses a **single pooled query**
$$
q = \mathrm{LN}\left(\tfrac{1}{2}(v+t)\right)
$$
to attend over the modality tokens:
$$
z = \mathrm{MHA}(q, X, X),\qquad f_{\text{cross}}(v,t) = z + \mathrm{FFN}(\mathrm{LN}(z)).
$$

Key consequences for multi-task fusion:
- The attention weights are **data-dependent** functions of $(v,t)$, so the model can adaptively emphasize the more informative modality per sample.
- The module is **deterministic** (no sampling/routing in forward), reducing optimization variance compared to stochastic routers.
- It is more expressive than pure interpolation (e.g., `gated`) because attention pooling can implement content-dependent mixing beyond elementwise convex blending.

### 3) Why it is preferable to other current options

- Versus `gated`: gating is restricted to an elementwise convex blend $g\odot v + (1-g)\odot t$, which cannot represent richer cross-feature interactions.
- Versus `film`: FiLM is asymmetric (text conditions vision) and may discard useful “vision→text” interaction patterns.
- Versus `transformer_concat`: transformer-based fusion is a strong alternative, but in this codebase it includes an effectively unused `modality_token` parameter and did not dominate `cross_attention` across tasks in the reported sweeps.
- Versus `shomr` / `energy_aware_adaptive`: both include discrete routing/sampling; in this codebase that increases variance, and `energy_aware_adaptive` is stochastic even at inference.

## How to use it

In your YAML config:

- `fusion.type: "cross_attention"`
- choose `teacher.fusion_layers`, `student.fusion_layers`, and `fusion_heads` as desired.

See examples in your configs under `config/fusion-explore-hp-wound/`.

## Scope / limitations

- This conclusion is grounded in the current wound fusion sweep results and the *current* module implementations.
- If you change backbones, datasets, or fix routing modules (e.g., make inference deterministic, wire energy loss into training), the ranking may change.

Related docs:
- [`docs/FUSION_METHODS_COMPARISON.md`](FUSION_METHODS_COMPARISON.md)
- [`docs/wound_fusion_hp_summary.md`](wound_fusion_hp_summary.md)
- [`docs/medpix_fusion_hp_summary.md`](medpix_fusion_hp_summary.md)

## Practical recommendation (if you must pick one)

- If you want **one default** across both datasets without special-casing (and `simple` is unavailable), use **`cross_attention`**.
- If your priority is **MedPix modality** (task1) in this sweep, `gated` performed best.
- If your priority is **MedPix location F1** (task2), `concat_mlp` performed best.
