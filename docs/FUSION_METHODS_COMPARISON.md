# Fusion Methods — Code-Level Comparison

This note compares the fusion modules implemented in `models/fusion/` by **reading the code** (not by theory), focusing on:
- How each module combines `img_emb` and `txt_emb`
- Symmetry (whether one modality conditions the other)
- Practical compute/behavior considerations (e.g., stochastic routing)

For a task-level recommendation based on current results + math, see: [`docs/BEST_FUSION_MODEL.md`](BEST_FUSION_MODEL.md).

## Common interface (all fusion modules)

In this repository, fusion modules operate on **projected** modality vectors:
- `img_emb`: shape `(B, D)`
- `txt_emb`: shape `(B, D)`
- output: fused representation `(B, D)`

They are created in the fusion factory in:
- `models/teacher.py` and `models/student.py`

## Summary table

| Fusion type | Implementation | What it does (exactly) | Symmetry | Compute / behavior notes |
|---|---|---|---|---|
| `concat_mlp` | [`models/fusion/concat_mlp.py`](../models/fusion/concat_mlp.py) (`ConcatMLPFusion`) | Concatenate then MLP: `MLP([img; txt]) -> D` | Symmetric | Cheap + stable baseline; interaction is learned only via concatenation. |
| `gated` | [`models/fusion/gated.py`](../models/fusion/gated.py) (`GatedFusion`) | Per-dim convex blend: `gate = sigmoid(W[img;txt])`; output `gate*img + (1-gate)*txt` | Symmetric | Very cheap; expressive mainly as a learned interpolation. |
| `film` | [`models/fusion/film.py`](../models/fusion/film.py) (`FiLMFusion`) | Text conditions vision: `gamma = Wg(txt)`, `beta = Wb(txt)`; output `gamma ⊙ img + beta` | **Asymmetric** | Cheap; treats text as conditioning signal, not a peer feature stream. |
| `modality_dropout` | [`models/fusion/modality_dropout.py`](../models/fusion/modality_dropout.py) (`ModalityDropoutFusion`) | During training may zero out img and/or txt, then `MLP([img;txt]) -> D` | Symmetric | **Batch-level dropout**: uses `torch.rand(1)`, so a modality can be dropped for the entire batch. This increases regularization but can add instability. |
| `transformer_concat` | [`models/fusion/transformer_concat.py`](../models/fusion/transformer_concat.py) (`TransformerConcatFusion`) | Same 2-token transformer pattern → mean | Symmetric | Contains `modality_token` parameter, but forward overwrites both token vectors with `img_emb/txt_emb`, so that parameter is effectively unused. |
| `cross_attention` | [`models/fusion/cross_attention.py`](../models/fusion/cross_attention.py) (`CrossAttentionFusion`) | Build 2 tokens `[img,txt]`; query = mean token; MHA(query→{img,txt}); add FFN residual | Symmetric-ish | Like attention pooling over the two modality vectors. Medium compute. |
| `shomr` | [`models/fusion/shomr.py`](../models/fusion/shomr.py) (`SHoMRFusion`) | Soft path: confidence-weighted sum + cross-attn; Hard path: route to vision-only/text-only/both | Symmetric | Has threshold switching + **stochastic routing** during training (`multinomial`). More moving parts ⇒ higher variance; can underperform if routing isn’t regularized well. |
| `energy_aware_adaptive` | [`models/fusion/energy_aware_adaptive.py`](../models/fusion/energy_aware_adaptive.py) (`EnergyAwareAdaptiveFusion`) | Router picks vision-only/text-only/both; “both” uses gated blend + cross-attn | Symmetric | Two notable code-level gotchas: (1) routing uses `multinomial` even in eval ⇒ **stochastic inference**; (2) computes `energy_loss` but does not return it, so the “energy budget” is not actually optimized unless wired into training. Docstring contains unverified efficiency claims. |

## Mathematical formulation (with brief justification)

Let $v \in \mathbb{R}^D$ be the projected vision embedding (`img_emb`) and $t \in \mathbb{R}^D$ be the projected text embedding (`txt_emb`). Each fusion module implements a function $f: \mathbb{R}^D \times \mathbb{R}^D \to \mathbb{R}^D$.

### `concat_mlp`

**Code:** `f(v,t) = \mathrm{MLP}([v;t])` where $[v;t] \in \mathbb{R}^{2D}$ is concatenation.

One can write a 2-layer version as:
$$
f(v,t) = W_2\,\phi(W_1 [v;t] + b_1) + b_2,
$$
with nonlinearity $\phi$ (GELU in code) and dropout between layers.

**Justification:** this is the simplest *universal* vector fusion family: with enough hidden width it can approximate many continuous fusion rules over $(v,t)$.

### `gated`

**Code:**
$$
g = \sigma(W_g [v;t] + b_g) \in (0,1)^D,\qquad f(v,t) = g \odot v + (\mathbf{1}-g) \odot t.
$$

**Justification:** this is a per-dimension mixture between modalities; it is a lightweight way to let the model select “how much to trust” each modality for each latent feature.

### `film`

**Code (text → vision conditioning):**
$$
\gamma = W_\gamma t + b_\gamma,\qquad \beta = W_\beta t + b_\beta,\qquad f(v,t) = \gamma \odot v + \beta.
$$

**Justification:** FiLM is a standard conditioning mechanism: text produces an affine transformation of vision features. It is intentionally asymmetric.

### `modality_dropout`

**Code behavior:** during training, the module samples two Bernoulli variables that are **batch-global**:
$$
d_v \sim \mathrm{Bernoulli}(p_{img}),\qquad d_t \sim \mathrm{Bernoulli}(p_{txt}),
$$
then applies masking
$$
	ilde v = (1-d_v)\,v,\qquad \tilde t = (1-d_t)\,t,
$$
and returns
$$
f(v,t) = \mathrm{MLP}([\tilde v; \tilde t]).
$$

**Justification:** modality dropout encourages robustness to missing/uninformative modalities by forcing the downstream MLP to perform well when one stream is absent.

### `transformer_concat`

**Code:** form a 2-token sequence $X_0 \in \mathbb{R}^{2\times D}$ with rows $v$ and $t$, apply a TransformerEncoder $\mathcal{T}$, then average the two output tokens:
$$
X_0 = \begin{bmatrix} v^\top \\ t^\top \end{bmatrix},\qquad X = \mathcal{T}(X_0),\qquad f(v,t) = \tfrac{1}{2}(X_{1,:} + X_{2,:}).
$$

**Implementation note:** the parameter `modality_token` is overwritten by assigning `tokens[:,0]=v` and `tokens[:,1]=t`, so it does not contribute to $f(v,t)$ as currently written.

### `cross_attention`

**Code:** create two tokens $X_0=[v;t] \in \mathbb{R}^{2\times D}$, define a single query token as the mean:
$$
q = \mathrm{LN}\left(\tfrac{1}{2}(v+t)\right) \in \mathbb{R}^D,
$$
then apply multi-head attention with query $q$ and keys/values $X_0$:
$$
z = \mathrm{MHA}(q, X_0, X_0) \in \mathbb{R}^D,
$$
followed by an FFN with residual:
$$
f(v,t) = z + \mathrm{FFN}(\mathrm{LN}(z)).
$$

**Justification:** this is attention pooling over the two modality tokens: the model learns weights that decide how to combine $v$ and $t$ for each sample.

### `shomr` (Soft-Hard Modality Routing)

Let $c(v,t) \in \Delta^2$ be the 2-way confidence distribution from `conf_net` (softmax over 2 logits), with weights $(w_v, w_t)$.

**Soft path (when not using hard routing):**
$$
	ext{base} = w_v\,v + w_t\,t,\qquad a = \mathrm{MHA}([v;t],[v;t],[v;t])_{\mathrm{mean\ over\ tokens}},
$$
$$
f_{soft}(v,t) = \mathrm{LN}(\text{base} + a) + \mathrm{FFN}(\mathrm{LN}(\text{base} + a)).
$$

**Hard path:** a 3-way router produces probabilities $r(v,t) \in \Delta^3$ over {vision-only, text-only, both}. The code samples during training and uses argmax in eval:
$$
\hat k \sim r(v,t)\ \text{(train)}\quad\text{or}\quad \hat k = \arg\max r(v,t)\ \text{(eval)}.
$$
Then
$$
f_{hard}(v,t)=
\begin{cases}
\mathrm{Proj}(v) & \hat k=0\\
\mathrm{Proj}(t) & \hat k=1\\
	ext{FuseBoth}(v,t) & \hat k=2
\end{cases}
$$
where `FuseBoth` is gated fusion plus cross-attn plus FFN as in the code.

**Justification:** SHoMR combines a smooth confidence-weighted fusion with an optional discrete routing mechanism to skip a modality when it is clearly unhelpful.

### `energy_aware_adaptive` (Energy-Aware Adaptive Fusion)

**Routing:** a 3-way router produces probabilities $r(v,t) \in \Delta^3$ and the code samples
$$
\hat k \sim r(v,t)
$$
both in training and eval.

Then the output is:
$$
f(v,t) =
\begin{cases}
v & \hat k=0\\
t & \hat k=1\\
\mathrm{FFN}(\mathrm{LN}(\text{gated}(v,t) + \text{attn}(v,t))) & \hat k=2
\end{cases}
$$
with `gated(v,t)` identical in form to the `gated` module and `attn(v,t)` computed from a 2-token self-attention block.

**Energy penalty in code:** the module computes an auxiliary energy loss
$$
\mathcal{L}_{energy} = \lambda\,\max\big(0, \mathbb{E}[E(\hat k)] - E_{budget}\big),
$$
but does **not** return or expose it from `forward`, so (as written) training does not optimize this penalty unless modified elsewhere.

**Justification:** conceptually this is a mixture-of-experts router over three fusion “experts” (vision-only, text-only, both). In practice, discrete sampling introduces variance; deterministic inference usually requires using argmax instead of sampling.

## Practical takeaways

- Deterministic continuous fusion (`cross_attention`, `transformer_concat`, `concat_mlp`, `gated`, `film`) tends to be easier to optimize.
- Discrete routing + sampling (`shomr`, `energy_aware_adaptive`) introduces variance; if used, consider adding explicit regularization and making inference deterministic.
- `film` is the only clearly **asymmetric** module (text→vision conditioning).

## Notes on usage in models

The Teacher/Student forward path projects backbone features into `fusion_dim` and then calls the fusion module as:
- `fused = dropout(fusion(v, t))`

See:
- [`models/teacher.py`](../models/teacher.py)
- [`models/student.py`](../models/student.py)
