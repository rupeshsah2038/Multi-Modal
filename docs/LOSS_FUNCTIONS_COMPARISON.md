# Knowledge Distillation Loss Functions: Mathematical Analysis & Comparison

## Overview
This document provides a rigorous mathematical comparison of five knowledge distillation loss functions implemented in the codebase, with theoretical justification and practical implications.

---

## 1. Vanilla Distillation Loss (DistillationLoss)

### Mathematical Formulation

$$\mathcal{L}_{\text{vanilla}} = \mathcal{L}_{\text{CE}} + \alpha \mathcal{L}_{\text{KL}} + \beta \mathcal{L}_{\text{feat}}$$

Where:

**Cross-Entropy Loss** (Task Loss):
$$\mathcal{L}_{\text{CE}} = -\sum_{c=1}^{C} y_c \log p_{\theta_S}(c|x)$$

**Knowledge Distillation Loss** (KL Divergence):
$$\mathcal{L}_{\text{KL}} = T^2 \cdot D_{KL}\left(p_{\theta_T}^{(T)} \parallel p_{\theta_S}^{(T)}\right)$$
$$= T^2 \sum_{c=1}^{C} p_{\theta_T}^{(T)}(c) \log \frac{p_{\theta_T}^{(T)}(c)}{p_{\theta_S}^{(T)}(c)}$$

Where $p^{(T)}(c) = \frac{\exp(z_c/T)}{\sum_{j}\exp(z_j/T)}$ (temperature-scaled softmax)

**Feature Matching Loss** (MSE):
$$\mathcal{L}_{\text{feat}} = \frac{1}{2D}\left(\|f_S^{\text{img}} - \phi(f_T^{\text{img}})\|^2 + \|f_S^{\text{txt}} - \psi(f_T^{\text{txt}})\|^2\right)$$

Where $\phi, \psi$ are projection layers.

### Theoretical Justification

1. **Temperature Scaling ($T^2$ term)**:
   - Softmax with temperature $T$: $\sigma_i^{(T)} = \frac{e^{z_i/T}}{\sum_j e^{z_j/T}}$
   - As $T \to \infty$: outputs converge to uniform distribution (soft targets)
   - As $T \to 0$: outputs converge to one-hot (hard targets)
   - The $T^2$ scaling compensates for the gradient magnitude change:
   $$\frac{\partial \mathcal{L}_{\text{KL}}}{\partial z_i} \propto \frac{1}{T}$$
   - Therefore, multiplying by $T^2$ restores gradient scale

2. **Feature Alignment**:
   - Minimizes $\|\cdot\|^2$ encourages student to match teacher's representation space
   - Projection layers $\phi, \psi$ allow dimension mismatch between teacher/student

### Hyperparameters
- $\alpha = 1.0$: balances hard vs soft targets
- $\beta = 100.0$: emphasizes feature alignment (high weight for representation learning)
- $T = 2.0$: moderate softening of logits

### Pros & Cons
✅ **Pros**:
- Simple, interpretable, fast convergence
- Proven effective for standard distillation tasks
- Balances task performance and knowledge transfer

❌ **Cons**:
- Treats all samples equally (no relation modeling)
- May ignore structural information in feature space
- High $\beta$ can lead to feature collapse

---

## 2. Combined Loss (MedKDCombinedLoss)

### Mathematical Formulation

$$\mathcal{L}_{\text{combined}} = \mathcal{L}_{\text{CE}} + \alpha \mathcal{L}_{\text{KL}} + \beta \mathcal{L}_{\text{MSE}} + \gamma \mathcal{L}_{\text{CRD}}$$

This extends Vanilla by adding **Contrastive Representation Distillation (CRD)**.

### Contrastive Component

$$\mathcal{L}_{\text{CRD}} = -\frac{1}{2B}\sum_{i=1}^{B}\left[\log\frac{\exp(s_i \cdot t_i / \tau)}{\sum_{j=1}^{B}\exp(s_i \cdot t_j / \tau)} + \log\frac{\exp(s'_i \cdot t'_i / \tau)}{\sum_{j=1}^{B}\exp(s'_i \cdot t'_j / \tau)}\right]$$

Where:
- $s_i, t_i$: student and teacher features for sample $i$ (normalized)
- $\tau$: temperature (default 0.1)
- First term: image modality, second term: text modality

### Theoretical Justification

**InfoNCE Bound** (van den Oord et al., 2018):
$$I(s; t) \geq \log(B) - \mathcal{L}_{\text{CRD}}$$

Where $I(s; t)$ is mutual information between student and teacher representations.

**Key Insight**: CRD maximizes mutual information, encouraging student to capture not just individual features but also **inter-sample relationships**.

**Contrastive Learning Benefits**:
1. **Positive pairs** $(s_i, t_i)$: pulled together in embedding space
2. **Negative pairs** $(s_i, t_j), j \neq i$: pushed apart
3. Prevents representation collapse (mode covering)

### Hyperparameters
- $\alpha = 1.0$, $\beta = 100.0$: same as vanilla
- $\gamma = 10.0$: moderate weight for contrastive term
- $T = 4.0$: higher temperature (softer logits, more dark knowledge)

### Pros & Cons
✅ **Pros**:
- Captures both instance-level and batch-level relationships
- Better representation quality (proven by mutual information bound)
- Robust to label noise

❌ **Cons**:
- Requires larger batch sizes for effective negative sampling
- Computationally expensive (pairwise comparisons: $O(B^2)$)
- Sensitive to $\tau$ (too low: hard negatives, too high: uniform)

---

## 3. Contrastive Representation Distillation (CRDLoss)

### Mathematical Formulation

Pure contrastive loss without CE/KL/MSE:

$$\mathcal{L}_{\text{CRD}} = \frac{1}{2}\left[\mathcal{L}_{\text{NCE}}^{\text{img}} + \mathcal{L}_{\text{NCE}}^{\text{txt}}\right]$$

Where:
$$\mathcal{L}_{\text{NCE}}^m = -\frac{1}{B}\sum_{i=1}^{B}\log\frac{\exp(\text{sim}(s_i^m, t_i^m) / \tau)}{\sum_{j=1}^{B}\exp(\text{sim}(s_i^m, t_j^m) / \tau)}$$

$m \in \{\text{img}, \text{txt}\}$

### Theoretical Justification

**Noise Contrastive Estimation** (Gutmann & Hyvärinen, 2010):
- Frames density estimation as binary classification problem
- Discriminates between data samples and noise samples
- As $B \to \infty$, converges to maximum likelihood estimation

**Connection to Mutual Information**:
$$\mathcal{L}_{\text{NCE}} \approx -I(s; t) + \log(B)$$

Therefore minimizing $\mathcal{L}_{\text{NCE}}$ maximizes $I(s; t)$.

**Normalization Effect**:
- $\text{sim}(s, t) = \frac{s \cdot t}{\|s\|\|t\|}$ (cosine similarity)
- Projects features onto unit hypersphere
- Makes loss invariant to feature magnitude

### Hyperparameters
- $\tau = 0.1$: low temperature (sharp distribution, focus on hard negatives)
- `base_temperature = 0.07`: for potential temperature scheduling

### Pros & Cons
✅ **Pros**:
- Strong representation learning (proven by MI maximization)
- No need for task-specific heads during distillation
- Invariant to feature scale

❌ **Cons**:
- Requires large batch sizes ($B \geq 256$ recommended in SimCLR paper)
- No explicit task supervision (may hurt task accuracy)
- Sensitive to initialization and temperature

---

## 4. Relational Knowledge Distillation (RKDLoss)

### Mathematical Formulation

$$\mathcal{L}_{\text{RKD}} = w_{\text{dist}} \mathcal{L}_{\text{dist}} + w_{\text{angle}} \mathcal{L}_{\text{angle}}$$

**Distance-wise Loss**:
$$\mathcal{L}_{\text{dist}} = \frac{1}{B^2}\sum_{i \neq j}\text{SmoothL1}\left(d_S(f_i, f_j), d_T(f_i, f_j)\right)$$

Where:
$$d(f_i, f_j) = \|f_i - f_j\|_2 = \sqrt{\sum_{k=1}^{D}(f_i^{(k)} - f_j^{(k)})^2}$$

**Angle-wise Loss**:
$$\mathcal{L}_{\text{angle}} = \frac{1}{B^2}\sum_{i,j}\text{SmoothL1}\left(\cos\theta_S^{ij}, \cos\theta_T^{ij}\right)$$

Where:
$$\cos\theta^{ij} = \frac{f_i \cdot f_j}{\|f_i\|\|f_j\|}$$

### Theoretical Justification

**Manifold Hypothesis**:
- Data lies on low-dimensional manifold in high-dimensional space
- Relational structure captures manifold geometry
- Teacher's manifold is richer due to higher capacity

**Geometric Interpretation**:
1. **Distance preservation**: Maintains relative distances between samples
   - If teacher places $x_i$ and $x_j$ close → student should too
   - Preserves neighborhood structure

2. **Angle preservation**: Maintains directional relationships
   - If teacher embeds $x_i, x_j, x_k$ forming angle $\theta$ → student should match
   - Captures higher-order structure

**Mathematical Property**:
- $d(f_i, f_j)$ is a metric on feature space
- $\cos\theta^{ij}$ is invariant to uniform scaling
- Together, they define Riemannian geometry of embedding space

### Hyperparameters
- $w_{\text{dist}} = 25.0$: weight for distance preservation
- $w_{\text{angle}} = 50.0$: higher weight for angular structure

### Pros & Cons
✅ **Pros**:
- Preserves structural information (not just pointwise features)
- Invariant to global transformations (rotation, scaling)
- Effective with small batch sizes (captures within-batch relations)

❌ **Cons**:
- Computational cost: $O(B^2)$ pairwise computations
- May over-constrain student (too rigid structure transfer)
- Ignores cross-batch relationships

---

## 5. Maximum Mean Discrepancy (MMDLoss)

### Mathematical Formulation

$$\mathcal{L}_{\text{MMD}} = \text{MMD}^2(p_S, p_T) = \left\|\mathbb{E}_{f_S \sim p_S}[\phi(f_S)] - \mathbb{E}_{f_T \sim p_T}[\phi(f_T)]\right\|_{\mathcal{H}}^2$$

**Kernel Trick** (unbiased estimator):
$$\text{MMD}^2(p_S, p_T) = \frac{1}{B^2}\sum_{i,j}k(f_S^i, f_S^j) + \frac{1}{B^2}\sum_{i,j}k(f_T^i, f_T^j) - \frac{2}{B^2}\sum_{i,j}k(f_S^i, f_T^j)$$

Where $k(\cdot, \cdot)$ is **Gaussian RBF kernel**:
$$k(x, y) = \frac{1}{|\mathcal{B}|}\sum_{\sigma \in \mathcal{B}}\exp\left(-\frac{\|x - y\|^2}{2\sigma^2}\right)$$

With multi-scale bandwidths: $\mathcal{B} = \{0.2, 0.5, 1, 2, 5\}$

### Theoretical Justification

**Integral Probability Metric** (Müller, 1997):
$$\text{MMD}(p, q) = \sup_{f \in \mathcal{F}}\left|\mathbb{E}_{x \sim p}[f(x)] - \mathbb{E}_{y \sim q}[f(y)]\right|$$

Where $\mathcal{F}$ is a function class (RKHS for characteristic kernels).

**Key Property**: If $k$ is characteristic (e.g., Gaussian), then:
$$\text{MMD}(p, q) = 0 \iff p = q$$

**Distribution Matching**:
- Matches all moments of distributions (not just mean/variance)
- Kernel implicitly defines feature space via $\phi: \mathcal{X} \to \mathcal{H}$
- Multi-scale kernels capture both local and global structure

**Computational Complexity**:
- Exact: $O(B^2 \cdot |\mathcal{B}|)$ for batch size $B$
- Can be approximated with random features for large-scale

### Hyperparameters
- Bandwidths $\mathcal{B} = \{0.2, 0.5, 1, 2, 5\}$: multi-scale kernel
- No additional weights (acts as standalone loss or combined with others)

### Pros & Cons
✅ **Pros**:
- Theoretically principled (minimizes divergence between distributions)
- Handles multi-modal distributions well
- No assumptions on distribution family (non-parametric)

❌ **Cons**:
- High computational cost: $O(B^2)$ comparisons
- Sensitive to bandwidth selection (requires tuning $\mathcal{B}$)
- May over-smooth (matches distributions too broadly)

---

## Comparative Summary

| Loss Function | Key Mechanism | Complexity | Batch Size Req. | Best Use Case |
|---------------|---------------|------------|-----------------|---------------|
| **Vanilla** | Logit + feature matching | $O(B)$ | Small (≥16) | Standard distillation, fast training |
| **Combined** | Vanilla + contrastive | $O(B^2)$ | Medium (≥32) | Robust representations + task accuracy |
| **CRD** | Pure contrastive (InfoNCE) | $O(B^2)$ | Large (≥128) | Self-supervised, representation learning |
| **RKD** | Pairwise relations (dist+angle) | $O(B^2)$ | Small (≥16) | Structural knowledge, metric learning |
| **MMD** | Distribution matching (kernel) | $O(B^2)$ | Medium (≥32) | Distribution alignment, domain adaptation |

---

## Mathematical Hierarchy

```
Pointwise (sample-level)
    ├── Vanilla: min ||f_S - f_T||² + KL(p_T || p_S)
    │
Pairwise (relation-level)
    ├── RKD: min Σᵢⱼ |d_S(i,j) - d_T(i,j)|
    ├── CRD: max I(f_S; f_T) via contrastive
    │
Set-wise (distribution-level)
    └── MMD: min ||E[φ(f_S)] - E[φ(f_T)]||²
```

---

## Theoretical Connections

### 1. KL Divergence ↔ Cross-Entropy
$$D_{KL}(p \| q) = H(p, q) - H(p)$$
- KL minimization = cross-entropy minimization when $H(p)$ constant
- Vanilla loss combines both perspectives

### 2. Contrastive ↔ Mutual Information
$$I(s; t) = H(s) - H(s|t) = H(t) - H(t|s)$$
- InfoNCE lower bounds MI
- CRD maximizes feature dependency

### 3. MMD ↔ Energy Distance
$$\text{MMD}^2_k = 2\mathbb{E}[k(x,y)] - \mathbb{E}[k(x,x')] - \mathbb{E}[k(y,y')]$$
- For kernel $k(x,y) = -\|x-y\|$, MMD² = energy distance
- Generalizes Wasserstein distance

### 4. RKD ↔ Metric Learning
- Distance/angle preservation → embedding space is quasi-isometric
- Related to triplet loss: $\|f(a) - f(p)\|² < \|f(a) - f(n)\|² + m$

---

## Practical Recommendations

### When to Use Each Loss:

1. **Vanilla** (DistillationLoss):
   - Default choice for standard distillation
   - Limited computational budget
   - Task accuracy is primary goal

2. **Combined** (MedKDCombinedLoss):
   - Best all-around performance
   - Sufficient batch size (≥32)
   - Balance task accuracy + representation quality

3. **CRD** (CRDLoss):
   - Pre-training or self-supervised scenarios
   - Large batch sizes available
   - Transfer learning applications

4. **RKD** (RKDLoss):
   - Metric learning tasks (face recognition, re-ID)
   - Small batch sizes
   - Structured output spaces

5. **MMD** (MMDLoss):
   - Domain adaptation
   - Distribution shift robustness
   - Non-parametric distribution matching

---

## Hyperparameter Tuning Guidelines

### Temperature $T$:
- Low ($T = 1$): hard targets, fast convergence, may overfit
- Medium ($T = 2-4$): balanced, recommended default
- High ($T > 5$): very soft, slow convergence, better generalization

### Loss Weights:
- Start with $\alpha = 1.0, \beta = 100.0$ (vanilla defaults)
- Increase $\beta$ if representation quality matters more than task accuracy
- Decrease $\beta$ if task accuracy is primary objective

### Contrastive Temperature $\tau$:
- $\tau = 0.07$: standard SimCLR default
- Lower $\tau$: focus on hard negatives, sharper gradients
- Higher $\tau$: smoother optimization, more negative pairs contribute

---

## Empirical Validation (Expected)

Based on literature and theoretical properties:

| Loss | Task Acc. | Rep. Quality | Speed | Memory |
|------|-----------|--------------|-------|--------|
| Vanilla | ★★★★☆ | ★★★☆☆ | ★★★★★ | ★★★★★ |
| Combined | ★★★★★ | ★★★★★ | ★★★☆☆ | ★★★☆☆ |
| CRD | ★★★☆☆ | ★★★★★ | ★★☆☆☆ | ★★★☆☆ |
| RKD | ★★★★☆ | ★★★★☆ | ★★☆☆☆ | ★★★☆☆ |
| MMD | ★★★★☆ | ★★★★☆ | ★★☆☆☆ | ★★☆☆☆ |

---

## References

1. Hinton et al. (2015) - Distilling the Knowledge in a Neural Network
2. Tian et al. (2019) - Contrastive Representation Distillation
3. Park et al. (2019) - Relational Knowledge Distillation
4. Gretton et al. (2012) - A Kernel Two-Sample Test
5. van den Oord et al. (2018) - Representation Learning with Contrastive Predictive Coding
