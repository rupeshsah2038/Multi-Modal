# System Architecture and Technical Details

## System Architecture Summary

**Multimodal Knowledge Distillation System for Medical Imaging**

This system implements teacher-student knowledge distillation for dual-task medical image classification. A large Teacher model (ViT-Base + Bio-ClinicalBERT, 197M parameters) processes radiology images and clinical descriptions through cross-attention fusion, learning modality/location classification (MedPix) or wound type/severity (Wound-1-0). The trained Teacher then distills knowledge to compact Student models (DeiT-Small/Tiny + DistilBERT/MiniLM, 30-90M parameters) using combined cross-entropy and KD loss with temperature scaling. Optuna-optimized hyperparameters (384-dim fusion, balanced loss weights, higher dropout) achieve 88-92% F1 scores across datasets. The system enables efficient edge deployment while maintaining strong performance through effective multimodal knowledge transfer.

---

## Technical Interaction Between Teacher and Student Models

### 1. Sequential Two-Phase Training

**Phase 1: Teacher Training**
- Teacher trains independently on ground-truth labels
- Uses cross-entropy (CE) loss for both tasks:
  ```python
  loss = CE(teacher.logits_modality, y_modality) + CE(teacher.logits_location, y_location)
  ```
- Optimized with AdamW (lr=4.79e-05, tuned from 1e-05)
- Trains for 3 epochs

**Phase 2: Student Training with Distillation**
- Teacher is frozen: `teacher.eval()`
- Teacher wrapped in `torch.no_grad()` to prevent gradient computation
- Student trains using combined supervision from teacher and ground-truth

### 2. Knowledge Transfer Mechanisms

The system uses **four complementary distillation losses** in the Combined Loss function:

#### A. Cross-Entropy Loss (Hard Label Learning)
```python
ce = CE(student.logits_modality, y_mod) + CE(student.logits_location, y_loc)
```
- Student learns from ground-truth one-hot labels
- Ensures task-specific performance
- Weight: `alpha = 0.518` (tuned, reduced from 1.0)

#### B. KL Divergence Loss (Soft Target Distillation)
```python
kl = KL(log_softmax(student_logits/T), softmax(teacher_logits/T)) × T²
# Applied to both modality and location tasks
```
- Teacher logits softened with temperature `T = 3.19` (tuned from 2.0)
- Student mimics teacher's probability distributions
- Captures inter-class relationships and uncertainty
- Weight: Implicit baseline weight of 1.0
- Temperature scaling `T²` compensates for gradient magnitude

#### C. MSE Feature Loss (Representation Alignment)
```python
# Teacher raw features projected to student dimension via lazy initialization
t_img_projected = Linear(teacher.img_raw) → student.img_proj dimension
t_txt_projected = Linear(teacher.txt_raw) → student.txt_proj dimension

mse = MSE(student.img_proj, t_img_projected) + MSE(student.txt_proj, t_txt_projected)
```
- Aligns student's projected features with teacher's raw backbone features
- Projection layers created lazily based on runtime tensor shapes
- Ensures device compatibility (GPU/CPU)
- Weight: `beta = 112.4` (tuned from 100.0)

#### D. CRD Loss (Contrastive Representation Distillation)
```python
crd = CRDLoss(student_features, teacher_features, y_mod, y_loc)
```
- Contrastive learning between student and teacher representations
- Encourages similar samples to have similar representations
- Weight: `gamma = 10.0` (default, not tuned in Optuna study)

**Combined Loss Formula:**
```python
total_loss = ce + alpha × kl + beta × mse + gamma × crd
```

With optimized hyperparameters:
```python
total_loss = ce + 0.518 × kl + 112.4 × mse + 10.0 × crd
```

### 3. Model Architecture and Outputs

#### Teacher Model Structure
```
Input: Image (224×224) + Text Description
  ↓
Vision Backbone (ViT-Base) → [batch, 768] CLS token (img_raw)
  ↓
Projection Layer → [batch, 384] (img_proj)
  ↓
Text Backbone (Bio-ClinicalBERT) → [batch, 768] CLS token (txt_raw)
  ↓
Projection Layer → [batch, 384] (txt_proj)
  ↓
Cross-Attention Fusion → [batch, 384] (fused representation)
  ↓
Dropout (p=0.185)
  ↓
Dual Classification Heads
  ├─ Modality Head → [batch, num_modality_classes]
  └─ Location Head → [batch, num_location_classes]
```

#### Student Model Structure
```
Input: Image (224×224) + Text Description
  ↓
Vision Backbone (DeiT-Small/Tiny) → [batch, 384/192] CLS token (img_raw)
  ↓
Projection Layer → [batch, 384] (img_proj)
  ↓
Text Backbone (DistilBERT/MiniLM) → [batch, 768/384] CLS token (txt_raw)
  ↓
Projection Layer → [batch, 384] (txt_proj)
  ↓
Cross-Attention Fusion → [batch, 384] (fused representation)
  - 2 fusion layers (tuned from 1)
  - 4 attention heads (tuned from 8)
  ↓
Dropout (p=0.238)
  ↓
Dual Classification Heads
  ├─ Modality Head → [batch, num_modality_classes]
  └─ Location Head → [batch, num_location_classes]
```

#### Output Dictionary Structure

Both Teacher and Student return identical output structure:
```python
{
    "logits_modality": Tensor[batch, num_modality_classes],  # Task 1 logits
    "logits_location": Tensor[batch, num_location_classes],  # Task 2 logits
    "img_raw": Tensor[batch, backbone_dim],                  # Vision backbone CLS token
    "txt_raw": Tensor[batch, backbone_dim],                  # Text backbone CLS token
    "img_proj": Tensor[batch, 384],                          # Projected vision features
    "txt_proj": Tensor[batch, 384]                           # Projected text features
}
```

### 4. Training Flow and Data Pipeline

#### Complete Training Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│ Phase 1: Teacher Training (3 epochs)                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ Batch → Teacher(trainable) → {logits, features}                │
│         ↓                                                       │
│         CE Loss (logits vs ground_truth)                        │
│         ↓                                                       │
│         Backprop → Update Teacher weights                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ Phase 2: Student Training with Distillation (10 epochs)        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ Batch → Teacher(frozen) ──────────┐                            │
│    │    with torch.no_grad()      │                            │
│    │                               ↓                            │
│    │    {t_logits_mod, t_logits_loc,                           │
│    │     t_img_raw, t_txt_raw,                                 │
│    │     t_img_proj, t_txt_proj} (detached)                    │
│    │                               │                            │
│    └─→ Student(trainable) ────────┤                            │
│        {s_logits_mod, s_logits_loc,│                            │
│         s_img_raw, s_txt_raw,      │                            │
│         s_img_proj, s_txt_proj}    │                            │
│                                    ↓                            │
│        Combined Loss:                                           │
│        ┌────────────────────────────────────────────┐          │
│        │ 1. CE Loss (student.logits vs ground_truth)│          │
│        │    weight: 1.0 (baseline in loss function) │          │
│        ├────────────────────────────────────────────┤          │
│        │ 2. KL Loss (student.logits vs teacher.logits)│        │
│        │    - Temperature scaling: T=3.19           │          │
│        │    - weight: alpha=0.518                   │          │
│        ├────────────────────────────────────────────┤          │
│        │ 3. MSE Loss (feature alignment)            │          │
│        │    - Project teacher raw → student dim     │          │
│        │    - weight: beta=112.4                    │          │
│        ├────────────────────────────────────────────┤          │
│        │ 4. CRD Loss (contrastive learning)         │          │
│        │    - weight: gamma=10.0                    │          │
│        └────────────────────────────────────────────┘          │
│                                    ↓                            │
│        Backprop → Update Student weights only                   │
│                   (Teacher frozen, no gradient flow)            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### Batch Data Structure

Each training batch contains:
```python
{
    'pixel_values': Tensor[batch, 3, 224, 224],          # Preprocessed images
    'input_ids_teacher': Tensor[batch, max_len],         # Teacher tokenized text
    'attention_mask_teacher': Tensor[batch, max_len],    # Teacher attention mask
    'input_ids_student': Tensor[batch, max_len],         # Student tokenized text
    'attention_mask_student': Tensor[batch, max_len],    # Student attention mask
    'modality': Tensor[batch],                           # Task 1 labels
    'location': Tensor[batch]                            # Task 2 labels
}
```

**Note:** Separate tokenization for teacher and student allows using different text backbones (e.g., Bio-ClinicalBERT vs DistilBERT with different vocabularies).

### 5. Key Technical Details

#### Lazy Projection Initialization
```python
# In MedKDCombinedLoss.forward()
# Projection layers created on first forward pass
dev = s_out.get('img_proj', next(iter(s_out.values()))).device

in_img = t_out['img_raw'].size(-1)   # Teacher backbone dimension
out_img = s_out['img_proj'].size(-1)  # Student fusion dimension

if self.proj_t_img is None or dimensions changed:
    self.proj_t_img = nn.Linear(in_img, out_img).to(dev)
```
- Handles different backbone dimensions dynamically
- Ensures device compatibility (cuda:0, cuda:1, etc.)
- Adapts to runtime tensor shapes

#### Temperature Scaling Effect
```python
# Without temperature (T=1.0): Sharp probabilities
softmax([2.0, 1.0, 0.5]) → [0.659, 0.242, 0.099]

# With temperature (T=3.19): Softer probabilities
softmax([2.0/3.19, 1.0/3.19, 0.5/3.19]) → [0.449, 0.314, 0.237]
```
- Higher T (3.19) reveals more inter-class relationships
- Student learns from teacher's uncertainty
- Multiplying by T² maintains gradient magnitude

#### Gradient Flow Control
```python
# Phase 2: Student training
teacher.eval()  # Disable dropout, batch norm updates

with torch.no_grad():
    t_out = teacher(pv, ids_t, mask_t)  # No gradient computation

s_out = student(pv, ids_s, mask_s)  # Gradients enabled
loss = distill_fn(s_out, t_out, y_mod, y_loc)
loss.backward()  # Only student parameters updated
```
- Teacher acts as static oracle
- Prevents teacher overfitting to student
- Reduces memory usage (no teacher gradients stored)

### 6. Hyperparameter Impact Analysis

#### Optuna-Optimized vs Baseline

| Parameter | Baseline | Optimized | Impact |
|-----------|----------|-----------|--------|
| **alpha (CE weight)** | 1.0 | 0.518 | Less emphasis on hard labels, more on soft targets |
| **beta (MSE weight)** | 100.0 | 112.4 | Stronger feature alignment between teacher-student |
| **T (Temperature)** | 2.0 | 3.19 | Softer distributions reveal more knowledge |
| **Teacher LR** | 1e-5 | 4.79e-5 | Faster convergence for larger model |
| **Student LR** | 3e-4 | 1.05e-4 | Slower, more stable distillation learning |
| **Fusion Dim** | 256/512 | 384 | Optimal capacity for multimodal alignment |
| **Student Fusion Layers** | 1 | 2 | More capacity needed for effective distillation |
| **Student Fusion Heads** | 8 | 4 | Reduced attention heads for efficiency |
| **Teacher Dropout** | 0.1 | 0.185 | Better generalization on medical data |
| **Student Dropout** | 0.1 | 0.238 | Even stronger regularization for smaller model |

#### Why These Changes Work

1. **Lower alpha, Higher beta:** Student learns more from teacher's representations than raw labels
2. **Higher Temperature:** Captures nuanced inter-class relationships (e.g., similar anatomical regions)
3. **384-dim Fusion:** Sweet spot - 256 too small, 512 over-parameterizes for these datasets
4. **2 Student Fusion Layers:** Single layer insufficient for complex multimodal alignment
5. **Divergent Learning Rates:** Teacher benefits from aggressive learning, student needs stability
6. **Higher Dropout:** Small medical datasets (1000-2000 samples) prone to overfitting

### 7. Performance Results

#### MedPix-2-0 Dataset
| Model | Parameters | Test Modality F1 | Test Location F1 | Avg F1 | Improvement |
|-------|------------|------------------|------------------|---------|-------------|
| Teacher | 197.07M | ~0.97 | ~0.88 | ~0.93 | Baseline |
| Student (deit-small + distilbert) | 90.40M | 0.965 | 0.862 | **0.914** | +6.4% vs baseline |
| Student (deit-tiny + minilm) | 30.27M | 0.970 | 0.797 | **0.883** | +8.3% vs baseline |

#### Wound-1-0 Dataset
| Model | Parameters | Test Type F1 | Test Severity F1 | Avg F1 | Improvement |
|-------|------------|--------------|------------------|---------|-------------|
| Teacher | 197.07M | ~0.88 | ~0.94 | ~0.91 | Baseline |
| Student (deit-small + minilm) | 46.60M | 0.905 | 0.938 | **0.922** | +5.2% vs baseline |
| Student (deit-tiny + minilm) | 30.28M | 0.825 | 0.939 | **0.882** | +4.2% vs baseline |

### 8. Implementation Notes

#### Device Management
```python
# Config-driven device selection
cfg_device = cfg.get('device', None)
if cfg_device:
    device = torch.device(cfg_device)  # e.g., 'cuda:0', 'cuda:3'
else:
    device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
```

#### Dynamic Class Counts
```python
# Supports varying number of classes per dataset
num_classes = get_num_classes(
    dataset_type=dataset_type,
    dataset_root=dataset_root,
    type_column=cfg['data'].get('type_column', 'type'),
    severity_column=cfg['data'].get('severity_column', 'severity'),
)
num_modality_classes = num_classes['modality']  # 2 for MedPix, 5 for Wound
num_location_classes = num_classes['location']  # 5 for MedPix, 3 for Wound
```

#### Tokenizer Matching
```python
# Automatically load correct tokenizers for configured backbones
t_pretrained = get_text_pretrained_name('bio-clinical-bert')  
# → 'emilyalsentzer/Bio_ClinicalBERT'

s_pretrained = get_text_pretrained_name('distilbert')
# → 'distilbert-base-uncased'

teacher_tokenizer = AutoTokenizer.from_pretrained(t_pretrained)
student_tokenizer = AutoTokenizer.from_pretrained(s_pretrained)
```

---

## Key Takeaways

1. **Multi-Level Knowledge Transfer:** System uses logits, features, and contrastive learning simultaneously
2. **No Feature Distillation Bypass:** Student doesn't directly copy teacher features - learns through MSE alignment
3. **Independent Architecture:** Student has its own backbones and fusion, not sharing weights with teacher
4. **Frozen Teacher:** Acts as static oracle during distillation, preventing catastrophic forgetting
5. **Hyperparameter Sensitivity:** Tuning provides 4-8% improvement, with temperature and loss weights most critical
6. **Efficiency-Accuracy Trade-off:** 30M parameter models achieve 88% F1 (only 4-5% drop from 197M teacher)

---

## Related Documentation

- **Configuration Guide:** `config/ultra-edge-tuned-hp/README.md`
- **Hyperparameter Tuning:** `docs/HYPERPARAMETER_TUNING_SUMMARY.md`
- **Results Analysis:** `docs/ULTRA_EDGE_TUNED_HP_RESULTS.md`
- **Loss Functions:** `docs/LOSS_FUNCTIONS_COMPARISON.md`
- **Pipeline Overview:** `docs/PIPELINE.md`
