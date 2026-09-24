# GPU Usage and Computational Resource Profiling

This document details the GPU memory (VRAM) consumption, computational latency, throughput, and parameter efficiency for the **Teacher Model** and **Student Model Distillation** across both the **MedPix** and **Wound** datasets.

---

## 1. Benchmarking Environment

- **Compute Device**: NVIDIA Quadro RTX 5000 (16,384 MB VRAM, Turing Architecture)
- **CUDA / Driver**: Active PyTorch CUDA Environment (`cuda:0`)
- **Precision**: Full Precision (FP32)
- **Batch Size ($B$)**: 16 (matching default training configs)
- **Image Inputs**: $224 \times 224 \times 3$ normalized RGB tensors
- **Architectures**:
  - **Teacher**: Vision Transformer Base (`google/vit-base-patch16-224`) + Clinical BERT (`emilyalsentzer/Bio_ClinicalBERT`) + 3-layer Cross-Attention Fusion
  - **Student**: MobileViT-xxs (`apple/mobilevit-xx-small`) + BERT-mini (`prajjwal1/bert-mini`) + 1-layer Cross-Attention Fusion
- **Distillation Loss Pipeline**: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \alpha \mathcal{L}_{\text{KL}} + \beta \mathcal{L}_{\text{MSE}} + \gamma \mathcal{L}_{\text{CRD}}$ (`MedKDCombinedLoss`)

---

## 2. Model Footprint & Parameter Efficiency

| Metric | Teacher Model (`ViT-Base + Bio_ClinicalBERT`) | Student Model (`MobileViT-xxs + BERT-mini`) | Reduction / Efficiency Gain |
| :--- | :---: | :---: | :---: |
| **Total Parameters** | **195.88 M** | **13.06 M** | **15.0× reduction (93.3% smaller)** |
| **Model Weights (Disk / Memory)** | **747.25 MB** | **49.85 MB** | **15.0× reduction** |
| **Static VRAM Footprint (Loaded in GPU)** | **747.38 MB** | **49.92 MB** | **15.0× less static VRAM** |

---

## 3. Training & Distillation Resource Usage (Batch Size = 16)

During **Teacher Training**, full forward and backward passes (with optimizer states) are computed for the 195.9M parameter model.  
During **Student Distillation**, the Teacher is frozen in evaluation mode (`eval()`, `torch.no_grad()`), and only the Student receives gradient updates and optimizer tracking.

### A. MedPix-2-0 Dataset (1,653 Train Samples, 200 Test Samples)

| Resource Metric | Teacher Model Training | Student Model Distillation | Student Distillation Advantage |
| :--- | :---: | :---: | :---: |
| **Peak Allocated VRAM** | **6,604.6 MB (6.45 GB)** | **2,843.5 MB (2.78 GB)** | **57.0% reduction (−3.76 GB)** |
| **Peak Reserved VRAM** | **7,252.0 MB (7.08 GB)** | **3,180.0 MB (3.11 GB)** | **56.2% reduction (−3.97 GB)** |
| **Batch Step Time** | **530.54 ms** | **222.76 ms** | **2.38× faster per step** |
| **Training Throughput** | **30.16 samples/sec** | **71.83 samples/sec** | **+138% throughput** |
| **Full Epoch Duration** | **54.81 seconds** | **23.01 seconds** | **31.80s saved per epoch** |

### B. Wound-1-0 Dataset (1,094 Train Samples, 235 Test Samples)

| Resource Metric | Teacher Model Training | Student Model Distillation | Student Distillation Advantage |
| :--- | :---: | :---: | :---: |
| **Peak Allocated VRAM** | **6,603.8 MB (6.45 GB)** | **2,842.9 MB (2.78 GB)** | **57.0% reduction (−3.76 GB)** |
| **Peak Reserved VRAM** | **7,276.0 MB (7.11 GB)** | **3,252.0 MB (3.18 GB)** | **55.3% reduction (−3.93 GB)** |
| **Batch Step Time** | **534.62 ms** | **219.83 ms** | **2.43× faster per step** |
| **Training Throughput** | **29.93 samples/sec** | **72.78 samples/sec** | **+143% throughput** |
| **Full Epoch Duration** | **36.55 seconds** | **15.03 seconds** | **21.52s saved per epoch** |

---

## 4. Inference & Edge Deployment Resources (Batch Size = 16)

When deploying to client/edge environments, the teacher network is omitted entirely, and the student runs standalone:

| Dataset | Metric | Teacher Model Inference | Student Standalone Inference | Gain / Factor |
| :--- | :--- | :---: | :---: | :---: |
| **MedPix** | **Peak Allocated VRAM** | 3,255.5 MB (3.18 GB) | **1,830.6 MB (1.79 GB)** | **43.8% less VRAM** |
| | **Peak Reserved VRAM** | 3,966.0 MB (3.87 GB) | **2,538.0 MB (2.48 GB)** | **36.0% less VRAM** |
| | **Latency per sample** | 9.64 ms / sample | **0.95 ms / sample** | **10.2× faster (sub-ms)** |
| | **Throughput** | 103.70 samples/sec | **1,057.16 samples/sec** | **10.2× higher throughput** |
| **Wound** | **Peak Allocated VRAM** | 3,252.8 MB (3.18 GB) | **1,832.3 MB (1.79 GB)** | **43.7% less VRAM** |
| | **Peak Reserved VRAM** | 4,152.0 MB (4.05 GB) | **2,754.0 MB (2.69 GB)** | **33.7% less VRAM** |
| | **Latency per sample** | 9.71 ms / sample | **0.95 ms / sample** | **10.2× faster (sub-ms)** |
| | **Throughput** | 102.98 samples/sec | **1,055.42 samples/sec** | **10.2× higher throughput** |

---

## 5. Architectural & System Insights

1. **VRAM Reduction During Distillation**:
   - Despite having *both* Teacher (195.9M params) and Student (13.1M params) loaded concurrently in GPU VRAM during distillation, the peak allocated VRAM drops from **6.45 GB to 2.78 GB** (−57%).
   - *Reason*: The frozen teacher operates under `torch.no_grad()`, requiring no gradient storage or intermediate activation caching for backpropagation. The AdamW optimizer only allocates momentum/variance buffers for the lightweight student (13.1M parameters vs 195.9M parameters).

2. **Ultra-Fast Edge Inference**:
   - The student model processes inputs at **0.95 ms per sample** (over **1,050 samples/second**), achieving a **10.2× acceleration** compared to the teacher (9.64–9.71 ms).
   - This makes the student suitable for real-time edge execution on resource-constrained medical devices.

3. **Low Memory Footprint**:
   - The standalone student model requires only **~49.9 MB** of parameter memory, allowing it to easily fit within mobile, embedded, and edge GPUs with 2GB–4GB memory budgets.
