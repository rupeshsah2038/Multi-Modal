# Comparative Analysis: Two-Phase Distillation vs. Single-Phase Joint Training

A direct empirical comparison evaluating whether training Teacher and Student simultaneously in a **Single Unified Phase** performs better or worse than the canonical **Two-Phase Knowledge Distillation** pipeline.

### Pipeline Overview:
- **Two-Phase Distillation (Offline KD)**: Phase 1 trains Teacher to convergence ($E_T=3$). Phase 2 freezes Teacher and distills Student ($E_S=10$).
- **Single-Phase Joint Training (Online KD)**: Teacher and Student start from pretrained backbones and co-evolve simultaneously ($E_{\text{joint}}=10$), with Student distilling from dynamic teacher features in every batch.

## Dataset: MEDPIX

### 1. Classification Performance Comparison (Student & Teacher)
| Model & Training Paradigm | Overall Acc (%) | Overall Macro-F1 (%) | Modality F1 (%) | Location F1 (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Teacher: Two-Phase (Pre-trained)** | 84.50 ± 0.00 | 83.43 ± 0.00 | — | — |
| **Teacher: Single-Phase (Co-trained)** | 90.05 ± 1.92 | 87.15 ± 3.02 | — | — |
| **Student: Two-Phase Distillation (Baseline)** | **90.65 ± 1.04** | **89.28 ± 1.19** | **96.20 ± 1.21** | **82.37 ± 2.89** |
| **Student: Single-Phase Joint Training (Proposed)** | 88.95 ± 2.74 | 86.69 ± 3.21 | 94.70 ± 1.08 | 78.68 ± 7.33 |


### 2. Hardware Resource & Efficiency Profiling
| Paradigm | Peak Allocated VRAM | Peak Reserved VRAM | Step Latency | Throughput | Training Duration |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Two-Phase Distillation** | 6.45 GB (Phase 1) / 2.78 GB (Phase 2) | 7.08 GB (Phase 1) / 3.11 GB (Phase 2) | ~530 ms (Phase 1) / ~222 ms (Phase 2) | ~30.2 s/s (Phase 1) / ~71.8 s/s (Phase 2) | ~395s total (3 ep Teacher + 10 ep Student) |
| **Single-Phase Joint Training** | **7.80 GB** | **8.54 GB** | **615.9 ms** | **26.0 samples/s** | **709.4 s** |

---

## Dataset: WOUND

### 1. Classification Performance Comparison (Student & Teacher)
| Model & Training Paradigm | Overall Acc (%) | Overall Macro-F1 (%) | Type F1 (%) | Severity F1 (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Teacher: Two-Phase (Pre-trained)** | 89.15 ± 0.00 | 89.03 ± 0.00 | — | — |
| **Teacher: Single-Phase (Co-trained)** | 91.45 ± 0.96 | 91.71 ± 1.02 | — | — |
| **Student: Two-Phase Distillation (Baseline)** | **86.55 ± 1.22** | **85.35 ± 1.30** | **79.08 ± 2.64** | **91.63 ± 0.89** |
| **Student: Single-Phase Joint Training (Proposed)** | 87.19 ± 1.31 | 87.18 ± 2.38 | 82.51 ± 4.17 | 91.86 ± 0.66 |


### 2. Hardware Resource & Efficiency Profiling
| Paradigm | Peak Allocated VRAM | Peak Reserved VRAM | Step Latency | Throughput | Training Duration |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Two-Phase Distillation** | 6.45 GB (Phase 1) / 2.78 GB (Phase 2) | 7.08 GB (Phase 1) / 3.11 GB (Phase 2) | ~530 ms (Phase 1) / ~222 ms (Phase 2) | ~30.2 s/s (Phase 1) / ~71.8 s/s (Phase 2) | ~260s total (3 ep Teacher + 10 ep Student) |
| **Single-Phase Joint Training** | **7.97 GB** | **8.73 GB** | **615.0 ms** | **26.0 samples/s** | **502.5 s** |

---
