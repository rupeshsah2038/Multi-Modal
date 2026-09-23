# Multimodal Ablation & Robustness Study: MobileViT-xxs + BERT-mini

Evaluated across both **MedPix** and **Wound** datasets averaged across seeds (42, 43, 44, 45, 46).

- **Image-Only Baseline**: MobileViT-xxs trained from scratch.
- **Text-Only Baseline**: BERT-mini trained from scratch.
- **Proposed Multimodal Model**: MobileViT-xxs + BERT-mini (Cross-Attention).
- **Perturbations on Trained Model**: Mismatch-Text, Noise ($\sigma=0.1, 0.2$), and Missing Modality (30% dropout).

## Dataset: MEDPIX

| Setting / Condition | Type | Overall Acc (%) | Overall Macro-F1 (%) | Modality F1 (%) | Location F1 (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Proposed Full Multimodal (Clean)** | Proposed (Upper Bound) | 90.65 ± 1.04 | 89.28 ± 1.19 | 96.20 ± 1.21 | 82.37 ± 2.89 |
| **Image-Only (MobileViT-xxs)** | Unimodal Baseline | 81.55 ± 0.99 | 76.84 ± 1.37 | 96.90 ± 1.60 | 56.79 ± 3.05 |
| **Text-Only (BERT-mini)** | Unimodal Baseline | 77.35 ± 2.02 | 75.71 ± 2.47 | 80.73 ± 2.75 | 70.70 ± 2.39 |
| **Mismatch-Text Pairing** | Cross-Modal Perturbation | 82.35 ± 1.10 | 79.43 ± 1.48 | 96.40 ± 1.60 | 62.47 ± 2.96 |
| **Noise (Image Gaussian $\sigma=0.1$)** | Robustness Perturbation | 87.25 ± 1.75 | 86.02 ± 1.64 | 90.87 ± 1.66 | 81.17 ± 3.80 |
| **Noise (Image Gaussian $\sigma=0.2$)** | Robustness Perturbation | 80.30 ± 2.93 | 78.28 ± 2.84 | 77.52 ± 4.72 | 79.04 ± 2.60 |
| **Missing Modality (30% Dropout)** | Missingness Perturbation | 79.40 ± 2.18 | 77.16 ± 2.02 | 87.05 ± 2.16 | 67.27 ± 2.87 |
| **Missing Image (Text Only Available)** | Degradation Evaluation | 77.20 ± 3.04 | 76.36 ± 2.68 | 77.27 ± 2.62 | 75.45 ± 4.17 |
| **Missing Text (Image Only Available)** | Degradation Evaluation | 79.20 ± 4.63 | 71.01 ± 5.83 | 96.79 ± 2.20 | 45.22 ± 12.02 |


## Dataset: WOUND

| Setting / Condition | Type | Overall Acc (%) | Overall Macro-F1 (%) | Type F1 (%) | Severity F1 (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Proposed Full Multimodal (Clean)** | Proposed (Upper Bound) | 86.55 ± 1.22 | 85.35 ± 1.30 | 79.08 ± 2.64 | 91.63 ± 0.89 |
| **Image-Only (MobileViT-xxs)** | Unimodal Baseline | 81.87 ± 1.10 | 80.97 ± 1.76 | 82.71 ± 1.41 | 79.23 ± 2.90 |
| **Text-Only (BERT-mini)** | Unimodal Baseline | 65.36 ± 1.24 | 65.93 ± 1.25 | 40.77 ± 2.88 | 91.09 ± 1.72 |
| **Mismatch-Text Pairing** | Cross-Modal Perturbation | 61.53 ± 1.97 | 54.42 ± 3.68 | 67.64 ± 5.01 | 41.19 ± 3.58 |
| **Noise (Image Gaussian $\sigma=0.1$)** | Robustness Perturbation | 64.43 ± 4.85 | 65.43 ± 3.45 | 41.38 ± 6.66 | 89.48 ± 1.00 |
| **Noise (Image Gaussian $\sigma=0.2$)** | Robustness Perturbation | 51.96 ± 3.11 | 53.02 ± 4.09 | 21.64 ± 3.92 | 84.40 ± 5.20 |
| **Missing Modality (30% Dropout)** | Missingness Perturbation | 65.79 ± 2.31 | 64.95 ± 2.09 | 58.62 ± 5.55 | 71.29 ± 7.34 |
| **Missing Image (Text Only Available)** | Degradation Evaluation | 40.00 ± 12.61 | 39.42 ± 12.80 | 13.28 ± 1.44 | 65.55 ± 24.71 |
| **Missing Text (Image Only Available)** | Degradation Evaluation | 73.11 ± 1.19 | 64.14 ± 4.52 | 73.41 ± 4.35 | 54.88 ± 5.40 |

