# Multimodal Ablation & Robustness Study: MobileViT-xxs + BERT-mini

Evaluated across both **MedPix** and **Wound** datasets averaged across seeds (42, 43, 44, 45, 46).

- **Image-Only Baseline**: MobileViT-xxs trained from scratch.
- **Text-Only Baseline**: BERT-mini trained from scratch.
- **Proposed Multimodal Model**: MobileViT-xxs + BERT-mini (Cross-Attention).
- **Perturbations on Trained Model**: Mismatch-Text, Noise ($\sigma=0.1, 0.2$), and Missing Modality (30% dropout).

## Dataset: MEDPIX

| Setting / Condition | Type | Overall Acc (%) | Overall Macro-F1 (%) | Modality F1 (%) | Location F1 (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Image-Only (MobileViT-xxs)** | Unimodal Baseline | 81.55 ± 0.99 | 76.84 ± 1.37 | 96.90 ± 1.60 | 56.79 ± 3.05 |
| **Text-Only (BERT-mini)** | Unimodal Baseline | 77.35 ± 2.02 | 75.71 ± 2.47 | 80.73 ± 2.75 | 70.70 ± 2.39 |


## Dataset: WOUND

| Setting / Condition | Type | Overall Acc (%) | Overall Macro-F1 (%) | Type F1 (%) | Severity F1 (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Image-Only (MobileViT-xxs)** | Unimodal Baseline | 81.87 ± 1.10 | 80.97 ± 1.76 | 82.71 ± 1.41 | 79.23 ± 2.90 |
| **Text-Only (BERT-mini)** | Unimodal Baseline | 65.36 ± 1.24 | 65.93 ± 1.25 | 40.77 ± 2.88 | 91.09 ± 1.72 |

