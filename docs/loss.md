# Loss Exploration — Test Set Performance

This document summarizes the test set performance across different knowledge distillation loss strategies (**Vanilla KD**, **CRD**, **RKD**, and **MMD**) for the ultra-edge student model (**MobileViT-XX-Small + BERT-Mini**, ~13.06M parameters) using **Cross Attention** fusion.

---

## 1. Wound-1-0 Dataset

- **Student Backbone**: MobileViT-XX-Small + BERT-Mini (13.06M parameters)
- **Teacher Backbone**: ViT-Base + Bio-ClinicalBERT (195.89M parameters)
- **Fusion**: Cross Attention
- **Tasks**:
  - **Task 1 (Type)**: 6 classes
  - **Task 2 (Severity)**: 3 classes

| Loss Strategy | Task 1 (Type) Acc | Task 1 (Type) F1 | Task 1 (Type) AUC | Task 2 (Severity) Acc | Task 2 (Severity) F1 | Task 2 (Severity) AUC | Average Acc | Macro F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Vanilla KD** | 80.43% | 81.14% | 0.9746 | 88.09% | 85.26% | 0.9867 | 84.26% | 83.20% |
| **CRD** | 83.40% | 83.35% | **0.9860** | 79.15% | 74.43% | 0.9331 | 81.28% | 78.89% |
| **RKD** | **84.26%** | **83.92%** | 0.9824 | **92.77%** | **91.67%** | **0.9910** | **88.51%** | **87.79%** |
| **MMD** | 74.04% | 76.40% | 0.9653 | 90.64% | 89.16% | 0.9827 | 82.34% | 82.78% |

---

## 2. MedPix Dataset

- **Student Backbone**: MobileViT-XX-Small + BERT-Mini (13.06M parameters)
- **Teacher Backbone**: ViT-Base + Bio-ClinicalBERT (195.89M parameters)
- **Fusion**: Cross Attention
- **Tasks**:
  - **Task 1 (Modality)**: 2 classes
  - **Task 2 (Location)**: 5 classes

| Loss Strategy | Task 1 (Modality) Acc | Task 1 (Modality) F1 | Task 1 (Modality) AUC | Task 2 (Location) Acc | Task 2 (Location) F1 | Task 2 (Location) AUC | Average Acc | Macro F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Vanilla KD** | **97.50%** | **97.50%** | 0.9876 | 73.50% | 62.89% | 0.8921 | 85.50% | 80.20% |
| **CRD** | 96.50% | 96.50% | 0.9864 | **81.00%** | **77.26%** | 0.9281 | **88.75%** | **86.88%** |
| **RKD** | 95.50% | 95.49% | 0.9829 | 80.50% | 75.49% | **0.9298** | 88.00% | 85.49% |
| **MMD** | 97.00% | 97.00% | **0.9964** | 67.00% | 53.25% | 0.8877 | 82.00% | 75.12% |

---

## Experiment Configurations & Log Directories

| Dataset | Loss | Config Path | Log Directory |
| :--- | :--- | :--- | :--- |
| Wound | Vanilla KD | `config/loss-explore-hp-wound/wound-mobilevit_xx_small-bert-mini-true-vanilla.yaml` | `logs/loss-explore-hp/wound-mobilevit_xx_small-bert-mini-true-vanilla` |
| Wound | CRD | `config/loss-explore-hp-wound/wound-mobilevit_xx_small-bert-mini-crd.yaml` | `logs/loss-explore-hp/wound-mobilevit_xx_small-bert-mini-crd` |
| Wound | RKD | `config/loss-explore-hp-wound/wound-mobilevit_xx_small-bert-mini-rkd.yaml` | `logs/loss-explore-hp/wound-mobilevit_xx_small-bert-mini-rkd` |
| Wound | MMD | `config/loss-explore-hp-wound/wound-mobilevit_xx_small-bert-mini-mmd.yaml` | `logs/loss-explore-hp/wound-mobilevit_xx_small-bert-mini-mmd` |
| MedPix | Vanilla KD | `config/loss-explore-hp-medpix/medpix-mobilevit_xx_small-bert-mini-true-vanilla.yaml` | `logs/loss-explore-hp/medpix-mobilevit_xx_small-bert-mini-true-vanilla` |
| MedPix | CRD | `config/loss-explore-hp-medpix/medpix-mobilevit_xx_small-bert-mini-crd.yaml` | `logs/loss-explore-hp/medpix-mobilevit_xx_small-bert-mini-crd` |
| MedPix | RKD | `config/loss-explore-hp-medpix/medpix-mobilevit_xx_small-bert-mini-rkd.yaml` | `logs/loss-explore-hp/medpix-mobilevit_xx_small-bert-mini-rkd` |
| MedPix | MMD | `config/loss-explore-hp-medpix/medpix-mobilevit_xx_small-bert-mini-mmd.yaml` | `logs/loss-explore-hp/medpix-mobilevit_xx_small-bert-mini-mmd` |
