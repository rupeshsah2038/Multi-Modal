# Loss-Explore Experiment Results

## Overview
Systematic comparison of **5 distillation losses** across **2 datasets** (MedPix-2-0, Wound-1-0) using a fixed fusion architecture and backbones.

**Configuration:**
- **Fusion:** `cross_attention` (teacher and student)
- **Teacher:** vit-large (vision) + bio-clinical-bert (text)
- **Student:** vit-base (vision) + distilbert (text)
- **Losses tested:** `vanilla`, `combined`, `crd`, `rkd`, `mmd`
- **Training:** teacher_epochs=3, student_epochs=10, batch_size=16
- **Device:** cuda:4

All numbers below are **test-set** metrics from `logs/loss-explore/*/results.json`.

---

## MedPix-2-0 Results (Loss Comparison)

Task mapping:
- Task 1: `modality` (CT vs MR)
- Task 2: `location` (body location)

### Test Performance Summary

| Loss      | Modality Acc | Modality F1 | Modality AUC | Location Acc | Location F1 | Location AUC | Infer (ms) |
|-----------|-------------:|------------:|-------------:|-------------:|------------:|-------------:|-----------:|
| vanilla   | 0.975 | 0.975 | 0.992 | 0.835 | 0.783 | 0.954 | 13.44 |
| combined  | 0.985 | 0.985 | 0.991 | 0.880 | 0.841 | 0.944 | 7.86 |
| crd       | 0.505 | 0.491 | 0.524 | 0.175 | 0.172 | 0.507 | 7.94 |
| mmd       | 0.555 | 0.531 | 0.585 | 0.140 | 0.133 | 0.506 | 8.22 |
| rkd       | 0.765 | 0.762 | 0.871 | 0.355 | 0.150 | 0.492 | 8.16 |

(F1 and AUC values rounded for readability.)

### Critical Observations — MedPix

**Top performer: `combined`**

- Highest overall performance on both tasks:
  - Modality F1 ≈ 0.985 (slightly better than `vanilla`).
  - Location F1 ≈ 0.84 (clearly better than `vanilla` ≈ 0.78).
- Despite extra loss terms, inference is actually **faster** than `vanilla` (7.86 ms vs 13.44 ms), due to differences in the trained student.
- Best choice when both tasks matter and latency is important.

**`vanilla` as strong but dominated baseline**

- Very strong modality F1 (≈ 0.975), but weaker location F1 (≈ 0.78) hurts average performance.
- Inference is significantly slower than `combined` in this experiment.
- Reasonable baseline, but consistently outperformed by `combined` on MedPix.

**Advanced feature losses (`crd`, `mmd`, `rkd`)**

- All three substantially underperform on **location**:
  - `crd` and `mmd`: location F1 collapses to ≈ 0.13–0.17.
  - `rkd`: modality F1 is moderate (~0.76) but location F1 ≈ 0.15, with AUC near 0.5.
- Location AUC values near 0.5 indicate near-random behavior for the harder task.
- In this cross-attention + vit-base setting, these losses are not competitive.

**MedPix takeaway**

- Best choice: `combined` loss (strong gains on the harder location task and faster inference than `vanilla`).
- Baseline: `vanilla` remains serviceable but is outperformed across most metrics.
- Not recommended in this configuration: `crd`, `mmd`, `rkd` (severe degradation on location).

---

## Wound-1-0 Results (Loss Comparison)

Task mapping (unified to modality/location style):
- Task 1: `type`     → modality-like task
- Task 2: `severity` → location-like task

### Test Performance Summary

| Loss      | Type Acc | Type F1 | Type AUC | Severity Acc | Severity F1 | Severity AUC | Infer (ms) |
|-----------|---------:|--------:|---------:|-------------:|------------:|-------------:|-----------:|
| vanilla   | 0.915 | 0.937 | 0.990 | 0.928 | 0.923 | 0.991 | 7.13 |
| combined  | 0.911 | 0.920 | 0.993 | 0.936 | 0.930 | 0.992 | 12.51 |
| crd       | 0.115 | 0.100 | 0.529 | 0.357 | 0.299 | 0.467 | 12.03 |
| mmd       | 0.136 | 0.030 | 0.498 | 0.485 | 0.468 | 0.674 | 11.78 |
| rkd       | 0.047 | 0.025 | 0.485 | 0.532 | 0.393 | 0.651 | 12.33 |

### Critical Observations — Wound

**Top performer: `vanilla`**

- Best overall average F1 (≈ 0.93) with:
  - Strongest type F1 (≈ 0.94).
  - Very good severity F1 (≈ 0.92).
- Fastest inference among all losses (7.13 ms), which makes it the best deployment candidate on Wound.

**`combined` as close second**

- Slightly better on severity F1 and AUC, but lower type F1 than `vanilla`.
- Noticeably slower (≈ 12.5 ms), so the small severity gain comes at a clear latency cost.
- Reasonable alternative when severity is the primary focus and extra latency is acceptable.

**Advanced feature losses (`crd`, `mmd`, `rkd`)**

- All three severely degrade **type** performance:
  - Type F1 drops to ≈ 0.03–0.10.
- Severity F1 is sometimes moderate (≈ 0.30–0.47), but that does not compensate for the collapse on type.
- Type AUCs close to 0.5 indicate near-random behavior on the primary task.
- Not viable choices for this dataset under the current architecture and training regime.

**Wound takeaway**

- Best choice: `vanilla` loss (best average F1 and best latency).
- Alternative: `combined` when maximum severity performance is desired and higher inference cost is acceptable.
- Not recommended here: `crd`, `mmd`, `rkd`.

---

## Cross-Dataset Loss Analysis

### High-level ranking

**MedPix:**

1. `combined` — best overall; stronger location performance and faster than `vanilla`.
2. `vanilla` — strong, but weaker on location and slower.
3. `rkd` — partly preserves modality, but fails on location.
4. `crd` / `mmd` — collapse location; not usable in this configuration.

**Wound:**

1. `vanilla` — best overall and fastest.
2. `combined` — strong, slightly better severity but slower.
3. `crd` / `mmd` / `rkd` — poor type performance; experimental only.

### Dataset-specific recommendations

| Scenario                      | MedPix Recommendation | Wound Recommendation |
|------------------------------|------------------------|----------------------|
| Single loss for both         | `combined`             | acceptable (slightly behind `vanilla`) |
| MedPix-focused               | `combined`             | —                    |
| Wound-focused                | —                      | `vanilla`            |
| Latency-critical (both)      | `combined` (MedPix)    | `vanilla` (Wound)    |

### When to use which loss

- `vanilla`  
  - Use when you want a simple, fast, and strong baseline, especially on Wound.
  - Recommended for deployment on Wound-1-0.

- `combined`  
  - Use when MedPix performance (especially location) is important, or when you want a single loss that works reasonably well on both datasets.
  - Provides a good compromise: best on MedPix and competitive on Wound.

- `crd`, `mmd`, `rkd`  
  - Not recommended as drop-in replacements in this cross-attention setup.
  - Only worth revisiting with significantly different architectures or training regimes.

---

## Summary and Practical Guidance

- For **MedPix-2-0**, the **combined loss** is clearly best, especially on the harder location task, and even offers better latency than `vanilla` in this experiment.
- For **Wound-1-0**, **vanilla loss** gives the best overall trade-off between accuracy and latency.
- If you need a **single configuration** across both datasets, choose **combined**; it is optimal on MedPix and close to optimal on Wound.
- Advanced representation-based losses (`crd`, `mmd`, `rkd`) do not improve results in this project’s current setting and generally harm robustness.

These conclusions are derived directly from the test metrics in `logs/loss-explore/*/results.json`.