# Ultra-Edge Experiment Results

## Overview
Comparison of **lightweight student configurations** designed for ultra-edge deployment across **MedPix-2-0** and **Wound-1-0**.

**Configuration:**
- **Teacher:** vit-base (vision) + bio-clinical-bert (text), fusion_dim=256, fusion_layers=2 — **~194.7M parameters**
- **Student variants (vision / text):**
  - `deit-small` / `distilbert` — **~89.3M parameters** (21.2M vision + 65.9M text + 2.2M fusion/proj/heads)
  - `deit-small` / `minilm` — **~45.5M parameters** (21.2M vision + 22.3M text + 2.0M fusion/proj/heads)
  - `deit-tiny` / `distilbert` — **~73.0M parameters** (5.3M vision + 65.9M text + 1.8M fusion/proj/heads)
  - `deit-tiny` / `minilm` — **~29.2M parameters** (5.3M vision + 22.3M text + 1.6M fusion/proj/heads)
- **Fusion:** `cross_attention`
- **Loss:** `combined`
- **Training:** teacher_epochs=3, student_epochs=10, teacher_lr=1e-5, student_lr=3e-4
- **Device:** cuda:4

All numbers below are **test-set** metrics from `logs/ultra-edge/*/results.json`.

---

## MedPix-2-0 Results (Ultra-Edge Students)

Task mapping:
- Task 1: `modality` (CT vs MR)
- Task 2: `location` (body location)

### Test Performance Summary

| Run ID                         | Student (vision / text)   | Params (M) | Modality Acc | Modality F1 | Modality AUC | Location Acc | Location F1 | Location AUC | Infer (ms) |
|--------------------------------|---------------------------|----------:|-------------:|------------:|-------------:|-------------:|------------:|-------------:|-----------:|
| medpix-deit_small-distilbert   | deit-small / distilbert   | 89.3 | 0.975 | 0.975 | 0.989 | 0.895 | 0.861 | 0.945 | 10.27 |
| medpix-deit_small-minilm       | deit-small / minilm       | 45.5 | 0.970 | 0.970 | 0.997 | 0.850 | 0.813 | 0.944 | 6.75 |
| medpix-deit_tiny-distilbert    | deit-tiny / distilbert    | 73.0 | 0.910 | 0.910 | 0.984 | 0.825 | 0.777 | 0.941 | 9.80 |
| medpix-deit_tiny-minilm        | deit-tiny / minilm        | 29.2 | 0.965 | 0.965 | 0.997 | 0.875 | 0.825 | 0.944 | 7.42 |

(F1 and AUC values rounded for readability.)

### Critical Observations — MedPix

**Best accuracy: `medpix-deit_small-distilbert`**

- Highest average performance:
  - Modality F1 ≈ 0.975.
  - Location F1 ≈ 0.86.
- However, it is the **slowest** configuration (~10.3 ms), so best suited when accuracy is the primary concern and latency is secondary.

**Best ultra-edge trade-off: `medpix-deit_tiny-minilm`**

- Very competitive accuracy:
  - Modality F1 ≈ 0.965.
  - Location F1 ≈ 0.825.
- Inference ~7.4 ms, significantly faster than the small/distilbert model and close to the fastest variant.
- Among truly small models, this is the best accuracy–latency compromise.

**`medpix-deit_small-minilm`**

- Slightly worse location performance than the tiny/minilm variant (location F1 ≈ 0.81 vs 0.83), with similar AUC.
- Fastest MedPix model (~6.75 ms) with still-strong modality F1 (≈ 0.97).
- Good option when latency is the absolute priority and small accuracy losses are acceptable.

**`medpix-deit_tiny-distilbert`**

- Lowest overall performance:
  - Modality F1 ≈ 0.91.
  - Location F1 ≈ 0.78.
- Latency (~9.8 ms) is not low enough to compensate for the drop in accuracy.
- Dominated by other configurations.

**MedPix takeaway**

- Best pure accuracy: `deit-small` + `distilbert`.
- Best ultra-edge trade-off: `deit-tiny` + `minilm` (strong performance and low latency).
- Lowest-latency option: `deit-small` + `minilm`, at the cost of slightly weaker location performance.

---

## Wound-1-0 Results (Ultra-Edge Students)

Task mapping (unified to modality/location style):
- Task 1: `type`     → modality-like task
- Task 2: `severity` → location-like task

### Test Performance Summary

| Run ID                          | Student (vision / text)   | Params (M) | Type Acc | Type F1 | Type AUC | Severity Acc | Severity F1 | Severity AUC | Infer (ms) |
|---------------------------------|---------------------------|----------:|---------:|--------:|---------:|-------------:|------------:|-------------:|-----------:|
| wound-deit_small-distilbert     | deit-small / distilbert   | 89.3 | 0.855 | 0.879 | 0.987 | 0.928 | 0.920 | 0.986 | 10.04 |
| wound-deit_small-minilm         | deit-small / minilm       | 45.5 | 0.860 | 0.885 | 0.979 | 0.940 | 0.940 | 0.993 | 7.84 |
| wound-deit_tiny-distilbert      | deit-tiny / distilbert    | 73.0 | 0.774 | 0.810 | 0.973 | 0.872 | 0.855 | 0.971 | 9.14 |
| wound-deit_tiny-minilm          | deit-tiny / minilm        | 29.2 | 0.762 | 0.777 | 0.976 | 0.919 | 0.905 | 0.990 | 6.62 |

### Critical Observations — Wound

**Best overall: `wound-deit_small-minilm`**

- Highest average performance:
  - Type F1 ≈ 0.885.
  - Severity F1 ≈ 0.94.
- Also significantly faster (~7.84 ms) than `wound-deit_small-distilbert` (~10.04 ms).
- Dominates the other configurations on both **accuracy** and **latency**.

**`wound-deit_small-distilbert`**

- Strong baseline with good type and severity F1 (≈ 0.88 and ≈ 0.92).
- Latency (~10.0 ms) is highest among Wound ultra-edge runs, so it is mainly a reference point.

**Latency-focused option: `wound-deit_tiny-minilm`**

- Fastest Wound configuration (~6.62 ms).
- Reasonable severity F1 (≈ 0.91) but notably lower type F1 (≈ 0.78) than the small models.
- Good for extremely tight latency budgets where some loss in type accuracy is acceptable.

**`wound-deit_tiny-distilbert`**

- Lower type and severity performance than the small models (F1 ≈ 0.81 and ≈ 0.86).
- Latency (~9.14 ms) is not low enough to justify the accuracy drop.
- Dominated by `wound-deit_small-minilm` and `wound-deit_tiny-minilm` depending on the priority.

**Wound takeaway**

- Best overall and recommended ultra-edge configuration: `deit-small` + `minilm`.
- Fastest configuration: `deit-tiny` + `minilm`, with a clear but controlled loss in type accuracy.

---

## Cross-Dataset Ultra-Edge Analysis

### Accuracy-focused ranking

**MedPix (approximate average over tasks):**
1. deit-small / distilbert — best F1 on both tasks.
2. deit-tiny / minilm — slightly lower but still strong, with better latency.
3. deit-small / minilm — strong modality, weaker location.
4. deit-tiny / distilbert — weakest.

**Wound (approximate average over tasks):**
1. deit-small / minilm — best type and severity balance.
2. deit-small / distilbert — strong, but slower.
3. deit-tiny / minilm — good severity, weaker type.
4. deit-tiny / distilbert — lowest overall.

### Latency-focused ranking

**MedPix:**
1. deit-small / minilm — fastest (~6.75 ms).
2. deit-tiny / minilm — second-fastest (~7.42 ms) with better accuracy.
3. deit-tiny / distilbert — ~9.80 ms.
4. deit-small / distilbert — ~10.27 ms.

**Wound:**
1. deit-tiny / minilm — fastest (~6.62 ms).
2. deit-small / minilm — ~7.84 ms.
3. deit-tiny / distilbert — ~9.14 ms.
4. deit-small / distilbert — ~10.04 ms.

### Recommended configurations

| Priority / Scenario              | MedPix Recommendation           | Wound Recommendation             |
|----------------------------------|---------------------------------|----------------------------------|
| Max accuracy                     | deit-small / distilbert         | deit-small / minilm              |
| Best ultra-edge trade-off        | deit-tiny / minilm              | deit-small / minilm              |
| Strict latency constraint        | deit-small / minilm             | deit-tiny / minilm               |
| Single student for both datasets | deit-small / minilm             | deit-small / minilm              |

---

## Summary and Practical Guidance

- For **MedPix-2-0**, use:
  - `deit-small` + `distilbert` when accuracy is the top priority.
  - `deit-tiny` + `minilm` when you need a strong ultra-edge trade-off between accuracy and latency.

- For **Wound-1-0**, use:
  - `deit-small` + `minilm` as the default ultra-edge configuration (best accuracy and very good latency).
  - `deit-tiny` + `minilm` only if you must minimize inference time further and can tolerate lower type accuracy.

- For a **single ultra-edge student** across both datasets, `deit-small` + `minilm` is the most balanced choice.

All conclusions are based directly on `logs/ultra-edge/*/results.json`.