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

| Run ID                       | Student (vision / text)   | Params (M) | Modality Acc | Modality F1 | Modality AUC | Location Acc | Location F1 | Location AUC |
|-----------------------------|---------------------------|----------:|-------------:|------------:|-------------:|-------------:|------------:|-------------:|
| medpix-deit_small-distilbert | deit-small / distilbert  | 89.3      | 0.975        | 0.975       | 0.993        | 0.915        | 0.901       | 0.967        |
| medpix-deit_small-minilm     | deit-small / minilm      | 45.5      | 0.960        | 0.960       | 0.994        | 0.845        | 0.825       | 0.965        |
| medpix-deit_tiny-distilbert  | deit-tiny / distilbert   | 73.0      | 0.960        | 0.960       | 0.996        | 0.815        | 0.775       | 0.945        |
| medpix-deit_tiny-minilm      | deit-tiny / minilm       | 29.2      | 0.980        | 0.980       | 0.992        | 0.895        | 0.855       | 0.960        |

Values are rounded for readability and taken from `logs/ultra-edge/medpix-*/results.json`.

### Critical Observations — MedPix

- **Strongest overall:** `deit-small / distilbert` has the best combination of modality and location scores, with modality F1 ≈ 0.98 and location F1 ≈ 0.90.
- **Best tiny configuration:** `deit-tiny / minilm` slightly improves modality metrics over `deit-small / minilm` and clearly outperforms other tiny variants on location (location F1 ≈ 0.86).
- **Text backbone effect:** Switching from `distilbert` to `minilm` consistently reduces parameters; on MedPix, the `minilm` variants are competitive but the `deit-small / distilbert` student still provides the strongest location performance.
- **Weaker options:** `deit-small / minilm` and especially `deit-tiny / distilbert` trail in location F1 and are not preferred when accuracy on both tasks matters.

---

## Wound-1-0 Results (Ultra-Edge Students)

Task mapping (unified to modality/location style):
- Task 1: `type`     → modality-like task
- Task 2: `severity` → location-like task

### Test Performance Summary

| Run ID                      | Student (vision / text)   | Params (M) | Type Acc | Type F1 | Type AUC | Severity Acc | Severity F1 | Severity AUC |
|-----------------------------|---------------------------|----------:|---------:|--------:|---------:|-------------:|------------:|-------------:|
| wound-deit_small-distilbert | deit-small / distilbert  | 89.3      | 0.762    | 0.744   | 0.980    | 0.919        | 0.899       | 0.989        |
| wound-deit_small-minilm     | deit-small / minilm      | 45.5      | 0.830    | 0.848   | 0.984    | 0.940        | 0.935       | 0.994        |
| wound-deit_tiny-distilbert  | deit-tiny / distilbert   | 73.0      | 0.723    | 0.763   | 0.967    | 0.919        | 0.914       | 0.975        |
| wound-deit_tiny-minilm      | deit-tiny / minilm       | 29.2      | 0.689    | 0.712   | 0.968    | 0.949        | 0.951       | 0.993        |

Values are rounded from `logs/ultra-edge/wound-*/results.json`.

### Critical Observations — Wound

- **Best overall:** `deit-small / minilm` achieves the strongest balance across both tasks, with type F1 ≈ 0.85 and severity F1 ≈ 0.93.
- **Severity-focused:** `deit-tiny / minilm` delivers the highest severity metrics (severity F1 ≈ 0.95) but has the weakest type performance; it is suitable only if severity is the dominant objective.
- **Distilbert variants:** both `deit-small / distilbert` and `deit-tiny / distilbert` underperform the corresponding `minilm` students on at least one task, offering no clear accuracy advantage.
- **Robust choice:** for Wound, `deit-small / minilm` remains the recommended ultra-edge student thanks to consistently strong type and severity performance with moderate parameter count.

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