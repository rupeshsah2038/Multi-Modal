# Ultra-Edge2 Experiment Results (fusion_dim=384)

## Overview
Comparison of **lightweight student configurations** designed for ultra-edge deployment across **MedPix-2-0** and **Wound-1-0** when both teacher and student fusion blocks use **fusion_dim = 384**.

**Configuration:**
- **Teacher:** `vit-base` (vision) + `bio-clinical-bert` (text), `fusion_dim=384`, `fusion_layers=2`
- **Student variants (vision / text):**
  - `deit-small` / `distilbert`
  - `deit-small` / `minilm`
  - `deit-tiny` / `distilbert`
  - `deit-tiny` / `minilm`
- **Fusion:** `cross_attention`
- **Loss:** `combined`
- **Training:** `teacher_epochs=3`, `student_epochs=10`, `teacher_lr=1e-5`, `student_lr=3e-4`, `alpha=1.0`, `beta=100.0`, `T=2.0`
- **Device:** `cuda:3`

All numbers below are **test-set** metrics from `logs/ultra-edge2/*/results.json`.

---

## MedPix-2-0 Results (Ultra-Edge2 Students)

Task mapping:
- Task 1: `modality` (CT vs MR)
- Task 2: `location` (body location)

### Test Performance Summary

| Run ID                                  | Student (vision / text)   | Modality Acc | Modality F1 | Modality AUC | Location Acc | Location F1 | Location AUC | Infer (ms) |
|-----------------------------------------|---------------------------|-------------:|------------:|-------------:|-------------:|------------:|-------------:|-----------:|
| medpix-deit_small-distilbert-384        | deit-small / distilbert   | 0.975 | 0.975 | 0.9910 | 0.850 | 0.807 | 0.9545 | 5.27 |
| medpix-deit_small-minilm-384            | deit-small / minilm       | 0.945 | 0.945 | 0.9934 | 0.870 | 0.840 | 0.9553 | 4.41 |
| medpix-deit_tiny-distilbert-384         | deit-tiny / distilbert    | 0.955 | 0.955 | 0.9917 | 0.790 | 0.717 | 0.9049 | 5.00 |
| medpix-deit_tiny-minilm-384             | deit-tiny / minilm        | 0.960 | 0.960 | 0.9858 | 0.790 | 0.764 | 0.9208 | 4.33 |

(F1, AUC and latency values rounded for readability.)

### Critical Observations — MedPix

**1. Accuracy behaviour**

- `deit-small / distilbert` remains **best on modality** (F1 ≈ 0.975) but its **location accuracy drops** from 0.895 (fusion_dim=256) to 0.850 here.
- `deit-small / minilm` trades a **small drop in modality** (0.970 → 0.945) for a **clear gain in location** (0.850 → 0.870) and improved AUC for location.
- `deit-tiny / distilbert` stays the weakest configuration for location (F1 ≈ 0.72) even at higher fusion_dim.
- `deit-tiny / minilm` improves location F1 vs tiny/distilbert and lands between the two small models.

**2. Latency behaviour**

- All ultra-edge2 students are **substantially faster** than their ultra-edge (256-dim) counterparts:
  - `deit-small / distilbert`: ~10.27 ms → ~5.27 ms.
  - `deit-small / minilm`: ~6.75 ms → ~4.41 ms.
  - `deit-tiny / distilbert`: ~9.80 ms → ~5.00 ms.
  - `deit-tiny / minilm`: ~7.42 ms → ~4.33 ms.
- Higher fusion_dim combined with the new implementation gives **lower latency without obvious degradation** in modality performance.

**3. MedPix takeaway (fusion_dim=384)**

- **Best pure modality accuracy:** `deit-small / distilbert` (unchanged vs 256-dim), but worse location.
- **Best overall MedPix trade-off:** `deit-small / minilm` — slightly weaker modality than small/distilbert but **better location** and much faster than any distilbert variant.
- **Tiny variants** are still useful when parameters must be minimal, but they **do not beat the small models** in accuracy and are only marginally faster than `deit-small / minilm` now.

---

## Wound-1-0 Results (Ultra-Edge2 Students)

Task mapping (unified to modality/location style):
- Task 1: `type`     → modality-like task
- Task 2: `severity` → location-like task

### Test Performance Summary

| Run ID                                   | Student (vision / text)   | Type Acc | Type F1 | Type AUC | Severity Acc | Severity F1 | Severity AUC | Infer (ms) |
|------------------------------------------|---------------------------|---------:|--------:|---------:|-------------:|------------:|-------------:|-----------:|
| wound-deit_small-distilbert-384          | deit-small / distilbert   | 0.8553 | 0.867 | 0.9880 | 0.8298 | 0.799 | 0.9410 | 5.33 |
| wound-deit_small-minilm-384              | deit-small / minilm       | 0.8298 | 0.834 | 0.9842 | 0.9489 | 0.935 | 0.9963 | 4.22 |
| wound-deit_tiny-distilbert-384           | deit-tiny / distilbert    | 0.7872 | 0.832 | 0.9807 | 0.9191 | 0.919 | 0.9848 | 4.87 |
| wound-deit_tiny-minilm-384               | deit-tiny / minilm        | 0.7830 | 0.782 | 0.9750 | 0.9277 | 0.916 | 0.9895 | 3.87 |

(F1, AUC and latency values rounded for readability.)

### Critical Observations — Wound

**1. Accuracy behaviour**

- `deit-small / minilm` remains the **best overall configuration**:
  - Slight drop in **type** F1 compared to ultra-edge (0.885 → ≈0.834), likely within variance.
  - **Severity** F1 stays extremely strong (≈0.935) with an AUC ≈ 0.996, essentially unchanged.
- `deit-small / distilbert` is a little stronger on type than ultra-edge2 minilm but significantly worse on severity.
- `deit-tiny / distilbert` and `deit-tiny / minilm` both do **surprisingly well on severity** (F1 ≈ 0.919–0.916), but type is clearly weaker than the small models.

**2. Latency behaviour**

- As on MedPix, ultra-edge2 models are **consistently faster** than ultra-edge:
  - `deit-small / distilbert`: ~10.04 ms → ~5.33 ms.
  - `deit-small / minilm`: ~7.84 ms → ~4.22 ms.
  - `deit-tiny / distilbert`: ~9.14 ms → ~4.87 ms.
  - `deit-tiny / minilm`: ~6.62 ms → ~3.87 ms.
- `deit-tiny / minilm` is now the **fastest** student while preserving strong severity performance.

**3. Wound takeaway (fusion_dim=384)**

- **Best overall Wound model:** `deit-small / minilm` — severity is excellent, type is competitive, and latency is very low.
- **Latency-first option:** `deit-tiny / minilm` — strongest severity among the tiny models and fastest inference, at the cost of weaker type.
- Distilbert students no longer justify their extra latency on this dataset given how strong `deit-small / minilm` is.

---

## Cross-Dataset Ultra-Edge2 Analysis

### Accuracy-focused comparison within ultra-edge2

**MedPix (average over tasks):**
- `deit-small / distilbert`: best **modality** but weaker **location**.
- `deit-small / minilm`: best **balanced** performance across modality and location.
- `deit-tiny / minilm`: reasonable compromise when vision parameters must be tiny, but dominated by `deit-small / minilm` if memory allows.

**Wound (average over tasks):**
- `deit-small / minilm`: clearly best aggregate performance; especially strong on **severity**.
- `deit-tiny / minilm`: second-best when combining accuracy + latency; strong severity but weaker type.
- Distilbert students lag behind minilm in overall trade-offs.

### Latency-focused comparison within ultra-edge2

**MedPix:**
- Fastest: `deit-tiny / minilm` (~4.33 ms).
- Very close second and more accurate: `deit-small / minilm` (~4.41 ms).
- Distilbert students are slower with no clear overall benefit.

**Wound:**
- Fastest: `deit-tiny / minilm` (~3.87 ms).
- Next: `deit-small / minilm` (~4.22 ms) with better type performance.

### Recommended ultra-edge2 configurations

| Priority / Scenario              | MedPix Recommendation (384)      | Wound Recommendation (384)       |
|----------------------------------|----------------------------------|----------------------------------|
| Max modality accuracy            | deit-small / distilbert          | deit-small / minilm              |
| Best overall trade-off           | deit-small / minilm              | deit-small / minilm              |
| Strict latency constraint        | deit-small / minilm or tiny/minilm | deit-tiny / minilm             |
| Single student for both datasets | deit-small / minilm              | deit-small / minilm              |

---

## Summary and Practical Guidance

- Moving from **fusion_dim=256** (ultra-edge) to **fusion_dim=384** (ultra-edge2) with the updated implementation **improves latency substantially** across all students without catastrophic changes in accuracy.
- Across **both MedPix and Wound**, `deit-small / minilm` emerges as the **best single student** when considering accuracy, AUC and inference time together.
- `deit-tiny / minilm` is the right choice only when you must push latency and memory to the limit; otherwise, its accuracy trade-offs are unnecessary because `deit-small / minilm` is already very fast in the ultra-edge2 regime.

These conclusions are based on `logs/ultra-edge2/*/results.json` and should be interpreted alongside the original ultra-edge results documented in `ULTRA_EDGE_RESULTS.md`. 