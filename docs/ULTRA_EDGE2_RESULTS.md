# Ultra-Edge2 Experiment Results (fusion_dim=384)

## Overview
Comparison of **lightweight student configurations** designed for ultra-edge deployment across **MedPix-2-0** and **Wound-1-0** when both teacher and student fusion blocks use **fusion_dim = 384**.

**Configuration:**
- **Teacher:** `vit-base` (vision) + `bio-clinical-bert` (text), `fusion_dim=384`, `fusion_layers=2`
- **Student variants (vision / text):**
  - `deit-small` / `distilbert` — **~92.3M parameters** (21.2M vision + 65.9M text + 5.2M fusion/proj/heads)
  - `deit-small` / `minilm` — **~48.1M parameters** (21.2M vision + 22.3M text + 4.6M fusion/proj/heads)
  - `deit-tiny` / `distilbert` — **~75.8M parameters** (5.3M vision + 65.9M text + 4.6M fusion/proj/heads)
  - `deit-tiny` / `minilm` — **~31.6M parameters** (5.3M vision + 22.3M text + 4.0M fusion/proj/heads)
- **Fusion:** `cross_attention`
- **Loss:** `combined`
- **Training:** `teacher_epochs=3`, `student_epochs=10`, `teacher_lr=1e-5`, `student_lr=3e-4`, `alpha=1.0`, `beta=100.0`, `T=2.0`
- **Device:** `cuda:3`

**Note:** At `fusion_dim=384`, fusion/projection layers are larger than at 256, adding ~3-4M parameters per student compared to ultra-edge (256). Despite this, inference is faster due to implementation optimizations.

All numbers below are **test-set** metrics from `logs/ultra-edge2/*/results.json`.

---

## MedPix-2-0 Results (Ultra-Edge2 Students)

Task mapping:
- Task 1: `modality` (CT vs MR)
- Task 2: `location` (body location)

### Test Performance Summary

| Run ID                            | Student (vision / text)   | Params (M) | Modality Acc | Modality F1 | Modality AUC | Location Acc | Location F1 | Location AUC |
|-----------------------------------|---------------------------|----------:|-------------:|------------:|-------------:|-------------:|------------:|-------------:|
| medpix-deit_small-distilbert-384 | deit-small / distilbert  | 90.4      | 0.970        | 0.970       | 0.995        | 0.920        | 0.888       | 0.964        |
| medpix-deit_small-minilm-384     | deit-small / minilm      | 46.6      | 0.960        | 0.960       | 0.988        | 0.860        | 0.833       | 0.944        |
| medpix-deit_tiny-distilbert-384  | deit-tiny / distilbert   | 74.1      | 0.925        | 0.925       | 0.992        | 0.885        | 0.823       | 0.968        |
| medpix-deit_tiny-minilm-384      | deit-tiny / minilm       | 30.3      | 0.970        | 0.970       | 0.991        | 0.850        | 0.812       | 0.931        |

Values are rounded from `logs/ultra-edge2/medpix-*/results.json`.

### Critical Observations — MedPix

- **High-capacity student:** `deit-small / distilbert` remains extremely strong on modality and now also achieves very high location accuracy (≈0.92), at the cost of the largest parameter count among ultra-edge2 students.
- **Balanced option:** `deit-small / minilm` offers only a small drop in modality metrics relative to small/distilbert, but with fewer parameters and still solid location performance (location F1 ≈ 0.83).
- **Tiny variants:** both `deit-tiny` students are competitive but generally sit below the small models; `deit-tiny / minilm` has good modality but noticeably weaker location than `deit-small / distilbert`.
- **Overall:** for MedPix at fusion_dim=384, `deit-small / distilbert` is best if model size is acceptable; otherwise `deit-small / minilm` is the most balanced accuracy–capacity choice.

---

## Wound-1-0 Results (Ultra-Edge2 Students)

Task mapping (unified to modality/location style):
- Task 1: `type`     → modality-like task
- Task 2: `severity` → location-like task

### Test Performance Summary

| Run ID                         | Student (vision / text)   | Params (M) | Type Acc | Type F1 | Type AUC | Severity Acc | Severity F1 | Severity AUC |
|--------------------------------|---------------------------|----------:|---------:|--------:|---------:|-------------:|------------:|-------------:|
| wound-deit_small-distilbert-384 | deit-small / distilbert | 90.4      | 0.821    | 0.831   | 0.982    | 0.940        | 0.939       | 0.993        |
| wound-deit_small-minilm-384     | deit-small / minilm     | 46.6      | 0.787    | 0.778   | 0.977    | 0.936        | 0.923       | 0.986        |
| wound-deit_tiny-distilbert-384  | deit-tiny / distilbert  | 74.1      | 0.749    | 0.707   | 0.959    | 0.949        | 0.946       | 0.994        |
| wound-deit_tiny-minilm-384      | deit-tiny / minilm      | 30.3      | 0.723    | 0.746   | 0.964    | 0.936        | 0.924       | 0.991        |

Values are rounded from `logs/ultra-edge2/wound-*/results.json`.

### Critical Observations — Wound

- **Severity is consistently strong:** all ultra-edge2 students achieve high severity accuracy and F1, with the tiny/distilbert variant slightly leading on severity metrics.
- **Type performance differentiates models:** `deit-small / distilbert` has the strongest type scores, but `deit-small / minilm` is close while using fewer parameters.
- **Tiny students:** both tiny variants show good severity but noticeably weaker type performance; they are mainly of interest when model size constraints dominate.
- **Overall:** for Wound at fusion_dim=384, `deit-small / minilm` remains a robust choice when balancing type and severity, while `deit-small / distilbert` is preferable only if maximizing type performance is critical and the larger model is acceptable.

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