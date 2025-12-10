# Ultra-Edge Experiments Summary

This document summarizes the ultra-edge student configurations under `logs/ultra-edge`,
covering both MedPix and Wound datasets. All runs use:

- Teacher: `vit-base` vision + `bio-clinical-bert` text, fusion_dim=256, fusion_layers=2
- Fusion: `cross_attention`
- Loss: `combined`
- Training: 3 teacher epochs, 10 student epochs, `teacher_lr=1e-5`, `student_lr=3e-4`

For each run we report key **test-set** metrics.

---

## MedPix Ultra-Edge Results

Task mapping:

- Task 1: `modality` (CT vs MR)
- Task 2: `location` (body location)

### Test Performance Overview

| Run ID                         | Student (vision / text)   | Test modality F1 | Test location F1 | Avg F1 | Test infer (ms) |
|--------------------------------|---------------------------|------------------|------------------|--------|------------------|
| `medpix-deit_small-distilbert` | deit-small / distilbert   | 0.97498          | 0.86052          | 0.91775 | 10.27           |
| `medpix-deit_small-minilm`     | deit-small / minilm       | 0.96997          | 0.81284          | 0.89141 | 6.75            |
| `medpix-deit_tiny-distilbert`  | deit-tiny / distilbert    | 0.90977          | 0.77730          | 0.84354 | 9.80            |
| `medpix-deit_tiny-minilm`      | deit-tiny / minilm        | 0.96500          | 0.82492          | 0.89496 | 7.42            |

(All latencies are `test_infer_ms` reported in `results.json`.)

### MedPix Critical Observations

- **`medpix-deit_small-distilbert`**  
  - Best pure accuracy: highest average F1 (≈0.918) across modality and location.  
  - Cost: slowest ultra-edge configuration (≈10.3 ms per example).

- **`medpix-deit_small-minilm`**  
  - Slightly lower average F1 (≈0.891) mainly due to lower location F1.  
  - Major upside: significantly faster inference (≈6.8 ms), offering a strong accuracy–latency trade-off.

- **`medpix-deit_tiny-distilbert`**  
  - Weakest MedPix configuration: both modality and location F1 substantially below the small models.  
  - Latency (~9.8 ms) is not low enough to justify the performance drop; dominated by `deit_small-*` and `deit_tiny-minilm`.

- **`medpix-deit_tiny-minilm`**  
  - Very competitive: average F1 (~0.895) slightly **better** than `deit_small-minilm`, and much stronger than `deit_tiny-distilbert`.  
  - Latency (~7.4 ms) close to the fastest MedPix model; best "tiny" option and arguably the best ultra-edge compromise.

**MedPix takeaway:**

- **Best pure accuracy:** `medpix-deit_small-distilbert`.
- **Best ultra-edge trade-off (accuracy vs latency):** `medpix-deit_tiny-minilm`.

---

## Wound Ultra-Edge Results

Task mapping (unified to modality/location style):

- Task 1: `type`   → modality-like task
- Task 2: `severity` → location-like task

### Test Performance Overview

| Run ID                          | Student (vision / text)   | Test type F1 | Test severity F1 | Avg F1 | Test infer (ms) |
|---------------------------------|---------------------------|--------------|------------------|--------|------------------|
| `wound-deit_small-distilbert`   | deit-small / distilbert   | 0.87907      | 0.92009          | 0.89958 | 10.04           |
| `wound-deit_small-minilm`       | deit-small / minilm       | 0.88459      | 0.93957          | 0.91208 | 7.84            |
| `wound-deit_tiny-distilbert`    | deit-tiny / distilbert    | 0.81035      | 0.85489          | 0.83262 | 9.14            |
| `wound-deit_tiny-minilm`        | deit-tiny / minilm        | 0.77666      | 0.90550          | 0.84108 | 6.62            |

### Wound Critical Observations

- **`wound-deit_small-distilbert`**  
  - Strong baseline: high type and severity F1 (avg ≈0.900).  
  - Slowest Wound ultra-edge configuration (~10.0 ms); good as a reference, less ideal for strict edge constraints.

- **`wound-deit_small-minilm`**  
  - Best overall Wound configuration: highest average F1 (~0.912) and best severity F1.  
  - Also clearly faster (~7.8 ms) than `small-distilbert`; dominates on both accuracy and efficiency.

- **`wound-deit_tiny-distilbert`**  
  - Noticeable degradation in both type and severity F1 (avg ≈0.833) compared to the small models.  
  - Latency (~9.1 ms) is not low enough to compensate, so it is dominated by `wound-deit_small-minilm`.

- **`wound-deit_tiny-minilm`**  
  - Moderate average F1 (~0.841) with good severity F1 but weaker type F1.  
  - Fastest Wound model (~6.6 ms); reasonable for very tight latency budgets, but with a clear accuracy trade-off versus `small-minilm`.

**Wound takeaway:**

- **Best overall:** `wound-deit_small-minilm` (dominates accuracy and latency versus other ultra-edge runs).
- **Best latency-only option:** `wound-deit_tiny-minilm`, at the cost of lower especially type F1.

---

## Cross-Dataset Recommendations

- If you prioritize **maximum accuracy** under ultra-edge constraints:  
  - MedPix: `medpix-deit_small-distilbert`  
  - Wound:  `wound-deit_small-minilm`

- If you prioritize **accuracy–latency trade-off**:  
  - MedPix: `medpix-deit_tiny-minilm` (strong F1, fast)  
  - Wound:  `wound-deit_small-minilm` remains the best compromise; `wound-deit_tiny-minilm` only if latency is critical.

- If you prioritize **minimum latency** while staying in a reasonable performance regime:  
  - MedPix: `medpix-deit_small-minilm` (fastest MedPix and solid F1)  
  - Wound:  `wound-deit_tiny-minilm` (fastest Wound configuration).
