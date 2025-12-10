# Loss Exploration: Cross-Attention Student

This document summarizes the **loss-explore** experiments under `logs/loss-explore/`,
where the fusion module is fixed to `cross_attention` and the student is
`vit-base` vision + `distilbert` text (with a `vit-large` + `bio-clinical-bert` teacher).

We compare the following distillation losses:

- `vanilla`
- `combined`
- `crd`
- `rkd`
- `mmd`

For each dataset we report key **test-set** metrics.

---

## MedPix: Effect of Loss Type

Task mapping:

- Task 1: `modality` (CT vs MR)
- Task 2: `location` (body location)

### Test Performance Overview (MedPix)

| Run ID                              | Loss     | Test modality F1 | Test location F1 | Avg F1 | Test infer (ms) |
|-------------------------------------|----------|------------------|------------------|--------|------------------|
| `medpix-cross_attention-vanilla`   | vanilla  | 0.9750           | 0.7825           | 0.8787 | 13.44           |
| `medpix-cross_attention-combined`  | combined | 0.9850           | 0.8414           | 0.9132 | 7.86            |
| `medpix-cross_attention-crd`       | crd      | 0.4911           | 0.1718           | 0.3315 | 7.94            |
| `medpix-cross_attention-mmd`       | mmd      | 0.5313           | 0.1331           | 0.3322 | 8.22            |
| `medpix-cross_attention-rkd`       | rkd      | 0.7624           | 0.1496           | 0.4560 | 8.16            |

(Avg F1 is a simple mean of modality and location F1.)

### MedPix Critical Observations

- **`combined` vs `vanilla`**  
  - `combined` achieves the **best overall performance**: highest average F1 (~0.913), notably stronger location F1 than `vanilla`.  
  - `vanilla` has excellent modality F1 but significantly weaker location F1; overall worse than `combined` and also **slower** inference (~13.4 ms vs ~7.9 ms).

- **`crd`, `mmd`, `rkd`**  
  - All three advanced feature-based losses underperform badly on MedPix location F1 and overall Avg F1, despite similar inference cost.  
  - `crd` and `mmd` collapse location performance (F1 ~0.13–0.17) and are not competitive as drop-in replacements in this setup.  
  - `rkd` preserves modality F1 somewhat (~0.76) but still fails on location, yielding poor average performance.

**MedPix takeaway:**

- **Best choice:** `combined` loss (strong gains on location without sacrificing modality, and faster than `vanilla`).
- **Safe baseline:** `vanilla` remains a solid baseline if you want a simpler loss, but it is dominated by `combined` here.
- **Not recommended in this configuration:** `crd`, `mmd`, `rkd` for MedPix with cross-attention + vit-base student.

---

## Wound: Effect of Loss Type

Task mapping (unified to modality/location style):

- Task 1: `type`   → modality-like task
- Task 2: `severity` → location-like task

### Test Performance Overview (Wound)

| Run ID                              | Loss     | Test type F1 | Test severity F1 | Avg F1 | Test infer (ms) |
|-------------------------------------|----------|--------------|------------------|--------|------------------|
| `wound-cross_attention-vanilla`    | vanilla  | 0.9369       | 0.9230           | 0.9300 | 7.13            |
| `wound-cross_attention-combined`   | combined | 0.9196       | 0.9299           | 0.9247 | 12.51           |
| `wound-cross_attention-crd`        | crd      | 0.0997       | 0.2992           | 0.1995 | 12.03           |
| `wound-cross_attention-mmd`        | mmd      | 0.0302       | 0.4684           | 0.2493 | 11.78           |
| `wound-cross_attention-rkd`        | rkd      | 0.0251       | 0.3933           | 0.2092 | 12.33           |

### Wound Critical Observations

- **`vanilla` vs `combined`**  
  - `vanilla` achieves the **best overall Wound performance**: highest average F1 (~0.93) with especially strong type F1.  
  - `combined` is close in average (~0.925), slightly better on severity F1 but lower on type F1, and notably **slower** (~12.5 ms vs ~7.1 ms).

- **`crd`, `mmd`, `rkd`**  
  - All three advanced losses severely degrade type F1 while sometimes maintaining moderate severity F1.  
  - None are competitive versus `vanilla` / `combined` on this dataset in the current setup.

**Wound takeaway:**

- **Best choice:** `vanilla` loss (strongest overall and clearly fastest).  
- **Alternative:** `combined` can be considered when slightly higher severity F1 is critical and extra latency is acceptable.
- **Not recommended in this configuration:** `crd`, `mmd`, `rkd` for Wound with cross-attention + vit-base student.

---

## Cross-Dataset Loss Recommendations

- **If you want a single loss for both datasets:**  
  - `combined` is the safer default, since it is best on MedPix and only slightly behind `vanilla` on Wound.

- **If you can tune per-dataset:**  
  - MedPix: prefer **`combined`** loss.  
  - Wound: prefer **`vanilla`** loss.

- **Advanced representation losses (`crd`, `mmd`, `rkd`):**  
  - In this particular setup, these do **not** improve performance and often hurt it substantially, especially on the more fine-grained Wound tasks.
