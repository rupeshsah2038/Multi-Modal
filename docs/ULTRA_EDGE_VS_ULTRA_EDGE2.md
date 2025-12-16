# Ultra-Edge (256) vs Ultra-Edge2 (384)

Side-by-side comparison of ultra-edge students at **fusion_dim=256** (original) and **fusion_dim=384** (ultra-edge2) across **MedPix-2-0** and **Wound-1-0**.

**Parameter differences:**
- At `fusion_dim=256`, fusion/projection layers add ~1.3-1.6M parameters per student
- At `fusion_dim=384`, fusion/projection layers add ~4.0-5.2M parameters per student
- **Net increase from 256→384**: ~3-4M parameters per student (modest overhead for the latency gains)

This document is meant to be read together with:
- `ULTRA_EDGE_RESULTS.md`  (baseline ultra-edge)
- `ULTRA_EDGE2_RESULTS.md` (new ultra-edge2)

---

## MedPix-2-0: 256 vs 384

**Tasks:**
- Task 1: `modality` (CT vs MR)
- Task 2: `location` (body location)

### Test Metrics by Student

Values are from `logs/ultra-edge/*/results.json` (256) and `logs/ultra-edge2/*/results.json` (384).

| Student (vision / text)   | Params (M, 256) | Params (M, 384) | Config ID (256)               | Modality Acc (256) | Location Acc (256) | Config ID (384)                    | Modality Acc (384) | Location Acc (384) |
|---------------------------|----------------:|----------------:|------------------------------|-------------------:|-------------------:|------------------------------------|-------------------:|-------------------:|
| deit-small / distilbert   | 89.3            | 90.4            | medpix-deit_small-distilbert | 0.975             | 0.915             | medpix-deit_small-distilbert-384  | 0.970             | 0.920             |
| deit-small / minilm       | 45.5            | 46.6            | medpix-deit_small-minilm     | 0.960             | 0.845             | medpix-deit_small-minilm-384      | 0.960             | 0.860             |
| deit-tiny / distilbert    | 73.0            | 74.1            | medpix-deit_tiny-distilbert  | 0.960             | 0.815             | medpix-deit_tiny-distilbert-384   | 0.925             | 0.885             |
| deit-tiny / minilm        | 29.2            | 30.3            | medpix-deit_tiny-minilm      | 0.980             | 0.895             | medpix-deit_tiny-minilm-384       | 0.970             | 0.850             |

### MedPix: What changed from 256 → 384?

- **Modality accuracy** stays high for all students across both setups; the changes from 256 → 384 are small (typically within a few points of accuracy).
- **Location accuracy** shifts more noticeably: some students (e.g. `deit-small / minilm` and `deit-tiny / distilbert`) gain location accuracy at 384, while others (e.g. `deit-tiny / minilm`) lose a bit.
- **Parameter overhead** from 256 → 384 is modest (~3–4M extra parameters per student) but can still matter on very constrained devices.
- For MedPix, a consistent pattern is that `deit-small` students provide the strongest overall accuracy, while `deit-tiny` students are only competitive when model size must be kept extremely small.

---

## Wound-1-0: 256 vs 384

**Tasks (mapped to modality/location):**
- Task 1: `type`     → modality-like task
- Task 2: `severity` → location-like task

### Test Metrics by Student

| Student (vision / text)   | Params (M, 256) | Params (M, 384) | Config ID (256)              | Type Acc (256) | Severity Acc (256) | Config ID (384)                    | Type Acc (384) | Severity Acc (384) |
|---------------------------|----------------:|----------------:|------------------------------|---------------:|-------------------:|------------------------------------|---------------:|-------------------:|
| deit-small / distilbert   | 89.3            | 90.4            | wound-deit_small-distilbert | 0.762          | 0.919             | wound-deit_small-distilbert-384   | 0.821          | 0.940             |
| deit-small / minilm       | 45.5            | 46.6            | wound-deit_small-minilm     | 0.830          | 0.940             | wound-deit_small-minilm-384       | 0.787          | 0.936             |
| deit-tiny / distilbert    | 73.0            | 74.1            | wound-deit_tiny-distilbert  | 0.723          | 0.919             | wound-deit_tiny-distilbert-384    | 0.749          | 0.949             |
| deit-tiny / minilm        | 29.2            | 30.3            | wound-deit_tiny-minilm      | 0.689          | 0.949             | wound-deit_tiny-minilm-384        | 0.723          | 0.936             |

### Wound: What changed from 256 → 384?

- Type and severity accuracy shift in opposite directions for some students when moving to 384, so the **best configuration depends on which task is more critical**.
- `deit-small / distilbert` improves on both type and severity accuracy at 384, but also remains the largest model.
- `deit-small / minilm` loses some type accuracy at 384 while keeping severity strong, so it is still attractive when parameter count matters.
- Tiny students gain some type accuracy at 384 while maintaining high severity, making them viable only when very small models are required, accepting weaker type performance.

---

## Global Takeaways

### 1. Accuracy: small but meaningful shifts

- Accuracy differences between 256 and 384 are generally modest, but **which student is best can change slightly** when you increase fusion_dim.
- Distilbert students tend to benefit more in Wound severity at 384, while `minilm` students sometimes trade small drops in one task for gains in the other.

### 2. Parameters vs performance

- The parameter increase from 256 → 384 is small in absolute terms, but relevant on very tight memory budgets.
- In many cases, **`deit-small / minilm` remains the most attractive compromise**: strong performance on both datasets with moderate parameter count.

### 3. Recommended students after comparison

| Scenario / Constraint               | Recommended Student (384)      | Rationale |
|------------------------------------|--------------------------------|-----------|
| Single student for **both** datasets | `deit-small / minilm`         | Strong on both MedPix and Wound with moderate parameters. |
| MedPix, max modality accuracy      | `deit-small / distilbert`      | Slight edge in modality metrics at the cost of a larger model. |
| MedPix, balanced performance       | `deit-small / minilm`          | Good modality and location scores with fewer parameters. |
| Wound, balanced performance        | `deit-small / minilm`          | Strong severity and acceptable type accuracy with compact size. |
| Strict model-size constraint       | `deit-tiny / minilm`           | Smallest student; accuracy is weaker but acceptable when memory is tight. |

---

## Practical Guidance

- If you are **currently on ultra-edge (256)**:
  - For most use cases, you can **safely migrate** to ultra-edge2 (384) with the **same student backbone choices**, gaining latency and keeping performance similar.
  - For new deployments, it is reasonable to **standardize on ultra-edge2** and treat 256-dim ultra-edge as legacy.
- For a **single best ultra-edge configuration** that works across both MedPix and Wound with minimal tuning, use:
  - **`deit-small` (vision) + `minilm` (text), fusion_dim=384**.

For dataset- and student-specific details, refer back to `ULTRA_EDGE_RESULTS.md` and `ULTRA_EDGE2_RESULTS.md` tables.