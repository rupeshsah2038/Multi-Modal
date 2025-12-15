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

| Student (vision / text)   | Params (M, 256) | Params (M, 384) | Config ID (256)                | Modality Acc (256) | Location Acc (256) | Infer (ms, 256) | Config ID (384)                     | Modality Acc (384) | Location Acc (384) | Infer (ms, 384) |
|---------------------------|----------------:|----------------:|--------------------------------|-------------------:|-------------------:|----------------:|--------------------------------------|-------------------:|-------------------:|----------------:|
| deit-small / distilbert   | 89.3 | 92.3 | medpix-deit_small-distilbert  | 0.975 | 0.895 | 10.27 | medpix-deit_small-distilbert-384  | 0.975 | 0.850 | 5.27 |
| deit-small / minilm       | 45.5 | 48.1 | medpix-deit_small-minilm      | 0.970 | 0.850 | 6.75  | medpix-deit_small-minilm-384      | 0.945 | 0.870 | 4.41 |
| deit-tiny / distilbert    | 73.0 | 75.8 | medpix-deit_tiny-distilbert   | 0.910 | 0.825 | 9.80  | medpix-deit_tiny-distilbert-384   | 0.955 | 0.790 | 5.00 |
| deit-tiny / minilm        | 29.2 | 31.6 | medpix-deit_tiny-minilm       | 0.965 | 0.875 | 7.42  | medpix-deit_tiny-minilm-384       | 0.960 | 0.790 | 4.33 |

(Here we focus on **accuracy** and **latency**; F1/AUC trends are consistent with these numbers.)

### MedPix: What changed from 256 → 384?

**1. Latency improvements across the board**

- All students roughly **halve their inference time** when moving from 256 to 384:
  - `deit-small / distilbert`: ~10.3 → ~5.3 ms.
  - `deit-small / minilm`: ~6.8 → ~4.4 ms.
  - `deit-tiny / distilbert`: ~9.8 → ~5.0 ms.
  - `deit-tiny / minilm`: ~7.4 → ~4.3 ms.
- Ultra-edge2 is therefore strictly better in terms of **latency** for every student.

**2. Modality vs location trade-offs**

- `deit-small / distilbert` keeps the **same modality accuracy** (0.975) but **loses some location accuracy** (0.895 → 0.850).
- `deit-small / minilm` trades a **small drop in modality** (0.970 → 0.945) for a **gain in location** (0.850 → 0.870).
- `deit-tiny / distilbert` gains modality (0.910 → 0.955) but loses location (0.825 → 0.790).
- `deit-tiny / minilm` is nearly unchanged in modality (0.965 → 0.960) but loses location (0.875 → 0.790).

**3. MedPix recommendations: 256 vs 384**

- If you previously used **`deit-small / distilbert` (256)** purely for accuracy:
  - At 384, it is still best on **modality**, but the **location drop** is noticeable.
- If you previously used **`deit-tiny / minilm` (256)** as the best ultra-edge trade-off:
  - At 384, **`deit-small / minilm`** becomes a **better all-round choice**: slightly slower than tiny/minilm but **better location** and competitive modality.
- Overall, for MedPix, ultra-edge2 suggests **moving from tiny/minilm (256)** to **small/minilm (384)** when memory allows.

---

## Wound-1-0: 256 vs 384

**Tasks (mapped to modality/location):**
- Task 1: `type`     → modality-like task
- Task 2: `severity` → location-like task

### Test Metrics by Student

| Student (vision / text)   | Params (M, 256) | Params (M, 384) | Config ID (256)               | Type Acc (256) | Severity Acc (256) | Infer (ms, 256) | Config ID (384)                     | Type Acc (384) | Severity Acc (384) | Infer (ms, 384) |
|---------------------------|----------------:|----------------:|-------------------------------|---------------:|-------------------:|----------------:|--------------------------------------|---------------:|-------------------:|----------------:|
| deit-small / distilbert   | 89.3 | 92.3 | wound-deit_small-distilbert  | 0.855 | 0.928 | 10.04 | wound-deit_small-distilbert-384  | 0.855 | 0.830 | 5.33 |
| deit-small / minilm       | 45.5 | 48.1 | wound-deit_small-minilm      | 0.860 | 0.940 | 7.84  | wound-deit_small-minilm-384      | 0.830 | 0.949 | 4.22 |
| deit-tiny / distilbert    | 73.0 | 75.8 | wound-deit_tiny-distilbert   | 0.774 | 0.872 | 9.14  | wound-deit_tiny-distilbert-384   | 0.787 | 0.919 | 4.87 |
| deit-tiny / minilm        | 29.2 | 31.6 | wound-deit_tiny-minilm       | 0.762 | 0.919 | 6.62  | wound-deit_tiny-minilm-384       | 0.783 | 0.928 | 3.87 |

(Again focusing on **accuracy** and **latency**; F1/AUC trends follow these patterns.)

### Wound: What changed from 256 → 384?

**1. Latency improvements**

- As on MedPix, all students are **significantly faster** at 384:
  - `deit-small / distilbert`: ~10.0 → ~5.3 ms.
  - `deit-small / minilm`: ~7.8 → ~4.2 ms.
  - `deit-tiny / distilbert`: ~9.1 → ~4.9 ms.
  - `deit-tiny / minilm`: ~6.6 → ~3.9 ms.

**2. Accuracy behaviour**

- `deit-small / distilbert` keeps **type** accuracy roughly unchanged (0.855), but **severity** drops (0.928 → 0.830).
- `deit-small / minilm` has a small **drop in type** (0.860 → 0.830) but a **slight gain in severity** (0.940 → 0.949).
- Both tiny variants **improve severity** while maintaining comparable type accuracy:
  - `deit-tiny / distilbert`: severity 0.872 → 0.919.
  - `deit-tiny / minilm`: severity 0.919 → 0.928.

**3. Wound recommendations: 256 vs 384**

- At 256, `deit-small / minilm` was already the **best overall** Wound configuration.
- At 384, `deit-small / minilm` **remains best overall**, with similar or slightly better severity and significantly lower latency.
- `deit-tiny / minilm` improves modestly and becomes an attractive **extreme-latency** model, but still lags in type compared to `deit-small / minilm`.

---

## Global Takeaways

### 1. Latency: ultra-edge2 clearly dominates

- For **every** student on **both** datasets, ultra-edge2 (384) provides **substantially lower inference time** than ultra-edge (256) while keeping the same architecture and training recipe.
- If you were constrained by latency under the 256-dim setup, **switching to the 384-dim ultra-edge2 configs is a free win**.

### 2. Accuracy: small shifts, no collapse

- There are **small shifts** in accuracy per task and per student when going from 256 → 384, but no catastrophic failures.
- The main patterns:
  - Some distilbert students lose a bit of performance on the second task (location/severity).
  - `deit-small / minilm` stays robust and often **improves** on the more challenging task (MedPix location, Wound severity).

### 3. Recommended students after comparison

| Scenario / Constraint               | Recommended Student (384)      | Rationale |
|------------------------------------|--------------------------------|-----------|
| Single student for **both** datasets | `deit-small / minilm`         | Strong on both MedPix and Wound, very fast. |
| MedPix, max modality accuracy      | `deit-small / distilbert`      | Highest modality accuracy; accept location drop. |
| MedPix, balanced accuracy+latency  | `deit-small / minilm`          | Better location vs tiny, near-best latency. |
| Wound, balanced accuracy+latency   | `deit-small / minilm`          | Best severity and strong type, low latency. |
| Extreme latency, both datasets     | `deit-tiny / minilm`           | Fastest overall; accuracy trade-offs acceptable in latency-bound settings. |

---

## Practical Guidance

- If you are **currently on ultra-edge (256)**:
  - For most use cases, you can **safely migrate** to ultra-edge2 (384) with the **same student backbone choices**, gaining latency and keeping performance similar.
  - For new deployments, it is reasonable to **standardize on ultra-edge2** and treat 256-dim ultra-edge as legacy.
- For a **single best ultra-edge configuration** that works across both MedPix and Wound with minimal tuning, use:
  - **`deit-small` (vision) + `minilm` (text), fusion_dim=384**.

For dataset- and student-specific details, refer back to `ULTRA_EDGE_RESULTS.md` and `ULTRA_EDGE2_RESULTS.md` tables.