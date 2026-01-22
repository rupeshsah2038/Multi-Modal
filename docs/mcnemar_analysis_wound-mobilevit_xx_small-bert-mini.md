# McNemar Analysis — Wound / mobilevit-xx-small + bert-mini

Source file:
- `logs/ultra-edge-hp-tuned-all/wound-mobilevit_xx_small-bert-mini/results.json`
- Timestamp: `2026-01-22T14:49:57.704951`

Config summary:
- Dataset: Wound (`task1_label=type`, `task2_label=severity`)
- Teacher: `vit-base` + `bio-clinical-bert` (fusion: cross_attention, 3 layers, dim 256)
- Student: `mobilevit-xx-small` + `bert-mini` (fusion: cross_attention, 1 layer, dim 256)
- McNemar enabled: `evaluation.mcnemar=true`, `mcnemar_exact=true`, `mcnemar_cc=true`

---

## What McNemar metrics mean in this repo

McNemar is computed on *paired* test examples (teacher vs student on the same images).

Discordant-pair counts (correctness vs correctness):
- **b**: teacher correct AND student wrong
- **c**: teacher wrong AND student correct
- **n = b + c**: number of samples where they differ in correctness

With `mcnemar_exact=true`, the repo uses the **exact two-sided binomial** McNemar p-value over the discordant pairs.

---

## Test metrics (Teacher vs Student)

### Task 1 — Type

**Student test**
- Acc: `0.8681`
- F1 (macro): `0.8388`

**Teacher test**
- Acc: `0.9064`
- F1 (macro): `0.8802`

**McNemar (teacher vs student)**
- b = `16`
- c = `7`
- n = `23`
- p = `0.09314`
- method = `exact-binomial`

Interpretation:
- Teacher outperforms student more often on the discordant cases (16 vs 7), but the difference is **not significant at α=0.05** (p≈0.093).

### Task 2 — Severity

**Student test**
- Acc: `0.9532`
- F1 (macro): `0.9491`

**Teacher test**
- Acc: `0.9106`
- F1 (macro): `0.8808`

**McNemar (teacher vs student)**
- b = `4`
- c = `14`
- n = `18`
- p = `0.03088`
- method = `exact-binomial`

Interpretation:
- Student outperforms teacher more often on the discordant cases (14 vs 4), and the difference **is significant at α=0.05** (p≈0.0309).

---

## Efficiency / size

- Student parameters: `13.06M`
- Teacher parameters: `195.89M`
- Student test inference: `4.75 ms/sample`
- Teacher test inference: `12.82 ms/sample`

Takeaway:
- Student is much smaller and faster, and it beats the teacher significantly on **severity**, while being slightly worse (not significantly) on **type**.

---

## Notes on multiple comparisons

Two hypothesis tests are being reported (type + severity). If you apply a strict Bonferroni correction for two tests, use α = 0.05/2 = 0.025.
- Under Bonferroni, `p=0.03088` (severity) would not pass.

If you plan to report significance formally, state whether (and how) you corrected for multiple tests.
