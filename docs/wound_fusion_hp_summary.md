# Wound Fusion HP Exploration: Test Results Summary

| Fusion Type           | test_type_acc | test_severity_acc | test_type_f1 | test_severity_f1 | test_type_auc | test_severity_auc |
|-----------------------|---------------|-------------------|--------------|------------------|---------------|-------------------|
| concat_mlp            | 0.796         | 0.919             | 0.758        | 0.904            | 0.970         | 0.976             |
| cross_attention       | 0.847         | 0.932             | 0.842        | 0.933            | 0.987         | 0.989             |
| energy_aware_adaptive | 0.583         | 0.851             | 0.536        | 0.839            | 0.868         | 0.966             |
| film                  | 0.838         | 0.940             | 0.831        | 0.939            | 0.988         | 0.990             |
| gated                 | 0.796         | 0.936             | 0.725        | 0.922            | 0.971         | 0.992             |
| modality_dropout      | 0.715         | 0.945             | 0.558        | 0.938            | 0.952         | 0.990             |
| shomr                 | 0.383         | 0.881             | 0.291        | 0.858            | 0.781         | 0.975             |
| transformer_concat    | 0.834         | 0.932             | 0.857        | 0.915            | 0.987         | 0.993             |

- All results are from the test split of each run in `logs/fusion-explore-hp-wound/`
- Metrics: accuracy, F1, and AUC for both wound type and severity tasks
- Student: mobilevit-xx-small + bert-mini, Teacher: vit-base + bio-clinical-bert

Note: This doc assumes `simple` / `SimpleFusion` is not available; the row is intentionally omitted.
