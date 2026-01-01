# MedPix Fusion HP Exploration: Test Results Summary

Source: `logs/fusion-explore-hp-medpix/summary.md` (test split only).

| Fusion Type           | test_modality_acc | test_location_acc | test_modality_f1 | test_location_f1 | test_modality_auc | test_location_auc |
|-----------------------|------------------:|------------------:|-----------------:|-----------------:|------------------:|------------------:|
| concat_mlp            | 0.955             | 0.875             | 0.95499          | 0.84745          | 0.9880            | 0.95949          |
| cross_attention       | 0.960             | **0.880**         | 0.95998          | 0.84266          | **0.9954**        | 0.95417          |
| energy_aware_adaptive | 0.905             | 0.765             | 0.90494          | 0.69390          | 0.96115           | 0.92118          |
| film                  | 0.975             | 0.860             | 0.97499          | 0.81323          | 0.99320           | 0.95041          |
| gated                 | 0.980             | 0.860             | 0.97999          | 0.83127          | 0.99190           | **0.96565**      |
| modality_dropout      | 0.955             | 0.855             | 0.95494          | 0.82484          | 0.99380           | 0.96543          |
| shomr                 | 0.970             | 0.725             | 0.97000          | 0.58388          | 0.99420           | 0.87394          |
| transformer_concat    | 0.970             | 0.865             | 0.96997          | 0.84173          | 0.99140           | 0.94681          |

Notes:
- Student: mobilevit-xx-small + bert-mini; Teacher: vit-base + bio-clinical-bert (per run configs)
- Tasks: modality (binary) and location (multi-class)

Note: This doc assumes `simple` / `SimpleFusion` is not available; the row is intentionally omitted.
