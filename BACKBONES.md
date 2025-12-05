# Backbone Comparison Reference

## Vision Backbones

| Model | HuggingFace ID | Params (M) | Latency (ms) | Tier |
|-------|---|---|---|---|
| **deit-base** (default) | `facebook/deit-base-distilled-patch16-224` | 86 | 8.5 | standard |
| vit-base | `google/vit-base-patch16-224` | 86 | 10.2 | standard |
| deit-tiny | `facebook/deit-tiny-distilled-patch16-224` | 5.7 | 2.1 | edge |
| mobile-vit | `facebook/deit-tiny-distilled-patch16-224` | 5.7 | 2.1 | ultra-edge |
| tiny-vit | `google/vit-base-patch16-224-in21k` | 86 | 9.8 | standard-pretrain |

## Text Backbones

| Model | HuggingFace ID | Params (M) | Latency (ms) | Tier |
|-------|---|---|---|---|
| **distilbert** (default) | `distilbert-base-uncased` | 66 | 4.2 | standard |
| bio-clinical-bert | `emilyalsentzer/Bio_ClinicalBERT` | 110 | 6.5 | standard-domain |
| bert-mini | `prajjwal1/bert-mini` | 11 | 1.8 | edge |
| bert-tiny | `prajjwal1/bert-tiny` | 4.4 | 0.9 | ultra-edge |
| mobile-bert | `google/mobilebert-uncased` | 25 | 2.3 | edge |

## Deployment Tiers

- **standard**: Full-capacity models (86-110M params); baseline performance
- **standard-domain**: Domain-specific large models; optimized for medical text
- **standard-pretrain**: Large models with ImageNet-21K pretraining; strong features
- **edge**: Compact models (5-25M params); 70-80% of standard accuracy; 2-5x faster
- **ultra-edge**: Ultra-compact models (<6M params); 40-60% of standard accuracy; 8-10x faster

## Example Run Commands

```bash
# Original baseline (standard-tier)
python tools/batch_runs.py --base config/default.yaml --runs original --execute --epochs 10

# Edge deployment (5.7M vision + 25M text)
python tools/batch_runs.py --base config/default.yaml --runs edge-vision,edge-text --execute --epochs 10

# Ultra-edge deployment (5.7M vision + 4.4M text)
python tools/batch_runs.py --base config/default.yaml --runs ultra-edge --execute --epochs 10

# Compare all variants
python tools/batch_runs.py --base config/default.yaml --runs original,edge-vision,edge-text,ultra-edge --execute --epochs 5
```
