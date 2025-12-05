from transformers import AutoModel, ViTModel, DeiTModel

# Mapping from friendly names to Hugging Face pretrained identifiers.
VISION_PRETRAINED = {
    "vit-large": "google/vit-large-patch16-224",
    "vit-base": "google/vit-base-patch16-224",
    "deit-base": "facebook/deit-base-distilled-patch16-224",
    # compact / edge-friendly vision backbones (real HF models)
    "tiny-vit": "google/vit-base-patch16-224-in21k",  # Smaller ViT variant
    "deit-tiny": "facebook/deit-tiny-patch16-224",
    "mobilevit-xx-small": "apple/mobilevit-xx-small",
    "mobilevit-small": "apple/mobilevit-small",
    "mobilevit-medium": "apple/mobilevit-medium",
    "efficientvit-b0": "mit-han-lab/efficientvit-b0",
    "mobilenet-v2": "google/mobilenet_v2_1.0_224",  # MobileNet v2
}

TEXT_PRETRAINED = {
    "bio-clinical-bert": "emilyalsentzer/Bio_ClinicalBERT",
    "distilbert": "distilbert-base-uncased",
    # compact / edge-friendly text backbones
    "mobile-bert": "google/mobilebert-uncased",
    "bert-tiny": "prajjwal1/bert-tiny",
    "bert-mini": "prajjwal1/bert-mini",
    "minilm": "nreimers/MiniLM-L6-H384-uncased",    
}


def get_vision_backbone(name: str):
    pretrained = VISION_PRETRAINED.get(name)
    if pretrained is None:
        raise KeyError(f"Unknown vision backbone: {name}")
    # prefer explicit classes where appropriate, fall back to AutoModel for other types
    if name.startswith('deit'):
        return DeiTModel.from_pretrained(pretrained)
    if name.startswith('vit') or name.startswith('tiny-vit'):
        return ViTModel.from_pretrained(pretrained)
    # fallback: some compact models may not have dedicated classes in this codebase
    return AutoModel.from_pretrained(pretrained)


def get_text_backbone(name: str):
    pretrained = TEXT_PRETRAINED.get(name)
    if pretrained is None:
        raise KeyError(f"Unknown text backbone: {name}")
    return AutoModel.from_pretrained(pretrained)


def get_vision_pretrained_name(name: str):
    return VISION_PRETRAINED.get(name)


def get_text_pretrained_name(name: str):
    return TEXT_PRETRAINED.get(name)
