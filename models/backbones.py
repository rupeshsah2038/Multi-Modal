from transformers import AutoModel, ViTModel, DeiTModel

# Mapping from friendly names to Hugging Face pretrained identifiers.
VISION_PRETRAINED = {
    "vit-large": "google/vit-large-patch16-224",
    "vit-base": "google/vit-base-patch16-224",
    "deit-base": "facebook/deit-base-distilled-patch16-224",
}

TEXT_PRETRAINED = {
    "bio-clinical-bert": "emilyalsentzer/Bio_ClinicalBERT",
    "distilbert": "distilbert-base-uncased",
}


def get_vision_backbone(name: str):
    pretrained = VISION_PRETRAINED.get(name)
    if pretrained is None:
        raise KeyError(f"Unknown vision backbone: {name}")
    if name.startswith('deit'):
        return DeiTModel.from_pretrained(pretrained)
    return ViTModel.from_pretrained(pretrained)


def get_text_backbone(name: str):
    pretrained = TEXT_PRETRAINED.get(name)
    if pretrained is None:
        raise KeyError(f"Unknown text backbone: {name}")
    return AutoModel.from_pretrained(pretrained)


def get_vision_pretrained_name(name: str):
    return VISION_PRETRAINED.get(name)


def get_text_pretrained_name(name: str):
    return TEXT_PRETRAINED.get(name)
