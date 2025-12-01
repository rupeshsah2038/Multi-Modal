from transformers import AutoModel, ViTModel, DeiTModel

VISION_BACKBONES = {
    "vit-large": lambda: ViTModel.from_pretrained("google/vit-large-patch16-224"),
    "vit-base": lambda: ViTModel.from_pretrained("google/vit-base-patch16-224"),
    "deit-base": lambda: DeiTModel.from_pretrained("facebook/deit-base-distilled-patch16-224"),
}

TEXT_BACKBONES = {
    "bio-clinical-bert": lambda: AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT"),
    "distilbert": lambda: AutoModel.from_pretrained("distilbert-base-uncased"),
}

def get_vision_backbone(name: str):
    return VISION_BACKBONES[name]()

def get_text_backbone(name: str):
    return TEXT_BACKBONES[name]()
