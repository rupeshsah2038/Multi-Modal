import torch
import torch.nn as nn
from .backbones import get_vision_backbone, get_text_backbone
from .heads import ClassificationHead
from .student import get_hidden_dim, extract_vision_features


class VisionOnlyStudent(nn.Module):
    """
    Image-only unimodal student baseline using MobileViT-xxs (or specified vision backbone).
    Takes only pixel_values and predicts Task 1 and Task 2.
    Accepts extra positional/keyword args to remain directly compatible with evaluate_detailed.
    """
    def __init__(
        self,
        vision: str = "mobilevit-xx-small",
        fusion_dim: int = 256,
        dropout: float = 0.1,
        num_modality_classes: int = 2,
        num_location_classes: int = 5,
        **kwargs,
    ):
        super().__init__()
        self.vision = get_vision_backbone(vision)
        vis_dim = get_hidden_dim(self.vision)
        self.proj = nn.Linear(vis_dim, fusion_dim)
        self.dropout = nn.Dropout(dropout)
        self.head_modality = ClassificationHead(fusion_dim, num_modality_classes)
        self.head_location = ClassificationHead(fusion_dim, num_location_classes)

    def forward(self, pixel_values, *args, **kwargs):
        v_out = self.vision(pixel_values)
        v_raw = extract_vision_features(v_out)
        feat = self.dropout(self.proj(v_raw))
        return {
            "logits_modality": self.head_modality(feat),
            "logits_location": self.head_location(feat),
            "img_raw": v_raw,
            "img_proj": feat,
        }


class TextOnlyStudent(nn.Module):
    """
    Text-only unimodal student baseline using BERT-mini (or specified text backbone).
    Takes only input_ids and attention_mask and predicts Task 1 and Task 2.
    Accepts extra positional/keyword args to remain directly compatible with evaluate_detailed.
    """
    def __init__(
        self,
        text: str = "bert-mini",
        fusion_dim: int = 256,
        dropout: float = 0.1,
        num_modality_classes: int = 2,
        num_location_classes: int = 5,
        **kwargs,
    ):
        super().__init__()
        self.text = get_text_backbone(text)
        txt_dim = get_hidden_dim(self.text)
        self.proj = nn.Linear(txt_dim, fusion_dim)
        self.dropout = nn.Dropout(dropout)
        self.head_modality = ClassificationHead(fusion_dim, num_modality_classes)
        self.head_location = ClassificationHead(fusion_dim, num_location_classes)

    def forward(self, pixel_values=None, input_ids=None, attention_mask=None, *args, **kwargs):
        # Support positional calling (pv, ids, mask) where pixel_values is 1st arg
        if input_ids is None and len(args) >= 2:
            input_ids = args[0]
            attention_mask = args[1]
        elif input_ids is None and len(args) == 1:
            input_ids = args[0]

        t_out = self.text(input_ids=input_ids, attention_mask=attention_mask)
        t_raw = t_out.last_hidden_state[:, 0]
        feat = self.dropout(self.proj(t_raw))
        return {
            "logits_modality": self.head_modality(feat),
            "logits_location": self.head_location(feat),
            "txt_raw": t_raw,
            "txt_proj": feat,
        }
