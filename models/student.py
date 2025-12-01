import torch.nn as nn
from .backbones import get_vision_backbone, get_text_backbone
from .fusion.simple import SimpleFusion
from .heads import ClassificationHead

class Student(nn.Module):
    def __init__(self, vision, text, fusion_dim=512, fusion_heads=8, fusion_layers=1, dropout=0.1):
        super().__init__()
        self.vision = get_vision_backbone(vision)
        self.text = get_text_backbone(text)
        vis_dim = self.vision.config.hidden_size
        txt_dim = self.text.config.hidden_size
        self.proj_vis = nn.Linear(vis_dim, fusion_dim)
        self.proj_txt = nn.Linear(txt_dim, fusion_dim)
        self.fusion = SimpleFusion(fusion_dim, fusion_heads, fusion_layers)
        self.dropout = nn.Dropout(dropout)
        self.head_modality = ClassificationHead(fusion_dim, 2)
        self.head_location = ClassificationHead(fusion_dim, 5)

    def forward(self, pixel_values, input_ids, attention_mask):
        v_out = self.vision(pixel_values)
        v = v_out.last_hidden_state[:, 0] if hasattr(v_out, 'last_hidden_state') else v_out[0][:, 0]
        t_out = self.text(input_ids=input_ids, attention_mask=attention_mask)
        t = t_out.last_hidden_state[:, 0]
        v = self.proj_vis(v)
        t = self.proj_txt(t)
        fused = self.dropout(self.fusion(v, t))
        return {
            "logits_modality": self.head_modality(fused),
            "logits_location": self.head_location(fused),
            "img_raw": v,
            "txt_raw": t,
            "img_proj": v,
            "txt_proj": t,
        }
