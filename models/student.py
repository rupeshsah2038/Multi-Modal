import torch.nn as nn
from .backbones import get_vision_backbone, get_text_backbone
from .fusion import (SimpleFusion, ConcatMLPFusion, CrossAttentionFusion,
                     GatedFusion, TransformerConcatFusion, ModalityDropoutFusion,
                     FiLMFusion, EnergyAwareAdaptiveFusion, SHoMRFusion)
from .heads import ClassificationHead


def get_hidden_dim(model):
    """Get hidden dimension from different model types."""
    config = model.config
    # ViT, DeiT, BERT, DistilBERT, etc.
    if hasattr(config, 'hidden_size'):
        return config.hidden_size
    # MobileViT models use neck_hidden_sizes
    elif hasattr(config, 'neck_hidden_sizes'):
        return config.neck_hidden_sizes[-1]
    else:
        raise AttributeError(f"Cannot determine hidden dimension for model config: {type(config)}")

class Student(nn.Module):
    def __init__(self, vision, text, fusion_dim, fusion_type='simple', fusion_heads=8, 
                 fusion_layers=1, dropout=0.1, num_modality_classes=2, num_location_classes=5,
                 fusion_params=None):
        super().__init__()
        self.vision = get_vision_backbone(vision)
        self.text = get_text_backbone(text)
        vis_dim = get_hidden_dim(self.vision)
        txt_dim = get_hidden_dim(self.text)
        self.proj_vis = nn.Linear(vis_dim, fusion_dim)
        self.proj_txt = nn.Linear(txt_dim, fusion_dim)
        # Pass fusion_params for module-specific configuration
        self.fusion = self._create_fusion(fusion_type, fusion_dim, fusion_heads, fusion_layers, fusion_params or {})
        self.dropout = nn.Dropout(dropout)
        self.head_modality = ClassificationHead(fusion_dim, num_modality_classes)
        self.head_location = ClassificationHead(fusion_dim, num_location_classes)

    def _create_fusion(self, fusion_type, fusion_dim, fusion_heads, fusion_layers, fusion_params):
        """Factory method to create fusion module based on type with configurable parameters."""
        fusion_type = fusion_type.lower().replace('_', '').replace('-', '')
        
        # Extract module-specific parameters with defaults (can be overridden via fusion_params)
        hidden_mult = fusion_params.get('hidden_mult', 2)
        dropout = fusion_params.get('dropout', 0.1)
        p_img = fusion_params.get('p_img', 0.3)
        p_txt = fusion_params.get('p_txt', 0.3)
        num_moments = fusion_params.get('num_moments', 4)
        
        # Map config names to classes and their preferred arguments
        if fusion_type in ['simple', 'simplefusion']:
            return SimpleFusion(fusion_dim, fusion_heads, fusion_layers)
        elif fusion_type in ['concatmlp', 'concatmlpfusion']:
            return ConcatMLPFusion(dim=fusion_dim, hidden_mult=hidden_mult, layers=fusion_layers)
        elif fusion_type in ['crossattention', 'crossattentionfusion']:
            return CrossAttentionFusion(dim=fusion_dim, heads=fusion_heads, dropout=dropout)
        elif fusion_type in ['gated', 'gatedfusion']:
            return GatedFusion(dim=fusion_dim)
        elif fusion_type in ['transformerconcat', 'transformerconcatfusion']:
            return TransformerConcatFusion(dim=fusion_dim, heads=fusion_heads, layers=fusion_layers)
        elif fusion_type in ['modalitydropout', 'modalitydropoutfusion']:
            return ModalityDropoutFusion(dim=fusion_dim, p_img=p_img, p_txt=p_txt)
        elif fusion_type in ['film', 'filmfusion']:
            return FiLMFusion(dim=fusion_dim)
        elif fusion_type in ['energyawareadaptive', 'energyawareadaptivefusion']:
            return EnergyAwareAdaptiveFusion(dim=fusion_dim, heads=fusion_heads, dropout=dropout)
        elif fusion_type in ['shomr', 'shomrfusion']:
            return SHoMRFusion(dim=fusion_dim, heads=fusion_heads, dropout=dropout)
        else:
            # Default to SimpleFusion
            return SimpleFusion(fusion_dim, fusion_heads, fusion_layers)

    def forward(self, pixel_values, input_ids, attention_mask):
        v_out = self.vision(pixel_values)
        # keep backbone raw vision features
        v_raw = v_out.last_hidden_state[:, 0] if hasattr(v_out, 'last_hidden_state') else v_out[0][:, 0]
        t_out = self.text(input_ids=input_ids, attention_mask=attention_mask)
        # keep backbone raw text features
        t_raw = t_out.last_hidden_state[:, 0]
        # projected features used for fusion and downstream heads
        v = self.proj_vis(v_raw)
        t = self.proj_txt(t_raw)
        fused = self.dropout(self.fusion(v, t))
        return {
            "logits_modality": self.head_modality(fused),
            "logits_location": self.head_location(fused),
            # `img_raw` / `txt_raw` are backbone outputs; `img_proj` / `txt_proj` are projected
            "img_raw": v_raw,
            "txt_raw": t_raw,
            "img_proj": v,
            "txt_proj": t,
        }
