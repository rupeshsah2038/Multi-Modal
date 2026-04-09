import torch
import torch.nn as nn


class TinyViT(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        d_model=192,
        depth=4,
        nhead=3,
        mlp_ratio=4.0,
        dropout=0.1,
    ):
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError("img_size must be divisible by patch_size")
        self.num_patches = (img_size // patch_size) ** 2

        self.patch = nn.Conv2d(in_chans, d_model, kernel_size=patch_size, stride=patch_size, bias=True)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.zeros(1, 1 + self.num_patches, d_model))
        self.pos_drop = nn.Dropout(dropout)

        ff_dim = int(d_model * mlp_ratio)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.norm = nn.LayerNorm(d_model)

        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)

    def forward(self, pixel_values):
        x = self.patch(pixel_values)
        x = x.flatten(2).transpose(1, 2)

        cls = self.cls.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)

        x = self.pos_drop(x + self.pos)
        x = self.enc(x)
        x = self.norm(x)
        return x[:, 0]


class TinyTextTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        max_len=256,
        d_model=128,
        depth=4,
        nhead=4,
        mlp_ratio=4.0,
        dropout=0.1,
        pad_token_id=0,
    ):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.tok = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.pos = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)

        ff_dim = int(d_model * mlp_ratio)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, input_ids, attention_mask):
        bsz, seq_len = input_ids.shape
        pos = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, seq_len)

        x = self.tok(input_ids) + self.pos(pos)
        x = self.drop(x)

        key_padding_mask = (attention_mask == 0)
        x = self.enc(x, src_key_padding_mask=key_padding_mask)
        x = self.norm(x)
        return x[:, 0]


class StudentCustom(nn.Module):
    def __init__(
        self,
        vocab_size,
        pad_token_id=0,
        fusion_dim=256,
        vision_d=192,
        vision_depth=4,
        vision_heads=3,
        vision_mlp_ratio=4.0,
        text_d=128,
        text_depth=4,
        text_heads=4,
        text_mlp_ratio=4.0,
        max_len=256,
        dropout=0.1,
        num_modality_classes=2,
        num_location_classes=5,
    ):
        super().__init__()
        self.vision = TinyViT(
            d_model=vision_d,
            depth=vision_depth,
            nhead=vision_heads,
            mlp_ratio=vision_mlp_ratio,
            dropout=dropout,
        )
        self.text = TinyTextTransformer(
            vocab_size=vocab_size,
            max_len=max_len,
            d_model=text_d,
            depth=text_depth,
            nhead=text_heads,
            mlp_ratio=text_mlp_ratio,
            dropout=dropout,
            pad_token_id=pad_token_id,
        )

        self.proj_vis = nn.Linear(vision_d, fusion_dim)
        self.proj_txt = nn.Linear(text_d, fusion_dim)

        self.gate = nn.Sequential(
            nn.Linear(2 * fusion_dim, fusion_dim),
            nn.GELU(),
            nn.Linear(fusion_dim, fusion_dim),
            nn.Sigmoid(),
        )
        self.drop = nn.Dropout(dropout)

        self.head_modality = nn.Linear(fusion_dim, num_modality_classes)
        self.head_location = nn.Linear(fusion_dim, num_location_classes)

    def forward(self, pixel_values, input_ids, attention_mask):
        v_raw = self.vision(pixel_values)
        t_raw = self.text(input_ids, attention_mask)

        v = self.proj_vis(v_raw)
        t = self.proj_txt(t_raw)

        g = self.gate(torch.cat([v, t], dim=-1))
        fused = self.drop(g * v + (1.0 - g) * t)

        return {
            "logits_modality": self.head_modality(fused),
            "logits_location": self.head_location(fused),
            "img_raw": v_raw,
            "txt_raw": t_raw,
            "img_proj": v,
            "txt_proj": t,
        }
