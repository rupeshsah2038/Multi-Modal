import torch
import torch.nn as nn

class CrossAttentionFusion(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim*4),
            nn.GELU(),
            nn.Linear(dim*4, dim),
            nn.Dropout(dropout)
        )

    def forward(self, img_emb, txt_emb):
        x = torch.stack([img_emb, txt_emb], dim=1)
        q = self.norm1(x.mean(dim=1, keepdim=True))
        x = self.cross_attn(q, x, x)[0]
        x = x + self.ffn(self.norm2(x))
        return x.squeeze(1)
