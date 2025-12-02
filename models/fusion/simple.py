import torch.nn as nn
import torch
class SimpleFusion(nn.Module):
    def __init__(self, dim: int, heads: int = 8, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim*4,
            dropout=dropout, activation="gelu", batch_first=True
        )
        self.fusion = nn.TransformerEncoder(encoder_layer, num_layers=layers)

    def forward(self, img_emb, txt_emb):
        x = torch.stack([img_emb, txt_emb], dim=1)
        x = self.fusion(x)
        return x.mean(dim=1)
