import torch.nn as nn

class TransformerConcatFusion(nn.Module):
    def __init__(self, dim: int, heads: int = 8, layers: int = 4, dropout: float = 0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim*4,
            dropout=dropout, activation="gelu", batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.modality_token = nn.Parameter(torch.randn(1, 2, dim))

    def forward(self, img_emb, txt_emb):
        B = img_emb.shape[0]
        tokens = self.modality_token.expand(B, -1, -1)
        tokens[:, 0] = img_emb
        tokens[:, 1] = txt_emb
        out = self.transformer(tokens)
        return out.mean(dim=1)
