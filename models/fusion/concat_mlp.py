import torch.nn as nn

class ConcatMLPFusion(nn.Module):
    def __init__(self, dim: int, hidden_mult: int = 2, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        blocks = []
        in_d = dim * 2
        for _ in range(layers-1):
            blocks.extend([
                nn.Linear(in_d, hidden_mult * dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            in_d = hidden_mult * dim
        blocks.append(nn.Linear(in_d, dim))
        self.mlp = nn.Sequential(*blocks)

    def forward(self, img_emb, txt_emb):
        return self.mlp(torch.cat([img_emb, txt_emb], dim=-1))
