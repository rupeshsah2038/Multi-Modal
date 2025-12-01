import torch.nn as nn

class GatedFusion(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim*2, dim),
            nn.Sigmoid()
        )

    def forward(self, img_emb, txt_emb):
        gate = self.gate(torch.cat([img_emb, txt_emb], dim=-1))
        return gate * img_emb + (1 - gate) * txt_emb
