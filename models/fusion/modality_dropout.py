# models/fusion/modality_dropout.py
import torch
import torch.nn as nn

class ModalityDropoutFusion(nn.Module):
    def __init__(self, dim, p_img=0.3, p_txt=0.3):
        super().__init__()
        self.p_img, self.p_txt = p_img, p_txt
        self.gated = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
    
    def forward(self, img_emb, txt_emb):
        if self.training:
            drop_img = torch.rand(1) < self.p_img
            drop_txt = torch.rand(1) < self.p_txt
            if drop_img: img_emb = img_emb * 0
            if drop_txt: txt_emb = txt_emb * 0
        return self.gated(torch.cat([img_emb, txt_emb], -1))