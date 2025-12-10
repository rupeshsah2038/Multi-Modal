# models/fusion/film.py
import torch
import torch.nn as nn

class FiLMFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp_gamma = nn.Linear(dim, dim)
        self.mlp_beta = nn.Linear(dim, dim)
    
    def forward(self, img_emb, txt_emb):
        gamma = self.mlp_gamma(txt_emb)
        beta  = self.mlp_beta(txt_emb)
        return gamma * img_emb + beta