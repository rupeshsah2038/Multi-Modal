import torch
import torch.nn as nn
import torch.nn.functional as F

def gaussian_kernel(x, y, bandwidths=[0.2, 0.5, 1, 2, 5]):
    B, N, D = x.shape
    _, M, _ = y.shape
    kernels = 0
    for bw in bandwidths:
        diff = x.unsqueeze(2) - y.unsqueeze(1)
        dist_sq = torch.sum(diff**2, dim=-1)
        kernel = torch.exp(-dist_sq / (2 * bw**2))
        kernels += kernel.mean(0)
    return kernels / len(bandwidths)

class MMDLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, s_feats, t_feats):
        s_feats = s_feats.unsqueeze(0)
        t_feats = t_feats.unsqueeze(0)
        k_xx = gaussian_kernel(s_feats, s_feats)
        k_yy = gaussian_kernel(t_feats, t_feats)
        k_xy = gaussian_kernel(s_feats, t_feats)
        mmd = k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()
        return mmd
