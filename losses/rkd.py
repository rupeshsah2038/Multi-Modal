import torch
import torch.nn as nn
import torch.nn.functional as F

class RKDLoss(nn.Module):
    def __init__(self, w_dist=25.0, w_angle=50.0):
        super().__init__()
        self.w_dist = w_dist
        self.w_angle = w_angle

    def pdist(self, x):
        x_norm = x.pow(2).sum(dim=-1, keepdim=True)
        dist = x_norm + x_norm.t() - 2 * torch.mm(x, x.t())
        return torch.sqrt(dist + 1e-8)

    def forward(self, s_feats, t_feats):
        s_dist = self.pdist(s_feats)
        t_dist = self.pdist(t_feats)
        loss_dist = F.smooth_l1_loss(s_dist, t_dist) * self.w_dist
        s_angle = torch.bmm(s_feats, s_feats.transpose(1,2))
        t_angle = torch.bmm(t_feats, t_feats.transpose(1,2))
        loss_angle = F.smooth_l1_loss(s_angle, t_angle) * self.w_angle
        return loss_dist + loss_angle
