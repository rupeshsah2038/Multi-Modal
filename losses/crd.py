import torch
import torch.nn as nn
import torch.nn.functional as F

class CRDLoss(nn.Module):
    def __init__(self, temperature=0.1, base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, s_img, s_txt, t_img, t_txt, labels):
        s_img = F.normalize(s_img, dim=-1)
        s_txt = F.normalize(s_txt, dim=-1)
        t_img = F.normalize(t_img, dim=-1)
        t_txt = F.normalize(t_txt, dim=-1)
        batch_size = s_img.shape[0]
        logits_img = torch.mm(s_img, t_img.t()) / self.temperature
        logits_txt = torch.mm(s_txt, t_txt.t()) / self.temperature
        targets = torch.arange(batch_size, device=s_img.device)
        loss_img = F.cross_entropy(logits_img, targets)
        loss_txt = F.cross_entropy(logits_txt, targets)
        return (loss_img + loss_txt) / 2
