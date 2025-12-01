import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillationLoss(nn.Module):
    def __init__(self, fusion_dim=512, alpha=1.0, beta=100.0, T=2.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.kl = nn.KLDivLoss(reduction="batchmean")
        self.mse = nn.MSELoss()
        self.alpha, self.beta, self.T = alpha, beta, T
        self.proj_t_img = nn.Linear(1024, fusion_dim)
        self.proj_t_txt = nn.Linear(768, fusion_dim)

    def forward(self, s_out, t_out, y_mod, y_loc):
        ce_mod = self.ce(s_out['logits_modality'], y_mod)
        ce_loc = self.ce(s_out['logits_location'], y_loc)
        kl_mod = self.kl(
            F.log_softmax(s_out['logits_modality'] / self.T, dim=-1),
            F.softmax(t_out['logits_modality'] / self.T, dim=-1)
        ) * (self.T ** 2)
        kl_loc = self.kl(
            F.log_softmax(s_out['logits_location'] / self.T, dim=-1),
            F.softmax(t_out['logits_location'] / self.T, dim=-1)
        ) * (self.T ** 2)
        dev = s_out['img_emb'].device
        t_img_proj = self.proj_t_img.to(dev)(t_out['img_emb'])
        t_txt_proj = self.proj_t_txt.to(dev)(t_out['txt_emb'])
        feat_img = self.mse(s_out['img_emb'], t_img_proj)
        feat_txt = self.mse(s_out['txt_emb'], t_txt_proj)
        loss = ce_mod + ce_loc + self.alpha * (kl_mod + kl_loc) + self.beta * (feat_img + feat_txt)
        return loss
