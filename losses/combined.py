import torch
import torch.nn as nn
import torch.nn.functional as F
from .crd import CRDLoss

class MedKDCombinedLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=100.0, gamma=10.0, T=4.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.kl = nn.KLDivLoss(reduction="batchmean")
        self.mse = nn.MSELoss()
        self.crd = CRDLoss()
        self.alpha, self.beta, self.gamma, self.T = alpha, beta, gamma, T
        self.proj_t_img = nn.Linear(1024, 512)
        self.proj_t_txt = nn.Linear(768, 512)

    def forward(self, s_out, t_out, y_mod, y_loc):
        ce = self.ce(s_out["logits_modality"], y_mod) + self.ce(s_out["logits_location"], y_loc)
        kl = self.kl(F.log_softmax(s_out["logits_modality"]/self.T, dim=-1),
                     F.softmax(t_out["logits_modality"]/self.T, dim=-1)) * (self.T**2)
        kl += self.kl(F.log_softmax(s_out["logits_location"]/self.T, dim=-1),
                      F.softmax(t_out["logits_location"]/self.T, dim=-1)) * (self.T**2)
        # Ensure projection layers are on the same device as model outputs
        dev = s_out.get('img_proj', next(iter(s_out.values()))).device
        t_img = self.proj_t_img.to(dev)(t_out["img_raw"])
        t_txt = self.proj_t_txt.to(dev)(t_out["txt_raw"])
        mse = self.mse(s_out["img_proj"], t_img) + self.mse(s_out["txt_proj"], t_txt)
        crd = self.crd(s_out, t_out, y_mod, y_loc)
        return ce + self.alpha * kl + self.beta * mse + self.gamma * crd
