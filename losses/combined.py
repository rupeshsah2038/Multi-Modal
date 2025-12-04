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
        # Projection layers will be created lazily on first forward pass
        self.proj_t_img = None
        self.proj_t_txt = None

    def forward(self, s_out, t_out, y_mod, y_loc):
        ce = self.ce(s_out["logits_modality"], y_mod) + self.ce(s_out["logits_location"], y_loc)
        kl = self.kl(F.log_softmax(s_out["logits_modality"]/self.T, dim=-1),
                     F.softmax(t_out["logits_modality"]/self.T, dim=-1)) * (self.T**2)
        kl += self.kl(F.log_softmax(s_out["logits_location"]/self.T, dim=-1),
                      F.softmax(t_out["logits_location"]/self.T, dim=-1)) * (self.T**2)
        # Ensure projection layers are on the same device as model outputs
        dev = s_out.get('img_proj', next(iter(s_out.values()))).device
        # Create projections lazily based on runtime tensor shapes
        in_img = t_out['img_raw'].size(-1)
        out_img = s_out['img_proj'].size(-1)
        if (self.proj_t_img is None) or (getattr(self.proj_t_img, 'in_features', None) != in_img) or (getattr(self.proj_t_img, 'out_features', None) != out_img):
            self.proj_t_img = nn.Linear(in_img, out_img).to(dev)
        t_img = self.proj_t_img(t_out["img_raw"].to(dev))
        # Text projection
        in_txt = t_out['txt_raw'].size(-1)
        out_txt = s_out['txt_proj'].size(-1)
        if (self.proj_t_txt is None) or (getattr(self.proj_t_txt, 'in_features', None) != in_txt) or (getattr(self.proj_t_txt, 'out_features', None) != out_txt):
            self.proj_t_txt = nn.Linear(in_txt, out_txt).to(dev)
        t_txt = self.proj_t_txt(t_out["txt_raw"].to(dev))
        mse = self.mse(s_out["img_proj"], t_img) + self.mse(s_out["txt_proj"], t_txt)
        crd = self.crd(s_out, t_out, y_mod, y_loc)
        return ce + self.alpha * kl + self.beta * mse + self.gamma * crd
