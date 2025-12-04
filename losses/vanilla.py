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
        self.fusion_dim = fusion_dim
        # Projection layers will be created lazily on first forward pass
        self.proj_t_img = None
        self.proj_t_txt = None

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
        # Use consistent keys with the rest of the project:
        # student provides `img_proj` / `txt_proj`; teacher provides `img_raw` / `txt_raw`
        dev = s_out['img_proj'].device
        # Create projections lazily based on runtime tensor shapes
        in_img = t_out['img_raw'].size(-1)
        out_img = s_out['img_proj'].size(-1)
        if (self.proj_t_img is None) or (getattr(self.proj_t_img, 'in_features', None) != in_img) or (getattr(self.proj_t_img, 'out_features', None) != out_img):
            self.proj_t_img = nn.Linear(in_img, out_img).to(dev)
        t_img_proj = self.proj_t_img(t_out['img_raw'].to(dev))
        # Text projection
        in_txt = t_out['txt_raw'].size(-1)
        out_txt = s_out['txt_proj'].size(-1)
        if (self.proj_t_txt is None) or (getattr(self.proj_t_txt, 'in_features', None) != in_txt) or (getattr(self.proj_t_txt, 'out_features', None) != out_txt):
            self.proj_t_txt = nn.Linear(in_txt, out_txt).to(dev)
        t_txt_proj = self.proj_t_txt(t_out['txt_raw'].to(dev))
        feat_img = self.mse(s_out['img_proj'], t_img_proj)
        feat_txt = self.mse(s_out['txt_proj'], t_txt_proj)
        loss = ce_mod + ce_loc + self.alpha * (kl_mod + kl_loc) + self.beta * (feat_img + feat_txt)
        return loss
