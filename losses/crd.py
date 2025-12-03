import torch
import torch.nn as nn
import torch.nn.functional as F

class CRDLoss(nn.Module):
    def __init__(self, temperature=0.1, base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        # Projection layers to match student feature dimension (512)
        self.proj_t_img = nn.Linear(1024, 512)
        self.proj_t_txt = nn.Linear(768, 512)

    def forward(self, s_out, t_out, y_mod, y_loc):
        # Accept both dict-based calls (from trainer) and direct tensor calls (for compatibility)
        if isinstance(s_out, dict):
            s_img = s_out["img_proj"]
            s_txt = s_out["txt_proj"]
            t_img_raw = t_out["img_raw"]
            t_txt_raw = t_out["txt_raw"]
            # Project teacher features to match student dimension
            dev = s_img.device
            t_img = self.proj_t_img.to(dev)(t_img_raw)
            t_txt = self.proj_t_txt.to(dev)(t_txt_raw)
        else:
            # Direct tensor calls (old interface for backward compatibility)
            s_img = s_out
            s_txt = t_out
            t_img = y_mod
            t_txt = y_loc
        
        s_img = F.normalize(s_img, dim=-1)
        s_txt = F.normalize(s_txt, dim=-1)
        t_img = F.normalize(t_img, dim=-1)
        t_txt = F.normalize(t_txt, dim=-1)
        batch_size = s_img.shape[0]
        logits_img = torch.mm(s_img, t_img.t()) / self.temperature
        logits_txt = torch.mm(s_txt, t_txt.t()) / self.temperature
        # Use identity targets (diagonal)
        targets = torch.arange(batch_size, device=s_img.device)
        loss_img = F.cross_entropy(logits_img, targets)
        loss_txt = F.cross_entropy(logits_txt, targets)
        return (loss_img + loss_txt) / 2
