import torch
import torch.nn as nn
import torch.nn.functional as F

class CRDLoss(nn.Module):
    def __init__(self, temperature=0.1, base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        # Projection layers will be created lazily on first forward pass
        self.proj_t_img = None
        self.proj_t_txt = None

    def forward(self, s_out, t_out, y_mod, y_loc):
        # Accept both dict-based calls (from trainer) and direct tensor calls (for compatibility)
        if isinstance(s_out, dict):
            s_img = s_out["img_proj"]
            s_txt = s_out["txt_proj"]
            t_img_raw = t_out["img_raw"]
            t_txt_raw = t_out["txt_raw"]
            # Project teacher features to match student dimension
            dev = s_img.device
            # Create projections lazily based on runtime tensor shapes
            in_img = t_img_raw.size(-1)
            out_img = s_img.size(-1)
            if (self.proj_t_img is None) or (getattr(self.proj_t_img, 'in_features', None) != in_img) or (getattr(self.proj_t_img, 'out_features', None) != out_img):
                self.proj_t_img = nn.Linear(in_img, out_img).to(dev)
            t_img = self.proj_t_img(t_img_raw.to(dev))
            # Text projection
            in_txt = t_txt_raw.size(-1)
            out_txt = s_txt.size(-1)
            if (self.proj_t_txt is None) or (getattr(self.proj_t_txt, 'in_features', None) != in_txt) or (getattr(self.proj_t_txt, 'out_features', None) != out_txt):
                self.proj_t_txt = nn.Linear(in_txt, out_txt).to(dev)
            t_txt = self.proj_t_txt(t_txt_raw.to(dev))
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
