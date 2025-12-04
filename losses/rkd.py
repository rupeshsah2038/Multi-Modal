import torch
import torch.nn as nn
import torch.nn.functional as F

class RKDLoss(nn.Module):
    def __init__(self, w_dist=25.0, w_angle=50.0):
        super().__init__()
        self.w_dist = w_dist
        self.w_angle = w_angle
        # Projection layers will be created lazily on first forward pass to
        # support swapping backbones with different feature sizes.
        self.proj_t_img = None
        self.proj_t_txt = None

    def pdist(self, x):
        x_norm = x.pow(2).sum(dim=-1, keepdim=True)
        dist = x_norm + x_norm.t() - 2 * torch.mm(x, x.t())
        # Clamp to avoid negative values due to numerical precision
        dist = torch.clamp(dist, min=0.0)
        return torch.sqrt(dist + 1e-8)

    def forward(self, s_out, t_out, y_mod=None, y_loc=None):
        # Accept both dict-based calls (from trainer) and direct tensor calls (for compatibility)
        if isinstance(s_out, dict):
            s_img = s_out["img_proj"]
            s_txt = s_out["txt_proj"]
            t_img_raw = t_out["img_raw"]
            t_txt_raw = t_out["txt_raw"]
            # Project teacher features to match student dimension. Create
            # projection layers lazily based on runtime tensor shapes so
            # swapping teacher/student backbones works without manual edits.
            dev = s_img.device
            # image projection
            in_img = t_img_raw.size(-1)
            out_img = s_img.size(-1)
            if (self.proj_t_img is None) or (getattr(self.proj_t_img, 'in_features', None) != in_img) or (getattr(self.proj_t_img, 'out_features', None) != out_img):
                self.proj_t_img = nn.Linear(in_img, out_img).to(dev)
            t_img = self.proj_t_img(t_img_raw.to(dev))
            # text projection
            in_txt = t_txt_raw.size(-1)
            out_txt = s_txt.size(-1)
            if (self.proj_t_txt is None) or (getattr(self.proj_t_txt, 'in_features', None) != in_txt) or (getattr(self.proj_t_txt, 'out_features', None) != out_txt):
                self.proj_t_txt = nn.Linear(in_txt, out_txt).to(dev)
            t_txt = self.proj_t_txt(t_txt_raw.to(dev))
            # Compute RKD loss for image and text separately, then average
            s_img = F.normalize(s_img, dim=-1)
            s_txt = F.normalize(s_txt, dim=-1)
            t_img = F.normalize(t_img, dim=-1)
            t_txt = F.normalize(t_txt, dim=-1)
            loss_img = self._compute_rkd(s_img, t_img)
            loss_txt = self._compute_rkd(s_txt, t_txt)
            return (loss_img + loss_txt) / 2
        else:
            # Direct tensor calls (old interface for backward compatibility)
            s_feats = s_out
            t_feats = t_out
            return self._compute_rkd(s_feats, t_feats)

    def _compute_rkd(self, s_feats, t_feats):
        # Compute pairwise distance loss
        s_dist = self.pdist(s_feats)
        t_dist = self.pdist(t_feats)
        loss_dist = F.smooth_l1_loss(s_dist, t_dist) * self.w_dist
        
        # Compute angle loss via cosine similarity (normalized dot product)
        # Normalize features and compute pairwise cosine similarity
        s_norm = F.normalize(s_feats, dim=-1)
        t_norm = F.normalize(t_feats, dim=-1)
        s_angle = torch.mm(s_norm, s_norm.t())  # (B, B) pairwise cosine similarity
        t_angle = torch.mm(t_norm, t_norm.t())  # (B, B) pairwise cosine similarity
        loss_angle = F.smooth_l1_loss(s_angle, t_angle) * self.w_angle
        return loss_dist + loss_angle
