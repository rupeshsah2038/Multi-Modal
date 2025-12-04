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
        # Projection layers will be created lazily on first forward pass
        self.proj_t_img = None
        self.proj_t_txt = None

    def forward(self, s_feats, t_feats=None, y_mod=None, y_loc=None):
        """
        Supports two calling conventions:
        - dict-based: forward(s_out, t_out, y_mod, y_loc) where s_out/t_out are dicts
        - tensor-based: forward(s_feats, t_feats) for backward compatibility
        We compute MMD per-modality (image/text) and return the average.
        """
        if isinstance(s_feats, dict):
            # dict-based call from trainer
            s_img = s_feats['img_proj']
            s_txt = s_feats['txt_proj']
            t_img_raw = t_feats['img_raw'] if isinstance(t_feats, dict) else t_feats['img_raw']
            t_txt_raw = t_feats['txt_raw'] if isinstance(t_feats, dict) else t_feats['txt_raw']
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
            # legacy tensor-based call: s_feats, t_feats
            s_img = s_feats
            t_img = t_feats
            # No text features in legacy call
            s_txt = None
            t_txt = None

        def compute_mmd(a, b):
            a3 = a.unsqueeze(0)
            b3 = b.unsqueeze(0)
            k_xx = gaussian_kernel(a3, a3)
            k_yy = gaussian_kernel(b3, b3)
            k_xy = gaussian_kernel(a3, b3)
            return k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()

        mmd_vals = []
        if s_img is not None and t_img is not None:
            mmd_vals.append(compute_mmd(s_img, t_img))
        if s_txt is not None and t_txt is not None:
            mmd_vals.append(compute_mmd(s_txt, t_txt))

        if not mmd_vals:
            return torch.tensor(0.0, device=s_img.device if isinstance(s_img, torch.Tensor) else None)

        return sum(mmd_vals) / len(mmd_vals)
