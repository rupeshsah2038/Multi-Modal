import torch
from losses.vanilla import DistillationLoss

# Create a small batch
B = 4
fusion_dim = 512

# Student outputs (projected)
s_out = {
    "logits_modality": torch.randn(B, 2),
    "logits_location": torch.randn(B, 5),
    "img_proj": torch.randn(B, fusion_dim),
    "txt_proj": torch.randn(B, fusion_dim),
}

# Teacher raw features
t_out = {
    "logits_modality": torch.randn(B, 2),
    "logits_location": torch.randn(B, 5),
    "img_raw": torch.randn(B, 1024),
    "txt_raw": torch.randn(B, 768),
}

# integer labels
y_mod = torch.randint(0, 2, (B,))
y_loc = torch.randint(0, 5, (B,))

loss_fn = DistillationLoss(fusion_dim=fusion_dim, alpha=1.0, beta=10.0, T=2.0)
loss = loss_fn(s_out, t_out, y_mod, y_loc)
print('Loss computed successfully:', loss.item())
