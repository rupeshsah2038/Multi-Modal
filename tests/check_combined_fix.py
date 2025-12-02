import torch
from losses.combined import MedKDCombinedLoss

# small batch
B = 4
fusion_dim = 512

s_out = {
    'logits_modality': torch.randn(B, 2),
    'logits_location': torch.randn(B, 5),
    'img_proj': torch.randn(B, fusion_dim),
    'txt_proj': torch.randn(B, fusion_dim),
}

# teacher outputs on CUDA (simulate GPU case if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
t_out = {
    'logits_modality': torch.randn(B, 2, device=device),
    'logits_location': torch.randn(B, 5, device=device),
    'img_raw': torch.randn(B, 1024, device=device),
    'txt_raw': torch.randn(B, 768, device=device),
}

y_mod = torch.randint(0, 2, (B,))
y_loc = torch.randint(0, 5, (B,))

loss_fn = MedKDCombinedLoss(alpha=1.0, beta=10.0, gamma=5.0, T=2.0)
# move s_out projected values to device to mimic training scenario
s_out['img_proj'] = s_out['img_proj'].to(device)
s_out['txt_proj'] = s_out['txt_proj'].to(device)
# student logits should also be on the same device in real runs
s_out['logits_modality'] = s_out['logits_modality'].to(device)
s_out['logits_location'] = s_out['logits_location'].to(device)

loss = loss_fn(s_out, t_out, y_mod.to(device), y_loc.to(device))
print('Combined loss computed successfully:', loss.item())
