import torch
from losses.rkd import RKDLoss

B = 4
feat_dim = 512

s_img = torch.randn(B, feat_dim)
s_txt = torch.randn(B, feat_dim)
t_img_raw = torch.randn(B, 1024)
t_txt_raw = torch.randn(B, 768)

y_mod = torch.randint(0, 2, (B,))
y_loc = torch.randint(0, 5, (B,))

rkd = RKDLoss()

# Test 1: Dict-based interface (from trainer)
s_out = {'img_proj': s_img, 'txt_proj': s_txt, 'logits_modality': torch.randn(B, 2), 'logits_location': torch.randn(B, 5)}
t_out = {'img_raw': t_img_raw, 'txt_raw': t_txt_raw, 'logits_modality': torch.randn(B, 2), 'logits_location': torch.randn(B, 5)}
loss1 = rkd(s_out, t_out, y_mod, y_loc)
print('RKD loss (dict interface):', loss1.item())

# Test 2: Direct tensor interface (backward compatibility)
t_img_proj = torch.randn(B, feat_dim)
t_txt_proj = torch.randn(B, feat_dim)
loss2 = rkd(s_img, t_img_proj)
print('RKD loss (tensor interface):', loss2.item())
