import torch
from losses.crd import CRDLoss

B = 4
feat_dim = 512

s_img = torch.randn(B, feat_dim)
s_txt = torch.randn(B, feat_dim)
t_img = torch.randn(B, feat_dim)
t_txt = torch.randn(B, feat_dim)

crd = CRDLoss()

# Test 1: without labels (should work now)
loss1 = crd(s_img, s_txt, t_img, t_txt)
print('CRD loss (no labels):', loss1.item())

# Test 2: with labels (should still work for backward compatibility)
labels = torch.randint(0, 2, (B,))
loss2 = crd(s_img, s_txt, t_img, t_txt, labels)
print('CRD loss (with labels):', loss2.item())
