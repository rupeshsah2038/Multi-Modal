# models/fusion/film.py
class FiLMFusion(nn.Module):
    def forward(self, img_emb, txt_emb):
        gamma = self.mlp_gamma(txt_emb).unsqueeze(1)
        beta  = self.mlp_beta(txt_emb).unsqueeze(1)
        return gamma * img_emb + beta * txt_emb