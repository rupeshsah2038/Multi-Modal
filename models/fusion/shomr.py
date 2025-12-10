# models/fusion/shomr.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SHoMRFusion(nn.Module):
    """
    Soft-Hard Modality Routing Fusion (SHoMR-Fusion)
    
    Combines soft confidence-based weighting with hard routing decisions for adaptive
    multimodal fusion. Features:
      • Soft path: Confidence-weighted fusion with cross-attention
      • Hard path: Discrete routing (vision-only, text-only, or both)
      • Dynamic threshold-based switching between soft and hard paths
      • Fallback mechanism for low-confidence samples
    
    → Balances computational efficiency with representation quality
    """
    def __init__(self, dim, heads=8, dropout=0.1, confidence_threshold=0.6, 
                 routing_temperature=1.0):
        super().__init__()
        self.dim = dim
        self.confidence_threshold = confidence_threshold
        self.routing_temperature = routing_temperature
        
        # Soft confidence network: estimates per-modality confidence scores
        self.conf_net = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 2),  # [vision_confidence, text_confidence]
        )
        
        # Hard routing network: discrete modality selection
        self.router = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 3),  # [vision_only, text_only, both]
        )
        
        # Cross-attention for rich multimodal interaction
        self.cross_attn = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        
        # Gated fusion for combining modalities
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
        
        # Post-fusion processing
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        
        # Lightweight projection for single-modality paths
        self.proj_single = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
    def forward(self, img_emb, txt_emb, use_hard_routing=None):
        """
        Args:
            img_emb: Vision embeddings (B, D)
            txt_emb: Text embeddings (B, D)
            use_hard_routing: If None, auto-decide based on confidence threshold
                            If True/False, force hard/soft routing
        
        Returns:
            fused: Fused representation (B, D)
            routing_info: Dict with routing statistics (for logging/analysis)
        """
        B, D = img_emb.shape
        
        # === Step 1: Compute soft confidence scores ===
        conf_input = torch.cat([img_emb, txt_emb], dim=-1)
        conf_logits = self.conf_net(conf_input)  # (B, 2)
        conf_weights = F.softmax(conf_logits, dim=-1)  # (B, 2)
        
        w_v = conf_weights[:, 0].unsqueeze(-1)  # (B, 1)
        w_t = conf_weights[:, 1].unsqueeze(-1)  # (B, 1)
        
        # Max confidence per sample
        max_conf = conf_weights.max(dim=-1)[0]  # (B,)
        
        # === Step 2: Decide routing strategy ===
        if use_hard_routing is None:
            # Auto-decide: use hard routing when confidence is high (clear winner)
            use_hard = max_conf > self.confidence_threshold
        else:
            use_hard = torch.tensor([use_hard_routing] * B, 
                                   device=img_emb.device, dtype=torch.bool)
        
        # === Step 3: Compute hard routing decisions ===
        route_logits = self.router(conf_input)  # (B, 3)
        route_probs = F.softmax(route_logits / self.routing_temperature, dim=-1)
        
        # During training: sample from distribution; inference: argmax
        if self.training:
            route_choice = torch.multinomial(route_probs, 1).squeeze(1)  # (B,)
        else:
            route_choice = route_probs.argmax(dim=-1)  # (B,)
        
        # === Step 4: Apply routing ===
        fused = torch.zeros_like(img_emb)
        routing_counts = {'soft': 0, 'hard_vision': 0, 'hard_text': 0, 'hard_both': 0}
        
        # Soft path: confidence-weighted fusion with cross-attention
        mask_soft = ~use_hard
        if mask_soft.any():
            idx_soft = mask_soft.nonzero(as_tuple=True)[0]
            v_soft = img_emb[idx_soft]
            t_soft = txt_emb[idx_soft]
            w_v_soft = w_v[idx_soft]
            w_t_soft = w_t[idx_soft]
            
            # Soft weighted fusion
            base_fused = w_v_soft * v_soft + w_t_soft * t_soft
            
            # Cross-attention enhancement
            attn_in = torch.stack([v_soft, t_soft], dim=1)  # (N, 2, D)
            attn_out = self.cross_attn(attn_in, attn_in, attn_in)[0]  # (N, 2, D)
            attn_mean = attn_out.mean(dim=1)  # (N, D)
            
            soft_out = base_fused + attn_mean
            soft_out = self.norm1(soft_out)
            soft_out = soft_out + self.ffn(soft_out)
            
            fused[idx_soft] = soft_out
            routing_counts['soft'] = idx_soft.size(0)
        
        # Hard path: discrete routing
        mask_hard = use_hard
        if mask_hard.any():
            idx_hard = mask_hard.nonzero(as_tuple=True)[0]
            route_hard = route_choice[idx_hard]
            
            # Vision-only path
            mask_v = route_hard == 0
            if mask_v.any():
                idx_v = idx_hard[mask_v]
                fused[idx_v] = self.proj_single(img_emb[idx_v])
                routing_counts['hard_vision'] = idx_v.size(0)
            
            # Text-only path
            mask_t = route_hard == 1
            if mask_t.any():
                idx_t = idx_hard[mask_t]
                fused[idx_t] = self.proj_single(txt_emb[idx_t])
                routing_counts['hard_text'] = idx_t.size(0)
            
            # Both modalities path (full fusion)
            mask_both = route_hard == 2
            if mask_both.any():
                idx_both = idx_hard[mask_both]
                v_both = img_emb[idx_both]
                t_both = txt_emb[idx_both]
                
                # Gated fusion
                gate_val = self.gate(torch.cat([v_both, t_both], dim=-1))
                gated_fused = gate_val * v_both + (1 - gate_val) * t_both
                
                # Cross-attention
                attn_in = torch.stack([v_both, t_both], dim=1)  # (N, 2, D)
                attn_out = self.cross_attn(attn_in, attn_in, attn_in)[0]
                attn_mean = attn_out.mean(dim=1)
                
                both_out = gated_fused + attn_mean
                both_out = self.norm2(both_out)
                both_out = both_out + self.ffn(both_out)
                
                fused[idx_both] = both_out
                routing_counts['hard_both'] = idx_both.size(0)
        
        # === Step 5: Fallback for very low confidence ===
        # When both confidences are very low, use simple average
        low_conf_mask = (max_conf < 0.3) & mask_soft
        if low_conf_mask.any():
            idx_low = low_conf_mask.nonzero(as_tuple=True)[0]
            fused[idx_low] = (img_emb[idx_low] + txt_emb[idx_low]) / 2
        
        routing_info = {
            'routing_counts': routing_counts,
            'mean_confidence': max_conf.mean().item(),
            'soft_ratio': mask_soft.float().mean().item(),
            'conf_weights': conf_weights.detach(),
            'route_probs': route_probs.detach(),
        }
        
        return fused, routing_info
    
    def get_routing_stats(self):
        """Returns statistics about routing decisions (for analysis)."""
        return {
            'confidence_threshold': self.confidence_threshold,
            'routing_temperature': self.routing_temperature,
        }
