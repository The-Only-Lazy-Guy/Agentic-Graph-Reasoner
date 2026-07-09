"""LatentProjector - replace explicit why-text decode with a learned projection from LM hidden -> mpnet space.

Speed: ~1ms forward pass vs ~200ms autoregressive decode per chain session.
Architecture mimics HRM's task encoder: spec -> LM hidden -> MLP -> query vector.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentProjector(nn.Module):
    def __init__(self, d_lm: int = 2048, d_proj: int = 768, d_hidden: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_lm, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_proj),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(h), dim=-1)


@torch.no_grad()
def project_lm_hidden(lm, tok, text: str, layer_idx: int = -1, device=None) -> torch.Tensor:
    """Run 1 LM forward pass, extract layer L's last-token hidden, mean-pool → [d_lm]."""
    enc = tok(text[:2000], return_tensors="pt", truncation=True, max_length=512)
    if device is not None:
        enc = {k: v.to(device) for k, v in enc.items()}
    out = lm(**enc, output_hidden_states=True)
    hs = out.hidden_states[layer_idx][0]
    return hs.mean(dim=0).float().cpu()
