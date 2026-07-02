from __future__ import annotations

import torch
import torch.nn as nn


class GatedMultimodalFusion(nn.Module):
    """Fuses textual and visual entity representations with a learned gate."""

    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )
        self.out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))

    def forward(self, z_text: torch.Tensor, z_vis: torch.Tensor) -> torch.Tensor:
        gate = self.gate(torch.cat([z_text, z_vis], dim=-1))
        fused = gate * z_text + (1.0 - gate) * z_vis
        return self.out(fused)
