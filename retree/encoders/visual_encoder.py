from __future__ import annotations

import torch
import torch.nn as nn


class VisualEncoder(nn.Module):
    """Projection network for pre-computed visual features."""

    def __init__(self, visual_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        return self.net(visual_features)
