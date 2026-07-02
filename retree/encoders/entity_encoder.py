from __future__ import annotations

from typing import Dict, Tuple
import torch
import torch.nn as nn

from .text_encoder import TextEncoder
from .visual_encoder import VisualEncoder
from .fusion import GatedMultimodalFusion


class MultimodalEntityEncoder(nn.Module):
    """Produces text, visual, and fused representations for entities."""

    def __init__(self, num_entities: int, text_dim: int, visual_dim: int, dim: int, dropout: float = 0.1):
        super().__init__()
        self.text_encoder = TextEncoder(num_entities, text_dim, dim, dropout)
        self.visual_encoder = VisualEncoder(visual_dim, dim, dropout)
        self.fusion = GatedMultimodalFusion(dim)

    def forward(self, entity_ids: torch.Tensor, visual_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        z_text = self.text_encoder(entity_ids)
        z_vis = self.visual_encoder(visual_features)
        z_fus = self.fusion(z_text, z_vis)
        return {"text": z_text, "vis": z_vis, "fus": z_fus}
