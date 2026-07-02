from __future__ import annotations

from typing import List, Optional
import torch
import torch.nn as nn


class TextEncoder(nn.Module):
    """Text encoder wrapper.

    The default implementation uses a trainable embedding table for reproducibility
    with pre-indexed entity descriptions. A PLM encoder can be plugged in through
    the same forward interface.
    """

    def __init__(self, num_entities: int, text_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(num_entities, text_dim)
        self.proj = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, entity_ids: torch.Tensor, texts: Optional[List[str]] = None) -> torch.Tensor:
        x = self.embedding(entity_ids)
        return self.proj(x)
