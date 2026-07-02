from __future__ import annotations

import torch
import torch.nn.functional as F


class TemporaryPrototypeConstructor:
    """Creates query-local temporary prototypes for hard unseen entities."""

    def __init__(self, momentum: float = 0.5):
        self.momentum = momentum

    def build(self, query_anchor: torch.Tensor, unseen_rep: torch.Tensor) -> torch.Tensor:
        proto = self.momentum * query_anchor + (1.0 - self.momentum) * unseen_rep
        return F.normalize(proto, dim=-1)
