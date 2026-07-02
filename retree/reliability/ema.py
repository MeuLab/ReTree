from __future__ import annotations

import copy
import torch
import torch.nn as nn


class EMAModel(nn.Module):
    """Maintains a slow branch for stop-gradient reliability computation."""

    def __init__(self, model: nn.Module, decay: float = 0.995):
        super().__init__()
        self.module = copy.deepcopy(model)
        self.decay = decay
        self.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for p_ema, p in zip(self.module.parameters(), model.parameters()):
            p_ema.data.mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)
        for b_ema, b in zip(self.module.buffers(), model.buffers()):
            b_ema.data.copy_(b.data)

    @torch.no_grad()
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
