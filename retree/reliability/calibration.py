from __future__ import annotations

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class RelationReferenceBank(nn.Module):
    """Maintains relation-level reference vectors for text, visual, and fused views."""

    def __init__(self, num_relations: int, dim: int, momentum: float = 0.95):
        super().__init__()
        self.momentum = momentum
        self.register_buffer("text_ref", torch.zeros(num_relations, dim))
        self.register_buffer("vis_ref", torch.zeros(num_relations, dim))
        self.register_buffer("fus_ref", torch.zeros(num_relations, dim))
        self.register_buffer("count", torch.zeros(num_relations))

    @torch.no_grad()
    def update(self, relation: torch.Tensor, reps: Dict[str, torch.Tensor]) -> None:
        for key, buf_name in [("text", "text_ref"), ("vis", "vis_ref"), ("fus", "fus_ref")]:
            buf = getattr(self, buf_name)
            for r in relation.unique():
                mask = relation == r
                mean = reps[key][mask].detach().mean(dim=0)
                rid = int(r.item())
                if self.count[rid] == 0:
                    buf[rid].copy_(mean)
                else:
                    buf[rid].mul_(self.momentum).add_(mean, alpha=1.0 - self.momentum)
                self.count[rid] += mask.sum().item()

    def mismatch(self, relation: torch.Tensor, reps: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = {}
        for key, buf_name in [("text", "text_ref"), ("vis", "vis_ref"), ("fus", "fus_ref")]:
            ref = getattr(self, buf_name)[relation]
            out[key] = 1.0 - F.cosine_similarity(reps[key], ref, dim=-1)
        return out
