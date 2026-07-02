from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
import math
import torch
import torch.nn.functional as F


@dataclass
class PrototypeConfig:
    beta_k: float = 1.0
    k_max: int = 16
    min_entities_per_relation: int = 4
    temperature: float = 0.10
    refinement_iters: int = 5


class AdaptivePrototypeAllocator:
    """Determines prototype counts from relation frequency and intra-relation dispersion."""

    def __init__(self, config: PrototypeConfig):
        self.config = config

    def dispersion(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(0) <= 1:
            return x.new_tensor(0.0)
        center = x.mean(dim=0, keepdim=True)
        return (1.0 - F.cosine_similarity(F.normalize(x, dim=-1), F.normalize(center, dim=-1), dim=-1)).mean()

    def allocate(self, relation_to_entities: Dict[int, torch.Tensor], entity_reps: torch.Tensor) -> Dict[int, int]:
        dispersions = {}
        for r, ids in relation_to_entities.items():
            dispersions[r] = self.dispersion(entity_reps[ids]).item()
        global_disp = sum(dispersions.values()) / max(1, len(dispersions))
        counts = {}
        for r, ids in relation_to_entities.items():
            n_r = max(1, int(ids.numel()))
            d_ratio = dispersions[r] / max(global_disp, 1e-6)
            raw = self.config.beta_k * math.sqrt(n_r) * (1.0 + d_ratio)
            k = max(1, min(self.config.k_max, int(round(raw))))
            if n_r < self.config.min_entities_per_relation:
                k = 1
            counts[r] = k
        return counts
