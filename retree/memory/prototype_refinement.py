from __future__ import annotations

from typing import Optional, Tuple
import torch
import torch.nn.functional as F


class ReliabilityWeightedKMeans:
    """Weighted K-means used to initialize and refine adaptive prototypes."""

    def __init__(self, num_iters: int = 5, eps: float = 1e-8):
        self.num_iters = num_iters
        self.eps = eps

    @torch.no_grad()
    def initialize(self, x: torch.Tensor, k: int) -> torch.Tensor:
        if x.size(0) <= k:
            pad = x[:1].repeat(k - x.size(0), 1) if x.size(0) < k else x.new_empty(0, x.size(1))
            return torch.cat([x, pad], dim=0)
        indices = torch.linspace(0, x.size(0) - 1, k, device=x.device).long()
        return x[indices].clone()

    @torch.no_grad()
    def fit(self, x: torch.Tensor, k: int, weights: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if weights is None:
            weights = torch.ones(x.size(0), device=x.device)
        centers = self.initialize(x, k)
        for _ in range(self.num_iters):
            sim = torch.matmul(F.normalize(x, dim=-1), F.normalize(centers, dim=-1).t())
            assignment = sim.argmax(dim=-1)
            new_centers = []
            for c in range(k):
                mask = assignment == c
                if not mask.any():
                    new_centers.append(centers[c])
                    continue
                w = weights[mask].unsqueeze(-1).clamp_min(self.eps)
                new_centers.append((x[mask] * w).sum(dim=0) / w.sum())
            centers = torch.stack(new_centers, dim=0)
        sim = torch.matmul(F.normalize(x, dim=-1), F.normalize(centers, dim=-1).t())
        assignment = sim.argmax(dim=-1)
        return centers, assignment
