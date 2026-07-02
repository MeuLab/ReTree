from __future__ import annotations

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from .soft_targets import predictive_distribution, build_reliability_soft_targets
from .neighborhood_consistency import NeighborhoodConsistencyLoss


class ReTreeRepresentationLoss(nn.Module):
    """Representation objective for reliability-aware representation learning."""

    def __init__(self, tau: float, gamma: float, lambda_stability: float, lambda_neighborhood: float, knn_size: int):
        super().__init__()
        self.tau = tau
        self.gamma = gamma
        self.lambda_stability = lambda_stability
        self.lambda_neighborhood = lambda_neighborhood
        self.neighborhood_loss = NeighborhoodConsistencyLoss(knn_size)

    def soft_target_loss(self, pos_score: torch.Tensor, neg_scores: torch.Tensor, reliability: Dict[str, torch.Tensor]) -> torch.Tensor:
        pred = predictive_distribution(pos_score, neg_scores, self.tau)
        target = build_reliability_soft_targets(pos_score, neg_scores, reliability["text"], reliability["vis"], self.gamma)
        return -(target * torch.log(pred.clamp_min(1e-12))).sum(dim=-1).mean()

    def stability_loss(self, clean_score: torch.Tensor, perturbed_score: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(torch.sigmoid(clean_score), torch.sigmoid(perturbed_score))

    def forward(self, pos_score: torch.Tensor, neg_scores: torch.Tensor, reliability: Dict[str, torch.Tensor],
                clean_score: torch.Tensor, perturbed_score: torch.Tensor,
                query_rep: torch.Tensor, entity_rep: torch.Tensor) -> Dict[str, torch.Tensor]:
        loss_soft = self.soft_target_loss(pos_score, neg_scores, reliability)
        loss_stab = self.stability_loss(clean_score, perturbed_score)
        loss_geo = self.neighborhood_loss(query_rep, entity_rep)
        total = loss_soft + self.lambda_stability * loss_stab + self.lambda_neighborhood * loss_geo
        return {"loss": total, "soft": loss_soft, "stability": loss_stab, "neighborhood": loss_geo}
