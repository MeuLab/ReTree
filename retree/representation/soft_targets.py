from __future__ import annotations

import torch
import torch.nn.functional as F


def build_reliability_soft_targets(pos_score: torch.Tensor, neg_scores: torch.Tensor,
                                   alpha_text: torch.Tensor, alpha_vis: torch.Tensor,
                                   gamma: float = 0.5) -> torch.Tensor:
    """Builds reliability-aware soft targets over one positive and multiple negatives."""
    batch_size, num_neg = neg_scores.shape
    hard = torch.zeros(batch_size, num_neg + 1, device=pos_score.device)
    hard[:, 0] = 1.0
    modality_weight = gamma * alpha_text + (1.0 - gamma) * alpha_vis
    positive_mass = modality_weight.clamp(0.05, 0.95)
    soft = torch.full_like(hard, fill_value=0.0)
    soft[:, 0] = positive_mass
    soft[:, 1:] = (1.0 - positive_mass).unsqueeze(-1) / max(1, num_neg)
    return soft


def predictive_distribution(pos_score: torch.Tensor, neg_scores: torch.Tensor, tau: float) -> torch.Tensor:
    logits = torch.cat([pos_score.unsqueeze(-1), neg_scores], dim=-1)
    return F.softmax(logits / tau, dim=-1)
