from __future__ import annotations

from typing import Tuple
import torch
import torch.nn.functional as F


class FinalReranker:
    """Reranks expanded candidates with query, relation, and reliability-aware scores."""

    def __init__(self):
        pass

    def rerank(self, query_anchor: torch.Tensor, candidate_reps: torch.Tensor, pair_reliability: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        score = torch.matmul(F.normalize(candidate_reps, dim=-1), F.normalize(query_anchor, dim=-1))
        if pair_reliability is not None:
            score = score + torch.log(pair_reliability.clamp_min(1e-6))
        order = torch.argsort(score, descending=True)
        return score[order], order
