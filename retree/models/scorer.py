from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BilinearKGScorer(nn.Module):
    """Bilinear query-candidate scoring for KG completion."""

    def __init__(self, num_relations: int, dim: int):
        super().__init__()
        self.relation = nn.Embedding(num_relations, dim)
        self.query_proj = nn.Sequential(nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, dim))

    def query_anchor(self, head_fus: torch.Tensor, relation_ids: torch.Tensor) -> torch.Tensor:
        rel = self.relation(relation_ids)
        return F.normalize(self.query_proj(torch.cat([head_fus, rel], dim=-1)), dim=-1)

    def score(self, head_fus: torch.Tensor, relation_ids: torch.Tensor, tail_fus: torch.Tensor) -> torch.Tensor:
        q = self.query_anchor(head_fus, relation_ids)
        return (q * F.normalize(tail_fus, dim=-1)).sum(dim=-1)

    def score_candidates(self, head_fus: torch.Tensor, relation_ids: torch.Tensor, candidate_fus: torch.Tensor) -> torch.Tensor:
        q = self.query_anchor(head_fus, relation_ids)
        return torch.einsum("bd,bnd->bn", q, F.normalize(candidate_fus, dim=-1))
