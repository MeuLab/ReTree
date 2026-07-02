from __future__ import annotations

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalCandidateSeparation(nn.Module):
    """Measures whether a candidate separates from local hard negatives for a relation."""

    def __init__(self, dim: int, temperature: float = 0.10, eps: float = 1e-8):
        super().__init__()
        self.temperature = temperature
        self.eps = eps
        self.query_proj = nn.Linear(dim * 2, dim)

    def make_query_anchor(self, head_fus: torch.Tensor, rel_emb: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.query_proj(torch.cat([head_fus, rel_emb], dim=-1)), dim=-1)

    def separation(self, anchor: torch.Tensor, candidate: torch.Tensor, negatives: torch.Tensor) -> torch.Tensor:
        pos = F.cosine_similarity(anchor, candidate, dim=-1)
        neg = torch.einsum("bd,bnd->bn", anchor, F.normalize(negatives, dim=-1))
        margin = pos - neg.max(dim=-1).values
        return torch.sigmoid(margin / self.temperature)

    def modality_separation(self, anchor: torch.Tensor, candidate_reps: Dict[str, torch.Tensor], negative_reps: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {m: self.separation(anchor, candidate_reps[m], negative_reps[m]) for m in ["text", "vis", "fus"]}
