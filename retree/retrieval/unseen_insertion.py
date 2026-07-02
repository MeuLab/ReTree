from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn.functional as F


@dataclass
class InsertionResult:
    prototype_scores: torch.Tensor
    selected: torch.Tensor
    use_temporary: torch.Tensor


class QueryLocalInsertion:
    """Computes how unseen candidate entities attach to stored prototypes for the current query."""

    def __init__(self, temperature: float = 0.10, temporary_threshold: float = 0.40):
        self.temperature = temperature
        self.temporary_threshold = temporary_threshold

    def __call__(self, unseen_reps: torch.Tensor, prototype_reps: torch.Tensor) -> InsertionResult:
        score = torch.matmul(F.normalize(unseen_reps, dim=-1), F.normalize(prototype_reps, dim=-1).t())
        prob = torch.softmax(score / self.temperature, dim=-1)
        best_score, selected = prob.max(dim=-1)
        use_temp = best_score < self.temporary_threshold
        return InsertionResult(prob, selected, use_temp)
