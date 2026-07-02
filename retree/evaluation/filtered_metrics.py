from __future__ import annotations

from typing import Dict, Iterable, Tuple
import torch

from retree.common.metrics import compute_ranking_metrics


class FilteredRankEvaluator:
    """Computes filtered ranking metrics for tail prediction."""

    def __init__(self, filter_dict: Dict[Tuple[int, int], set]):
        self.filter_dict = filter_dict

    def rank_tail(self, scores: torch.Tensor, head: int, relation: int, tail: int) -> int:
        filt = self.filter_dict.get((head, relation), set())
        filtered_scores = scores.clone()
        for t in filt:
            if t != tail and 0 <= t < filtered_scores.numel():
                filtered_scores[t] = -1e9
        target = filtered_scores[tail]
        return int((filtered_scores > target).sum().item() + 1)

    def evaluate(self, ranks: Iterable[int]):
        return compute_ranking_metrics(ranks)
