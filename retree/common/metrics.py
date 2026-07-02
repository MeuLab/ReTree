from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List


@dataclass
class RankingMetrics:
    mrr: float
    hits1: float
    hits3: float
    hits10: float

    def to_dict(self) -> Dict[str, float]:
        return {"MRR": self.mrr, "Hits@1": self.hits1, "Hits@3": self.hits3, "Hits@10": self.hits10}


def compute_ranking_metrics(ranks: Iterable[int]) -> RankingMetrics:
    ranks = list(ranks)
    if not ranks:
        return RankingMetrics(0.0, 0.0, 0.0, 0.0)
    n = len(ranks)
    mrr = sum(1.0 / r for r in ranks) / n
    hits1 = sum(r <= 1 for r in ranks) / n
    hits3 = sum(r <= 3 for r in ranks) / n
    hits10 = sum(r <= 10 for r in ranks) / n
    return RankingMetrics(mrr, hits1, hits3, hits10)
