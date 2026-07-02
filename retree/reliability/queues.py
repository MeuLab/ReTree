from __future__ import annotations

from collections import defaultdict, deque
from typing import DefaultDict, Deque, Dict, Iterable, Tuple
import torch


class CalibrationQueue:
    """Stores relation-modality mismatch scores for conformal-style calibration."""

    def __init__(self, max_size: int = 4096, min_relation_size: int = 32):
        self.max_size = max_size
        self.min_relation_size = min_relation_size
        self._queues: DefaultDict[Tuple[int, str], Deque[float]] = defaultdict(lambda: deque(maxlen=max_size))
        self._global: DefaultDict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=max_size))

    def update(self, relation: torch.Tensor, modality: str, scores: torch.Tensor) -> None:
        rels = relation.detach().cpu().tolist()
        vals = scores.detach().cpu().float().tolist()
        for r, v in zip(rels, vals):
            self._queues[(int(r), modality)].append(float(v))
            self._global[modality].append(float(v))

    def rank_score(self, relation: torch.Tensor, modality: str, scores: torch.Tensor) -> torch.Tensor:
        ranks = []
        for r, s in zip(relation.detach().cpu().tolist(), scores.detach().cpu().float().tolist()):
            q = self._queues.get((int(r), modality), None)
            if q is None or len(q) < self.min_relation_size:
                q = self._global.get(modality, [])
            values = list(q)
            if not values:
                ranks.append(0.5)
            else:
                leq = sum(v <= float(s) for v in values)
                ranks.append(leq / max(1, len(values)))
        return torch.tensor(ranks, dtype=scores.dtype, device=scores.device)
