from __future__ import annotations

from typing import Dict, List
import torch


class TripleCollator:
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        return {
            "head": torch.tensor([b["head"] for b in batch], dtype=torch.long),
            "relation": torch.tensor([b["relation"] for b in batch], dtype=torch.long),
            "tail": torch.tensor([b["tail"] for b in batch], dtype=torch.long),
        }
