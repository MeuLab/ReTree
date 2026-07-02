from __future__ import annotations

from typing import Dict
import torch


class ReliabilityAnalyzer:
    """Collects clean-noisy reliability gaps for analysis figures."""

    def __init__(self):
        self.history = []

    def add_epoch(self, epoch: int, clean: Dict[str, torch.Tensor], noisy: Dict[str, torch.Tensor]) -> None:
        item = {"epoch": epoch}
        for key in ["text", "vis", "pair"]:
            item[f"delta_{key}"] = float((clean[key].mean() - noisy[key].mean()).detach().cpu().item())
        self.history.append(item)

    def to_rows(self):
        return list(self.history)
