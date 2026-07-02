from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class NeighborhoodConsistencyLoss(nn.Module):
    """Aligns local neighborhoods between query anchors and entity representations."""

    def __init__(self, knn_size: int = 16, temperature: float = 0.10):
        super().__init__()
        self.knn_size = knn_size
        self.temperature = temperature

    @torch.no_grad()
    def nearest_neighbors(self, x: torch.Tensor) -> torch.Tensor:
        sim = torch.matmul(F.normalize(x, dim=-1), F.normalize(x, dim=-1).t())
        sim.fill_diagonal_(-1e9)
        return sim.topk(min(self.knn_size, x.size(0) - 1), dim=-1).indices

    def forward(self, query_rep: torch.Tensor, entity_rep: torch.Tensor) -> torch.Tensor:
        if query_rep.size(0) <= 1:
            return query_rep.new_tensor(0.0)
        q_nn = self.nearest_neighbors(query_rep.detach())
        e_nn = self.nearest_neighbors(entity_rep.detach())
        q_sim = torch.gather(torch.matmul(query_rep, query_rep.t()) / self.temperature, 1, q_nn)
        e_sim = torch.gather(torch.matmul(entity_rep, entity_rep.t()) / self.temperature, 1, e_nn)
        q_dist = F.log_softmax(q_sim, dim=-1)
        e_dist = F.softmax(e_sim.detach(), dim=-1)
        return F.kl_div(q_dist, e_dist, reduction="batchmean")
