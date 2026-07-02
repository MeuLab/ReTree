from __future__ import annotations

from typing import List, Tuple
import torch
import torch.nn.functional as F

from retree.memory.tree_index import HierarchicalTreeMemory, PrototypeNode


class PrototypeRetriever:
    """Selects top prototypes from the relation branch."""

    def __init__(self, b1: int = 4):
        self.b1 = b1

    def retrieve(self, memory: HierarchicalTreeMemory, relation_id: int, query_anchor: torch.Tensor) -> List[Tuple[PrototypeNode, float]]:
        root = memory.get_relation_root(int(relation_id))
        if not root.prototypes:
            return []
        vectors = torch.stack([p.vector.to(query_anchor.device) for p in root.prototypes], dim=0)
        score = torch.matmul(F.normalize(vectors, dim=-1), F.normalize(query_anchor, dim=-1))
        topk = score.topk(min(self.b1, score.numel())).indices.detach().cpu().tolist()
        return [(root.prototypes[i], float(score[i].item())) for i in topk]
