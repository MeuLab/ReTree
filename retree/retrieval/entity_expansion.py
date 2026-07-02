from __future__ import annotations

from typing import Iterable, List, Tuple
import torch
import torch.nn.functional as F

from retree.memory.tree_index import HierarchicalTreeMemory, PrototypeNode


class EntityExpansion:
    """Expands entity nodes from selected prototypes and keeps top candidates."""

    def __init__(self, b2: int = 32):
        self.b2 = b2

    def expand(self, memory: HierarchicalTreeMemory, selected_prototypes: Iterable[Tuple[PrototypeNode, float]], query_anchor: torch.Tensor) -> torch.Tensor:
        ids: List[int] = []
        for proto, _ in selected_prototypes:
            ids.extend(proto.entity_ids)
        if not ids:
            return torch.empty(0, dtype=torch.long, device=query_anchor.device)
        unique = torch.tensor(sorted(set(ids)), dtype=torch.long, device=query_anchor.device)
        reps = memory.entity_reps[unique].to(query_anchor.device)
        score = torch.matmul(F.normalize(reps, dim=-1), F.normalize(query_anchor, dim=-1))
        keep = score.topk(min(self.b2, score.numel())).indices
        return unique[keep]
