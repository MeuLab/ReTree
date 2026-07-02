from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import torch


@dataclass
class PrototypeNode:
    relation_id: int
    prototype_id: int
    vector: torch.Tensor
    entity_ids: List[int] = field(default_factory=list)


@dataclass
class RelationRoot:
    relation_id: int
    prototypes: List[PrototypeNode] = field(default_factory=list)


@dataclass
class HierarchicalTreeMemory:
    """Tree memory with relation roots, adaptive prototypes, and entity nodes."""

    roots: Dict[int, RelationRoot]
    entity_reps: torch.Tensor
    relation_reps: torch.Tensor

    def get_relation_root(self, relation_id: int) -> RelationRoot:
        return self.roots[relation_id]

    def to(self, device: torch.device | str) -> "HierarchicalTreeMemory":
        for root in self.roots.values():
            for proto in root.prototypes:
                proto.vector = proto.vector.to(device)
        self.entity_reps = self.entity_reps.to(device)
        self.relation_reps = self.relation_reps.to(device)
        return self

    def state_dict(self) -> Dict:
        roots = {}
        for r, root in self.roots.items():
            roots[r] = [{"prototype_id": p.prototype_id, "vector": p.vector.cpu(), "entity_ids": p.entity_ids} for p in root.prototypes]
        return {"roots": roots, "entity_reps": self.entity_reps.cpu(), "relation_reps": self.relation_reps.cpu()}

    @staticmethod
    def from_state_dict(state: Dict) -> "HierarchicalTreeMemory":
        roots = {}
        for r, protos in state["roots"].items():
            rid = int(r)
            roots[rid] = RelationRoot(rid, [PrototypeNode(rid, int(p["prototype_id"]), p["vector"], list(p["entity_ids"])) for p in protos])
        return HierarchicalTreeMemory(roots, state["entity_reps"], state["relation_reps"])
