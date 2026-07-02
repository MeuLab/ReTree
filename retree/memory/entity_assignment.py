from __future__ import annotations

from typing import Dict
import torch
import torch.nn.functional as F

from .tree_index import HierarchicalTreeMemory, RelationRoot, PrototypeNode
from .adaptive_prototypes import AdaptivePrototypeAllocator, PrototypeConfig
from .prototype_refinement import ReliabilityWeightedKMeans


class TreeMemoryBuilder:
    """Builds relation-rooted tree memory from trained entity representations."""

    def __init__(self, prototype_config: PrototypeConfig):
        self.allocator = AdaptivePrototypeAllocator(prototype_config)
        self.kmeans = ReliabilityWeightedKMeans(prototype_config.refinement_iters)

    @torch.no_grad()
    def build(self, relation_to_entities: Dict[int, torch.Tensor], entity_reps: torch.Tensor,
              relation_reps: torch.Tensor, pair_reliability: Dict[int, torch.Tensor] | None = None) -> HierarchicalTreeMemory:
        k_per_relation = self.allocator.allocate(relation_to_entities, entity_reps)
        roots = {}
        for r, ids in relation_to_entities.items():
            x = entity_reps[ids]
            weights = None if pair_reliability is None else pair_reliability.get(r, None)
            centers, assignment = self.kmeans.fit(x, k_per_relation[r], weights)
            root = RelationRoot(relation_id=r)
            for pid in range(k_per_relation[r]):
                local_ids = ids[assignment == pid].detach().cpu().tolist()
                root.prototypes.append(PrototypeNode(r, pid, centers[pid].detach().clone(), local_ids))
            roots[r] = root
        return HierarchicalTreeMemory(roots, entity_reps.detach().clone(), relation_reps.detach().clone())
