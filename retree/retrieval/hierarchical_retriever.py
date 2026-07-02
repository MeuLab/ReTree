from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import torch

from .prototype_retriever import PrototypeRetriever
from .entity_expansion import EntityExpansion
from .reranker import FinalReranker
from retree.memory.tree_index import HierarchicalTreeMemory


@dataclass
class RetrievalTrace:
    relation_id: int
    selected_prototypes: list
    expanded_entities: torch.Tensor
    final_scores: torch.Tensor
    final_indices: torch.Tensor


class HierarchicalRetriever:
    """End-to-end hierarchical retrieval for inductive reasoning."""

    def __init__(self, b1: int = 4, b2: int = 32):
        self.prototype_retriever = PrototypeRetriever(b1)
        self.entity_expansion = EntityExpansion(b2)
        self.reranker = FinalReranker()

    def retrieve(self, memory: HierarchicalTreeMemory, relation_id: int, query_anchor: torch.Tensor,
                 pair_reliability: torch.Tensor | None = None) -> RetrievalTrace:
        selected = self.prototype_retriever.retrieve(memory, relation_id, query_anchor)
        expanded = self.entity_expansion.expand(memory, selected, query_anchor)
        candidate_reps = memory.entity_reps[expanded].to(query_anchor.device) if expanded.numel() > 0 else memory.entity_reps.new_empty(0, memory.entity_reps.size(-1))
        scores, order = self.reranker.rerank(query_anchor, candidate_reps, pair_reliability)
        return RetrievalTrace(relation_id, selected, expanded, scores, order)
