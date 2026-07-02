from __future__ import annotations

from typing import Dict, List
import torch
from tqdm import tqdm

from retree.common.metrics import compute_ranking_metrics
from retree.retrieval.hierarchical_retriever import HierarchicalRetriever


def evaluate_hierarchical(model, memory, triples, dataset_index, filter_dict, b1: int, b2: int, device: str = "cuda") -> Dict[str, float]:
    retriever = HierarchicalRetriever(b1, b2)
    ranks: List[int] = []
    model.eval()
    with torch.no_grad():
        for tri in tqdm(triples, desc="hierarchical-eval"):
            h = dataset_index.entity_to_id[tri.head]
            r = dataset_index.relation_to_id[tri.relation]
            t = dataset_index.entity_to_id[tri.tail]
            head_rep = memory.entity_reps[h].to(device).unsqueeze(0)
            rel_id = torch.tensor([r], dtype=torch.long, device=device)
            anchor = model.scorer.query_anchor(head_rep, rel_id).squeeze(0)
            trace = retriever.retrieve(memory, r, anchor)
            candidate_ids = trace.expanded_entities[trace.final_indices]
            if candidate_ids.numel() == 0 or not (candidate_ids == t).any():
                rank = b2 + 1
            else:
                rank = int((candidate_ids == t).nonzero(as_tuple=True)[0][0].item() + 1)
            ranks.append(rank)
    return compute_ranking_metrics(ranks).to_dict()
