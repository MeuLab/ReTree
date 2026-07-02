from __future__ import annotations

from typing import Dict, List


def format_retrieval_trace(trace, id_to_entity: List[str]) -> Dict:
    prototypes = []
    for proto, score in trace.selected_prototypes:
        prototypes.append({
            "relation_id": proto.relation_id,
            "prototype_id": proto.prototype_id,
            "prototype_score": score,
            "entities": [id_to_entity[i] for i in proto.entity_ids[:10]],
        })
    final_entities = [id_to_entity[int(i)] for i in trace.expanded_entities[trace.final_indices].detach().cpu().tolist()]
    return {"selected_prototypes": prototypes, "final_reranking": final_entities}
