from __future__ import annotations

from typing import Dict, Iterable, Tuple
import random
import torch


class RelationAwareNegativeSampler:
    """Samples negatives from relation-tail pools with filtered true-tail removal."""

    def __init__(self, num_entities: int, relation_tail_pool: Dict[int, Iterable[int]], filter_dict: Dict[Tuple[int, int], set], num_negatives: int):
        self.num_entities = num_entities
        self.relation_tail_pool = {r: list(v) for r, v in relation_tail_pool.items()}
        self.filter_dict = filter_dict
        self.num_negatives = num_negatives

    def sample(self, heads: torch.Tensor, relations: torch.Tensor, tails: torch.Tensor) -> torch.Tensor:
        batch_neg = []
        for h, r, t in zip(heads.tolist(), relations.tolist(), tails.tolist()):
            true_tails = self.filter_dict.get((h, r), set())
            pool = self.relation_tail_pool.get(r)
            if not pool:
                pool = range(self.num_entities)
            candidates = []
            attempts = 0
            while len(candidates) < self.num_negatives and attempts < self.num_negatives * 20:
                neg = random.choice(list(pool))
                if neg not in true_tails and neg != t:
                    candidates.append(neg)
                attempts += 1
            while len(candidates) < self.num_negatives:
                neg = random.randrange(self.num_entities)
                if neg not in true_tails and neg != t:
                    candidates.append(neg)
            batch_neg.append(candidates)
        return torch.tensor(batch_neg, dtype=torch.long, device=heads.device)


def build_relation_tail_pool(triples, index) -> Dict[int, set]:
    pool: Dict[int, set] = {}
    for tri in triples:
        if tri.relation in index.relation_to_id and tri.tail in index.entity_to_id:
            r = index.relation_to_id[tri.relation]
            t = index.entity_to_id[tri.tail]
            pool.setdefault(r, set()).add(t)
    return pool
