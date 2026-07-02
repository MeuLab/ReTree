from __future__ import annotations

from typing import Dict
import torch
import torch.nn as nn

from retree.encoders.entity_encoder import MultimodalEntityEncoder
from retree.models.scorer import BilinearKGScorer
from retree.reliability.assessment import ReliabilityAssessment
from retree.reliability.ema import EMAModel


class ReTreeModel(nn.Module):
    """Full ReTree model before tree memory construction."""

    def __init__(self, num_entities: int, num_relations: int, text_dim: int, visual_dim: int, dim: int,
                 dropout: float, queue_size: int, min_relation_queue: int, lambda_x: float, alpha_min: float,
                 ema_decay: float = 0.995):
        super().__init__()
        self.entity_encoder = MultimodalEntityEncoder(num_entities, text_dim, visual_dim, dim, dropout)
        self.scorer = BilinearKGScorer(num_relations, dim)
        self.reliability = ReliabilityAssessment(num_relations, dim, queue_size, min_relation_queue, lambda_x, alpha_min)
        self.ema_encoder = EMAModel(self.entity_encoder, decay=ema_decay)

    def encode_entities(self, entity_ids: torch.Tensor, visual_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.entity_encoder(entity_ids, visual_features)

    @torch.no_grad()
    def encode_entities_ema(self, entity_ids: torch.Tensor, visual_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.ema_encoder(entity_ids, visual_features)

    def update_ema(self) -> None:
        self.ema_encoder.update(self.entity_encoder)

    def forward(self, batch: Dict[str, torch.Tensor], visual_head: torch.Tensor, visual_tail: torch.Tensor, visual_neg: torch.Tensor):
        head = self.encode_entities(batch["head"], visual_head)
        tail = self.encode_entities(batch["tail"], visual_tail)
        pos_score = self.scorer.score(head["fus"], batch["relation"], tail["fus"])
        neg_shape = batch["negative_tail"].shape
        flat_neg = batch["negative_tail"].reshape(-1)
        neg_reps = self.encode_entities(flat_neg, visual_neg.reshape(flat_neg.size(0), -1))
        neg_fus = neg_reps["fus"].reshape(neg_shape[0], neg_shape[1], -1)
        neg_scores = self.scorer.score_candidates(head["fus"], batch["relation"], neg_fus)
        return {"head": head, "tail": tail, "neg": neg_reps, "pos_score": pos_score, "neg_scores": neg_scores}
