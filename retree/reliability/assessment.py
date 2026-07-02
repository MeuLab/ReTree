from __future__ import annotations

from typing import Dict, Optional
import torch
import torch.nn as nn

from .calibration import RelationReferenceBank
from .candidate_separation import LocalCandidateSeparation
from .queues import CalibrationQueue


class ReliabilityAssessment(nn.Module):
    """Computes textual, visual, and pair reliability.

    Reliability is estimated from relation-level calibration and local candidate
    separation. The output is detached by default before it is used to construct
    soft targets or prototype refinement weights.
    """

    def __init__(self, num_relations: int, dim: int, queue_size: int, min_relation_queue: int, lambda_x: float, alpha_min: float):
        super().__init__()
        self.reference_bank = RelationReferenceBank(num_relations, dim)
        self.local_separation = LocalCandidateSeparation(dim)
        self.queue = CalibrationQueue(queue_size, min_relation_queue)
        self.lambda_x = lambda_x
        self.alpha_min = alpha_min

    @torch.no_grad()
    def update_calibration(self, relation: torch.Tensor, candidate_reps: Dict[str, torch.Tensor]) -> None:
        self.reference_bank.update(relation, candidate_reps)
        mismatch = self.reference_bank.mismatch(relation, candidate_reps)
        for modality in ["text", "vis", "fus"]:
            self.queue.update(relation, modality, mismatch[modality])

    def forward(self, relation: torch.Tensor, head_fus: torch.Tensor, rel_emb: torch.Tensor,
                candidate_reps: Dict[str, torch.Tensor], negative_reps: Dict[str, torch.Tensor], detach: bool = True) -> Dict[str, torch.Tensor]:
        mismatch = self.reference_bank.mismatch(relation, candidate_reps)
        calib = {}
        for modality in ["text", "vis", "fus"]:
            rank = self.queue.rank_score(relation, modality, mismatch[modality])
            calib[modality] = 1.0 - rank

        anchor = self.local_separation.make_query_anchor(head_fus, rel_emb)
        sep = self.local_separation.modality_separation(anchor, candidate_reps, negative_reps)

        alpha_text = self.lambda_x * calib["text"] + (1.0 - self.lambda_x) * sep["text"]
        alpha_vis = self.lambda_x * calib["vis"] + (1.0 - self.lambda_x) * sep["vis"]
        alpha_pair = self.lambda_x * calib["fus"] + (1.0 - self.lambda_x) * sep["fus"]

        out = {
            "text": alpha_text.clamp_min(self.alpha_min),
            "vis": alpha_vis.clamp_min(self.alpha_min),
            "pair": alpha_pair.clamp_min(self.alpha_min),
        }
        return {k: v.detach() for k, v in out.items()} if detach else out
