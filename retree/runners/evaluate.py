from __future__ import annotations

from pathlib import Path
import torch

from retree.common.config import load_config
from retree.common.logging import get_logger
from retree.data.dataset import InductiveMKGCData
from retree.models.retree import ReTreeModel
from retree.memory.serialization import load_tree_memory
from retree.evaluation.ranking import evaluate_hierarchical


def run_evaluate(config_path: str, checkpoint: str, memory_path: str) -> None:
    cfg = load_config(config_path)
    logger = get_logger("ReTreeEval", str(Path(cfg.optimization.save_dir) / "eval.log"))
    data = InductiveMKGCData(
        cfg.dataset.root, cfg.dataset.train_file, cfg.dataset.valid_file, cfg.dataset.test_file,
        cfg.dataset.entity_file, cfg.dataset.relation_file, cfg.dataset.text_file,
        cfg.dataset.image_file, cfg.dataset.image_index_file,
    )
    device = torch.device(cfg.optimization.device if torch.cuda.is_available() else "cpu")
    visual_dim = data.image_features.shape[1] if data.image_features.ndim == 2 else cfg.model.visual_dim
    model = ReTreeModel(
        data.index.num_entities, data.index.num_relations, cfg.model.text_dim, visual_dim, cfg.model.dim,
        cfg.model.dropout, cfg.reliability.calibration_queue_size, cfg.reliability.min_relation_queue,
        cfg.reliability.lambda_x, cfg.reliability.alpha_min, cfg.reliability.ema_decay,
    ).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model"], strict=False)
    memory = load_tree_memory(memory_path, map_location=device).to(device)
    metrics = evaluate_hierarchical(model, memory, data.test_triples, data.index, data.filter_dict, cfg.retrieval.b1, cfg.retrieval.b2, str(device))
    logger.info("test metrics: %s", metrics)
