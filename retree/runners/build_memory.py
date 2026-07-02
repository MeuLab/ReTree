from __future__ import annotations

from pathlib import Path
from collections import defaultdict
import torch

from retree.common.config import load_config
from retree.common.logging import get_logger
from retree.data.dataset import InductiveMKGCData
from retree.models.retree import ReTreeModel
from retree.memory.adaptive_prototypes import PrototypeConfig
from retree.memory.entity_assignment import TreeMemoryBuilder
from retree.memory.serialization import save_tree_memory


def run_build_memory(config_path: str, checkpoint: str) -> None:
    cfg = load_config(config_path)
    save_dir = Path(cfg.optimization.save_dir)
    logger = get_logger("ReTreeMemory", str(save_dir / "build_memory.log"))
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
    model.eval()

    all_ids = torch.arange(data.index.num_entities, dtype=torch.long, device=device)
    visual = torch.zeros(data.index.num_entities, visual_dim, device=device)
    with torch.no_grad():
        reps = model.encode_entities(all_ids, visual)["fus"].detach()
        rel_reps = model.scorer.relation.weight.detach()

    relation_to_entities = defaultdict(set)
    for tri in data.train_triples:
        r = data.index.relation_to_id[tri.relation]
        t = data.index.entity_to_id[tri.tail]
        relation_to_entities[r].add(t)
    relation_to_entities = {r: torch.tensor(sorted(v), dtype=torch.long, device=device) for r, v in relation_to_entities.items()}

    builder = TreeMemoryBuilder(PrototypeConfig(
        beta_k=cfg.memory.beta_k,
        k_max=cfg.memory.k_max,
        min_entities_per_relation=cfg.memory.min_entities_per_relation,
        temperature=cfg.memory.prototype_temperature,
        refinement_iters=cfg.memory.refinement_iters,
    ))
    memory = builder.build(relation_to_entities, reps, rel_reps)
    path = save_dir / "memory" / "tree_memory.pt"
    save_tree_memory(memory, path)
    logger.info("tree memory saved to %s", path)
