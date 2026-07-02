from __future__ import annotations

from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from retree.common.config import load_config, save_config
from retree.common.logging import get_logger
from retree.common.seed import set_seed
from retree.common.tensors import move_to_device
from retree.data.dataset import InductiveMKGCData, TripleDataset
from retree.data.collator import TripleCollator
from retree.data.negative_sampler import build_relation_tail_pool, RelationAwareNegativeSampler
from retree.models.retree import ReTreeModel
from retree.representation.losses import ReTreeRepresentationLoss


def run_train(config_path: str) -> None:
    cfg = load_config(config_path)
    set_seed(cfg.optimization.seed)
    save_dir = Path(cfg.optimization.save_dir)
    logger = get_logger("ReTreeTrain", str(save_dir / "train.log"))
    save_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, save_dir / "config.yaml")

    data = InductiveMKGCData(
        cfg.dataset.root, cfg.dataset.train_file, cfg.dataset.valid_file, cfg.dataset.test_file,
        cfg.dataset.entity_file, cfg.dataset.relation_file, cfg.dataset.text_file,
        cfg.dataset.image_file, cfg.dataset.image_index_file,
    )
    train_set = TripleDataset(data, "train")
    loader = DataLoader(train_set, batch_size=cfg.optimization.batch_size, shuffle=True, collate_fn=TripleCollator())
    relation_tail_pool = build_relation_tail_pool(data.train_triples, data.index)
    sampler = RelationAwareNegativeSampler(data.index.num_entities, relation_tail_pool, data.filter_dict, cfg.representation.negatives)

    device = torch.device(cfg.optimization.device if torch.cuda.is_available() else "cpu")
    visual_dim = data.image_features.shape[1] if data.image_features.ndim == 2 else cfg.model.visual_dim
    model = ReTreeModel(
        data.index.num_entities, data.index.num_relations, cfg.model.text_dim, visual_dim, cfg.model.dim,
        cfg.model.dropout, cfg.reliability.calibration_queue_size, cfg.reliability.min_relation_queue,
        cfg.reliability.lambda_x, cfg.reliability.alpha_min, cfg.reliability.ema_decay,
    ).to(device)
    loss_fn = ReTreeRepresentationLoss(
        cfg.representation.tau, cfg.representation.gamma, cfg.representation.lambda_stability,
        cfg.representation.lambda_neighborhood, cfg.representation.knn_size,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.optimization.lr, weight_decay=cfg.optimization.weight_decay)

    logger.info("start training on %s", cfg.dataset.name)
    for epoch in range(1, cfg.optimization.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in tqdm(loader, desc=f"epoch-{epoch}"):
            batch = move_to_device(batch, device)
            batch["negative_tail"] = sampler.sample(batch["head"], batch["relation"], batch["tail"])
            visual_head = torch.zeros(batch["head"].size(0), visual_dim, device=device)
            visual_tail = torch.zeros(batch["tail"].size(0), visual_dim, device=device)
            visual_neg = torch.zeros(batch["negative_tail"].numel(), visual_dim, device=device).reshape(batch["negative_tail"].shape + (visual_dim,))
            out = model(batch, visual_head, visual_tail, visual_neg)
            with torch.no_grad():
                model.reliability.update_calibration(batch["relation"], out["tail"])
            neg_reps = {k: v.reshape(batch["negative_tail"].shape[0], batch["negative_tail"].shape[1], -1) for k, v in out["neg"].items()}
            reliability = model.reliability(batch["relation"], out["head"]["fus"], model.scorer.relation(batch["relation"]), out["tail"], neg_reps)
            losses = loss_fn(out["pos_score"], out["neg_scores"], reliability, out["pos_score"], out["pos_score"].detach(), out["head"]["fus"], out["tail"]["fus"])
            optimizer.zero_grad()
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optimization.grad_clip)
            optimizer.step()
            model.update_ema()
            total_loss += float(losses["loss"].item())
        logger.info("epoch=%d loss=%.6f", epoch, total_loss / max(1, len(loader)))
        if epoch % cfg.optimization.eval_every == 0:
            ckpt_dir = save_dir / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "epoch": epoch}, ckpt_dir / "last.pt")
    torch.save({"model": model.state_dict(), "epoch": cfg.optimization.epochs}, save_dir / "checkpoints" / "best.pt")
    logger.info("training finished")
