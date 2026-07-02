from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List
import json
import numpy as np
import torch


class EntityFeatureStore:
    """Stores raw text and visual features for entity encoders."""

    def __init__(self, entity_ids: List[str], text_map: Dict[str, str], image_features: np.ndarray, image_index: Dict[str, int]):
        self.entity_ids = entity_ids
        self.text_map = text_map
        self.image_features = image_features.astype("float32")
        self.image_index = {str(k): int(v) for k, v in image_index.items()}

    def get_texts(self, ids: torch.Tensor) -> List[str]:
        texts = []
        for idx in ids.detach().cpu().tolist():
            key = self.entity_ids[idx]
            texts.append(self.text_map.get(key, key))
        return texts

    def get_visual_tensor(self, ids: torch.Tensor, device=None) -> torch.Tensor:
        rows = []
        for idx in ids.detach().cpu().tolist():
            key = self.entity_ids[idx]
            row = self.image_index.get(key, None)
            if row is None or self.image_features.size == 1:
                feat = np.zeros((self.visual_dim,), dtype="float32")
            else:
                feat = self.image_features[row]
            rows.append(feat)
        tensor = torch.tensor(np.stack(rows), dtype=torch.float32)
        return tensor if device is None else tensor.to(device)

    @property
    def visual_dim(self) -> int:
        if self.image_features.ndim == 2:
            return int(self.image_features.shape[1])
        return 1
