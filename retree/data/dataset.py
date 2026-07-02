from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import json

import numpy as np
import torch
from torch.utils.data import Dataset

from .schema import KGIndex, Triple


def read_id_file(path: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            key = parts[0]
            name = parts[1] if len(parts) > 1 else parts[0]
            mapping[key] = name
    return mapping


def read_triples(path: Path) -> List[Triple]:
    triples: List[Triple] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            h, r, t = line.split("\t")[:3]
            triples.append(Triple(h, r, t))
    return triples


class InductiveMKGCData:
    """Dataset container for inductive multimodal KG completion."""

    def __init__(self, root: str | Path, train_file: str, valid_file: str, test_file: str,
                 entity_file: str, relation_file: str, text_file: str, image_file: str,
                 image_index_file: str):
        self.root = Path(root)
        entities = read_id_file(self.root / entity_file)
        relations = read_id_file(self.root / relation_file)
        self.index = KGIndex(
            entity_to_id={e: i for i, e in enumerate(entities.keys())},
            relation_to_id={r: i for i, r in enumerate(relations.keys())},
            id_to_entity=list(entities.keys()),
            id_to_relation=list(relations.keys()),
        )
        self.entity_names = entities
        self.relation_names = relations
        self.train_triples = read_triples(self.root / train_file)
        self.valid_triples = read_triples(self.root / valid_file)
        self.test_triples = read_triples(self.root / test_file)
        self.entity_text = self._load_json(self.root / text_file)
        self.image_index = self._load_json(self.root / image_index_file)
        self.image_features = self._load_image_features(self.root / image_file)
        self.filter_dict = self._build_filter_dict(self.train_triples + self.valid_triples + self.test_triples)

    @staticmethod
    def _load_json(path: Path) -> Dict[str, str]:
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _load_image_features(path: Path) -> np.ndarray:
        if path.exists():
            return np.load(path)
        return np.zeros((1, 1), dtype=np.float32)

    def _build_filter_dict(self, triples: Iterable[Triple]) -> Dict[Tuple[int, int], set]:
        filt: Dict[Tuple[int, int], set] = {}
        for tri in triples:
            if tri.head not in self.index.entity_to_id or tri.tail not in self.index.entity_to_id:
                continue
            h = self.index.entity_to_id[tri.head]
            r = self.index.relation_to_id[tri.relation]
            t = self.index.entity_to_id[tri.tail]
            filt.setdefault((h, r), set()).add(t)
        return filt


class TripleDataset(Dataset):
    def __init__(self, data: InductiveMKGCData, split: str):
        self.data = data
        self.triples = getattr(data, f"{split}_triples")

    def __len__(self) -> int:
        return len(self.triples)

    def __getitem__(self, idx: int):
        tri = self.triples[idx]
        index = self.data.index
        return {
            "head": index.entity_to_id[tri.head],
            "relation": index.relation_to_id[tri.relation],
            "tail": index.entity_to_id[tri.tail],
            "head_key": tri.head,
            "tail_key": tri.tail,
        }
