from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class Triple:
    head: str
    relation: str
    tail: str


@dataclass
class EntityRecord:
    entity_id: str
    name: str
    description: str = ""
    image_index: int | None = None


@dataclass
class KGIndex:
    entity_to_id: Dict[str, int]
    relation_to_id: Dict[str, int]
    id_to_entity: List[str]
    id_to_relation: List[str]

    @property
    def num_entities(self) -> int:
        return len(self.id_to_entity)

    @property
    def num_relations(self) -> int:
        return len(self.id_to_relation)
