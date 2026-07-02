from __future__ import annotations

from pathlib import Path
import torch

from .tree_index import HierarchicalTreeMemory


def save_tree_memory(memory: HierarchicalTreeMemory, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(memory.state_dict(), path)


def load_tree_memory(path: str | Path, map_location=None) -> HierarchicalTreeMemory:
    state = torch.load(path, map_location=map_location)
    return HierarchicalTreeMemory.from_state_dict(state)
