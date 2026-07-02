from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
import yaml


class ConfigNode(dict):
    """Dictionary with attribute access used by training and evaluation runners."""

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc
        if isinstance(value, dict) and not isinstance(value, ConfigNode):
            value = ConfigNode(value)
            self[key] = value
        return value

    def copy(self) -> "ConfigNode":
        return ConfigNode(super().copy())


def _deep_update(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _set_dotted(config: Dict[str, Any], dotted_key: str, value: Any) -> None:
    node = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    if isinstance(value, str):
        if value.lower() == "true":
            value = True
        elif value.lower() == "false":
            value = False
        else:
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass
    node[parts[-1]] = value


def load_config(path: str | Path) -> ConfigNode:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    if "base" in raw:
        base_path = (path.parent / raw.pop("base")).resolve()
        base = load_config(base_path)
        merged = dict(base)
    else:
        merged = {}

    normal_items = {k: v for k, v in raw.items() if "." not in k}
    dotted_items = {k: v for k, v in raw.items() if "." in k}
    _deep_update(merged, normal_items)
    for key, value in dotted_items.items():
        _set_dotted(merged, key, value)
    return ConfigNode(merged)


def save_config(config: ConfigNode, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(dict(config), f, sort_keys=False)
