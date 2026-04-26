"""Small Hydra/OmegaConf helpers used by runtime entrypoints."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from omegaconf import OmegaConf


def apply_schema(cfg: Any, schema_cls: type[Any]):
    """Apply a structured schema and enable strict key validation."""

    schema = OmegaConf.structured(schema_cls())
    merged = OmegaConf.merge(schema, cfg)
    OmegaConf.set_struct(merged, True)
    return merged


def to_namespace(cfg: Any) -> SimpleNamespace:
    data = OmegaConf.to_container(cfg, resolve=True)
    if isinstance(data, dict):
        return SimpleNamespace(**data)
    return SimpleNamespace(value=data)
