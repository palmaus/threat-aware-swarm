from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from scripts.common.metrics_utils import METRIC_KEYS


@dataclass(frozen=True)
class MetricsSchema:
    version: str
    keys: list[str]


@dataclass
class MetricManifest:
    created_at: str
    schema_version: str
    schema_path: str | None
    schema_keys: list[str]
    actual_keys: list[str]
    missing_keys: list[str]
    extra_keys: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "created_at": self.created_at,
            "schema_version": self.schema_version,
            "schema_path": self.schema_path,
            "schema_keys": self.schema_keys,
            "actual_keys": self.actual_keys,
            "missing_keys": self.missing_keys,
            "extra_keys": self.extra_keys,
        }


def load_metrics_schema(path: Path | None) -> MetricsSchema:
    """Читает схему метрик из YAML, либо возвращает встроенную схему."""
    if path is None or not path.exists():
        return MetricsSchema(version="builtin@v1", keys=list(METRIC_KEYS))
    cfg = OmegaConf.load(path)
    data = OmegaConf.to_container(cfg, resolve=True)
    version = "unknown"
    keys: list[str] | None = None
    if isinstance(data, dict):
        version = str(data.get("version", "unknown"))
        raw = data.get("metrics") or data.get("keys")
        if isinstance(raw, list):
            keys = [str(k) for k in raw]
    elif isinstance(data, list):
        keys = [str(k) for k in data]
    if not keys:
        return MetricsSchema(version=version, keys=list(METRIC_KEYS))
    return MetricsSchema(version=version, keys=keys)


def validate_metrics(
    actual_keys: Iterable[str],
    schema_keys: Iterable[str],
) -> tuple[list[str], list[str]]:
    actual = set(actual_keys)
    schema = set(schema_keys)
    missing = sorted(schema - actual)
    extra = sorted(actual - schema)
    return missing, extra


def write_metrics_manifest(
    out_dir: Path,
    *,
    schema_path: Path | None,
    schema: MetricsSchema,
    actual_keys: Iterable[str],
    missing: Iterable[str],
    extra: Iterable[str],
) -> Path:
    payload = MetricManifest(
        created_at=datetime.now().isoformat(timespec="seconds"),
        schema_version=schema.version,
        schema_path=str(schema_path) if schema_path is not None else None,
        schema_keys=list(schema.keys),
        actual_keys=sorted(set(actual_keys)),
        missing_keys=list(missing),
        extra_keys=list(extra),
    )
    path = out_dir / "metric_manifest.json"
    path.write_text(json.dumps(payload.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return path
