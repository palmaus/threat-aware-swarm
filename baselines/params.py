from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from pathlib import Path

import yaml

DEFAULT_PATHS = [
    os.environ.get("POLICY_BEST_PARAMS", ""),
    "configs/best_policy_params.json",
    "configs/best_policy_params.yaml",
    "configs/best_policy_params.yml",
]

logger = logging.getLogger(__name__)


def _load_file(path: Path) -> dict:
    if path.suffix in {".yaml", ".yml"}:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {}
    if "policies" in data and isinstance(data["policies"], dict):
        return data["policies"]
    return data


@lru_cache(maxsize=1)
def load_best_params() -> dict[str, dict]:
    # Флаг отключения нужен для воспроизводимых экспериментов без автоподхвата.
    if str(os.environ.get("POLICY_BEST_PARAMS_DISABLE", "")).lower() in {"1", "true", "yes"}:
        return {}
    params = {}
    for p in DEFAULT_PATHS:
        if not p:
            continue
        path = Path(p)
        if path.exists():
            try:
                # Берём первый найденный файл, чтобы порядок путей был предсказуем.
                params = _load_file(path)
                logger.info("Loaded best policy params from %s", path)
                break
            except Exception as exc:
                logger.warning("Failed to load policy params from %s: %s", path, exc)
    return params if isinstance(params, dict) else {}


def get_best_params(policy_name: str) -> dict:
    data = load_best_params()
    if not isinstance(data, dict):
        return {}
    params = data.get(policy_name, {})
    if isinstance(params, dict):
        return params
    return {}
