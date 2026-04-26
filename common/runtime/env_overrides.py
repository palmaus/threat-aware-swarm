"""Helpers for splitting EnvConfig fields from runtime-only env overrides."""

from __future__ import annotations

from dataclasses import fields
from typing import Any


def coerce_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(value):
            data = OmegaConf.to_container(value, resolve=True) or {}
            return dict(data) if isinstance(data, dict) else {}
    except Exception:
        return {}
    return {}


def split_env_constructor_overrides(
    env_config_cls: type,
    *payloads: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Separate EnvConfig dataclass fields from runtime/curriculum-only params."""

    config_keys = {field.name for field in fields(env_config_cls)}
    constructor_payload: dict[str, Any] = {}
    runtime_payloads: list[dict[str, Any]] = []
    for payload in payloads:
        payload = dict(payload or {})
        cfg_part = {key: val for key, val in payload.items() if key in config_keys}
        runtime_part = {key: val for key, val in payload.items() if key not in config_keys}
        constructor_payload.update(cfg_part)
        runtime_payloads.append(runtime_part)
    return constructor_payload, runtime_payloads


def split_env_runtime_fields(
    payload: Any,
    *,
    default_max_steps: int = 600,
    default_goal_radius: float = 3.0,
) -> tuple[dict[str, Any], int, float]:
    """Extract episode/runtime fields that are intentionally outside EnvConfig."""

    env_payload = coerce_mapping(payload)
    max_steps = int(env_payload.pop("max_steps", default_max_steps))
    goal_radius = float(env_payload.pop("goal_radius", default_goal_radius))
    return env_payload, max_steps, goal_radius
