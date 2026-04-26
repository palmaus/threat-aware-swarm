"""Единая схема cost‑метрик для eval/tune/UI."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

COST_KEYS = (
    "cost_progress",
    "cost_risk",
    "cost_wall",
    "cost_collision",
    "cost_energy",
    "cost_jerk",
    "cost_time",
)

REWARD_TO_COST = {
    "rew_progress": "cost_progress",
    "rew_risk": "cost_risk",
    "rew_wall": "cost_wall",
    "rew_collision": "cost_collision",
    "rew_energy": "cost_energy",
    "rew_action_change": "cost_jerk",
    "rew_time": "cost_time",
}


@dataclass(frozen=True)
class CostSchema:
    keys: tuple[str, ...] = COST_KEYS


def ensure_costs(info: Mapping[str, float]) -> dict[str, float]:
    """Гарантирует наличие ключей cost_* (NaN по умолчанию)."""
    out = dict(info)
    for key in COST_KEYS:
        out.setdefault(key, float("nan"))
    return out


def costs_from_parts(parts: Mapping[str, float] | None, *, include_time: bool = True) -> dict[str, float]:
    """Строит cost‑метрики из reward‑частей."""
    out: dict[str, float] = {}
    parts = parts or {}
    for rew_key, cost_key in REWARD_TO_COST.items():
        try:
            out[cost_key] = -float(parts.get(rew_key, 0.0))
        except Exception:
            out[cost_key] = float("nan")
    out["cost_time"] = 1.0 if include_time else 0.0
    return ensure_costs(out)
