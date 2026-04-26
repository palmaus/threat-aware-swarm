"""DTO для разбиения награды."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RewardBreakdown:
    total: float
    parts: dict[str, float]

    @staticmethod
    def from_output(rew_out: Any) -> RewardBreakdown:
        if isinstance(rew_out, RewardBreakdown):
            return rew_out
        if isinstance(rew_out, tuple) and len(rew_out) == 2:
            total, parts = rew_out
            return RewardBreakdown(float(total), _coerce_parts(parts))
        if isinstance(rew_out, dict):
            return RewardBreakdown(0.0, _coerce_parts(rew_out))
        return RewardBreakdown(float(rew_out), {})


def _coerce_parts(parts: Any) -> dict[str, float]:
    out: dict[str, float] = {}
    if not isinstance(parts, dict):
        return out
    for key, val in parts.items():
        try:
            out[str(key)] = float(val)
        except Exception:
            continue
    return out


@dataclass(frozen=True)
class RewardOutput:
    """Типизированный результат расчёта наград за один шаг."""

    rewards: dict[str, float]
    infos: dict[str, dict]
    terminations: dict[str, bool]
    truncations: dict[str, bool]

    def as_tuple(self) -> tuple[dict[str, float], dict[str, dict], dict[str, bool], dict[str, bool]]:
        return self.rewards, self.infos, self.terminations, self.truncations


@dataclass(frozen=True)
class RewardRuntimeView:
    """Read-only runtime data needed by reward/info assembly.

    The reward package consumes this DTO instead of reaching into SwarmEngine
    private fields directly.
    """

    config: Any
    path_len: np.ndarray
    optimal_len: np.ndarray
    optimality_gap: np.ndarray
    threat_collisions: np.ndarray
    min_threat_dist: np.ndarray
    death_step: np.ndarray
    start_dists: np.ndarray | None
    last_collision: np.ndarray
