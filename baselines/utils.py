"""Утилиты для базовых политик и безопасной математики векторов."""

from __future__ import annotations

import numpy as np

from common.policy.obs_schema import OBS_LAYOUT, normalize, obs_grid, obs_to_target, obs_vel, obs_walls, split_obs
from common.policy.waypoint_controller import (
    apply_control_mode,
    maybe_accel_action,
    velocity_tracking_action,
    velocity_tracking_action_batch,
)

_GRID_OFFSETS_CACHE: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}


def agent_index_from_id(agent_id: str | int, info: dict | None) -> int | None:
    """Извлекает индекс агента из info или строки идентификатора."""
    if info and info.get("agent_index") is not None:
        try:
            return int(info["agent_index"])
        except Exception:
            return None
    if isinstance(agent_id, int):
        return int(agent_id)
    if isinstance(agent_id, str):
        parts = agent_id.split("_")
        if parts:
            tail = parts[-1]
            if tail.isdigit():
                return int(tail)
    return None


def wall_avoid_from_distances(walls: np.ndarray) -> np.ndarray:
    """Простая эвристика отталкивания от стен по дистанциям до границ."""
    left, right, down, up = [float(x) for x in walls]
    eps = 1e-3
    x = (1.0 / (left + eps)) - (1.0 / (right + eps))
    y = (1.0 / (down + eps)) - (1.0 / (up + eps))
    return normalize(np.array([x, y], dtype=np.float32))


def grid_avoid_from_cost(grid: np.ndarray, radius: int | None = None) -> np.ndarray:
    """Отталкивание от риск‑карты: чем выше cost, тем сильнее уходим."""
    cost = np.asarray(grid, dtype=np.float32)
    if cost.size == 0:
        return np.zeros((2,), dtype=np.float32)
    shape = (int(cost.shape[0]), int(cost.shape[1]))
    cached = _GRID_OFFSETS_CACHE.get(shape)
    if cached is None:
        ys, xs = np.indices(shape)
        cy = int(shape[0] // 2)
        cx = int(shape[1] // 2)
        dx = xs.astype(np.float32) - float(cx)
        dy = ys.astype(np.float32) - float(cy)
        _GRID_OFFSETS_CACHE[shape] = (dx, dy)
    else:
        dx, dy = cached
    cy = int(shape[0] // 2)
    cx = int(shape[1] // 2)
    mask = cost > 0.0
    if radius is not None:
        r = int(radius)
        mask &= (np.abs(dx) <= float(r)) & (np.abs(dy) <= float(r))
    mask[cy, cx] = False
    if not np.any(mask):
        return np.zeros((2,), dtype=np.float32)
    weights = cost * mask.astype(np.float32)
    vec_x = -float(np.sum(weights * dx))
    vec_y = -float(np.sum(weights * dy))
    return normalize(np.array([vec_x, vec_y], dtype=np.float32))


def predict_target_single(
    target_pos: np.ndarray,
    target_vel,
    *,
    pos: np.ndarray | None = None,
    max_speed: float | None = None,
    gain: float = 0.6,
    max_time: float = 3.0,
    enabled: bool = True,
) -> np.ndarray:
    """Линейная экстраполяция цели для перехвата с ограничением по времени."""
    if not enabled or target_vel is None:
        return np.asarray(target_pos, dtype=np.float32)
    try:
        vel = np.asarray(target_vel, dtype=np.float32).reshape(2)
    except Exception:
        return np.asarray(target_pos, dtype=np.float32)
    if pos is None or max_speed is None or max_speed <= 1e-6:
        t = float(max_time)
    else:
        try:
            dist = float(np.linalg.norm(np.asarray(target_pos, dtype=np.float32) - np.asarray(pos, dtype=np.float32)))
        except Exception:
            dist = 0.0
        t = float(min(max_time, gain * (dist / float(max_speed))))
    return np.asarray(target_pos, dtype=np.float32) + (vel * t)


def predict_target_batch(
    target_pos: np.ndarray,
    target_vel,
    agents_pos: np.ndarray,
    *,
    max_speed: float | None = None,
    gain: float = 0.6,
    max_time: float = 3.0,
    enabled: bool = True,
) -> np.ndarray:
    """Пакетная экстраполяция цели для нескольких агентов одним проходом."""
    if not enabled or target_vel is None:
        return np.asarray(target_pos, dtype=np.float32)
    vel = np.asarray(target_vel, dtype=np.float32)
    if vel.shape != (2,):
        return np.asarray(target_pos, dtype=np.float32)
    pos = np.asarray(agents_pos, dtype=np.float32)
    if pos.ndim != 2 or pos.shape[1] != 2:
        return np.asarray(target_pos, dtype=np.float32)
    dists = np.linalg.norm(np.asarray(target_pos, dtype=np.float32) - pos, axis=1)
    if max_speed is None or max_speed <= 1e-6:
        t = np.full_like(dists, float(max_time), dtype=np.float32)
    else:
        t = np.minimum(float(max_time), float(gain) * (dists / float(max_speed))).astype(np.float32)
    return np.asarray(target_pos, dtype=np.float32) + (t[:, None] * vel[None, :])
