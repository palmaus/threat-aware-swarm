"""Shared observation vector layout for obs@1694:v5 consumers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ObsVectorLayout:
    to_target: slice = slice(0, 2)
    vel: slice = slice(2, 4)
    walls: slice = slice(4, 8)
    last_action: slice = slice(8, 10)
    measured_accel: slice = slice(10, 12)
    energy_level: slice = slice(12, 13)
    grid_start: int = 13


OBS_LAYOUT = ObsVectorLayout()
OBS_VECTOR_DIM = OBS_LAYOUT.grid_start


def obs_vector_to_fields(vector: Any) -> dict[str, Any] | None:
    vec = np.asarray(vector, dtype=np.float32).reshape(-1)
    if vec.size < OBS_VECTOR_DIM:
        return None
    return {
        "vector": [float(x) for x in vec[:OBS_VECTOR_DIM]],
        "to_target": [float(x) for x in vec[OBS_LAYOUT.to_target]],
        "vel": [float(x) for x in vec[OBS_LAYOUT.vel]],
        "walls": [float(x) for x in vec[OBS_LAYOUT.walls]],
        "last_action": [float(x) for x in vec[OBS_LAYOUT.last_action]],
        "measured_accel": [float(x) for x in vec[OBS_LAYOUT.measured_accel]],
        "energy_level": float(vec[OBS_LAYOUT.energy_level][0]),
    }


def fields_to_obs_vector(fields: dict[str, Any], *, vector_dim: int = OBS_VECTOR_DIM) -> np.ndarray:
    if not isinstance(fields, dict):
        return np.zeros((int(vector_dim),), dtype=np.float32)
    raw = fields.get("vector")
    if raw is not None:
        vec = np.asarray(raw, dtype=np.float32).reshape(-1)
        if vec.size >= int(vector_dim):
            return vec[: int(vector_dim)].astype(np.float32, copy=True)
    vec = np.zeros((int(vector_dim),), dtype=np.float32)

    def _fill(name: str, sl: slice) -> None:
        val = fields.get(name)
        if val is None:
            return
        arr = np.asarray(val, dtype=np.float32).reshape(-1)
        width = min(arr.size, sl.stop - sl.start)
        if width > 0 and sl.start < vec.size:
            vec[sl.start : min(sl.start + width, vec.size)] = arr[:width]

    _fill("to_target", OBS_LAYOUT.to_target)
    _fill("vel", OBS_LAYOUT.vel)
    _fill("walls", OBS_LAYOUT.walls)
    _fill("last_action", OBS_LAYOUT.last_action)
    _fill("measured_accel", OBS_LAYOUT.measured_accel)
    if OBS_LAYOUT.energy_level.start < vec.size and fields.get("energy_level") is not None:
        try:
            vec[OBS_LAYOUT.energy_level.start] = float(fields["energy_level"])
        except Exception:
            pass
    return vec.astype(np.float32, copy=False)


def obs_to_target(obs: dict[str, np.ndarray]) -> np.ndarray:
    vec = _obs_vector(obs)
    if vec.size >= 2:
        return vec[OBS_LAYOUT.to_target]
    return np.zeros((2,), dtype=np.float32)


def obs_vel(obs: dict[str, np.ndarray]) -> np.ndarray:
    vec = _obs_vector(obs)
    if vec.size >= 4:
        return vec[OBS_LAYOUT.vel]
    return np.zeros((2,), dtype=np.float32)


def obs_walls(obs: dict[str, np.ndarray]) -> np.ndarray:
    vec = _obs_vector(obs)
    if vec.size >= OBS_VECTOR_DIM:
        return vec[OBS_LAYOUT.walls]
    return np.zeros((4,), dtype=np.float32)


def obs_grid(obs: dict[str, np.ndarray]) -> np.ndarray | None:
    if isinstance(obs, dict):
        grid = obs.get("grid")
        if grid is None:
            return None
        grid_arr = np.asarray(grid, dtype=np.float32)
        if grid_arr.ndim == 3 and grid_arr.shape[0] == 1:
            return grid_arr[0]
        if grid_arr.ndim == 2:
            return grid_arr
        if grid_arr.ndim == 1:
            width = int(np.sqrt(grid_arr.size))
            if width > 0 and (width * width) == grid_arr.size:
                return grid_arr.reshape(width, width)
        return None
    arr = _obs_vector(obs)
    if arr.size <= OBS_VECTOR_DIM:
        return None
    grid_len = arr.size - OBS_VECTOR_DIM
    width = int(np.sqrt(grid_len))
    if width <= 0 or (width * width) != grid_len:
        return None
    return arr[-grid_len:].reshape(width, width)


def split_obs(obs: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    vec = _obs_vector(obs)
    to_target = vec[OBS_LAYOUT.to_target] if vec.size >= 2 else np.zeros((2,), dtype=np.float32)
    vel = vec[OBS_LAYOUT.vel] if vec.size >= 4 else np.zeros((2,), dtype=np.float32)
    walls = vec[OBS_LAYOUT.walls] if vec.size >= OBS_VECTOR_DIM else np.zeros((4,), dtype=np.float32)
    return to_target, vel, walls, obs_grid(obs)


def normalize(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm < 1e-6:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr / norm).astype(np.float32)


def _obs_vector(obs: Any) -> np.ndarray:
    if isinstance(obs, dict):
        vec = np.asarray(obs.get("vector", []), dtype=np.float32).reshape(-1)
        if vec.size:
            return vec
        grid = obs.get("grid")
        if grid is not None:
            return np.asarray(grid, dtype=np.float32).reshape(-1)
        return np.zeros((0,), dtype=np.float32)
    return np.asarray(obs, dtype=np.float32).reshape(-1)
