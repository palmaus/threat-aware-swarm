from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np


def _copy_threats(threats: list[Any]) -> list[Any]:
    copied: list[Any] = []
    for threat in threats or []:
        if hasattr(threat, "copy") and callable(threat.copy):
            copied.append(threat.copy())
            continue
        copied.append(copy.deepcopy(threat))
    return copied


@dataclass
class RuntimeSnapshot:
    field_size: float
    max_steps: int
    target_pos: np.ndarray
    target_vel: np.ndarray
    target_motion: dict[str, Any] | None
    target_angle: float
    target_velocity_internal: np.ndarray
    walls: list[tuple[float, float, float, float]]
    static_circles: list[tuple[np.ndarray, float]]
    agents_pos: np.ndarray
    agents_vel: np.ndarray
    agents_active: np.ndarray
    agent_state: np.ndarray
    energy: np.ndarray
    threats: list[Any]
    time_step: int
    threat_speed_scale: float
    last_collision: np.ndarray
    last_collision_speed: np.ndarray
    prev_dists: np.ndarray
    was_alive: np.ndarray
    in_goal_steps: np.ndarray
    finished: np.ndarray
    path_len: np.ndarray
    threat_collisions: np.ndarray
    min_threat_dist: np.ndarray
    death_step: np.ndarray
    start_dists: np.ndarray | None
    last_actions: np.ndarray


__all__ = ["RuntimeSnapshot", "_copy_threats"]
