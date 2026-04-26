from __future__ import annotations

import numpy as np

from baselines.controllers import WaypointController
from baselines.policies import ObsDict, PlannerPolicy
from baselines.utils import (
    grid_avoid_from_cost,
    normalize,
    split_obs,
    wall_avoid_from_distances,
)
from common.policy.context import PolicyContext


class SeparationSteeringPolicy(PlannerPolicy):
    """Эвристика: движение к цели + отталкивание от угроз + локальная сепарация."""

    def __init__(
        self,
        w_goal: float = 1.0,
        w_avoid: float = 1.5,
        w_sep: float = 0.0,
        **_,
    ):
        self.w_goal = float(w_goal)
        self.w_avoid = float(w_avoid)
        self.w_sep = float(w_sep)
        self.stop_risk_threshold = 0.4
        self._controller = WaypointController(
            goal_radius_control=4.0,
            near_goal_speed_cap=0.6,
            near_goal_damping=0.7,
            near_goal_kp=0.8,
            risk_speed_scale=0.65,
            risk_speed_floor=0.3,
        )

    def reset(self, seed: int | None = None) -> None:
        return

    def plan(self, agent_id: str, obs: ObsDict, state: PolicyContext, info: dict | None = None) -> np.ndarray:
        to_target, _, walls, grid = split_obs(obs)
        if grid is None:
            action = normalize(wall_avoid_from_distances(walls) + normalize(to_target))
            return action

        goal_vec = normalize(to_target)
        # Грид интерпретируется как риск, поэтому веса направлены «от» опасных клеток.
        avoid_vec = wall_avoid_from_distances(walls) + grid_avoid_from_cost(grid, radius=None)
        sep_vec = grid_avoid_from_cost(grid, radius=2)

        return self.w_goal * goal_vec + self.w_avoid * avoid_vec + self.w_sep * sep_vec
