from __future__ import annotations

import numpy as np

from baselines.controllers import WaypointController
from baselines.policies import ObsDict, PlannerPolicy
from baselines.utils import agent_index_from_id, normalize, obs_to_target
from common.policy.context import PolicyContext


class GlobalFlowFieldPolicy(PlannerPolicy):
    """Глобальная политика потока на основе oracle distance field + локальное раздвижение."""

    def __init__(
        self,
        speed_gain: float = 1.0,
        separation_gain: float = 0.35,
        separation_radius: float | None = None,
        separation_power: float = 1.5,
        risk_gain: float = 0.5,
        plan_interval: int = 1,
        stop_risk_threshold: float = 0.4,
        goal_radius_control: float = 4.0,
        near_goal_speed_cap: float = 0.6,
        near_goal_damping: float = 0.7,
        near_goal_kp: float = 0.8,
        risk_speed_scale: float = 0.65,
        risk_speed_floor: float = 0.3,
    ):
        self.speed_gain = float(speed_gain)
        self.separation_gain = float(separation_gain)
        self.separation_radius = separation_radius
        self.separation_power = float(separation_power)
        self.risk_gain = float(risk_gain)
        self.plan_interval = int(max(1, plan_interval))
        self.stop_risk_threshold = float(stop_risk_threshold)
        self._controller = WaypointController(
            goal_radius_control=goal_radius_control,
            near_goal_speed_cap=near_goal_speed_cap,
            near_goal_damping=near_goal_damping,
            near_goal_kp=near_goal_kp,
            risk_speed_scale=risk_speed_scale,
            risk_speed_floor=risk_speed_floor,
        )
        self._cache: dict[int, dict[str, np.ndarray | int]] = {}
        self._sep_cache_step: int | None = None
        self._sep_cache_vecs: np.ndarray | None = None

    def reset(self, seed: int | None = None) -> None:
        self._cache = {}
        self._sep_cache_step = None
        self._sep_cache_vecs = None

    def _compute_separation_vectors(self, state: PolicyContext) -> np.ndarray:
        pos = np.asarray(state.pos, dtype=np.float32)
        if pos.size == 0:
            return np.zeros((0, 2), dtype=np.float32)
        alive = getattr(state, "alive", None)
        if alive is None:
            alive = np.ones((pos.shape[0],), dtype=bool)
        else:
            alive = np.asarray(alive, dtype=bool)
        radius = self.separation_radius
        if radius is None or radius <= 0.0:
            radius = max(1.0, float(getattr(state, "agent_radius", 0.5)) * 4.0)
        radius = float(max(1e-6, radius))
        diff = pos[:, None, :] - pos[None, :, :]
        dist = np.linalg.norm(diff, axis=2)
        mask = (dist > 1e-6) & (dist < radius)
        if alive is not None:
            mask &= alive[:, None] & alive[None, :]
        weight = (1.0 - (dist / radius)).clip(0.0, 1.0)
        if self.separation_power != 1.0:
            weight = np.power(weight, float(self.separation_power))
        weight = np.where(mask, weight, 0.0)
        vec = (diff / (dist[..., None] + 1e-6)) * weight[..., None]
        sep = vec.sum(axis=1)
        norms = np.linalg.norm(sep, axis=1, keepdims=True)
        safe_norms = np.where(norms > 1e-6, norms, 1.0)
        sep = sep / safe_norms
        sep = np.where(norms > 1e-6, sep, 0.0)
        return sep.astype(np.float32)

    def _get_separation_vectors(self, state: PolicyContext, step: int) -> np.ndarray:
        if (
            self._sep_cache_vecs is None
            or self._sep_cache_step != step
            or self._sep_cache_vecs.shape[0] != state.pos.shape[0]
        ):
            self._sep_cache_vecs = self._compute_separation_vectors(state)
            self._sep_cache_step = step
        return self._sep_cache_vecs

    def plan(
        self,
        agent_id: str,
        obs: ObsDict,
        state: PolicyContext,
        info: dict | None = None,
    ) -> np.ndarray:
        idx = agent_index_from_id(agent_id, info)
        step = int(getattr(state, "decision_step", getattr(state, "timestep", 0)))
        cache_key = -1 if idx is None else idx
        cached = self._cache.get(cache_key)
        age = step - int(cached.get("step", 0)) if cached is not None else self.plan_interval
        use_cache = cached is not None and 0 <= age < self.plan_interval
        if use_cache:
            desired = np.asarray(cached.get("desired", np.zeros((2,), dtype=np.float32)), dtype=np.float32)
        else:
            flow_dir = None
            oracle_dir = getattr(state, "oracle_dir", None)
            if oracle_dir is not None and idx is not None:
                try:
                    if 0 <= idx < oracle_dir.shape[0]:
                        flow_dir = np.asarray(oracle_dir[idx], dtype=np.float32)
                except Exception:
                    flow_dir = None
            if flow_dir is None or float(np.linalg.norm(flow_dir)) <= 1e-6:
                flow_dir = np.asarray(obs_to_target(obs), dtype=np.float32)
            direction = normalize(flow_dir)

            # Локальное раздвижение: удерживаем рой от схлопывания в одну точку.
            sep_vec = None
            if self.separation_gain > 0.0 and idx is not None:
                try:
                    sep_vectors = self._get_separation_vectors(state, step)
                    if 0 <= idx < sep_vectors.shape[0]:
                        sep_vec = np.asarray(sep_vectors[idx], dtype=np.float32)
                        if float(np.linalg.norm(sep_vec)) <= 1e-6:
                            sep_vec = None
                except Exception:
                    sep_vec = None

            if sep_vec is not None:
                direction = normalize(direction + (sep_vec * float(self.separation_gain)))

            if self.risk_gain > 0.0 and idx is not None:
                risk_grad = getattr(state, "oracle_risk_grad", None)
                if risk_grad is not None:
                    try:
                        pos = np.asarray(state.pos[idx], dtype=np.float32)
                        height, width = risk_grad.shape[:2]
                        cell_size = float(state.field_size) / max(float(width - 1), 1.0)
                        gx = int(np.clip(np.floor(pos[0] / cell_size), 0, width - 1))
                        gy = int(np.clip(np.floor(pos[1] / cell_size), 0, height - 1))
                        risk_vec = np.asarray(risk_grad[gy, gx], dtype=np.float32)
                        if np.all(np.isfinite(risk_vec)) and float(np.linalg.norm(risk_vec)) > 1e-6:
                            direction = normalize(direction + (risk_vec * float(self.risk_gain)))
                    except Exception:
                        pass
            gain = float(max(0.0, self.speed_gain))
            if gain > 1.0:
                gain = 1.0
            desired = direction * gain
            if idx is not None:
                self._cache[idx] = {"desired": desired, "step": step}

        return desired


class FlowFieldPolicy(GlobalFlowFieldPolicy):
    """Совместимый алиас для GlobalFlowFieldPolicy."""
