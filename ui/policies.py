"""Политики, доступные в интерфейсе, включая загрузку обученных PPO моделей."""

from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np
from gymnasium import spaces

from baselines.factory import create_baseline_policy
from baselines.policies import BasePolicy, adapt_policy
from common.policy.waypoint_controller import WaypointController

__all__ = [
    "PPO_AVAILABLE",
    "PolicySpec",
    "PpoPolicy",
    "create_policy",
]

try:
    from stable_baselines3 import PPO
except Exception:
    PPO = None

try:
    from sb3_contrib import RecurrentPPO
except Exception:
    RecurrentPPO = None

PPO_AVAILABLE = PPO is not None


def ensure_numpy_pickle_compat() -> None:
    """Мапит numpy._core.* на numpy.core.* для совместимости pickle с numpy>=2."""
    try:
        import numpy as _np
    except Exception:
        return
    if "numpy._core" not in sys.modules:
        sys.modules["numpy._core"] = _np.core
    if "numpy._core.numeric" not in sys.modules:
        sys.modules["numpy._core.numeric"] = _np.core.numeric
    if "numpy._core._multiarray_umath" not in sys.modules and hasattr(_np.core, "_multiarray_umath"):
        sys.modules["numpy._core._multiarray_umath"] = _np.core._multiarray_umath


@dataclass
class PolicySpec:
    name: str
    model_path: str | None = None
    deterministic: bool = True


class PpoPolicy(BasePolicy):
    def __init__(self, model_path: str, deterministic: bool = True):
        if PPO is None:
            raise RuntimeError("stable_baselines3 не доступен")
        ensure_numpy_pickle_compat()
        # Стабилизируем загрузку, чтобы не влияли расписания и прогресс обучения.
        custom_objects = {
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
        }
        self.model = None
        try:
            self.model = PPO.load(model_path, custom_objects=custom_objects)
        except Exception:
            if RecurrentPPO is not None:
                try:
                    self.model = RecurrentPPO.load(model_path, custom_objects=custom_objects)
                except Exception:
                    self.model = None
        if self.model is None:
            self.model = PPO.load(model_path)
        self.deterministic = bool(deterministic)
        self.is_recurrent = bool(getattr(self.model.policy, "lstm_actor", None))
        self.lstm_states: dict[int, object] = {}
        self.episode_starts: dict[int, bool] = {}
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
        self.lstm_states = {}
        self.episode_starts = {}
        return None

    def get_action(self, agent_id: str, obs: np.ndarray | dict, state, info: dict | None = None) -> np.ndarray:
        info = self._ensure_info(info, state)
        o = obs
        space = getattr(self.model, "observation_space", None)
        if not isinstance(space, spaces.Dict) or "vector" not in space.spaces or "grid" not in space.spaces:
            raise RuntimeError("PPO‑модель должна использовать Dict‑наблюдение (vector/grid).")
        if not isinstance(o, dict):
            raise RuntimeError("UI PPO принимает только Dict‑наблюдения (vector/grid).")
        grid_space = space.spaces["grid"]
        vec = np.asarray(o.get("vector", []), dtype=np.float32).reshape(-1)
        grid = np.asarray(o.get("grid", []), dtype=np.float32)
        if grid.size == 0:
            grid = np.zeros(grid_space.shape, dtype=np.float32)
        model_obs = {"vector": vec, "grid": grid.reshape(grid_space.shape)}
        if self.is_recurrent:
            idx = 0
            if info is not None and info.get("agent_index") is not None:
                try:
                    idx = int(info.get("agent_index", 0))
                except Exception:
                    idx = 0
            elif isinstance(agent_id, str):
                parts = agent_id.split("_")
                if parts and parts[-1].isdigit():
                    idx = int(parts[-1])
            state = self.lstm_states.get(idx, None)
            start = np.asarray([self.episode_starts.get(idx, True)], dtype=bool)
            act, next_state = self.model.predict(
                model_obs,
                state=state,
                episode_start=start,
                deterministic=self.deterministic,
            )
            self.lstm_states[idx] = next_state
            self.episode_starts[idx] = False
        else:
            act, _ = self.model.predict(model_obs, deterministic=self.deterministic)
        act = np.asarray(act, dtype=np.float32)
        if state is None:
            return act
        return self._apply_controller(act, obs, state, info)

    def get_actions(self, obs_map: dict, state, infos: dict | None = None) -> dict[str, np.ndarray]:
        infos = infos or {}
        actions: dict[str, np.ndarray] = {}
        for agent_id, obs in obs_map.items():
            info = infos.get(agent_id, None)
            actions[agent_id] = self.get_action(agent_id, obs, state, info)
        return actions


def create_policy(name: str, env, model_path: str | None, deterministic: bool) -> tuple[str, object]:
    if name in {"ppo", "ppo:loaded"}:
        if not model_path:
            raise ValueError("Не указан model_path для PPO")
        return name, adapt_policy(PpoPolicy(model_path, deterministic=deterministic))
    try:
        return name, create_baseline_policy(name, env=env, seed=0, ui_safe=True, adapt=True)
    except KeyError as exc:
        raise KeyError(f"Неизвестная политика: {name}") from exc
