"""Адаптеры совместимости и VecEnv-обёртки для обучения."""

from __future__ import annotations

from typing import Any

import numpy as np
from pettingzoo import ParallelEnv
from stable_baselines3.common.vec_env import VecEnvWrapper

from common.policy.context import context_from_state
from common.policy.obs_schema import obs_to_target
from common.policy.oracle_visibility import oracle_visible
from common.policy.waypoint_controller import WaypointController


class WaypointActionWrapper(ParallelEnv):
    """Прокси-обёртка: превращает waypoint-векторы в управляемое ускорение через контроллер."""

    def __init__(
        self,
        env: Any,
        *,
        stop_risk_threshold: float = 0.4,
        goal_radius_control: float = 4.0,
        near_goal_speed_cap: float = 0.6,
        near_goal_damping: float = 0.7,
        near_goal_kp: float = 0.8,
        risk_speed_scale: float = 0.65,
        risk_speed_floor: float = 0.3,
    ) -> None:
        self.env = env
        self.metadata = getattr(env, "metadata", {})
        self.stop_risk_threshold = float(stop_risk_threshold)
        self.controller = WaypointController(
            goal_radius_control=goal_radius_control,
            near_goal_speed_cap=near_goal_speed_cap,
            near_goal_damping=near_goal_damping,
            near_goal_kp=near_goal_kp,
            risk_speed_scale=risk_speed_scale,
            risk_speed_floor=risk_speed_floor,
        )

    def reset(self, *args: Any, **kwargs: Any):
        return self.env.reset(*args, **kwargs)

    def step(self, actions: Any):
        if actions is None:
            return self.env.step(actions)
        include_oracle = oracle_visible(self.env.config, consumer="agent")
        public_state = self.env.get_public_state(include_oracle=include_oracle)
        context = context_from_state(public_state, include_oracle=include_oracle)
        if isinstance(actions, dict):
            action_map = actions
        else:
            arr = np.asarray(actions, dtype=np.float32)
            action_map = {}
            agents = getattr(self.env, "possible_agents", [])
            if arr.ndim == 2 and len(agents) == arr.shape[0]:
                for i, agent_id in enumerate(agents):
                    action_map[agent_id] = arr[i]
            elif arr.ndim == 1 and len(agents) == 1:
                action_map[agents[0]] = arr
            else:
                return self.env.step(actions)

        out_actions: dict[str, np.ndarray] = {}
        agents = getattr(self.env, "possible_agents", [])
        for i, agent_id in enumerate(agents):
            desired = np.asarray(action_map.get(agent_id, np.zeros((2,), dtype=np.float32)), dtype=np.float32)
            obs = self.env.get_agent_observation(i)
            dist_m = float(context.dists[i]) if context.dists is not None and i < context.dists.shape[0] else None
            in_goal = (
                bool(context.in_goal[i]) if context.in_goal is not None and i < context.in_goal.shape[0] else False
            )
            risk_p = float(context.risk_p[i]) if context.risk_p is not None and i < context.risk_p.shape[0] else 0.0
            info = {
                "agent_index": i,
                "control_mode": context.control_mode,
                "max_speed": context.max_speed,
                "max_accel": context.max_accel,
                "dt": context.dt,
                "drag": context.drag,
                "grid_res": context.grid_res,
            }
            try:
                if getattr(public_state, "vel", None) is not None and i < public_state.vel.shape[0]:
                    info["cur_vel"] = public_state.vel[i]
            except Exception:
                pass
            to_target = obs_to_target(obs)
            out_actions[agent_id] = self.controller.compute_action(
                desired,
                obs,
                dist_m,
                False,
                float(context.field_size),
                in_goal=in_goal,
                risk_p=risk_p,
                stop_risk_threshold=self.stop_risk_threshold,
                info=info,
                to_target=to_target,
            )
        return self.env.step(out_actions)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def observation_space(self, agent):
        return self.env.observation_space(agent)

    def action_space(self, agent):
        return self.env.action_space(agent)


class ResetStepAdapter:
    """
    Нормализует reset/step из Gymnasium под формат, ожидаемый SB3.
    reset(): возвращает только obs, без info.
    step(): приводит 5‑кортеж к 4‑кортежу с done = terminated or truncated.
    """

    def __init__(self, env: Any):
        self.env = env

    def reset(self, *args: Any, **kwargs: Any):
        out = self.env.reset(*args, **kwargs)
        if isinstance(out, tuple) and len(out) == 2:
            obs, _info = out
            return obs
        return out

    def step(self, action: Any) -> tuple[Any, Any, Any, Any]:
        out = self.env.step(action)
        if isinstance(out, tuple) and len(out) == 5:
            obs, reward, terminated, truncated, info = out
            try:
                done = terminated | truncated
            except Exception:
                done = terminated or truncated
            return obs, reward, done, info
        return out

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)


class ResetStepVecWrapper(VecEnvWrapper):
    def reset(self):
        out = self.venv.reset()
        if isinstance(out, tuple) and len(out) == 2:
            obs, _info = out
            return obs
        return out

    def step_wait(self):
        out = self.venv.step_wait()
        if isinstance(out, tuple) and len(out) == 5:
            obs, rewards, terminated, truncated, infos = out
            dones = np.logical_or(terminated, truncated)
            return obs, rewards, dones, infos
        return out


class SafeEnvIsWrapped(VecEnvWrapper):
    def reset(self):
        return self.venv.reset()

    def step_wait(self):
        return self.venv.step_wait()

    def env_is_wrapped(self, wrapper_class, indices=None):
        try:
            return self.venv.env_is_wrapped(wrapper_class, indices=indices)
        except TypeError:
            try:
                return self.venv.env_is_wrapped(wrapper_class)
            except TypeError:
                n = getattr(self, "num_envs", None)
                if n is None:
                    return [False]
                return [False for _ in range(int(n))]
