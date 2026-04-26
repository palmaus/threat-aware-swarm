"""PettingZoo ParallelEnv для роя с учетом угроз.

Среда рассчитана на стек PettingZoo (Parallel API) + SuperSuit + Stable-Baselines3 (PPO).
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from common.runtime.contracts import maybe_validate_reset
from env.config import EnvConfig
from env.engine import SwarmEngine
from env.observations.observer import SwarmObserver
from env.rewards.dto import RewardRuntimeView
from env.rewards.rewarder import RewardConfig, SwarmRewarder
from env.state import PublicState, SimState

ENV_SCHEMA_VERSION = EnvConfig().obs_schema_version


class SwarmPZEnv(ParallelEnv):
    metadata: ClassVar[dict[str, object]] = {"render_modes": ["human"], "name": "swarm_env"}

    def __init__(
        self,
        config: EnvConfig | None = None,
        reward_cfg: RewardConfig | None = None,
        *,
        goal_radius: float = 3.0,
        goal_hold_steps: int = 10,
        max_steps: int = 600,
        oracle_enabled: bool = True,
        oracle_cell_size: float = 1.0,
        oracle_async: bool = False,
        shared_curriculum: dict | None = None,
    ) -> None:
        if config is None:
            config = EnvConfig()
        self.render_mode = None
        self.reward = reward_cfg if reward_cfg is not None else RewardConfig()

        self.engine = SwarmEngine(
            config,
            reward_cfg=self.reward,
            goal_radius=goal_radius,
            goal_hold_steps=goal_hold_steps,
            max_steps=max_steps,
            oracle_enabled=oracle_enabled,
            oracle_cell_size=oracle_cell_size,
            oracle_async=oracle_async,
            shared_curriculum=shared_curriculum,
        )
        self.config = self.engine.config
        self.observer = SwarmObserver(self.config, rng=self.engine.rng)
        self.rewarder = SwarmRewarder(
            self.reward,
            field_size=self.config.field_size,
            goal_radius=self.engine.goal_radius,
        )
        self.obs_builder = self.observer.obs_builder
        self.reward_fn = self.rewarder.reward_fn
        self.metrics_fn = self.rewarder.metrics_fn

        self.possible_agents = [f"drone_{i}" for i in range(self.config.n_agents)]
        self.agents = self.possible_agents[:]

        self.env_schema_version = self.observer.env_schema_version
        self.grid_width = int(self.observer.grid_width)
        self.vector_dim = int(self.observer.vector_dim)
        self.obs_dim = int(self.vector_dim + (self.grid_width**2))

        self.observation_spaces = self.observer.make_observation_spaces(self.possible_agents)
        self.action_spaces = {
            agent: spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32) for agent in self.possible_agents
        }

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        state = self.engine.reset(seed=seed, options=options)
        self.observer.set_rng(self.engine.get_rng("obs"))
        self.observer.sync_from_engine(self.engine.config)
        self.rewarder.sync_from_engine(
            field_size=self.engine.config.field_size,
            goal_radius=self.engine.goal_radius,
            battery_drain_hover=getattr(self.engine.config.battery, "drain_hover", None),
            battery_drain_thrust=getattr(self.engine.config.battery, "drain_thrust", None),
        )
        observations = self.observer.build_all(state, self.possible_agents)
        infos = self.rewarder.build_reset_infos(state, self._reward_runtime_view(), self.possible_agents)
        runtime_mode = str(getattr(self.engine.config, "runtime_mode", "full")).strip().lower()
        if runtime_mode not in {"train_fast", "fast"}:
            try:
                public_state = self.engine.get_public_state(include_oracle=True)
            except Exception:
                public_state = None
            if public_state is not None:
                maybe_validate_reset(
                    state=state,
                    public_state=public_state,
                    observations=observations,
                    grid_width=int(getattr(self.engine.config, "grid_width", 41)),
                )
        return observations, infos

    def step(self, actions):
        total_rewards: dict[str, float] | None = None
        merged_infos: dict[str, dict[str, Any]] | None = None
        observations = {}
        terminations = {}
        truncations = {}
        infos = {}
        debug_metrics = bool(getattr(self.config, "debug_metrics", True))
        decision = self.engine.step(actions)
        last_state = None
        for step in decision.steps:
            last_state = step.state
            reward_out = self.rewarder.build_step_result(
                step,
                self._reward_runtime_view(),
                self.possible_agents,
                debug_metrics=debug_metrics,
            )
            rewards, infos, terminations, truncations = reward_out.as_tuple()
            try:
                self.engine.event_stats.record_costs(infos)
            except Exception:
                pass
            if total_rewards is None:
                total_rewards = {k: float(v) for k, v in rewards.items()}
            else:
                for key, val in rewards.items():
                    total_rewards[key] = float(total_rewards.get(key, 0.0) + float(val))
            merged_infos = self._merge_step_infos(merged_infos, infos)
            if all(terminations.values()) or all(truncations.values()):
                break
        if last_state is not None:
            observations = self.observer.build_all(last_state, self.possible_agents)
        if total_rewards is None:
            total_rewards = dict.fromkeys(self.possible_agents, 0.0)
        if merged_infos is not None:
            infos = merged_infos
            for agent_id, reward_val in total_rewards.items():
                if agent_id in infos:
                    infos[agent_id]["rew_total"] = float(reward_val)
        return observations, total_rewards, terminations, truncations, infos

    @staticmethod
    def _merge_step_infos(
        accumulated: dict[str, dict[str, Any]] | None,
        current: dict[str, dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        if accumulated is None:
            return {agent_id: dict(info) for agent_id, info in current.items()}
        event_max_keys = {"newly_finished", "died_this_step"}
        for agent_id, info in current.items():
            if not isinstance(info, dict):
                continue
            prev = accumulated.setdefault(agent_id, {})
            summed: dict[str, float] = {}
            maxed: dict[str, float] = {}
            for key, val in info.items():
                if key.startswith(("rew_", "cost_")):
                    try:
                        summed[key] = float(prev.get(key, 0.0)) + float(val)
                    except Exception:
                        pass
                elif key in event_max_keys:
                    try:
                        maxed[key] = max(float(prev.get(key, 0.0)), float(val))
                    except Exception:
                        pass
            prev.update(info)
            prev.update(summed)
            prev.update(maxed)
        return accumulated

    def apply_curriculum(self, stage_params: dict) -> None:
        self.engine.apply_curriculum(stage_params)
        self.rewarder.sync_from_engine(field_size=self.engine.config.field_size, goal_radius=self.engine.goal_radius)
        self.observer.sync_from_engine(self.engine.config)

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def get_state(self) -> SimState:
        return self.engine.get_state_copy()

    def get_public_state(self, *, include_oracle: bool = True) -> PublicState:
        return self.engine.get_public_state(include_oracle=include_oracle)

    @property
    def n_agents(self) -> int:
        return int(self.config.n_agents)

    @property
    def goal_radius(self) -> float:
        return float(self.engine.goal_radius)

    @goal_radius.setter
    def goal_radius(self, value: float) -> None:
        self.engine.goal_radius = float(value)
        self.rewarder.sync_from_engine(field_size=self.engine.config.field_size, goal_radius=self.engine.goal_radius)

    def sample_wind(self, agent_idx: int) -> np.ndarray | None:
        return self.engine.sample_wind(agent_idx)

    def set_threat_speed_scale(self, value: float) -> None:
        self.engine.set_threat_speed_scale(value)

    def get_threat_speed_scale(self) -> float:
        return self.engine.get_threat_speed_scale()

    def set_oracle_options(
        self,
        *,
        enabled: bool | None = None,
        async_mode: bool | None = None,
        update_interval: int | None = None,
        recompute: bool = True,
    ) -> None:
        self.engine.set_oracle_options(
            enabled=enabled,
            async_mode=async_mode,
            update_interval=update_interval,
            recompute=recompute,
        )

    def set_runtime_defaults(
        self,
        *,
        max_speed: float | None = None,
        mass: float | None = None,
        max_thrust: float | None = None,
        drag_coeff: float | None = None,
    ) -> None:
        self.engine.set_runtime_defaults(
            max_speed=max_speed,
            mass=mass,
            max_thrust=max_thrust,
            drag_coeff=drag_coeff,
        )
        self.rewarder.sync_from_engine(field_size=self.engine.config.field_size, goal_radius=self.engine.goal_radius)
        self.observer.sync_from_engine(self.engine.config)

    def set_walls(self, walls) -> None:
        self.engine.set_walls(walls)

    def set_static_circles(self, circles) -> None:
        self.engine.set_static_circles(circles)

    def set_agent_positions(self, positions) -> None:
        self.engine.set_agent_positions(positions)

    def clear_threats(self) -> None:
        self.engine.clear_threats()

    def replace_threats(self, threats) -> None:
        self.engine.replace_threats(threats)

    def add_threat_object(self, threat) -> None:
        self.engine.add_threat_object(threat)

    def add_static_threat(self, position, radius: float, intensity: float, *, oracle_block: bool = True) -> None:
        self.engine.add_static_threat(position, radius, intensity, oracle_block=oracle_block)

    def configure_target_motion(self, cfg: dict | None) -> None:
        self.engine.configure_target_motion(cfg)

    def get_oracle_path(self) -> list[list[float]]:
        oracle = self.engine.oracle
        if oracle is None:
            return []
        try:
            return [[float(x), float(y)] for x, y in (oracle.path or [])]
        except Exception:
            return []

    def get_agent_observation(self, agent_idx: int, state: SimState | None = None) -> dict[str, np.ndarray]:
        obs = self._get_obs(agent_idx, state)
        return {key: np.array(value, copy=True) for key, value in obs.items()}

    def get_runtime_snapshot(
        self,
        *,
        include_oracle: bool = True,
        policy: object | None = None,
        policy_name: str | None = None,
    ) -> dict[str, Any]:
        from common.policy.oracle_visibility import oracle_visible_for_policy

        oracle_visible = bool(include_oracle)
        if policy is not None or policy_name is not None:
            oracle_visible = bool(oracle_visible_for_policy(self.config, policy=policy, policy_name=policy_name))
        state = self.get_public_state(include_oracle=oracle_visible)
        threats = []
        for threat in state.threats or []:
            velocity = getattr(threat, "velocity", getattr(threat, "vel", np.zeros(2, dtype=np.float32)))
            threats.append(
                (
                    np.array(getattr(threat, "position", np.zeros(2, dtype=np.float32)), dtype=np.float32, copy=True),
                    float(getattr(threat, "radius", 0.0)),
                    float(getattr(threat, "intensity", 0.0)),
                    np.array(velocity, dtype=np.float32, copy=True),
                )
            )
        return {
            "agents_pos": np.array(state.pos, dtype=np.float32, copy=True),
            "agents_vel": np.array(state.vel, dtype=np.float32, copy=True),
            "agents_active": np.array(state.alive, dtype=bool, copy=True),
            "agent_state": np.array(state.agent_state, dtype=np.int8, copy=True),
            "finished": np.array(self.engine.finished, dtype=bool, copy=True),
            "in_goal": np.array(state.in_goal, dtype=bool, copy=True),
            "threats": threats,
            "walls": [list(w) for w in state.static_walls],
            "static_circles": [list(c) for c in state.static_circles],
            "field_size": float(state.field_size),
            "max_speed": float(state.max_speed),
            "dt": float(state.dt),
            "max_accel": float(state.max_accel),
            "max_thrust": float(state.max_thrust),
            "mass": float(state.mass),
            "drag_coeff": float(state.drag_coeff),
            "drag": float(state.drag),
            "agent_radius": float(state.agent_radius),
            "wall_friction": float(state.wall_friction),
            "grid_res": float(state.grid_res),
            "target_pos": np.array(state.target_pos, dtype=np.float32, copy=True),
            "target_vel": np.array(state.target_vel, dtype=np.float32, copy=True),
            "oracle_dir": None
            if state.oracle_dir is None
            else np.array(state.oracle_dir, dtype=np.float32, copy=True),
            "oracle_visible": oracle_visible,
            "goal_radius": float(self.goal_radius),
            "timestep": int(state.timestep),
            "decision_step": int(state.decision_step),
            "control_mode": str(state.control_mode),
            "energy_level": np.array(state.energy_level, dtype=np.float32, copy=True),
            "measured_accel": np.array(state.measured_accel, dtype=np.float32, copy=True),
        }

    def get_episode_summary(self):
        return self.engine.get_episode_summary()

    def close(self) -> None:
        self.engine.close()

    @property
    def decision_loop(self):
        return self.engine.decision_loop

    @property
    def physics_loop(self):
        return self.engine.physics_loop

    @property
    def event_stats(self):
        return self.engine.event_stats

    @property
    def rng_registry(self):
        return self.engine.rng_registry

    def get_rng(self, name: str):
        return self.engine.get_rng(name)

    def _set_rng(self, seed: int) -> None:
        self.engine._set_rng(seed)

    def _get_obs(self, idx: int, state: SimState | None = None) -> dict[str, np.ndarray]:
        if state is None:
            state = self.engine.get_state()
        return self.observer.build(state, idx)

    def _zero_obs(self) -> dict[str, np.ndarray]:
        return self.observer.zero_obs()

    def _reward_runtime_view(self) -> RewardRuntimeView:
        return self.engine.get_reward_runtime_view()

    @property
    def sim(self):
        return self.engine.sim

    @property
    def spawn(self):
        return self.engine.spawn

    @property
    def scene(self):
        return self.engine.scene

    @property
    def oracle(self):
        return self.engine.oracle

    @oracle.setter
    def oracle(self, value) -> None:
        self.engine.oracle = value
        if hasattr(value, "optimal_len"):
            self.engine._optimal_len = value.optimal_len
        self.engine._invalidate_export_cache()

    @property
    def target_pos(self) -> np.ndarray:
        return self.engine.target_pos

    @target_pos.setter
    def target_pos(self, value) -> None:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if arr.size < 2:
            raise ValueError("target_pos must contain at least two coordinates")
        self.engine.target_pos = arr[:2].copy()
        self.engine._invalidate_export_cache()

    @property
    def target_vel(self) -> np.ndarray:
        return self.engine.target_vel

    @target_vel.setter
    def target_vel(self, value) -> None:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if arr.size < 2:
            raise ValueError("target_vel must contain at least two coordinates")
        self.engine.target_vel = arr[:2].copy()
        self.engine._invalidate_export_cache()

    @property
    def finished(self) -> np.ndarray:
        return self.engine.finished

    @finished.setter
    def finished(self, value) -> None:
        self.engine.finished = np.asarray(value, dtype=bool).copy()
        self.engine._invalidate_export_cache()

    @property
    def max_steps(self) -> int:
        return int(self.engine.max_steps)

    @max_steps.setter
    def max_steps(self, value: int) -> None:
        self.engine.max_steps = int(value)

    @property
    def goal_hold_steps(self) -> int:
        return int(self.engine.goal_hold_steps)

    @goal_hold_steps.setter
    def goal_hold_steps(self, value: int) -> None:
        self.engine.goal_hold_steps = int(value)

    @property
    def oracle_enabled(self) -> bool:
        return bool(self.engine.oracle_enabled)

    @oracle_enabled.setter
    def oracle_enabled(self, value: bool) -> None:
        self.engine.set_oracle_options(enabled=bool(value), recompute=False)

    @property
    def oracle_async(self) -> bool:
        return bool(self.engine.oracle_async)

    @oracle_async.setter
    def oracle_async(self, value: bool) -> None:
        self.engine.set_oracle_options(async_mode=bool(value), recompute=False)

    @property
    def oracle_cell_size(self) -> float:
        return float(self.engine.oracle_cell_size)

    @oracle_cell_size.setter
    def oracle_cell_size(self, value: float) -> None:
        self.engine.oracle_cell_size = float(value)
        self.engine._invalidate_export_cache()

    @property
    def _optimal_len(self) -> np.ndarray:
        return self.engine._optimal_len

    @property
    def _path_len(self) -> np.ndarray:
        return self.engine._path_len

    @property
    def _threat_collisions(self) -> np.ndarray:
        return self.engine._threat_collisions

    @property
    def _min_threat_dist(self) -> np.ndarray:
        return self.engine._min_threat_dist

    @property
    def _death_step(self) -> np.ndarray:
        return self.engine._death_step

    @property
    def _base_max_speed(self) -> float:
        return float(self.engine._base_max_speed)

    @_base_max_speed.setter
    def _base_max_speed(self, value: float) -> None:
        self.engine.set_runtime_defaults(max_speed=float(value))

    @property
    def _base_mass(self) -> float:
        return float(self.engine._base_mass)

    @_base_mass.setter
    def _base_mass(self, value: float) -> None:
        self.engine.set_runtime_defaults(mass=float(value))

    @property
    def _base_max_thrust(self) -> float:
        return float(self.engine._base_max_thrust)

    @_base_max_thrust.setter
    def _base_max_thrust(self, value: float) -> None:
        self.engine.set_runtime_defaults(max_thrust=float(value))

    @property
    def _base_drag_coeff(self) -> float:
        return float(self.engine._base_drag_coeff)

    @_base_drag_coeff.setter
    def _base_drag_coeff(self, value: float) -> None:
        self.engine.set_runtime_defaults(drag_coeff=float(value))
