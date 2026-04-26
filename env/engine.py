from __future__ import annotations

import copy
from typing import Any

import numpy as np

from common.policy.oracle_visibility import oracle_visible
from env.config import EnvConfig
from env.decision_loop import DecisionLoop
from env.event_handlers import EpisodeStatsCollector
from env.events import DecisionStepEvent, EpisodeEndEvent, EpisodeStartEvent, EventBus
from env.oracles.manager import OracleManager
from env.physics.core import PhysicsCore
from env.physics.loop import PhysicsLoop
from env.rng_registry import RNGRegistry
from env.runtime.curriculum import apply_curriculum_params
from env.runtime.episode_reset import (
    apply_episode_domain_randomization,
    apply_scene_wind_override,
    configure_episode_wind,
)
from env.runtime.results import DecisionStepResult, StepResult, ThreatMeta
from env.runtime.snapshots import RuntimeSnapshot, _copy_threats
from env.scenes.providers import DefaultMapProvider, DefaultThreatProvider, MapProvider, ThreatProvider
from env.scenes.scene_manager import SceneManager
from env.scenes.spawn_controller import SpawnController
from env.state import PublicState, SimState, copy_public_state, copy_sim_state, public_state_from_state
from env.utils.geometry import normalize_walls

_UNSAFE_RUNTIME_CONFIG_KEYS = {"n_agents", "grid_width", "grid_res", "obs_schema_version"}


def _row_norm(values: np.ndarray, *, out: np.ndarray) -> np.ndarray:
    np.sqrt(np.sum(values * values, axis=1, dtype=np.float32), out=out)
    return out


class SwarmEngine:
    def __init__(
        self,
        config: EnvConfig | None = None,
        *,
        reward_cfg: Any | None = None,
        goal_radius: float = 3.0,
        goal_hold_steps: int = 10,
        max_steps: int = 600,
        oracle_enabled: bool = True,
        oracle_cell_size: float = 1.0,
        oracle_async: bool = False,
        shared_curriculum: dict | None = None,
        map_provider: MapProvider | None = None,
        threat_provider: ThreatProvider | None = None,
    ) -> None:
        if config is None:
            config = EnvConfig()
        else:
            config = copy.deepcopy(config)
        self.config = config
        self.rng_registry = RNGRegistry()
        self.rng = self.rng_registry.get("engine")
        self.sim = PhysicsCore(config, rng=self.rng_registry.get("physics"))
        self.physics_loop = PhysicsLoop()
        self.decision_loop = DecisionLoop(
            ticks_per_action=getattr(config, "physics_ticks_per_action", 1),
            physics_loop=self.physics_loop,
        )
        self.reward_cfg = reward_cfg
        self.map_provider = map_provider or DefaultMapProvider()
        self.threat_provider = threat_provider or DefaultThreatProvider()
        self.events = EventBus()
        self.event_stats = EpisodeStatsCollector(self.events)

        self._base_max_speed = float(self.config.max_speed)
        self._base_mass = float(getattr(self.config.physics, "mass", 1.0))
        self._base_drag_coeff = float(getattr(self.config.physics, "drag_coeff", 0.0))
        self._base_max_thrust = float(getattr(self.config.physics, "max_thrust", 0.0))
        self._base_wind_enabled = bool(getattr(self.config.wind, "enabled", False))
        self._base_wind_theta = float(getattr(self.config.wind, "ou_theta", 0.15))
        self._base_wind_sigma = float(getattr(self.config.wind, "ou_sigma", 0.3))
        self._base_wind_seed = getattr(self.config.wind, "seed", None)
        self._base_battery_capacity = float(getattr(self.config.battery, "capacity", 0.0))
        self._base_battery_drain_hover = float(getattr(self.config.battery, "drain_hover", 0.0))
        self._base_battery_drain_thrust = float(getattr(self.config.battery, "drain_thrust", 0.0))

        self.goal_radius = float(goal_radius)
        self.goal_hold_steps = int(goal_hold_steps)
        self.max_steps = int(max_steps)
        self._base_max_steps = int(max_steps)
        self._base_field_size = float(self.config.field_size)
        self.oracle_enabled = bool(oracle_enabled)
        self.oracle_cell_size = float(oracle_cell_size)
        self.oracle_async = bool(oracle_async)
        self._walls: list[tuple[float, float, float, float]] = []

        self.spawn = SpawnController(self, rng=self.rng_registry.get("spawn"))
        self.oracle = OracleManager(self)
        self.scene = SceneManager(self, self.spawn, rng=self.rng_registry.get("scene"))
        self._optimal_len = self.oracle.optimal_len

        self._path_len = np.zeros(self.config.n_agents, dtype=np.float32)
        self._threat_collisions = np.zeros(self.config.n_agents, dtype=np.int32)
        self._min_threat_dist = np.full(self.config.n_agents, np.inf, dtype=np.float32)
        self._death_step = np.full(self.config.n_agents, -1, dtype=np.int32)
        self._start_dists: np.ndarray | None = None
        self._last_actions = np.zeros((self.config.n_agents, 2), dtype=np.float32)
        self._prev_pos: np.ndarray | None = None
        self._prev_vel: np.ndarray | None = None
        self._prev_alive: np.ndarray | None = None
        self._prev_energy: np.ndarray | None = None
        self._prev_agent_state: np.ndarray | None = None
        self._finished_before: np.ndarray | None = None
        self._actions_scaled: np.ndarray | None = None
        self._cur_actions_buf: np.ndarray | None = None
        self._prev_last_actions: np.ndarray | None = None
        self._step_delta_buf: np.ndarray | None = None
        self._step_dist_buf: np.ndarray | None = None
        self._dists_buf: np.ndarray | None = None
        self._measured_accel_buf: np.ndarray | None = None
        self._energy_level_buf: np.ndarray | None = None
        self._shared_curriculum = shared_curriculum
        self._shared_curriculum_version: int | None = None

        self.target_pos = np.zeros(2, dtype=np.float32)
        self.target_vel = np.zeros(2, dtype=np.float32)
        self._target_motion: dict | None = None
        self._target_angle = 0.0
        self._target_vel = np.zeros(2, dtype=np.float32)
        self.prev_dists = np.zeros(self.config.n_agents, dtype=np.float32)
        self.was_alive = np.ones(self.config.n_agents, dtype=bool)
        self.in_goal_steps = np.zeros(self.config.n_agents, dtype=np.int32)
        self.finished = np.zeros(self.config.n_agents, dtype=bool)
        self._state: SimState | None = None
        self._public_state_cache: dict[bool, PublicState] = {}
        self._wind_field = None
        self._static_circles: list[tuple[np.ndarray, float]] = []
        self._static_walls_payload: list[tuple[float, float, float, float]] = []
        self._static_circles_payload: list[tuple[float, float, float]] = []
        self._dt = 0.0
        self._mass = 1.0
        self._max_thrust = 0.0
        self._drag_coeff = 0.0
        self._max_accel = 0.0
        self._drag = 0.0
        self._battery_capacity = 0.0
        self._sync_runtime_params()
        self._validate_control_mode()

    def _apply_profile(self) -> None:
        """Применяет профиль среды, чтобы быстро включать/выключать тяжелые подсистемы."""
        profile = str(getattr(self.config, "profile", "full")).lower()
        if profile == "lite":
            self.config.wind.enabled = False
            self.config.battery.capacity = 0.0
            self.config.battery.drain_hover = 0.0
            self.config.battery.drain_thrust = 0.0
            return
        # Полный профиль восстанавливает базовые параметры, зафиксированные при инициализации.
        self.config.wind.enabled = bool(self._base_wind_enabled)
        self.config.wind.ou_theta = float(self._base_wind_theta)
        self.config.wind.ou_sigma = float(self._base_wind_sigma)
        self.config.wind.seed = self._base_wind_seed
        self.config.battery.capacity = float(self._base_battery_capacity)
        self.config.battery.drain_hover = float(self._base_battery_drain_hover)
        self.config.battery.drain_thrust = float(self._base_battery_drain_thrust)

    def _runtime_mode(self) -> str:
        return str(getattr(self.config, "runtime_mode", "full")).strip().lower()

    def _runtime_is_fast(self) -> bool:
        return self._runtime_mode() in {"train_fast", "fast"}

    def _should_materialize_oracle_dir(self) -> bool:
        if not self.oracle_enabled or getattr(self, "oracle", None) is None:
            return False
        return bool(
            oracle_visible(self.config, consumer="baseline") or oracle_visible(self.config, consumer="agent")
        )

    def _validate_control_mode(self) -> None:
        mode = str(getattr(self.config, "control_mode", "waypoint")).lower()
        if mode != "waypoint":
            raise ValueError(f"Режим управления '{mode}' больше не поддерживается; используйте control_mode=waypoint.")

    def reset(self, seed: int | None = None, options: dict | None = None) -> SimState:
        self.decision_loop.reset()
        self.physics_loop.reset()
        self._validate_control_mode()
        self._restore_episode_defaults()
        scene = (options or {}).get("scene") if options else None
        if scene and scene.get("seed") is not None:
            self._set_rng(int(scene["seed"]))
        elif seed is not None:
            self._set_rng(int(seed))
        apply_episode_domain_randomization(self)
        self._last_actions = np.zeros((self.config.n_agents, 2), dtype=np.float32)
        self._invalidate_export_cache()

        self._apply_profile()
        apply_scene_wind_override(self.config, scene if isinstance(scene, dict) else None)
        self.sim.reset()
        self._static_circles = []
        self._sync_shared_curriculum()
        self._configure_target_motion(None)
        self.target_vel = np.zeros(2, dtype=np.float32)
        configure_episode_wind(self)

        with self.scene.transaction():
            self.map_provider.apply(self, scene if isinstance(scene, dict) else None)
            self.threat_provider.apply(self, scene if isinstance(scene, dict) else None)
            self._refresh_static_payloads()
        self._sync_runtime_params()

        self.prev_dists = np.linalg.norm(self.sim.agents_pos - self.target_pos, axis=1).astype(np.float32)
        self._start_dists = self.prev_dists.copy()
        self.was_alive[:] = True
        self.in_goal_steps[:] = 0
        self.finished[:] = False
        self._path_len[:] = 0.0
        self._threat_collisions[:] = 0
        self._min_threat_dist[:] = np.inf
        self._death_step[:] = -1
        self.oracle.compute_lengths(clear_state=True)
        state = self._build_state(
            pos=self.sim.agents_pos,
            vel=self.sim.agents_vel,
            alive=self.sim.agents_active,
            dists=self.prev_dists,
            in_goal=(self.prev_dists <= self.goal_radius) & self.sim.agents_active,
            in_goal_steps=self.in_goal_steps.copy(),
            finished=self.finished.copy(),
            newly_finished=np.zeros(self.sim.n, dtype=bool),
            risk_p=self._compute_risk_probs(self.sim.agents_pos, alive_override=self.sim.agents_active),
            min_neighbor_dist=self._compute_min_neighbor_dist(self.sim.agents_pos, self.sim.agents_active),
            last_action=self._last_actions.copy(),
            walls=None,
            collision_speed=self.sim.last_collision_speed.copy(),
            timestep=int(self.sim.time_step),
            threats=self.sim.threats,
        )
        self._state = state
        self._invalidate_export_cache()
        try:
            self.events.emit(
                EpisodeStartEvent(state=state, seed=seed, scene=scene if isinstance(scene, dict) else None)
            )
        except Exception:
            pass
        return state

    def step(self, actions: dict | None) -> DecisionStepResult:
        ticks = max(1, int(getattr(self.config, "physics_ticks_per_action", 1)))
        self.decision_loop.sync(ticks)
        steps = self.decision_loop.run(self._step_physics, actions)
        if steps:
            last = steps[-1]
            if self.decision_loop.is_timeout(self.max_steps) and not last.done:
                last.done = True
                last.is_timeout = True
            for _idx, step in enumerate(steps):
                try:
                    self.events.emit(
                        DecisionStepEvent(
                            step=step,
                            decision_index=int(self.decision_loop.decision_step),
                            done=bool(step.done),
                            is_timeout=bool(step.is_timeout),
                        )
                    )
                except Exception:
                    pass
            if last.done:
                try:
                    self.events.emit(
                        EpisodeEndEvent(
                            state=last.state,
                            steps=int(self.decision_loop.decision_step),
                            done=bool(last.done),
                            is_timeout=bool(last.is_timeout),
                        )
                    )
                except Exception:
                    pass
        return DecisionStepResult(steps=steps)

    def _ensure_buffers(self) -> None:
        n = int(self.sim.n)
        if self._prev_pos is None or self._prev_pos.shape != self.sim.agents_pos.shape:
            self._prev_pos = np.empty_like(self.sim.agents_pos)
            self._prev_vel = np.empty_like(self.sim.agents_vel)
            self._prev_alive = np.empty_like(self.sim.agents_active)
            self._prev_energy = np.empty_like(self.sim.energy)
            self._prev_agent_state = np.empty_like(self.sim.agent_state)
            self._finished_before = np.empty_like(self.finished)
            self._actions_scaled = np.zeros((n, 2), dtype=np.float32)
            self._cur_actions_buf = np.zeros((n, 2), dtype=np.float32)
            self._prev_last_actions = np.empty_like(self._last_actions)
            self._step_delta_buf = np.empty_like(self.sim.agents_pos)
            self._step_dist_buf = np.empty((n,), dtype=np.float32)
            self._dists_buf = np.empty((n,), dtype=np.float32)
            self._measured_accel_buf = np.empty_like(self.sim.agents_vel)
            self._energy_level_buf = np.empty((n,), dtype=np.float32)

    def _step_physics(self, actions: dict | None) -> StepResult:
        self._ensure_buffers()
        prev_alive = self._prev_alive
        np.copyto(prev_alive, self.sim.agents_active)
        alive_before = prev_alive
        finished_before = self._finished_before
        np.copyto(finished_before, self.finished)

        # Мертвые и завершившие агенты игнорируются, чтобы не вносить шум в симуляцию.
        actions_scaled = self._actions_scaled
        actions_scaled.fill(0.0)
        scale = float(self._max_thrust)
        prev_last_actions = self._prev_last_actions
        np.copyto(prev_last_actions, self._last_actions)
        cur_actions = self._cur_actions_buf
        cur_actions.fill(0.0)
        for agent_id, action in (actions or {}).items():
            try:
                idx = int(str(agent_id).split("_")[1])
            except Exception:
                continue

            if idx < 0 or idx >= self.sim.n:
                continue
            if not self.sim.agents_active[idx]:
                continue
            if self.finished[idx]:
                continue

            act = np.asarray(action, dtype=np.float32)
            act = np.clip(act, -1.0, 1.0)
            cur_actions[idx] = act
            actions_scaled[idx] = act * scale
        np.copyto(self._last_actions, cur_actions)

        prev_pos = self._prev_pos
        prev_vel = self._prev_vel
        prev_energy = self._prev_energy
        prev_agent_state = self._prev_agent_state
        np.copyto(prev_pos, self.sim.agents_pos)
        np.copyto(prev_vel, self.sim.agents_vel)
        np.copyto(prev_energy, self.sim.energy)
        np.copyto(prev_agent_state, self.sim.agent_state)
        prev_state = self._state

        # Симулятор — единственная точка истины для физики.
        self.sim.step(actions_scaled)
        if np.any(finished_before):
            # Finished agents are terminal for task metrics; keep them alive but frozen
            # while the rest of the swarm completes the episode.
            self.sim.agents_pos[finished_before] = prev_pos[finished_before]
            self.sim.agents_vel[finished_before] = 0.0
            self.sim.energy[finished_before] = prev_energy[finished_before]
            self.sim.agent_state[finished_before] = prev_agent_state[finished_before]
            self.sim.agents_active[finished_before] = prev_alive[finished_before]
            self.sim.last_collision[finished_before] = False
            self.sim.last_collision_speed[finished_before] = 0.0
        self._update_target_motion()

        # Накопление длины пути нужно для метрики path_ratio.
        pos = self.sim.agents_pos
        step_delta = self._step_delta_buf
        np.subtract(pos, prev_pos, out=step_delta)
        step_dist = self._step_dist_buf
        _row_norm(step_delta, out=step_dist)
        self._path_len += step_dist
        vel = self.sim.agents_vel
        measured_accel = self._measured_accel_buf
        np.subtract(vel, prev_vel, out=measured_accel)
        measured_accel *= np.float32(1.0 / max(self._dt, 1e-6))
        alive = self.sim.agents_active
        np.subtract(pos, self.target_pos[None, :], out=step_delta)
        dists = self._dists_buf
        _row_norm(step_delta, out=dists)
        energy = self.sim.energy.copy()
        if self._battery_capacity > 0.0:
            energy_level = self._energy_level_buf
            np.multiply(energy, np.float32(1.0 / self._battery_capacity), out=energy_level)
            np.clip(energy_level, 0.0, 1.0, out=energy_level)
        else:
            energy_level = self._energy_level_buf
            energy_level.fill(0.0)

        # Фиксируем факт нахождения в зоне цели на текущем шаге.
        in_goal = (dists <= self.goal_radius) & alive

        # Требуем удержание в цели несколько шагов, чтобы исключить случайные касания.
        self.in_goal_steps[in_goal] += 1
        self.in_goal_steps[~in_goal] = 0

        newly_finished = (~self.finished) & (self.in_goal_steps >= self.goal_hold_steps)
        self.finished[newly_finished] = True
        if np.any(newly_finished):
            self.sim.agents_vel[newly_finished] = 0.0
            measured_accel[newly_finished] = 0.0

        debug_metrics = bool(getattr(self.config, "debug_metrics", True))
        debug_mode = str(getattr(self.config, "debug_metrics_mode", "full")).lower()
        full_debug = debug_metrics and debug_mode != "lite" and not self._runtime_is_fast()

        # Риск нужен для награды и базовых метрик.
        risk_p, threat_meta = self._compute_threat_metrics(pos, alive_override=prev_alive, with_meta=True)

        # Минимальная дистанция до соседа используется в регуляризации поведения.
        min_neighbor_dist = self._compute_min_neighbor_dist(pos, alive)

        # Метаданные угрозы для текущей позиции нужны для метрик и логирования.
        prev_threat_meta = (
            self._compute_threat_metrics(prev_pos, alive_override=prev_alive, with_meta=True)[1] if full_debug else None
        )

        track_metrics = alive_before & (~finished_before)
        if np.any(track_metrics):
            self._min_threat_dist[track_metrics] = np.minimum(
                self._min_threat_dist[track_metrics],
                threat_meta.dist[track_metrics],
            )
            self._threat_collisions[track_metrics] += threat_meta.inside_any[track_metrics].astype(np.int32)

        if hasattr(self.sim, "get_wall_distances_batch"):
            walls = self.sim.get_wall_distances_batch()
        else:
            walls = np.zeros((self.sim.n, 4), dtype=np.float32)
            for i in range(self.sim.n):
                walls[i] = self.sim.get_wall_distances(i)

        if prev_state is None:
            prev_state = self._build_state(
                pos=prev_pos,
                vel=prev_vel,
                alive=prev_alive,
                dists=self.prev_dists.copy(),
                in_goal=np.zeros_like(prev_alive, dtype=bool),
                in_goal_steps=self.in_goal_steps.copy(),
                finished=self.finished.copy(),
                newly_finished=np.zeros_like(prev_alive, dtype=bool),
                risk_p=risk_p,
                min_neighbor_dist=self._compute_min_neighbor_dist(prev_pos, prev_alive),
                last_action=prev_last_actions,
                walls=walls.copy(),
                collision_speed=np.zeros_like(self.prev_dists, dtype=np.float32),
                timestep=int(self.sim.time_step - 1),
                threats=self.sim.threats,
                measured_accel=np.zeros_like(prev_vel, dtype=np.float32),
                energy=prev_energy,
                energy_level=np.clip(prev_energy / max(self._battery_capacity, 1e-6), 0.0, 1.0).astype(np.float32)
                if self._battery_capacity > 0.0
                else np.zeros_like(prev_energy, dtype=np.float32),
                agent_state=prev_agent_state,
            )

        state = self._build_state(
            pos=pos,
            vel=vel,
            alive=alive,
            dists=dists.copy(),
            in_goal=in_goal,
            in_goal_steps=self.in_goal_steps.copy(),
            finished=self.finished.copy(),
            newly_finished=newly_finished.copy(),
            risk_p=risk_p,
            min_neighbor_dist=min_neighbor_dist,
            last_action=self._last_actions.copy(),
            walls=walls,
            collision_speed=self.sim.last_collision_speed.copy(),
            timestep=int(self.sim.time_step),
            threats=self.sim.threats,
            measured_accel=measured_accel.copy(),
            energy=energy,
            energy_level=energy_level.copy(),
            agent_state=self.sim.agent_state.copy(),
        )

        # Таймаут рассчитывается на уровне decision‑шага.
        is_timeout = False
        all_dead = not bool(np.any(alive))
        active_mask = alive
        all_finished = bool(np.all(self.finished[active_mask])) if np.any(active_mask) else False
        done = bool(all_dead or all_finished)

        died_this_step = alive_before & (~alive)
        if np.any(died_this_step):
            for i in range(self.sim.n):
                if died_this_step[i] and self._death_step[i] < 0:
                    self._death_step[i] = int(self.sim.time_step)

        # Сохраняем расстояния для награды следующего шага.
        self.prev_dists = dists
        self._state = state
        self._invalidate_export_cache()

        return StepResult(
            prev_state=prev_state,
            state=state,
            done=done,
            is_timeout=is_timeout,
            alive_before=alive_before,
            died_this_step=died_this_step,
            threat_meta=threat_meta,
            prev_threat_meta=prev_threat_meta,
        )

    def get_state(self) -> SimState:
        if self._state is None:
            self._state = self._build_state()
        return self._state

    def get_state_copy(self) -> SimState:
        return copy_sim_state(self.get_state())

    def get_public_state(self, *, include_oracle: bool = True) -> PublicState:
        cached = self._public_state_cache.get(bool(include_oracle))
        if cached is not None:
            return copy_public_state(cached)
        public_state = self._build_public_state(self.get_state(), include_oracle=include_oracle)
        self._public_state_cache[bool(include_oracle)] = public_state
        return copy_public_state(public_state)

    def _invalidate_export_cache(self) -> None:
        self._public_state_cache.clear()

    def _invalidate_spatial_caches(self) -> None:
        try:
            self.spawn._mask_cache.clear()
        except Exception:
            pass
        if getattr(self, "oracle", None) is not None:
            try:
                self.oracle.grid = None
                self.oracle.cache_key = None
                self.oracle.risk_grid = None
                self.oracle.risk_cache_key = None
                self.oracle.risk_grad = None
                self.oracle.risk_grad_cache_key = None
                self.oracle.dist_field = None
                self.oracle.dist_cache_key = None
                self.oracle.dist_grad = None
                self.oracle.dist_grad_cache_key = None
                self.oracle._path = []
                self.oracle._path_cache_key = None
                self.oracle._last_target_pos = None
            except Exception:
                pass
        try:
            self.sim.set_walls(self._walls)
        except Exception:
            pass
        self._invalidate_export_cache()

    def set_oracle_options(
        self,
        *,
        enabled: bool | None = None,
        async_mode: bool | None = None,
        update_interval: int | None = None,
        recompute: bool = True,
    ) -> None:
        if enabled is not None:
            self.oracle_enabled = bool(enabled)
        if async_mode is not None:
            self.oracle_async = bool(async_mode)
        if update_interval is not None:
            self.config.oracle_update_interval = int(update_interval)
        self._invalidate_export_cache()
        if recompute and self.oracle_enabled and self.oracle is not None:
            self.oracle.compute_lengths(clear_state=True)

    def set_runtime_defaults(
        self,
        *,
        max_speed: float | None = None,
        mass: float | None = None,
        max_thrust: float | None = None,
        drag_coeff: float | None = None,
    ) -> None:
        if max_speed is not None:
            value = float(max_speed)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"max_speed must be finite and non-negative, got {max_speed!r}")
            self.config.max_speed = value
            self._base_max_speed = value
        if mass is not None:
            value = float(mass)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"mass must be positive and finite, got {mass!r}")
            self.config.physics.mass = value
            self._base_mass = value
        if max_thrust is not None:
            value = float(max_thrust)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"max_thrust must be finite and non-negative, got {max_thrust!r}")
            self.config.physics.max_thrust = value
            self._base_max_thrust = value
        if drag_coeff is not None:
            value = float(drag_coeff)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"drag_coeff must be finite and non-negative, got {drag_coeff!r}")
            self.config.physics.drag_coeff = value
            self._base_drag_coeff = value
        self._sync_runtime_params()
        self._state = None
        self._invalidate_export_cache()

    def set_episode_field_size(self, field_size: float) -> None:
        self._set_field_size(field_size, update_base=False, clip_current=False)

    def set_walls(self, walls) -> None:
        """Replace wall geometry and invalidate all spatial consumers."""

        rects = normalize_walls(walls)
        self._walls = rects
        self.sim.set_walls(rects)
        self._refresh_static_payloads()
        if self.oracle is not None:
            self.oracle.set_walls(rects)
        self._state = None
        self._invalidate_spatial_caches()

    def set_static_circles(self, circles) -> None:
        """Replace static circle obstacles used by physics, observations and spawn checks."""

        normalized: list[tuple[np.ndarray, float]] = []
        for pos, radius in circles or []:
            arr = np.asarray(pos, dtype=np.float32).reshape(-1)
            if arr.size < 2:
                continue
            normalized.append((arr[:2].copy(), float(radius)))
        self._static_circles = normalized
        self.sim.set_circle_obstacles(normalized)
        self._refresh_static_payloads()
        try:
            self.spawn._mask_cache.clear()
        except Exception:
            pass
        self._state = None
        self._invalidate_export_cache()

    def set_agent_positions(self, positions) -> None:
        arr = np.asarray(positions, dtype=np.float32)
        arr = arr.reshape((-1, 2))
        n = int(self.config.n_agents)
        if arr.shape[0] != n:
            raise ValueError(f"agent positions must have shape ({n}, 2), got {arr.shape}")
        self.sim.agents_pos = np.clip(arr, 0.0, float(self.config.field_size)).astype(np.float32)
        self._state = None
        self._invalidate_export_cache()

    def replace_threats(self, threats) -> None:
        from env.scenes.threats import is_dynamic_threat

        copied = _copy_threats(list(threats or []))
        self.sim.threats = copied
        self.sim.static_threats = [threat for threat in copied if not is_dynamic_threat(threat)]
        self.sim.dynamic_threats = [threat for threat in copied if is_dynamic_threat(threat)]
        self.sim._sync_threat_arrays()
        self._state = None
        self._invalidate_spatial_caches()

    def clear_threats(self) -> None:
        self.sim.threats = []
        self.sim.static_threats = []
        self.sim.dynamic_threats = []
        self.sim._sync_threat_arrays()
        self._state = None
        self._invalidate_spatial_caches()

    def add_threat_object(self, threat) -> None:
        self.sim.add_threat_obj(threat)
        self._state = None
        self._invalidate_spatial_caches()

    def add_static_threat(self, position, radius: float, intensity: float, *, oracle_block: bool = True) -> None:
        self.sim.add_threat(position, radius, intensity, oracle_block=oracle_block)
        self._state = None
        self._invalidate_spatial_caches()

    def sample_wind(self, agent_idx: int) -> np.ndarray | None:
        wind_field = getattr(self.sim, "_wind_field", None)
        if wind_field is None:
            return None
        idx = int(np.clip(agent_idx, 0, max(int(self.config.n_agents) - 1, 0)))
        pos = np.asarray(self.sim.agents_pos[idx], dtype=np.float32)
        sample = np.asarray(wind_field.sample(pos), dtype=np.float32)
        if sample.shape != (2,):
            return None
        return sample.copy()

    def set_threat_speed_scale(self, value: float) -> None:
        self.sim.threat_speed_scale = float(value)
        self._state = None
        self._invalidate_export_cache()

    def get_threat_speed_scale(self) -> float:
        return float(getattr(self.sim, "threat_speed_scale", 1.0))

    def configure_target_motion(self, cfg: dict | None) -> None:
        self._configure_target_motion(cfg)
        self._state = None
        self._invalidate_export_cache()

    def get_agent_positions(self, *, copy: bool = True) -> np.ndarray:
        arr = np.asarray(self.sim.agents_pos, dtype=np.float32)
        return arr.copy() if copy else arr

    def get_threats(self) -> tuple:
        return tuple(self.sim.threats or ())

    def get_static_walls(self) -> tuple[tuple[float, float, float, float], ...]:
        return tuple(tuple(map(float, wall)) for wall in (self._walls or ()))

    def get_static_circles(self) -> tuple[tuple[np.ndarray, float], ...]:
        return tuple((np.asarray(pos, dtype=np.float32).copy(), float(radius)) for pos, radius in self._static_circles)

    def get_time_step(self) -> int:
        return int(getattr(self.sim, "time_step", 0))

    def get_path_lengths(self, *, copy: bool = True) -> np.ndarray:
        arr = np.asarray(self._path_len, dtype=np.float32)
        return arr.copy() if copy else arr

    def capture_runtime_snapshot(self) -> RuntimeSnapshot:
        return RuntimeSnapshot(
            field_size=float(getattr(self.config, "field_size", 100.0)),
            max_steps=int(self.max_steps),
            target_pos=np.asarray(self.target_pos, dtype=np.float32).copy(),
            target_vel=np.asarray(self.target_vel, dtype=np.float32).copy(),
            target_motion=copy.deepcopy(self._target_motion),
            target_angle=float(self._target_angle),
            target_velocity_internal=np.asarray(self._target_vel, dtype=np.float32).copy(),
            walls=list(self.get_static_walls()),
            static_circles=[
                (np.asarray(pos, dtype=np.float32).copy(), float(radius))
                for pos, radius in self.get_static_circles()
            ],
            agents_pos=np.asarray(self.sim.agents_pos, dtype=np.float32).copy(),
            agents_vel=np.asarray(self.sim.agents_vel, dtype=np.float32).copy(),
            agents_active=np.asarray(self.sim.agents_active, dtype=bool).copy(),
            agent_state=np.asarray(self.sim.agent_state, dtype=np.int8).copy(),
            energy=np.asarray(self.sim.energy, dtype=np.float32).copy(),
            threats=_copy_threats(list(self.get_threats())),
            time_step=int(self.get_time_step()),
            threat_speed_scale=float(self.get_threat_speed_scale()),
            last_collision=np.asarray(self.sim.last_collision, dtype=bool).copy(),
            last_collision_speed=np.asarray(self.sim.last_collision_speed, dtype=np.float32).copy(),
            prev_dists=np.asarray(self.prev_dists, dtype=np.float32).copy(),
            was_alive=np.asarray(self.was_alive, dtype=bool).copy(),
            in_goal_steps=np.asarray(self.in_goal_steps, dtype=np.int32).copy(),
            finished=np.asarray(self.finished, dtype=bool).copy(),
            path_len=np.asarray(self._path_len, dtype=np.float32).copy(),
            threat_collisions=np.asarray(self._threat_collisions, dtype=np.int32).copy(),
            min_threat_dist=np.asarray(self._min_threat_dist, dtype=np.float32).copy(),
            death_step=np.asarray(self._death_step, dtype=np.int32).copy(),
            start_dists=None if self._start_dists is None else np.asarray(self._start_dists, dtype=np.float32).copy(),
            last_actions=np.asarray(self._last_actions, dtype=np.float32).copy(),
        )

    def restore_runtime_snapshot(self, snapshot: RuntimeSnapshot) -> None:
        field_size = float(snapshot.field_size)
        if abs(float(getattr(self.config, "field_size", field_size)) - field_size) > 1.0e-6:
            self._set_field_size(field_size, update_base=False, clip_current=False)
        self.max_steps = int(snapshot.max_steps)
        self.target_pos = np.asarray(snapshot.target_pos, dtype=np.float32).copy()
        self.target_vel = np.asarray(snapshot.target_vel, dtype=np.float32).copy()
        self._target_motion = copy.deepcopy(snapshot.target_motion)
        self._target_angle = float(snapshot.target_angle)
        self._target_vel = np.asarray(snapshot.target_velocity_internal, dtype=np.float32).copy()
        self._walls = [tuple(map(float, wall)) for wall in snapshot.walls]
        self._static_circles = [
            (np.asarray(pos, dtype=np.float32).copy(), float(radius))
            for pos, radius in snapshot.static_circles
        ]
        self.sim.set_walls(self._walls)
        self.sim.set_circle_obstacles(self._static_circles)
        self.sim.agents_pos = np.asarray(snapshot.agents_pos, dtype=np.float32).copy()
        self.sim.agents_vel = np.asarray(snapshot.agents_vel, dtype=np.float32).copy()
        self.sim.agents_active = np.asarray(snapshot.agents_active, dtype=bool).copy()
        self.sim.agent_state = np.asarray(snapshot.agent_state, dtype=np.int8).copy()
        self.sim.energy = np.asarray(snapshot.energy, dtype=np.float32).copy()
        self.sim.time_step = int(snapshot.time_step)
        self.sim.threat_speed_scale = float(snapshot.threat_speed_scale)
        self.sim.last_collision = np.asarray(snapshot.last_collision, dtype=bool).copy()
        self.sim.last_collision_speed = np.asarray(snapshot.last_collision_speed, dtype=np.float32).copy()
        self.replace_threats(snapshot.threats)
        self.prev_dists = np.asarray(snapshot.prev_dists, dtype=np.float32).copy()
        self.was_alive = np.asarray(snapshot.was_alive, dtype=bool).copy()
        self.in_goal_steps = np.asarray(snapshot.in_goal_steps, dtype=np.int32).copy()
        self.finished = np.asarray(snapshot.finished, dtype=bool).copy()
        self._path_len = np.asarray(snapshot.path_len, dtype=np.float32).copy()
        self._threat_collisions = np.asarray(snapshot.threat_collisions, dtype=np.int32).copy()
        self._min_threat_dist = np.asarray(snapshot.min_threat_dist, dtype=np.float32).copy()
        self._death_step = np.asarray(snapshot.death_step, dtype=np.int32).copy()
        self._start_dists = (
            None if snapshot.start_dists is None else np.asarray(snapshot.start_dists, dtype=np.float32).copy()
        )
        self._last_actions = np.asarray(snapshot.last_actions, dtype=np.float32).copy()
        self._refresh_static_payloads()
        self._state = None
        self._invalidate_spatial_caches()

    def get_reward_runtime_view(self):
        from env.rewards.dto import RewardRuntimeView

        n = int(self.config.n_agents)
        oracle = self.oracle
        optimal_len = np.full((n,), np.nan, dtype=np.float32)
        optimality_gap = np.full((n,), np.nan, dtype=np.float32)
        if oracle is not None:
            try:
                raw_len = np.asarray(getattr(oracle, "optimal_len", optimal_len), dtype=np.float32).reshape(-1)
                optimal_len[: min(n, raw_len.size)] = raw_len[:n]
            except Exception:
                pass
            for i in range(n):
                try:
                    optimality_gap[i] = float(oracle.optimality_gap(i))
                except Exception:
                    optimality_gap[i] = float("nan")
        return RewardRuntimeView(
            config=self.config,
            path_len=np.asarray(self._path_len, dtype=np.float32),
            optimal_len=optimal_len,
            optimality_gap=optimality_gap,
            threat_collisions=np.asarray(self._threat_collisions, dtype=np.int32),
            min_threat_dist=np.asarray(self._min_threat_dist, dtype=np.float32),
            death_step=np.asarray(self._death_step, dtype=np.int32),
            start_dists=None if self._start_dists is None else np.asarray(self._start_dists, dtype=np.float32),
            last_collision=np.asarray(getattr(self.sim, "last_collision", np.zeros((n,), dtype=bool)), dtype=bool),
        )

    def _set_field_size(self, field_size: float, *, update_base: bool = False, clip_current: bool = True) -> None:
        value = float(field_size)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"field_size must be a positive finite number, got {field_size!r}")
        self.config.field_size = value
        if update_base:
            self._base_field_size = value
        if clip_current:
            try:
                np.clip(self.target_pos, 0.0, value, out=self.target_pos)
            except Exception:
                pass
            try:
                np.clip(self.sim.agents_pos, 0.0, value, out=self.sim.agents_pos)
            except Exception:
                pass
        self._state = None
        self._invalidate_spatial_caches()

    def _restore_episode_defaults(self) -> None:
        self.max_steps = int(self._base_max_steps)
        base_field_size = float(self._base_field_size)
        if abs(float(getattr(self.config, "field_size", base_field_size)) - base_field_size) > 1e-6:
            self._set_field_size(base_field_size, update_base=False, clip_current=False)

    def _build_public_state(self, state: SimState, *, include_oracle: bool) -> PublicState:
        return public_state_from_state(state, include_oracle=include_oracle)

    def get_episode_summary(self):
        """Возвращает последнюю сводку эпизода (если она доступна)."""
        return getattr(self.event_stats, "last_summary", None)

    def _build_state(
        self,
        *,
        pos: np.ndarray | None = None,
        vel: np.ndarray | None = None,
        alive: np.ndarray | None = None,
        dists: np.ndarray | None = None,
        in_goal: np.ndarray | None = None,
        in_goal_steps: np.ndarray | None = None,
        finished: np.ndarray | None = None,
        newly_finished: np.ndarray | None = None,
        risk_p: np.ndarray | None = None,
        min_neighbor_dist: np.ndarray | None = None,
        last_action: np.ndarray | None = None,
        walls: np.ndarray | None = None,
        collision_speed: np.ndarray | None = None,
        timestep: int | None = None,
        threats: list | None = None,
        measured_accel: np.ndarray | None = None,
        energy: np.ndarray | None = None,
        energy_level: np.ndarray | None = None,
        agent_state: np.ndarray | None = None,
        decision_step: int | None = None,
    ) -> SimState:
        pos = np.asarray(pos if pos is not None else self.sim.agents_pos, dtype=np.float32)
        vel = np.asarray(vel if vel is not None else self.sim.agents_vel, dtype=np.float32)
        alive = np.asarray(alive if alive is not None else self.sim.agents_active, dtype=bool)
        if dists is None:
            dists = np.linalg.norm(pos - self.target_pos, axis=1).astype(np.float32)
        if in_goal is None:
            in_goal = (dists <= self.goal_radius) & alive
        if in_goal_steps is None:
            in_goal_steps = self.in_goal_steps.copy()
        if finished is None:
            finished = self.finished.copy()
        if newly_finished is None:
            newly_finished = np.zeros_like(alive, dtype=bool)
        if risk_p is None:
            risk_p = self._compute_risk_probs(pos, alive_override=alive)
        if min_neighbor_dist is None:
            min_neighbor_dist = self._compute_min_neighbor_dist(pos, alive)
        if last_action is None:
            last_action = self._last_actions.copy()
        if measured_accel is None:
            measured_accel = np.zeros_like(vel, dtype=np.float32)
        if energy is None:
            energy = self.sim.energy.copy()
        if energy_level is None:
            cap = float(getattr(self.config.battery, "capacity", 0.0))
            if cap > 0.0:
                energy_level = np.clip(energy / cap, 0.0, 1.0)
            else:
                energy_level = np.zeros_like(energy, dtype=np.float32)
        if agent_state is None:
            agent_state = self.sim.agent_state.copy()
        oracle_dir = None
        if self._should_materialize_oracle_dir():
            if hasattr(self.oracle, "direction_to_goal_batch"):
                oracle_dir = self.oracle.direction_to_goal_batch(pos)
            else:
                oracle_dir = np.zeros_like(pos, dtype=np.float32)
                for i in range(pos.shape[0]):
                    oracle_dir[i] = self.oracle.direction_to_goal(pos[i])
        if walls is None:
            if hasattr(self.sim, "get_wall_distances_batch"):
                walls = self.sim.get_wall_distances_batch()
            else:
                walls = np.zeros((self.sim.n, 4), dtype=np.float32)
                for i in range(self.sim.n):
                    walls[i] = self.sim.get_wall_distances(i)
        if collision_speed is None:
            collision_speed = self.sim.last_collision_speed.copy()
        if timestep is None:
            timestep = int(self.sim.time_step)
        if threats is None:
            threats = self.sim.threats
        if decision_step is None:
            decision_step = (
                int(getattr(self, "decision_loop", None).decision_step)
                if getattr(self, "decision_loop", None) is not None
                else 0
            )
        return SimState(
            pos=pos,
            vel=vel,
            alive=alive,
            target_pos=self.target_pos.copy(),
            target_vel=self.target_vel.copy(),
            timestep=int(timestep),
            threats=threats,
            dists=np.asarray(dists, dtype=np.float32),
            in_goal=np.asarray(in_goal, dtype=bool),
            in_goal_steps=np.asarray(in_goal_steps, dtype=np.int32),
            finished=np.asarray(finished, dtype=bool),
            newly_finished=np.asarray(newly_finished, dtype=bool),
            risk_p=np.asarray(risk_p, dtype=np.float32),
            min_neighbor_dist=np.asarray(min_neighbor_dist, dtype=np.float32),
            last_action=np.asarray(last_action, dtype=np.float32),
            walls=np.asarray(walls, dtype=np.float32),
            oracle_dir=oracle_dir,
            static_walls=self._static_walls_payload,
            static_circles=self._static_circles_payload,
            collision_speed=np.asarray(collision_speed, dtype=np.float32),
            field_size=float(self.config.field_size),
            max_speed=float(self.config.max_speed),
            max_accel=float(self._max_accel),
            dt=float(self._dt),
            drag=float(self._drag),
            grid_res=float(getattr(self.config, "grid_res", 1.0)),
            agent_radius=float(getattr(self.config, "agent_radius", 0.0)),
            wall_friction=float(getattr(self.config, "wall_friction", 0.0)),
            measured_accel=np.asarray(measured_accel, dtype=np.float32),
            energy=np.asarray(energy, dtype=np.float32),
            energy_level=np.asarray(energy_level, dtype=np.float32),
            agent_state=np.asarray(agent_state, dtype=np.int8),
            control_mode=str(getattr(self.config, "control_mode", "waypoint")),
            max_thrust=float(self._max_thrust),
            mass=float(self._mass),
            drag_coeff=float(self._drag_coeff),
            decision_step=int(decision_step),
            oracle_dist=(
                getattr(self.oracle, "dist_field", None)
                if getattr(self, "oracle", None) is not None and not self._runtime_is_fast()
                else None
            ),
            oracle_risk=(
                getattr(self.oracle, "risk_grid", None)
                if getattr(self, "oracle", None) is not None and not self._runtime_is_fast()
                else None
            ),
            oracle_risk_grad=(
                getattr(self.oracle, "risk_grad", None)
                if getattr(self, "oracle", None) is not None and not self._runtime_is_fast()
                else None
            ),
        )

    def _sync_shared_curriculum(self) -> None:
        if not self._shared_curriculum:
            return
        try:
            version = int(self._shared_curriculum.get("version", 0))
            if self._shared_curriculum_version == version:
                return
            params = dict(self._shared_curriculum.get("params") or {})
        except Exception:
            return
        if params:
            try:
                self.apply_curriculum(params)
            except Exception:
                pass
        self._shared_curriculum_version = version

    def apply_curriculum(self, stage_params: dict) -> None:
        """Обновляет конфиг среды на лету. Изменения применятся на следующем reset()."""
        apply_curriculum_params(self, stage_params, unsafe_keys=_UNSAFE_RUNTIME_CONFIG_KEYS)

    def _apply_rng_streams(self) -> None:
        """Подкладывает детерминированные RNG для подсистем."""
        self.rng = self.rng_registry.get("engine")
        self.sim.set_rng(self.rng_registry.get("physics"))
        self.spawn.set_rng(self.rng_registry.get("spawn"))
        self.scene.set_rng(self.rng_registry.get("scene"))

    def get_rng(self, name: str) -> np.random.Generator:
        return self.rng_registry.get(name)

    def _set_rng(self, seed: int) -> None:
        self.rng_registry.reset(int(seed))
        self._apply_rng_streams()

    def close(self) -> None:
        oracle = getattr(self, "oracle", None)
        if oracle is None or not hasattr(oracle, "close"):
            return
        oracle.close()

    def _refresh_static_payloads(self) -> None:
        self._static_walls_payload = [tuple(map(float, wall)) for wall in (self._walls or [])]
        circles_payload: list[tuple[float, float, float]] = []
        for c_pos, c_rad in (self._static_circles or []):
            arr = np.asarray(c_pos, dtype=np.float32)
            circles_payload.append((float(arr[0]), float(arr[1]), float(c_rad)))
        self._static_circles_payload = circles_payload

    def _sync_runtime_params(self) -> None:
        self._dt = float(getattr(self.config, "dt", 0.0))
        self._mass = float(getattr(self.config.physics, "mass", 1.0))
        self._max_thrust = float(getattr(self.config.physics, "max_thrust", 0.0))
        self._drag_coeff = float(getattr(self.config.physics, "drag_coeff", 0.0))
        self._max_accel = self._max_thrust / max(self._mass, 1e-6)
        self._drag = (self._drag_coeff / max(self._mass, 1e-6)) * self._dt
        self._battery_capacity = float(getattr(self.config.battery, "capacity", 0.0))

    def _configure_target_motion(self, cfg: dict | None) -> None:
        self._target_motion = None
        self._target_angle = 0.0
        self._target_vel = np.zeros(2, dtype=np.float32)
        self.target_vel = np.zeros(2, dtype=np.float32)
        if not cfg:
            return
        data = dict(cfg)
        mode = str(data.get("type", "linear")).lower()
        if mode in {"linear", "line"}:
            vel = data.get("vel") or data.get("velocity")
            if vel is None:
                speed = float(data.get("speed", 2.0))
                if data.get("direction") is not None:
                    vec = np.asarray(data.get("direction"), dtype=np.float32)
                    norm = float(np.linalg.norm(vec))
                    if norm > 1e-6:
                        vec = vec / norm
                    vel = (vec * speed).astype(np.float32)
                elif data.get("angle") is not None:
                    ang_raw = data.get("angle")
                    if isinstance(ang_raw, str) and ang_raw.lower() == "random":
                        rng_target = self.rng_registry.get("target")
                        ang = float(rng_target.uniform(0.0, 360.0))
                    else:
                        ang = float(ang_raw)
                    vel = np.array([np.cos(np.deg2rad(ang)), np.sin(np.deg2rad(ang))], dtype=np.float32) * speed
                else:
                    vel = np.array([speed, 0.0], dtype=np.float32)
            self._target_vel = np.asarray(vel, dtype=np.float32)
            self._target_motion = {"type": "linear"}
        elif mode in {"circle", "circular"}:
            center = data.get("center") or data.get("center_pos") or self.target_pos
            center = np.asarray(center, dtype=np.float32)
            radius = float(data.get("radius", 10.0))
            omega = data.get("angular_speed", data.get("omega", None))
            if omega is None and data.get("speed") is not None:
                omega = float(data.get("speed")) / max(radius, 1e-6)
            omega = float(omega if omega is not None else 0.4)
            phase = float(data.get("phase", 0.0))
            self._target_motion = {"type": "circle", "center": center, "radius": radius, "omega": omega}
            self._target_angle = phase
            self.target_pos = center + radius * np.array([np.cos(phase), np.sin(phase)], dtype=np.float32)
            self._target_vel = (
                np.array([-np.sin(phase), np.cos(phase)], dtype=np.float32) * float(omega) * float(radius)
            )
        self.target_vel = self._target_vel.copy()

    def _update_target_motion(self) -> None:
        if not self._target_motion:
            return
        dt = float(self.config.dt)
        mode = self._target_motion.get("type")
        if mode == "linear":
            pos = self.target_pos + (self._target_vel * dt)
            vel = self._target_vel.copy()
            margin = float(max(self.goal_radius, 0.0))
            field = float(self.config.field_size)
            for axis in (0, 1):
                if pos[axis] < margin:
                    pos[axis] = margin
                    vel[axis] = abs(vel[axis])
                elif pos[axis] > (field - margin):
                    pos[axis] = field - margin
                    vel[axis] = -abs(vel[axis])
            self._target_vel = vel
            self.target_pos = pos.astype(np.float32)
            self.target_vel = self._target_vel.copy()
        elif mode == "circle":
            center = np.asarray(self._target_motion.get("center", self.target_pos), dtype=np.float32)
            radius = float(self._target_motion.get("radius", 10.0))
            omega = float(self._target_motion.get("omega", 0.4))
            self._target_angle += omega * dt
            self.target_pos = center + radius * np.array(
                [np.cos(self._target_angle), np.sin(self._target_angle)],
                dtype=np.float32,
            )
            self._target_vel = (
                np.array(
                    [-np.sin(self._target_angle), np.cos(self._target_angle)],
                    dtype=np.float32,
                )
                * float(omega)
                * float(radius)
            )
            self.target_vel = self._target_vel.copy()
        if self.oracle_enabled:
            self.oracle.compute_lengths()

    def _compute_nearest_threat(self, positions: np.ndarray):
        n = positions.shape[0]
        if not self.sim.threats:
            return (
                np.full(n, np.inf, dtype=np.float32),
                np.full(n, -np.inf, dtype=np.float32),
                np.full(n, np.nan, dtype=np.float32),
            )
        dists = np.full(n, np.inf, dtype=np.float32)
        margins = np.full(n, -np.inf, dtype=np.float32)
        radii = np.full(n, np.nan, dtype=np.float32)
        for i in range(n):
            p = positions[i]
            best_d2 = np.inf
            best_margin = -np.inf
            best_radius = np.nan
            for t in self.sim.threats:
                diff = np.asarray(p, dtype=np.float32) - np.asarray(t.position, dtype=np.float32)
                d2 = float(diff[0] * diff[0] + diff[1] * diff[1])
                if d2 < best_d2:
                    best_d2 = d2
                    d = float(np.sqrt(d2))
                    best_margin = float(t.radius) - d
                    best_radius = float(t.radius)
            dists[i] = float(np.sqrt(best_d2)) if np.isfinite(best_d2) else float("inf")
            margins[i] = float(best_margin)
            radii[i] = float(best_radius)
        return dists, margins, radii

    def _compute_threat_metrics(
        self,
        positions: np.ndarray,
        alive_override: np.ndarray | None = None,
        *,
        with_meta: bool = True,
    ) -> tuple[np.ndarray, ThreatMeta | None]:
        n = positions.shape[0]
        risk = np.zeros(n, dtype=np.float32)
        if self.sim.threat_pos.size == 0:
            if with_meta:
                return (
                    risk,
                    ThreatMeta(
                        dist=np.full(n, np.inf, dtype=np.float32),
                        margin=np.full(n, -np.inf, dtype=np.float32),
                        radius=np.full(n, np.nan, dtype=np.float32),
                        intensity=np.full(n, np.nan, dtype=np.float32),
                        idx=np.full(n, -1, dtype=np.int32),
                        inside_any=np.zeros(n, dtype=bool),
                        inside_any_intensity_sum=np.zeros(n, dtype=np.float32),
                    ),
                )
            return risk, None

        t_pos = np.asarray(self.sim.threat_pos, dtype=np.float32)
        t_rad = np.asarray(self.sim.threat_radius, dtype=np.float32)
        t_int = np.asarray(self.sim.threat_intensity, dtype=np.float32)
        diff = positions[:, None, :] - t_pos[None, :, :]
        d2 = np.sum(diff * diff, axis=2)
        rad2 = t_rad[None, :] * t_rad[None, :]
        inside = d2 <= rad2
        inside_int = np.where(inside, t_int[None, :], 0.0)
        prod = np.prod(1.0 - inside_int, axis=1)
        risk = (1.0 - prod).astype(np.float32)
        if alive_override is not None:
            risk[~alive_override] = 0.0

        if not with_meta:
            return risk, None

        best_idx = np.argmin(d2, axis=1)
        rows = np.arange(n, dtype=np.int32)
        best_d2 = d2[rows, best_idx]
        best_d = np.sqrt(best_d2).astype(np.float32)
        best_radius = t_rad[best_idx].astype(np.float32)
        best_intensity = t_int[best_idx].astype(np.float32)
        best_margin = (best_radius - best_d).astype(np.float32)
        inside_any = np.any(inside, axis=1)
        inside_sum = (t_int[None, :] * inside).sum(axis=1).astype(np.float32)
        meta = ThreatMeta(
            dist=best_d.astype(np.float32),
            margin=best_margin.astype(np.float32),
            radius=best_radius,
            intensity=best_intensity,
            idx=best_idx.astype(np.int32),
            inside_any=inside_any.astype(bool),
            inside_any_intensity_sum=inside_sum,
        )
        return risk, meta

    def _compute_threat_meta(self, positions: np.ndarray) -> ThreatMeta:
        _, meta = self._compute_threat_metrics(positions, alive_override=None, with_meta=True)
        assert meta is not None
        return meta

    def _compute_risk_probs(self, positions: np.ndarray, alive_override: np.ndarray | None = None) -> np.ndarray:
        risk, _meta = self._compute_threat_metrics(positions, alive_override=alive_override, with_meta=False)
        return risk

    def _compute_min_neighbor_dist(self, positions: np.ndarray, alive: np.ndarray) -> np.ndarray:
        n = positions.shape[0]
        out = np.full(n, np.inf, dtype=np.float32)
        idx = np.where(alive)[0]
        if idx.size <= 1:
            return out

        pts = positions[idx]
        diff = pts[:, None, :] - pts[None, :, :]
        d2 = np.sum(diff * diff, axis=2)
        np.fill_diagonal(d2, np.inf)
        mins = np.min(d2, axis=1)
        out[idx] = np.sqrt(mins).astype(np.float32)
        return out
