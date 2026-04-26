from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

from baselines.mpc_support import (
    dist_from_info as _dist_from_info,
    extract_plan_action as _extract_plan_action,
    finalize_batch_waypoint_actions,
)
from baselines.policies import ObsDict, PlannerPolicy
from baselines.planner_fallbacks import create_astar_fallback
from baselines.utils import (
    agent_index_from_id,
    normalize,
    obs_to_target,
    obs_vel,
    predict_target_single,
)
from common.physics.model import apply_accel_dynamics_step, apply_accel_dynamics_vel, inverse_accel_for_velocity
from common.physics.walls import resolve_wall_slide as _resolve_wall_slide
from common.physics.walls import resolve_wall_slide_batch as _resolve_wall_slide_batch
from common.physics.walls_numba import circle_hits_any_numba as _circle_hits_any_numba
from common.physics.walls_numba import circle_rect_normal_numba as _circle_rect_normal_numba
from common.physics.walls_numba import resolve_wall_slide_numba as _resolve_wall_slide_numba
from common.policy.context import PolicyContext

try:  # Опциональное ускорение через Numba.
    from numba import njit
except Exception:  # pragma: no cover - опциональная зависимость
    njit = None

_NUMBA_AVAILABLE = njit is not None
logger = logging.getLogger(__name__)


@dataclass
class RolloutContext:
    pos0: np.ndarray
    vel0: np.ndarray
    dist0: float
    dt: float
    max_speed: float
    max_accel: float
    drag: float
    field_size: float | None
    steps: int
    walls: list | None
    wall_arr: np.ndarray | None
    wall_min: np.ndarray | None
    wall_max: np.ndarray | None
    has_walls: bool
    radius: float
    friction: float
    t_pos: np.ndarray | None
    t_rad: np.ndarray | None
    t_int: np.ndarray | None
    scale: np.ndarray | None
    has_threats: bool
    others: np.ndarray | None
    has_others: bool


@dataclass
class SharedRolloutContext:
    walls: list | None
    wall_arr: np.ndarray | None
    wall_min: np.ndarray | None
    wall_max: np.ndarray | None
    has_walls: bool
    t_pos: np.ndarray | None
    t_rad: np.ndarray | None
    t_int: np.ndarray | None
    scale: np.ndarray | None
    has_threats: bool


_rollout_scores_numba = None
_rollout_scores_numba_walls = None
_NUMBA_WARMED = False

if _NUMBA_AVAILABLE:

    @njit(cache=True, fastmath=True)
    def _rollout_scores_numba(
        pos0,
        vel0,
        actions,
        steps,
        target_pos,
        dist0,
        dt,
        max_speed,
        max_accel,
        drag,
        field_size,
        use_field,
        t_pos,
        t_rad,
        t_int,
        scale,
        has_threats,
        risk_mode_binary,
        safety_first,
        safety_margin,
        others,
        has_others,
        w_progress,
        w_risk,
        risk_hard_penalty,
        w_wall,
        w_collision,
        idle_penalty,
    ):
        n_actions = actions.shape[0]
        pos = np.empty((n_actions, 2), dtype=np.float32)
        vel = np.empty((n_actions, 2), dtype=np.float32)
        for i in range(n_actions):
            pos[i, 0] = pos0[0]
            pos[i, 1] = pos0[1]
            vel[i, 0] = vel0[0]
            vel[i, 1] = vel0[1]

        risk_sum = np.zeros(n_actions, dtype=np.float32)
        wall_pen = np.zeros(n_actions, dtype=np.float32)
        coll_pen = np.zeros(n_actions, dtype=np.float32)
        safe_hit = np.zeros(n_actions, dtype=np.uint8)
        inside_any = np.zeros(n_actions, dtype=np.uint8)

        for _ in range(steps):
            for a in range(n_actions):
                vx, vy = apply_accel_dynamics_step(
                    vel[a, 0],
                    vel[a, 1],
                    actions[a, 0] * max_accel,
                    actions[a, 1] * max_accel,
                    dt,
                    drag,
                    max_speed,
                    0.0,
                    0.0,
                )
                pos[a, 0] += vx * dt
                pos[a, 1] += vy * dt
                if use_field:
                    if pos[a, 0] < 0.0:
                        pos[a, 0] = 0.0
                    elif pos[a, 0] > field_size:
                        pos[a, 0] = field_size
                    if pos[a, 1] < 0.0:
                        pos[a, 1] = 0.0
                    elif pos[a, 1] > field_size:
                        pos[a, 1] = field_size
                vel[a, 0] = vx
                vel[a, 1] = vy

            if has_threats:
                for a in range(n_actions):
                    prod = 1.0
                    inside = False
                    safe = False
                    for t in range(t_pos.shape[0]):
                        dx = pos[a, 0] - t_pos[t, 0]
                        dy = pos[a, 1] - t_pos[t, 1]
                        dist = math.sqrt(dx * dx + dy * dy)
                        margin = dist - t_rad[t]
                        if margin <= 0.0:
                            p = 1.0
                            inside = True
                        else:
                            if risk_mode_binary:
                                p = 0.0
                            else:
                                s = scale[t]
                                if s <= 1e-6:
                                    p = 0.0
                                else:
                                    p = math.exp(-margin / s)
                        if safety_first and safety_margin > 0.0:
                            if dist <= (t_rad[t] + safety_margin):
                                safe = True
                        p = p * t_int[t]
                        if p > 1.0:
                            p = 1.0
                        prod *= 1.0 - p
                    risk = 1.0 - prod
                    risk_sum[a] += risk
                    if safe:
                        safe_hit[a] = 1
                    if inside:
                        inside_any[a] = 1

            if use_field:
                for a in range(n_actions):
                    size = field_size
                    d_left = pos[a, 0] / size
                    d_right = (size - pos[a, 0]) / size
                    d_down = pos[a, 1] / size
                    d_up = (size - pos[a, 1]) / size
                    min_wall = d_left
                    if d_right < min_wall:
                        min_wall = d_right
                    if d_down < min_wall:
                        min_wall = d_down
                    if d_up < min_wall:
                        min_wall = d_up
                    if min_wall < 0.05:
                        wall_pen[a] += 0.05 - min_wall

            if has_others:
                for a in range(n_actions):
                    min_dist = 1.0e9
                    for j in range(others.shape[0]):
                        dx = pos[a, 0] - others[j, 0]
                        dy = pos[a, 1] - others[j, 1]
                        dist = math.sqrt(dx * dx + dy * dy)
                        if dist < min_dist:
                            min_dist = dist
                    if min_dist < 1.5:
                        coll_pen[a] += 1.5 - min_dist

        scores = np.empty(n_actions, dtype=np.float32)
        for a in range(n_actions):
            dx = pos[a, 0] - target_pos[0]
            dy = pos[a, 1] - target_pos[1]
            dist1 = math.sqrt(dx * dx + dy * dy)
            progress = dist0 - dist1
            hard = 1.0 if inside_any[a] == 1 else 0.0
            score = (
                w_progress * progress
                - w_risk * (risk_sum[a] / steps)
                - risk_hard_penalty * hard
                - w_wall * (wall_pen[a] / steps)
                - w_collision * (coll_pen[a] / steps)
            )
            ax = actions[a, 0]
            ay = actions[a, 1]
            if math.sqrt(ax * ax + ay * ay) <= 1e-3:
                score -= idle_penalty
            scores[a] = score
        return scores, safe_hit


class MPCLitePolicy(PlannerPolicy):
    """Упрощённый MPC с дискретным перебором действий и коротким роллаутом.

    Цель — дать быстрый и воспроизводимый бейзлайн без тяжёлой симуляции.
    """

    def __init__(
        self,
        horizon: int = 2,
        n_directions: int = 16,
        accel_levels: tuple[float, ...] | None = None,
        speed_levels: tuple[float, ...] | None = None,
        use_numba: bool = True,
        w_progress: float = 5.0,
        w_risk: float = 6.0,
        w_wall: float = 6.0,
        w_collision: float = 0.0,
        risk_hard_penalty: float = 6.0,
        risk_mode: str = "soft",
        risk_soft_margin: float = 3.0,
        risk_soft_radius_scale: float = 0.6,
        safety_first: bool = True,
        safety_margin: float = 2.5,
        safety_min_score: float = -0.5,
        idle_penalty: float = 0.2,
        jerk_penalty: float = 0.15,
        terminal_speed_weight: float = 0.1,
        terminal_brake_radius: float = 6.0,
        threat_predict_time: float = 0.6,
        threat_predict_inflate: float = 0.6,
        cem_target_velocity: bool = True,
        cem_tau: float = 0.35,
        fallback_astar: bool = True,
        fallback_score: float = -0.2,
        fallback_cooldown: int = 3,
        fallback_confirm_steps: int = 2,
        stuck_steps: int = 20,
        stuck_dist_eps: float = 0.5,
        dynamic_horizon: bool = True,
        horizon_max: int = 6,
        horizon_speed_scale: float = 3.0,
        intercept_enabled: bool = True,
        intercept_gain: float = 0.6,
        intercept_max_time: float = 3.0,
        cem_enabled: bool = False,
        cem_iters: int = 2,
        cem_samples: int = 24,
        cem_elite: int = 6,
        cem_sigma: float = 0.7,
        cem_min_sigma: float = 0.1,
        cem_include_zero: bool = True,
        cem_seed: int | None = None,
        **_,
    ):
        self.horizon = int(max(1, horizon))
        self.use_numba = bool(use_numba)
        self.n_directions = int(max(4, n_directions))
        accel = (1.0,) if accel_levels is None else tuple(float(x) for x in accel_levels)
        accel = tuple(sorted({max(0.0, min(1.0, lv)) for lv in accel}))
        if not accel:
            accel = (1.0,)
        self.accel_levels = accel
        speed = accel if speed_levels is None else tuple(float(x) for x in speed_levels)
        speed = tuple(sorted({max(0.0, min(1.0, lv)) for lv in speed}))
        if not speed:
            speed = accel
        self.speed_levels = speed
        self.w_progress = float(w_progress)
        self.w_risk = float(w_risk)
        self.w_wall = float(w_wall)
        self.w_collision = float(w_collision)
        self.risk_hard_penalty = float(risk_hard_penalty)
        self.risk_mode = str(risk_mode)
        self.risk_soft_margin = float(risk_soft_margin)
        self.risk_soft_radius_scale = float(risk_soft_radius_scale)
        self.safety_first = bool(safety_first)
        self.safety_margin = float(safety_margin)
        self.safety_min_score = float(safety_min_score)
        self.idle_penalty = float(idle_penalty)
        self.jerk_penalty = float(jerk_penalty)
        self.terminal_speed_weight = float(terminal_speed_weight)
        self.terminal_brake_radius = float(terminal_brake_radius)
        self.threat_predict_time = float(threat_predict_time)
        self.threat_predict_inflate = float(threat_predict_inflate)
        self.cem_target_velocity = bool(cem_target_velocity)
        self.cem_tau = float(cem_tau)
        self.fallback_score = float(fallback_score)
        self.fallback_cooldown = int(max(0, fallback_cooldown))
        self.fallback_confirm_steps = int(max(1, fallback_confirm_steps))
        self.stuck_steps = int(max(0, stuck_steps))
        self.stuck_dist_eps = float(stuck_dist_eps)
        self.dynamic_horizon = bool(dynamic_horizon)
        self.horizon_max = int(max(self.horizon, horizon_max))
        self.horizon_speed_scale = float(horizon_speed_scale)
        self.intercept_enabled = bool(intercept_enabled)
        self.intercept_gain = float(intercept_gain)
        self.intercept_max_time = float(intercept_max_time)
        self.cem_enabled = bool(cem_enabled)
        self.cem_iters = int(max(1, cem_iters))
        self.cem_samples = int(max(4, cem_samples))
        self.cem_elite = int(max(1, cem_elite))
        self.cem_sigma = float(cem_sigma)
        self.cem_min_sigma = float(cem_min_sigma)
        self.cem_include_zero = bool(cem_include_zero)
        self._rng = np.random.default_rng(cem_seed)
        self._astar_fallback = None
        if fallback_astar:
            self._astar_fallback = create_astar_fallback()
        self._state: PolicyContext | None = None
        self._target_pos = None
        self._field_size = None
        self._max_speed = None
        self._max_accel = None
        self._drag = None
        self._dt = None
        self._agent_radius = 0.0
        self._wall_friction = 0.0
        self._walls = None
        self._agent_state = {}
        self._fallback_cooldowns: dict[int, int] = {}
        self._fallback_low_score_counts: dict[int, int] = {}
        self._fallback_stats: dict[str, int] = {}
        self._soft_failures: dict[str, int] = {}

        angles = np.linspace(0.0, 2.0 * np.pi, self.n_directions, endpoint=False)
        dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(np.float32)

        def _build_actions(levels: tuple[float, ...]) -> np.ndarray:
            actions = []
            for lv in levels:
                if lv <= 0.0:
                    continue
                actions.append(dirs * lv)
            if actions:
                actions_arr = np.vstack(actions)
            else:
                actions_arr = np.zeros((0, 2), dtype=np.float32)
            return np.vstack([actions_arr, np.zeros((1, 2), dtype=np.float32)])

        self._actions_accel = _build_actions(self.accel_levels)
        self._actions_vel = _build_actions(self.speed_levels)
        self._empty_vec2 = np.zeros((0, 2), dtype=np.float32)
        self._empty_vec1 = np.zeros((0,), dtype=np.float32)
        if _NUMBA_AVAILABLE and self.use_numba:
            self._warmup_numba()

    def _warmup_numba(self) -> None:
        global _NUMBA_WARMED
        if _NUMBA_WARMED or not _NUMBA_AVAILABLE:
            return
        try:
            actions = np.zeros((1, 2), dtype=np.float32)
            pos0 = np.zeros((2,), dtype=np.float32)
            vel0 = np.zeros((2,), dtype=np.float32)
            target_pos = np.zeros((2,), dtype=np.float32)
            t_pos = np.zeros((1, 2), dtype=np.float32)
            t_rad = np.zeros((1,), dtype=np.float32)
            t_int = np.zeros((1,), dtype=np.float32)
            scale = np.zeros((1,), dtype=np.float32)
            others = np.zeros((1, 2), dtype=np.float32)
            _rollout_scores_numba(
                pos0,
                vel0,
                actions,
                1,
                target_pos,
                0.0,
                0.1,
                1.0,
                1.0,
                0.0,
                50.0,
                1,
                t_pos,
                t_rad,
                t_int,
                scale,
                0,
                0,
                0,
                0.0,
                others,
                0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                0.0,
            )
            walls = np.zeros((1, 4), dtype=np.float32)
            _rollout_scores_numba_walls(
                pos0,
                vel0,
                actions,
                1,
                target_pos,
                0.0,
                0.1,
                1.0,
                1.0,
                0.0,
                50.0,
                1,
                walls,
                0,
                1.0,
                0.0,
                t_pos,
                t_rad,
                t_int,
                scale,
                0,
                0,
                0,
                0.0,
                others,
                0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                0.0,
            )
            _NUMBA_WARMED = True
        except Exception:
            return

    def set_context(self, state: PolicyContext | None = None) -> None:
        self._state = state
        if state is None:
            return
        self._target_pos = np.asarray(state.target_pos, dtype=np.float32)
        self._field_size = float(state.field_size)
        self._max_speed = float(state.max_speed)
        self._walls = list(state.static_walls) if state.static_walls is not None else None
        self._dt = float(state.dt)
        self._agent_radius = float(state.agent_radius)
        self._wall_friction = float(state.wall_friction)
        self._max_accel = float(state.max_accel)
        self._drag = float(state.drag)
        if self._astar_fallback is not None:
            if hasattr(self._astar_fallback, "set_context"):
                self._astar_fallback.set_context(state)
            try:
                grid_res = float(state.grid_res)
                self._astar_fallback.cell_size = grid_res
                if self._astar_fallback.global_plan_cell_size is not None:
                    self._astar_fallback.global_plan_cell_size = max(grid_res * 2.0, grid_res)
            except Exception as exc:
                self._record_soft_failure("fallback_context", exc)

    def reset(self, seed: int | None = None) -> None:
        self._agent_state = {}
        self._fallback_cooldowns = {}
        self._fallback_low_score_counts = {}
        self._fallback_stats = {}
        self._soft_failures = {}
        if self._astar_fallback is not None and hasattr(self._astar_fallback, "reset"):
            self._astar_fallback.reset(seed)
        return

    def _record_soft_failure(self, key: str, exc: Exception) -> None:
        count = int(self._soft_failures.get(key, 0)) + 1
        self._soft_failures[key] = count
        if count == 1:
            logger.warning("%s soft fallback [%s]: %s", self.__class__.__name__, key, exc)

    def _record_fallback_stat(self, key: str) -> None:
        self._fallback_stats[key] = int(self._fallback_stats.get(key, 0)) + 1

    def _fallback_ready(self, idx: int) -> bool:
        if self._astar_fallback is None:
            return False
        if self.fallback_cooldown <= 0:
            return True
        cooldown = int(self._fallback_cooldowns.get(idx, 0))
        if cooldown > 0:
            self._fallback_cooldowns[idx] = cooldown - 1
            self._record_fallback_stat("cooldown_block")
            return False
        return True

    def _arm_fallback_cooldown(self, idx: int) -> None:
        if self.fallback_cooldown > 0:
            self._fallback_cooldowns[idx] = self.fallback_cooldown

    def _should_fallback_on_score(self, idx: int, chosen_score: float) -> bool:
        if chosen_score >= self.fallback_score:
            self._fallback_low_score_counts[idx] = 0
            return False
        count = int(self._fallback_low_score_counts.get(idx, 0)) + 1
        self._fallback_low_score_counts[idx] = count
        if count < self.fallback_confirm_steps:
            self._record_fallback_stat("low_score_deferred")
            return False
        self._fallback_low_score_counts[idx] = 0
        return True

    def _fallback_action(
        self,
        reason: str,
        agent_id: str,
        idx: int,
        obs: ObsDict,
        state: PolicyContext,
        info: dict,
        target_pos: np.ndarray | None,
    ) -> np.ndarray | None:
        if self._astar_fallback is None:
            return None
        fb_info = dict(info)
        if target_pos is not None:
            fb_info["target_pos"] = target_pos
        self._arm_fallback_cooldown(idx)
        self._record_fallback_stat(f"used_{reason}")
        return _extract_plan_action(self._astar_fallback.plan(agent_id, obs, state, fb_info))

    def _prepare_threats(
        self,
        threats: list | None,
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None, bool]:
        if not threats:
            return None, None, None, None, False
        t_pos = np.asarray([t.position for t in threats], dtype=np.float32)
        t_rad = np.asarray([t.radius for t in threats], dtype=np.float32)
        t_int = np.asarray([t.intensity for t in threats], dtype=np.float32)
        if self.threat_predict_time > 0.0:
            t_vel = np.zeros_like(t_pos, dtype=np.float32)
            for i, t in enumerate(threats):
                vel = getattr(t, "velocity", None)
                if vel is None:
                    vel = getattr(t, "vel", None)
                if vel is None:
                    continue
                t_vel[i] = np.asarray(vel, dtype=np.float32)
            lead = float(self.threat_predict_time)
            t_pos = t_pos + (t_vel * lead)
            t_rad = t_rad + (np.linalg.norm(t_vel, axis=1) * lead * float(self.threat_predict_inflate))
        if self.risk_mode == "binary":
            scale = None
        else:
            scale = np.maximum(self.risk_soft_margin, self.risk_soft_radius_scale * t_rad)
            scale = np.maximum(scale, 1e-3).astype(np.float32)
        return t_pos, t_rad, t_int, scale, True

    def _prepare_others(self, state: PolicyContext, idx: int) -> tuple[np.ndarray | None, bool]:
        if self.w_collision <= 0.0:
            return None, False
        if not bool(state.alive[idx]):
            return None, False
        mask = np.ones(state.pos.shape[0], dtype=bool)
        mask[idx] = False
        mask &= np.asarray(state.alive, dtype=bool)
        if np.any(mask):
            return state.pos[mask].astype(np.float32), True
        return None, False

    def _prepare_walls(
        self, walls: list | None
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, bool]:
        if not walls:
            return None, None, None, False
        try:
            wall_arr = np.asarray(walls, dtype=np.float32)
        except Exception:
            return None, None, None, False
        if wall_arr.ndim != 2 or wall_arr.shape[1] < 4 or wall_arr.shape[0] == 0:
            return None, None, None, False
        wall_min = wall_max = None
        if _resolve_wall_slide is not None:
            wall_min = wall_arr[:, :2].min(axis=0)
            wall_max = wall_arr[:, 2:4].max(axis=0)
        return wall_arr, wall_min, wall_max, True

    def _prepare_shared_context(self, state: PolicyContext) -> SharedRolloutContext:
        walls = self._walls if self._walls is not None else list(state.static_walls or [])
        wall_arr, wall_min, wall_max, has_walls = self._prepare_walls(walls)
        t_pos, t_rad, t_int, scale, has_threats = self._prepare_threats(list(state.threats or []))
        return SharedRolloutContext(
            walls=walls,
            wall_arr=wall_arr,
            wall_min=wall_min,
            wall_max=wall_max,
            has_walls=has_walls,
            t_pos=t_pos,
            t_rad=t_rad,
            t_int=t_int,
            scale=scale,
            has_threats=has_threats,
        )

    def _prepare_rollout_context(
        self,
        idx: int,
        target_pos: np.ndarray,
        horizon: int,
        shared: SharedRolloutContext | None = None,
    ) -> RolloutContext | None:
        state = self._state
        if state is None:
            return None
        pos0 = state.pos[idx].astype(np.float32)
        try:
            vel0 = np.asarray(state.vel[idx], dtype=np.float32)
        except Exception:
            vel0 = np.zeros((2,), dtype=np.float32)
        dist0 = float(np.linalg.norm(pos0 - target_pos))
        dt = float(self._dt) if self._dt is not None else 1.0
        max_speed = float(self._max_speed) if self._max_speed is not None else 0.0
        max_accel = float(self._max_accel) if self._max_accel is not None else float(state.max_accel)
        drag = float(self._drag) if self._drag is not None else float(state.drag)
        field_size = float(self._field_size) if self._field_size is not None else None
        steps = max(1, int(horizon))
        if shared is None:
            shared = self._prepare_shared_context(state)
        walls = shared.walls
        wall_arr = shared.wall_arr
        wall_min = shared.wall_min
        wall_max = shared.wall_max
        has_walls = shared.has_walls
        t_pos = shared.t_pos
        t_rad = shared.t_rad
        t_int = shared.t_int
        scale = shared.scale
        has_threats = shared.has_threats
        others, has_others = self._prepare_others(state, idx)
        return RolloutContext(
            pos0=pos0,
            vel0=vel0,
            dist0=dist0,
            dt=dt,
            max_speed=max_speed,
            max_accel=max_accel,
            drag=drag,
            field_size=field_size,
            steps=steps,
            walls=walls,
            wall_arr=wall_arr,
            wall_min=wall_min,
            wall_max=wall_max,
            has_walls=has_walls,
            radius=float(self._agent_radius),
            friction=float(self._wall_friction),
            t_pos=t_pos,
            t_rad=t_rad,
            t_int=t_int,
            scale=scale,
            has_threats=has_threats,
            others=others,
            has_others=has_others,
        )

    def plan(
        self,
        agent_id: str,
        obs: ObsDict,
        state: PolicyContext,
        info: dict | None = None,
    ) -> np.ndarray:
        self.set_context(state)
        if self._state is None:
            return np.zeros((2,), dtype=np.float32)
        info = dict(self._ensure_info(info, state) or {})
        info.setdefault("target_pos", state.target_pos)
        info.setdefault("target_vel", state.target_vel)
        idx = agent_index_from_id(agent_id, info)
        if idx is None:
            return np.zeros((2,), dtype=np.float32)
        idx = int(idx)
        if 0 <= idx < state.pos.shape[0]:
            info.setdefault("pos", state.pos[idx])
        info.setdefault("max_speed", state.max_speed)
        info.setdefault("max_accel", state.max_accel)
        info.setdefault("dt", state.dt)
        info.setdefault("drag", state.drag)
        info.setdefault("grid_res", state.grid_res)
        if "target_pos" in info:
            try:
                self._target_pos = np.asarray(info["target_pos"], dtype=np.float32)
            except Exception as exc:
                self._record_soft_failure("target_pos_info", exc)
        target_pos = self._target_pos
        if target_pos is None:
            target_pos = None
        else:
            target_vel = info.get("target_vel")
            if target_vel is not None:
                try:
                    target_pos = predict_target_single(
                        target_pos,
                        target_vel,
                        pos=info.get("pos"),
                        max_speed=self._max_speed,
                        gain=self.intercept_gain,
                        max_time=self.intercept_max_time,
                        enabled=self.intercept_enabled,
                    )
                except Exception as exc:
                    self._record_soft_failure("target_predict", exc)
                    target_pos = self._target_pos
        stuck = self._is_stuck(idx, info, obs)
        allow_fallback = self._fallback_ready(idx)
        if allow_fallback and stuck:
            # Если агент застрял, переключаемся на локальный A* как на «спасательный» план.
            fallback_action = self._fallback_action("stuck", agent_id, idx, obs, state, info, target_pos)
            if fallback_action is not None:
                return fallback_action
        if stuck:
            oracle_action = self._oracle_direction_action(idx, obs, state, info)
            if oracle_action is not None:
                return oracle_action
        if target_pos is None:
            return normalize(obs_to_target(obs))

        horizon = self._effective_horizon(obs)
        actions_mode = "velocity"
        if self.cem_enabled:
            chosen_action, chosen_score = self._choose_action_cem(
                idx,
                horizon,
                target_pos,
                obs,
                info,
                actions_mode=actions_mode,
            )
        else:
            chosen_action, chosen_score = self._choose_action_discrete(
                idx,
                horizon,
                target_pos,
                actions_mode=actions_mode,
            )

        if allow_fallback and self._should_fallback_on_score(idx, float(chosen_score)):
            # Если оценка слабая, доверяемся A* как более устойчивому планировщику.
            fallback_action = self._fallback_action("low_score", agent_id, idx, obs, state, info, target_pos)
            if fallback_action is not None:
                return fallback_action
        if chosen_score < self.fallback_score:
            oracle_action = self._oracle_direction_action(idx, obs, state, info)
            if oracle_action is not None:
                return oracle_action
        else:
            self._fallback_low_score_counts[idx] = 0
        state = self._agent_state.setdefault(idx, {"best_dist": None, "since_improve": 0, "last_action": None})
        state["last_action"] = np.asarray(chosen_action, dtype=np.float32)
        pos = info.get("pos")
        if pos is not None and target_pos is not None:
            try:
                to_target = np.asarray(target_pos, dtype=np.float32) - np.asarray(pos, dtype=np.float32)
                return chosen_action.astype(np.float32), {"to_target": to_target}
            except Exception as exc:
                self._record_soft_failure("plan_tuple", exc)
        return chosen_action.astype(np.float32)

    def step_batch(
        self,
        obs_map: dict[str, ObsDict],
        state: PolicyContext,
        infos: dict | None = None,
    ) -> dict[str, np.ndarray]:
        return self.get_actions(obs_map, state, infos)

    def get_actions(self, obs_dict: dict[str, ObsDict], state: PolicyContext, infos_dict: dict | None = None) -> dict:
        self.set_context(state)
        if self._state is None:
            return {agent_id: np.zeros((2,), dtype=np.float32) for agent_id in obs_dict}
        infos_dict = infos_dict or {}
        shared = self._prepare_shared_context(state)
        agent_ids = list(obs_dict.keys())
        n = len(agent_ids)
        desired = np.zeros((n, 2), dtype=np.float32)
        to_target = np.zeros((n, 2), dtype=np.float32)
        dist_m = np.full((n,), np.nan, dtype=np.float32)
        in_goal = np.zeros((n,), dtype=bool)
        risk_p = np.full((n,), np.nan, dtype=np.float32)
        cur_vel = np.zeros((n, 2), dtype=np.float32)
        obs_list: list[ObsDict] = []
        info_list: list[dict] = []

        for i, agent_id in enumerate(agent_ids):
            obs = obs_dict[agent_id]
            info = dict(self._ensure_info(infos_dict.get(agent_id), state) or {})
            info_list.append(info)
            obs_list.append(obs)
            idx = agent_index_from_id(agent_id, info)
            if idx is None:
                continue
            idx = int(idx)
            if 0 <= idx < state.pos.shape[0]:
                info.setdefault("pos", state.pos[idx])
            info.setdefault("target_pos", state.target_pos)
            info.setdefault("target_vel", state.target_vel)
            info.setdefault("max_speed", state.max_speed)
            info.setdefault("max_accel", state.max_accel)
            info.setdefault("dt", state.dt)
            info.setdefault("drag", state.drag)
            info.setdefault("grid_res", state.grid_res)
            if state.dists is not None and 0 <= idx < state.dists.shape[0]:
                dist_m[i] = float(state.dists[idx])
            if state.in_goal is not None and 0 <= idx < state.in_goal.shape[0]:
                in_goal[i] = bool(state.in_goal[idx])
            if state.risk_p is not None and 0 <= idx < state.risk_p.shape[0]:
                risk_p[i] = float(state.risk_p[idx])
            if getattr(state, "vel", None) is not None and 0 <= idx < state.vel.shape[0]:
                cur_vel[i] = state.vel[idx]
            else:
                cur_vel[i] = obs_vel(obs) * float(getattr(state, "max_speed", 0.0))

            target_pos = info.get("target_pos")
            if target_pos is None:
                target_pos = self._target_pos
            else:
                target_pos = np.asarray(target_pos, dtype=np.float32)
            if target_pos is None:
                desired[i] = normalize(obs_to_target(obs))
                to_target[i] = obs_to_target(obs)
                continue
            stuck = self._is_stuck(idx, info, obs)
            allow_fallback = self._fallback_ready(idx)
            if allow_fallback and stuck:
                fallback_desired = self._fallback_action("stuck", agent_id, idx, obs, state, info, target_pos)
                if fallback_desired is not None:
                    desired[i] = fallback_desired.astype(np.float32)
                    to_target[i] = np.asarray(target_pos, dtype=np.float32) - np.asarray(info["pos"], dtype=np.float32)
                    continue
            if stuck:
                oracle_action = self._oracle_direction_action(idx, obs, state, info)
                if oracle_action is not None:
                    desired[i] = oracle_action.astype(np.float32)
                    to_target[i] = np.asarray(target_pos, dtype=np.float32) - np.asarray(info["pos"], dtype=np.float32)
                    continue

            horizon = self._effective_horizon(obs)
            ctx = self._prepare_rollout_context(idx, target_pos, horizon, shared=shared)
            if ctx is None:
                continue
            actions_mode = "velocity"
            if self.cem_enabled:
                chosen_action, chosen_score = self._choose_action_cem(
                    idx,
                    horizon,
                    target_pos,
                    obs,
                    info,
                    ctx=ctx,
                    actions_mode=actions_mode,
                )
            else:
                chosen_action, chosen_score = self._choose_action_discrete(
                    idx,
                    horizon,
                    target_pos,
                    ctx=ctx,
                    actions_mode=actions_mode,
                )

            if allow_fallback and self._should_fallback_on_score(idx, float(chosen_score)):
                fallback_desired = self._fallback_action("low_score", agent_id, idx, obs, state, info, target_pos)
                if fallback_desired is not None:
                    desired[i] = fallback_desired.astype(np.float32)
                    to_target[i] = np.asarray(target_pos, dtype=np.float32) - np.asarray(info["pos"], dtype=np.float32)
                    continue
            if chosen_score < self.fallback_score:
                oracle_action = self._oracle_direction_action(idx, obs, state, info)
                if oracle_action is not None:
                    desired[i] = oracle_action.astype(np.float32)
                    to_target[i] = np.asarray(target_pos, dtype=np.float32) - np.asarray(info["pos"], dtype=np.float32)
                    continue
            else:
                self._fallback_low_score_counts[idx] = 0
            agent_state = self._agent_state.setdefault(
                idx, {"best_dist": None, "since_improve": 0, "last_action": None}
            )
            agent_state["last_action"] = np.asarray(chosen_action, dtype=np.float32)
            desired[i] = chosen_action.astype(np.float32)
            to_target[i] = np.asarray(target_pos, dtype=np.float32) - np.asarray(info["pos"], dtype=np.float32)

        return finalize_batch_waypoint_actions(
            agent_ids=agent_ids,
            desired=desired,
            to_target=to_target,
            dist_m=dist_m,
            in_goal=in_goal,
            risk_p=risk_p,
            cur_vel=cur_vel,
            obs_list=obs_list,
            info_list=info_list,
            state=state,
            controller=getattr(self, "_controller", None),
            stop_risk_threshold=float(getattr(self, "stop_risk_threshold", 0.4)),
        )

    def _oracle_direction_action(
        self,
        idx: int,
        obs: ObsDict,
        state: PolicyContext,
        info: dict | None,
    ) -> np.ndarray | None:
        oracle_dir = getattr(state, "oracle_dir", None)
        if oracle_dir is None:
            return None
        if idx < 0 or idx >= oracle_dir.shape[0]:
            return None
        vec = np.asarray(oracle_dir[idx], dtype=np.float32)
        if not np.all(np.isfinite(vec)):
            return None
        norm = float(np.linalg.norm(vec))
        if norm <= 1e-6:
            return None
        return (vec / norm).astype(np.float32)

    def _choose_action_discrete(
        self,
        idx: int,
        horizon: int,
        target_pos: np.ndarray,
        ctx: RolloutContext | None = None,
        *,
        actions_mode: str = "accel",
    ) -> tuple[np.ndarray, float]:
        actions = self._actions_vel if actions_mode == "velocity" else self._actions_accel
        if ctx is None:
            scores, safe_hit = self._rollout_scores(idx, actions, horizon, target_pos, actions_mode=actions_mode)
        else:
            scores, safe_hit = self._rollout_scores_with_context(
                idx, ctx, actions, target_pos, actions_mode=actions_mode
            )
        return self._select_action(actions, scores, safe_hit)

    def _choose_action_cem(
        self,
        idx: int,
        horizon: int,
        target_pos: np.ndarray,
        obs: np.ndarray,
        info: dict,
        ctx: RolloutContext | None = None,
        *,
        actions_mode: str = "velocity",
    ) -> tuple[np.ndarray, float]:
        pos = info.get("pos")
        if pos is not None:
            to_target = np.asarray(target_pos, dtype=np.float32) - np.asarray(pos, dtype=np.float32)
        else:
            to_target = obs_to_target(obs)
        mean = normalize(to_target)
        sigma = np.full((2,), float(max(self.cem_min_sigma, self.cem_sigma)), dtype=np.float32)

        best_action = np.zeros((2,), dtype=np.float32)
        best_score = -1e9

        for _ in range(self.cem_iters):
            actions = self._sample_cem_actions(mean, sigma)
            mode = "velocity" if actions_mode == "velocity" else "accel"
            if ctx is None:
                scores, safe_hit = self._rollout_scores(idx, actions, horizon, target_pos, actions_mode=mode)
            else:
                scores, safe_hit = self._rollout_scores_with_context(idx, ctx, actions, target_pos, actions_mode=mode)

            if safe_hit is not None:
                penalty = float(np.max(np.abs(scores)) + 1.0) if scores.size else 1.0
                adjusted_scores = scores - (penalty * safe_hit.astype(np.float32))
            else:
                adjusted_scores = scores
            best_idx = int(np.argmax(adjusted_scores))
            best_step_score = float(scores[best_idx])
            if best_step_score > best_score:
                best_score = best_step_score
                best_action = actions[best_idx]

            elite_actions = self._select_elite(actions, adjusted_scores)
            if elite_actions is None or elite_actions.size == 0:
                break
            mean = np.mean(elite_actions, axis=0)
            sigma = np.maximum(self.cem_min_sigma, np.std(elite_actions, axis=0)).astype(np.float32)

        return best_action, float(best_score)

    def _sample_cem_actions(self, mean: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        sigma = np.asarray(sigma, dtype=np.float32).reshape(1, 2)
        noise = self._rng.normal(size=(self.cem_samples, 2)).astype(np.float32)
        actions = mean[None, :].astype(np.float32) + (sigma * noise)
        actions = self._normalize_actions(actions)
        if self.cem_include_zero:
            actions = np.vstack([actions, np.zeros((1, 2), dtype=np.float32)])
        return actions

    def _normalize_actions(self, actions: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(actions, axis=1, keepdims=True)
        scale = np.ones_like(norms, dtype=np.float32)
        scale = np.where(norms > 1.0, 1.0 / np.maximum(norms, 1e-6), 1.0)
        return actions * scale

    def _select_elite(self, actions: np.ndarray, scores: np.ndarray) -> np.ndarray | None:
        scores = np.asarray(scores, dtype=np.float32)
        count = int(min(self.cem_elite, scores.shape[0]))
        if count <= 0:
            return None
        idx = np.argpartition(scores, -count)[-count:]
        return actions[idx]

    def _select_action(
        self,
        actions: np.ndarray,
        scores: np.ndarray,
        safe_hit: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        scores = np.asarray(scores, dtype=np.float32)
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        best_action = actions[best_idx]
        safe_mask = ~safe_hit
        if np.any(safe_mask):
            safe_scores = np.where(safe_mask, scores, -1.0e9)
            safe_idx = int(np.argmax(safe_scores))
            best_safe_score = float(safe_scores[safe_idx])
            if best_safe_score >= self.safety_min_score:
                return actions[safe_idx], best_safe_score
        return best_action, best_score

    def _apply_action_penalties(self, idx: int, actions: np.ndarray, scores: np.ndarray) -> np.ndarray:
        if self.jerk_penalty <= 0.0:
            return scores
        state = self._agent_state.setdefault(idx, {"best_dist": None, "since_improve": 0, "last_action": None})
        last = state.get("last_action")
        if last is None:
            return scores
        try:
            last_vec = np.asarray(last, dtype=np.float32).reshape(1, 2)
            diff = np.linalg.norm(actions - last_vec, axis=1)
            return scores - (self.jerk_penalty * diff.astype(np.float32))
        except Exception:
            return scores

    def _apply_terminal_speed_penalty(
        self,
        scores: np.ndarray,
        pos_end: np.ndarray,
        vel_end: np.ndarray,
        target_pos: np.ndarray,
        max_speed: float,
    ) -> np.ndarray:
        if self.terminal_speed_weight <= 0.0 or max_speed <= 0.0:
            return scores
        to_target = target_pos[None, :] - pos_end
        dist = np.linalg.norm(to_target, axis=1)
        dir_vec = to_target / (dist[:, None] + 1e-6)
        speed = np.linalg.norm(vel_end, axis=1)
        align = np.sum(vel_end * dir_vec, axis=1) / max_speed
        scores = scores + (self.terminal_speed_weight * align.astype(np.float32))
        brake_radius = float(self.terminal_brake_radius)
        if brake_radius > 0.0:
            mask = dist < brake_radius
            if np.any(mask):
                scores = scores - (
                    self.terminal_speed_weight * (speed / max_speed).astype(np.float32) * mask.astype(np.float32)
                )
        return scores

    def _simulate_terminal(
        self,
        pos0: np.ndarray,
        vel0: np.ndarray,
        actions: np.ndarray,
        steps: int,
        dt: float,
        max_speed: float,
        max_accel: float,
        drag: float,
        field_size: float | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        n_actions = actions.shape[0]
        pos = np.repeat(pos0[None, :], n_actions, axis=0)
        vel = np.repeat(vel0[None, :], n_actions, axis=0)
        accel = actions * float(max_accel)
        for _ in range(int(max(1, steps))):
            vel = apply_accel_dynamics_vel(
                vel,
                accel,
                dt,
                drag=drag,
                max_speed=max_speed,
            )
            pos = pos + (vel * float(dt))
            if field_size is not None:
                pos = np.clip(pos, 0.0, float(field_size))
        return pos.astype(np.float32), vel.astype(np.float32)

    def _simulate_terminal_velocity(
        self,
        pos0: np.ndarray,
        vel0: np.ndarray,
        actions: np.ndarray,
        steps: int,
        dt: float,
        max_speed: float,
        max_accel: float,
        drag: float,
        field_size: float | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        n_actions = actions.shape[0]
        pos = np.repeat(pos0[None, :], n_actions, axis=0)
        vel = np.repeat(vel0[None, :], n_actions, axis=0)
        tau = max(float(self.cem_tau), 1e-3)
        for _ in range(int(max(1, steps))):
            desired_vel = actions * float(max_speed)
            accel = inverse_accel_for_velocity(desired_vel, vel, dt=tau, drag=drag)
            accel = np.clip(accel, -max_accel, max_accel)
            vel = apply_accel_dynamics_vel(
                vel,
                accel,
                dt,
                drag=drag,
                max_speed=max_speed,
            )
            pos = pos + (vel * float(dt))
            if field_size is not None:
                pos = np.clip(pos, 0.0, float(field_size))
        return pos.astype(np.float32), vel.astype(np.float32)

    def _simulate_actions(
        self,
        actions: np.ndarray,
        ctx: RolloutContext,
        *,
        mode: str = "accel",
    ) -> tuple[np.ndarray, np.ndarray]:
        actions = np.asarray(actions, dtype=np.float32)
        n_actions = actions.shape[0]
        pos = np.repeat(ctx.pos0[None, :], n_actions, axis=0)
        vel = np.repeat(ctx.vel0[None, :], n_actions, axis=0)
        pos_hist = np.empty((ctx.steps, n_actions, 2), dtype=np.float32)
        vel_hist = np.empty((ctx.steps, n_actions, 2), dtype=np.float32)
        tau = max(float(self.cem_tau), 1e-3)
        accel_const = None
        if mode != "velocity":
            accel_const = actions * float(ctx.max_accel)

        for step in range(ctx.steps):
            if mode == "velocity":
                desired_vel = actions * float(ctx.max_speed)
                accel = inverse_accel_for_velocity(desired_vel, vel, dt=tau, drag=ctx.drag)
                accel = np.clip(accel, -ctx.max_accel, ctx.max_accel)
            else:
                accel = accel_const

            vel = apply_accel_dynamics_vel(
                vel,
                accel,
                ctx.dt,
                drag=ctx.drag,
                max_speed=ctx.max_speed,
            )

            if _resolve_wall_slide_batch is not None and ctx.walls:
                pos, vel, _impact = _resolve_wall_slide_batch(pos, vel, ctx.dt, ctx.walls, ctx.radius, ctx.friction)
            elif _resolve_wall_slide is not None and ctx.walls:
                delta = vel * ctx.dt
                if ctx.wall_min is not None and ctx.wall_max is not None:
                    pad = float(ctx.radius + ctx.max_speed * ctx.dt + 1e-3)
                    seg_min = np.minimum(pos, pos + delta)
                    seg_max = np.maximum(pos, pos + delta)
                    in_box = ~(
                        (seg_max[:, 0] < (ctx.wall_min[0] - pad))
                        | (seg_min[:, 0] > (ctx.wall_max[0] + pad))
                        | (seg_max[:, 1] < (ctx.wall_min[1] - pad))
                        | (seg_min[:, 1] > (ctx.wall_max[1] + pad))
                    )
                    if np.any(in_box):
                        if np.any(~in_box):
                            pos[~in_box] = pos[~in_box] + delta[~in_box]
                        for a in np.where(in_box)[0]:
                            pos[a], vel[a], _impact = _resolve_wall_slide(
                                pos[a], vel[a], ctx.dt, ctx.walls, ctx.radius, ctx.friction
                            )
                    else:
                        pos = pos + delta
                else:
                    for a in range(n_actions):
                        pos[a], vel[a], _impact = _resolve_wall_slide(
                            pos[a], vel[a], ctx.dt, ctx.walls, ctx.radius, ctx.friction
                        )
            else:
                pos = pos + (vel * ctx.dt)

            if ctx.field_size is not None:
                pos = np.clip(pos, 0.0, ctx.field_size)

            pos_hist[step] = pos
            vel_hist[step] = vel

        return pos_hist, vel_hist

    def _rollout_scores_velocity_streaming_core(
        self,
        actions: np.ndarray,
        ctx: RolloutContext,
        target_pos: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        actions = np.asarray(actions, dtype=np.float32)
        n_actions = actions.shape[0]
        pos = np.repeat(ctx.pos0[None, :], n_actions, axis=0)
        vel = np.repeat(ctx.vel0[None, :], n_actions, axis=0)
        desired_vel = actions * float(ctx.max_speed)
        tau = max(float(self.cem_tau), 1e-3)

        risk_sum = np.zeros(n_actions, dtype=np.float32)
        wall_pen = np.zeros(n_actions, dtype=np.float32)
        coll_pen = np.zeros(n_actions, dtype=np.float32)
        safe_hit = np.zeros(n_actions, dtype=bool)
        inside_any = np.zeros(n_actions, dtype=bool)

        for _ in range(ctx.steps):
            accel = inverse_accel_for_velocity(desired_vel, vel, dt=tau, drag=ctx.drag)
            accel = np.clip(accel, -ctx.max_accel, ctx.max_accel)
            vel = apply_accel_dynamics_vel(
                vel,
                accel,
                ctx.dt,
                drag=ctx.drag,
                max_speed=ctx.max_speed,
            )
            pos = pos + (vel * ctx.dt)
            if ctx.field_size is not None:
                pos = np.clip(pos, 0.0, ctx.field_size)

            if ctx.has_threats and ctx.t_pos is not None and ctx.t_rad is not None and ctx.t_int is not None:
                diff = pos[:, None, :] - ctx.t_pos[None, :, :]
                d2 = np.sum(diff * diff, axis=2)
                dists = np.sqrt(d2).astype(np.float32)
                margin = dists - ctx.t_rad[None, :]
                if self.risk_mode == "binary":
                    p = (margin <= 0.0).astype(np.float32)
                else:
                    p = np.exp(-np.maximum(margin, 0.0) / ctx.scale[None, :])
                    p = np.where(margin <= 0.0, 1.0, p)
                p = p * ctx.t_int[None, :]
                prod = np.prod(1.0 - p, axis=1)
                risk = 1.0 - prod
                risk_sum += risk.astype(np.float32)
                if self.safety_first and self.safety_margin > 0.0:
                    safe_hit |= np.any(dists <= (ctx.t_rad[None, :] + self.safety_margin), axis=1)
                inside_any |= np.any(dists <= ctx.t_rad[None, :], axis=1)

            if ctx.field_size is not None:
                size = float(ctx.field_size)
                d_left = pos[:, 0] / size
                d_right = (size - pos[:, 0]) / size
                d_down = pos[:, 1] / size
                d_up = (size - pos[:, 1]) / size
                min_wall = np.minimum(np.minimum(d_left, d_right), np.minimum(d_down, d_up))
                wall_pen += np.maximum(0.0, 0.05 - min_wall).astype(np.float32)

            if ctx.others is not None:
                diff = pos[:, None, :] - ctx.others[None, :, :]
                d2 = np.sum(diff * diff, axis=2)
                min_dist = np.sqrt(np.min(d2, axis=1)).astype(np.float32)
                coll_pen += np.maximum(0.0, 1.5 - min_dist).astype(np.float32)

        dist1 = np.linalg.norm(pos - target_pos[None, :], axis=1)
        progress = ctx.dist0 - dist1
        hard = inside_any.astype(np.float32)
        score = (
            self.w_progress * progress
            - self.w_risk * (risk_sum / max(ctx.steps, 1))
            - self.risk_hard_penalty * hard
            - self.w_wall * (wall_pen / max(ctx.steps, 1))
            - self.w_collision * (coll_pen / max(ctx.steps, 1))
        )
        idle = np.linalg.norm(actions, axis=1) <= 1e-3
        score = np.where(idle, score - self.idle_penalty, score)
        return score.astype(np.float32), safe_hit, pos.astype(np.float32), vel.astype(np.float32)

    def _evaluate_trajectories(
        self,
        pos_hist: np.ndarray,
        actions: np.ndarray,
        ctx: RolloutContext,
        target_pos: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        n_actions = actions.shape[0]
        steps = int(ctx.steps)
        risk_sum = np.zeros(n_actions, dtype=np.float32)
        wall_pen = np.zeros(n_actions, dtype=np.float32)
        coll_pen = np.zeros(n_actions, dtype=np.float32)
        safe_hit = np.zeros(n_actions, dtype=bool)
        inside_any = np.zeros(n_actions, dtype=bool)

        if ctx.has_threats and ctx.t_pos is not None and ctx.t_rad is not None and ctx.t_int is not None:
            diff = pos_hist[:, :, None, :] - ctx.t_pos[None, None, :, :]
            d2 = np.sum(diff * diff, axis=3)
            dists = np.sqrt(d2).astype(np.float32)
            margin = dists - ctx.t_rad[None, None, :]
            if self.risk_mode == "binary":
                p = (margin <= 0.0).astype(np.float32)
            else:
                p = np.exp(-np.maximum(margin, 0.0) / ctx.scale[None, None, :])
                p = np.where(margin <= 0.0, 1.0, p)
            p = p * ctx.t_int[None, None, :]
            prod = np.prod(1.0 - p, axis=2)
            risk = 1.0 - prod
            risk_sum = risk.sum(axis=0).astype(np.float32)
            if self.safety_first and self.safety_margin > 0.0:
                safe_hit = np.any(dists <= (ctx.t_rad[None, None, :] + self.safety_margin), axis=(0, 2))
            inside_any = np.any(dists <= ctx.t_rad[None, None, :], axis=(0, 2))

        if ctx.field_size is not None:
            size = float(ctx.field_size)
            d_left = pos_hist[:, :, 0] / size
            d_right = (size - pos_hist[:, :, 0]) / size
            d_down = pos_hist[:, :, 1] / size
            d_up = (size - pos_hist[:, :, 1]) / size
            min_wall = np.minimum(np.minimum(d_left, d_right), np.minimum(d_down, d_up))
            wall_pen = np.maximum(0.0, 0.05 - min_wall).sum(axis=0).astype(np.float32)

        if ctx.others is not None:
            diff = pos_hist[:, :, None, :] - ctx.others[None, None, :, :]
            dists = np.linalg.norm(diff, axis=3)
            min_dist = np.min(dists, axis=2)
            coll_pen = np.maximum(0.0, 1.5 - min_dist).sum(axis=0).astype(np.float32)

        dist1 = np.linalg.norm(pos_hist[-1] - target_pos[None, :], axis=1)
        progress = ctx.dist0 - dist1
        hard = inside_any.astype(np.float32)
        score = (
            self.w_progress * progress
            - self.w_risk * (risk_sum / max(steps, 1))
            - self.risk_hard_penalty * hard
            - self.w_wall * (wall_pen / max(steps, 1))
            - self.w_collision * (coll_pen / max(steps, 1))
        )
        idle = np.linalg.norm(actions, axis=1) <= 1e-3
        score = np.where(idle, score - self.idle_penalty, score)
        return score.astype(np.float32), safe_hit

    def _rollout_scores(
        self,
        idx: int,
        actions: np.ndarray,
        horizon: int,
        target_pos: np.ndarray,
        *,
        actions_mode: str = "accel",
    ) -> tuple[np.ndarray, np.ndarray]:
        ctx = self._prepare_rollout_context(idx, target_pos, horizon)
        return self._rollout_scores_with_context(idx, ctx, actions, target_pos, actions_mode=actions_mode)

    def _rollout_scores_with_context(
        self,
        idx: int,
        ctx: RolloutContext | None,
        actions: np.ndarray,
        target_pos: np.ndarray,
        *,
        actions_mode: str = "accel",
    ) -> tuple[np.ndarray, np.ndarray]:
        if ctx is None:
            return np.zeros((actions.shape[0],), dtype=np.float32), np.zeros((actions.shape[0],), dtype=bool)

        if actions_mode == "velocity":
            if not ctx.has_walls:
                scores, safe_hit, pos_end, vel_end = self._rollout_scores_velocity_streaming_core(actions, ctx, target_pos)
            else:
                pos_hist, vel_hist = self._simulate_actions(actions, ctx, mode="velocity")
                scores, safe_hit = self._evaluate_trajectories(pos_hist, actions, ctx, target_pos)
                pos_end = pos_hist[-1]
                vel_end = vel_hist[-1]
            scores = self._apply_action_penalties(idx, actions, scores.astype(np.float32))
            if self.terminal_speed_weight > 0.0 and ctx.max_speed > 0.0:
                scores = self._apply_terminal_speed_penalty(scores, pos_end, vel_end, target_pos, ctx.max_speed)
            return scores.astype(np.float32), safe_hit.astype(bool)

        actions = np.ascontiguousarray(actions, dtype=np.float32)
        scores = None
        safe_hit = None
        if _NUMBA_AVAILABLE and self.use_numba:
            try:
                t_pos = ctx.t_pos if ctx.t_pos is not None else self._empty_vec2
                t_rad = ctx.t_rad if ctx.t_rad is not None else self._empty_vec1
                t_int = ctx.t_int if ctx.t_int is not None else self._empty_vec1
                scale = ctx.scale if ctx.scale is not None else self._empty_vec1
                others = ctx.others if ctx.others is not None else self._empty_vec2
                target_pos = np.asarray(target_pos, dtype=np.float32)
                if ctx.has_walls and ctx.wall_arr is not None:
                    scores, safe_hit = _rollout_scores_numba_walls(
                        ctx.pos0,
                        ctx.vel0,
                        actions,
                        int(ctx.steps),
                        target_pos,
                        float(ctx.dist0),
                        float(ctx.dt),
                        float(ctx.max_speed),
                        float(ctx.max_accel),
                        float(ctx.drag),
                        float(ctx.field_size) if ctx.field_size is not None else 0.0,
                        bool(ctx.field_size is not None),
                        ctx.wall_arr,
                        True,
                        float(ctx.radius),
                        float(ctx.friction),
                        t_pos,
                        t_rad,
                        t_int,
                        scale,
                        bool(ctx.has_threats),
                        bool(self.risk_mode == "binary"),
                        bool(self.safety_first),
                        float(self.safety_margin),
                        others,
                        bool(ctx.has_others),
                        float(self.w_progress),
                        float(self.w_risk),
                        float(self.risk_hard_penalty),
                        float(self.w_wall),
                        float(self.w_collision),
                        float(self.idle_penalty),
                    )
                else:
                    scores, safe_hit = _rollout_scores_numba(
                        ctx.pos0,
                        ctx.vel0,
                        actions,
                        int(ctx.steps),
                        target_pos,
                        float(ctx.dist0),
                        float(ctx.dt),
                        float(ctx.max_speed),
                        float(ctx.max_accel),
                        float(ctx.drag),
                        float(ctx.field_size) if ctx.field_size is not None else 0.0,
                        bool(ctx.field_size is not None),
                        t_pos,
                        t_rad,
                        t_int,
                        scale,
                        bool(ctx.has_threats),
                        bool(self.risk_mode == "binary"),
                        bool(self.safety_first),
                        float(self.safety_margin),
                        others,
                        bool(ctx.has_others),
                        float(self.w_progress),
                        float(self.w_risk),
                        float(self.risk_hard_penalty),
                        float(self.w_wall),
                        float(self.w_collision),
                        float(self.idle_penalty),
                    )
                scores = scores.astype(np.float32)
                safe_hit = safe_hit.astype(bool)
            except Exception:
                scores = None
                safe_hit = None
        if scores is None or safe_hit is None:
            if not ctx.has_walls:
                scores, safe_hit = self._rollout_scores_vectorized(
                    pos0=ctx.pos0,
                    vel0=ctx.vel0,
                    actions=actions,
                    steps=int(ctx.steps),
                    target_pos=np.asarray(target_pos, dtype=np.float32),
                    dist0=float(ctx.dist0),
                    dt=float(ctx.dt),
                    max_speed=float(ctx.max_speed),
                    max_accel=float(ctx.max_accel),
                    drag=float(ctx.drag),
                    field_size=ctx.field_size,
                    t_pos=ctx.t_pos,
                    t_rad=ctx.t_rad,
                    t_int=ctx.t_int,
                    scale=ctx.scale,
                    others=ctx.others,
                )
            else:
                pos_hist, _vel_hist = self._simulate_actions(actions, ctx, mode=actions_mode)
                scores, safe_hit = self._evaluate_trajectories(pos_hist, actions, ctx, target_pos)

        scores = self._apply_action_penalties(idx, actions, scores.astype(np.float32))
        if self.terminal_speed_weight > 0.0 and ctx.max_speed > 0.0:
            pos_end, vel_end = self._simulate_terminal(
                ctx.pos0,
                ctx.vel0,
                actions,
                ctx.steps,
                ctx.dt,
                ctx.max_speed,
                ctx.max_accel,
                ctx.drag,
                ctx.field_size,
            )
            scores = self._apply_terminal_speed_penalty(scores, pos_end, vel_end, target_pos, ctx.max_speed)
        return scores.astype(np.float32), safe_hit.astype(bool)

    @njit(cache=True, fastmath=True)
    def _rollout_scores_numba_walls(
        pos0,
        vel0,
        actions,
        steps,
        target_pos,
        dist0,
        dt,
        max_speed,
        max_accel,
        drag,
        field_size,
        use_field,
        walls,
        has_walls,
        radius,
        friction,
        t_pos,
        t_rad,
        t_int,
        scale,
        has_threats,
        risk_mode_binary,
        safety_first,
        safety_margin,
        others,
        has_others,
        w_progress,
        w_risk,
        risk_hard_penalty,
        w_wall,
        w_collision,
        idle_penalty,
    ):
        n_actions = actions.shape[0]
        pos = np.empty((n_actions, 2), dtype=np.float32)
        vel = np.empty((n_actions, 2), dtype=np.float32)
        for i in range(n_actions):
            pos[i, 0] = pos0[0]
            pos[i, 1] = pos0[1]
            vel[i, 0] = vel0[0]
            vel[i, 1] = vel0[1]

        risk_sum = np.zeros(n_actions, dtype=np.float32)
        wall_pen = np.zeros(n_actions, dtype=np.float32)
        coll_pen = np.zeros(n_actions, dtype=np.float32)
        safe_hit = np.zeros(n_actions, dtype=np.uint8)
        inside_any = np.zeros(n_actions, dtype=np.uint8)

        for _ in range(steps):
            for a in range(n_actions):
                px = pos[a, 0]
                py = pos[a, 1]
                vx, vy = apply_accel_dynamics_step(
                    vel[a, 0],
                    vel[a, 1],
                    actions[a, 0] * max_accel,
                    actions[a, 1] * max_accel,
                    dt,
                    drag,
                    max_speed,
                    0.0,
                    0.0,
                )
                if has_walls:
                    px, py, vx, vy, _hs = _resolve_wall_slide_numba(px, py, vx, vy, dt, walls, radius, friction)
                else:
                    px = px + (vx * dt)
                    py = py + (vy * dt)
                if use_field:
                    if px < 0.0:
                        px = 0.0
                    elif px > field_size:
                        px = field_size
                    if py < 0.0:
                        py = 0.0
                    elif py > field_size:
                        py = field_size
                pos[a, 0] = px
                pos[a, 1] = py
                vel[a, 0] = vx
                vel[a, 1] = vy

            if has_threats:
                for a in range(n_actions):
                    prod = 1.0
                    inside = False
                    safe = False
                    for t in range(t_pos.shape[0]):
                        dx = pos[a, 0] - t_pos[t, 0]
                        dy = pos[a, 1] - t_pos[t, 1]
                        dist = math.sqrt(dx * dx + dy * dy)
                        margin = dist - t_rad[t]
                        if margin <= 0.0:
                            p = 1.0
                            inside = True
                        else:
                            if risk_mode_binary:
                                p = 0.0
                            else:
                                s = scale[t]
                                if s <= 1e-6:
                                    p = 0.0
                                else:
                                    p = math.exp(-margin / s)
                        if safety_first and safety_margin > 0.0:
                            if dist <= (t_rad[t] + safety_margin):
                                safe = True
                        p = p * t_int[t]
                        if p > 1.0:
                            p = 1.0
                        prod *= 1.0 - p
                    risk = 1.0 - prod
                    risk_sum[a] += risk
                    if safe:
                        safe_hit[a] = 1
                    if inside:
                        inside_any[a] = 1

            if use_field:
                for a in range(n_actions):
                    size = field_size
                    d_left = pos[a, 0] / size
                    d_right = (size - pos[a, 0]) / size
                    d_down = pos[a, 1] / size
                    d_up = (size - pos[a, 1]) / size
                    min_wall = d_left
                    if d_right < min_wall:
                        min_wall = d_right
                    if d_down < min_wall:
                        min_wall = d_down
                    if d_up < min_wall:
                        min_wall = d_up
                    if min_wall < 0.05:
                        wall_pen[a] += 0.05 - min_wall

            if has_others:
                for a in range(n_actions):
                    min_dist = 1e9
                    for j in range(others.shape[0]):
                        dx = pos[a, 0] - others[j, 0]
                        dy = pos[a, 1] - others[j, 1]
                        dist = math.sqrt(dx * dx + dy * dy)
                        if dist < min_dist:
                            min_dist = dist
                    if min_dist < 1.5:
                        coll_pen[a] += 1.5 - min_dist

        dist1 = np.empty(n_actions, dtype=np.float32)
        for a in range(n_actions):
            dx = pos[a, 0] - target_pos[0]
            dy = pos[a, 1] - target_pos[1]
            dist1[a] = math.sqrt(dx * dx + dy * dy)
        progress = dist0 - dist1
        hard = inside_any.astype(np.float32)
        steps_safe = steps if steps > 0 else 1
        score = (
            w_progress * progress
            - w_risk * (risk_sum / steps_safe)
            - risk_hard_penalty * hard
            - w_wall * (wall_pen / steps_safe)
            - w_collision * (coll_pen / steps_safe)
        )
        if idle_penalty > 0.0:
            for a in range(n_actions):
                idle = math.sqrt((actions[a, 0] * actions[a, 0]) + (actions[a, 1] * actions[a, 1]))
                if idle <= 1e-3:
                    score[a] -= idle_penalty

        return score.astype(np.float32), safe_hit

    def _rollout_scores_vectorized(
        self,
        *,
        pos0: np.ndarray,
        vel0: np.ndarray,
        actions: np.ndarray,
        steps: int,
        target_pos: np.ndarray,
        dist0: float,
        dt: float,
        max_speed: float,
        max_accel: float,
        drag: float,
        field_size: float | None,
        t_pos: np.ndarray | None,
        t_rad: np.ndarray | None,
        t_int: np.ndarray | None,
        scale: np.ndarray | None,
        others: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        n_actions = actions.shape[0]
        pos = np.repeat(pos0[None, :], n_actions, axis=0)
        vel = np.repeat(vel0[None, :], n_actions, axis=0)
        risk_sum = np.zeros(n_actions, dtype=np.float32)
        wall_pen = np.zeros(n_actions, dtype=np.float32)
        coll_pen = np.zeros(n_actions, dtype=np.float32)
        safe_hit = np.zeros(n_actions, dtype=bool)
        inside_any = np.zeros(n_actions, dtype=bool)
        accel = actions * float(max_accel)

        for _ in range(steps):
            vel = apply_accel_dynamics_vel(
                vel,
                accel,
                dt,
                drag=drag,
                max_speed=max_speed,
            )
            pos = pos + (vel * float(dt))
            if field_size is not None:
                pos = np.clip(pos, 0.0, float(field_size))

            if t_pos is not None and t_rad is not None and t_int is not None:
                diff = pos[:, None, :] - t_pos[None, :, :]
                dists = np.linalg.norm(diff, axis=2)
                margin = dists - t_rad[None, :]
                if self.risk_mode == "binary":
                    p = (margin <= 0.0).astype(np.float32)
                else:
                    p = np.exp(-np.maximum(margin, 0.0) / scale[None, :])
                    p = np.where(margin <= 0.0, 1.0, p)
                p = p * t_int[None, :]
                prod = np.prod(1.0 - p, axis=1)
                risk = 1.0 - prod
                risk_sum += risk.astype(np.float32)
                if self.safety_first and self.safety_margin > 0.0:
                    safe_hit |= np.any(dists <= (t_rad[None, :] + self.safety_margin), axis=1)
                inside_any |= np.any(dists <= t_rad[None, :], axis=1)

            if field_size is not None:
                size = float(field_size)
                d_left = pos[:, 0] / size
                d_right = (size - pos[:, 0]) / size
                d_down = pos[:, 1] / size
                d_up = (size - pos[:, 1]) / size
                min_wall = np.minimum(np.minimum(d_left, d_right), np.minimum(d_down, d_up))
                wall_pen += np.maximum(0.0, 0.05 - min_wall).astype(np.float32)

            if others is not None:
                diff = pos[:, None, :] - others[None, :, :]
                dists = np.linalg.norm(diff, axis=2)
                min_dist = np.min(dists, axis=1)
                coll_pen += np.maximum(0.0, 1.5 - min_dist).astype(np.float32)

        dist1 = np.linalg.norm(pos - target_pos[None, :], axis=1)
        progress = dist0 - dist1
        hard = inside_any.astype(np.float32)
        score = (
            self.w_progress * progress
            - self.w_risk * (risk_sum / steps)
            - self.risk_hard_penalty * hard
            - self.w_wall * (wall_pen / steps)
            - self.w_collision * (coll_pen / steps)
        )
        idle = np.linalg.norm(actions, axis=1) <= 1e-3
        score = np.where(idle, score - self.idle_penalty, score)
        return score.astype(np.float32), safe_hit

    def _rollout_scores_velocity(
        self,
        idx: int,
        actions: np.ndarray,
        horizon: int,
        target_pos: np.ndarray,
        ctx: RolloutContext | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if ctx is None:
            ctx = self._prepare_rollout_context(idx, target_pos, horizon)
        if ctx is None:
            return np.zeros((actions.shape[0],), dtype=np.float32), np.zeros((actions.shape[0],), dtype=bool)

        actions = np.asarray(actions, dtype=np.float32)
        pos_hist, _vel_hist = self._simulate_actions(actions, ctx, mode="velocity")
        return self._evaluate_trajectories(pos_hist, actions, ctx, target_pos)

    def _rollout_metrics(self, idx: int, act: np.ndarray, horizon: int, target_pos: np.ndarray) -> tuple[float, bool]:
        scores, safe_hit = self._rollout_scores(idx, np.asarray(act, dtype=np.float32)[None, :], horizon, target_pos)
        return float(scores[0]), bool(safe_hit[0])

    def _risk_prob(self, pos: np.ndarray, threats: list) -> float:
        if not threats:
            return 0.0
        # Считаем угрозы независимыми, чтобы получить общую вероятность опасности.
        prod = 1.0
        for t in threats:
            dist = float(np.linalg.norm(pos - t.position))
            margin = dist - float(t.radius)
            if self.risk_mode == "binary":
                p = 1.0 if margin <= 0.0 else 0.0
            else:
                scale = max(self.risk_soft_margin, self.risk_soft_radius_scale * float(t.radius), 1e-3)
                if margin <= 0.0:
                    p = 1.0
                else:
                    p = float(np.exp(-margin / scale))
            p = float(np.clip(p, 0.0, 1.0)) * float(t.intensity)
            prod *= 1.0 - p
        return float(1.0 - prod)

    def _effective_horizon(self, obs: dict | None) -> int:
        h = int(self.horizon)
        if not self.dynamic_horizon:
            return h
        # При высокой скорости увеличиваем горизонт, чтобы заранее замечать риск.
        if obs is None:
            return h
        try:
            speed = float(np.linalg.norm(obs_vel(obs)))
        except Exception:
            return h
        extra = round(speed * self.horizon_speed_scale)
        return int(np.clip(h + extra, h, self.horizon_max))

    def _is_stuck(self, idx: int, info: dict, obs: dict | None) -> bool:
        if self.stuck_steps <= 0:
            return False
        dist = _dist_from_info(info, obs)
        if dist is None:
            return False
        state = self._agent_state.setdefault(idx, {"best_dist": None, "since_improve": 0, "last_action": None})
        best = state["best_dist"]
        if best is None or dist < best - self.stuck_dist_eps:
            state["best_dist"] = dist
            state["since_improve"] = 0
        else:
            state["since_improve"] += 1
        return state["since_improve"] >= self.stuck_steps


if _NUMBA_AVAILABLE:
    _rollout_scores_numba_walls = MPCLitePolicy.__dict__["_rollout_scores_numba_walls"]
