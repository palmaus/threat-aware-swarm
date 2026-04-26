from __future__ import annotations

import heapq
import logging
import math
from collections import OrderedDict

import numpy as np

from baselines.astar_support import (
    MOVES_CARDINAL,
    MOVES_DIAGONAL,
    MOVES_DX,
    MOVES_DY,
    dist_from_info as _dist_from_info,
    escape_direction as _escape_direction,
    maybe_denorm_dist as _maybe_denorm_dist,
    maybe_denorm_vec as _maybe_denorm_vec,
    memory_penalty_from_entries,
    reuse_direction_to_cell,
)

try:  # Опциональное ускорение через Numba.
    from numba import njit
except Exception:  # pragma: no cover - опциональная зависимость
    njit = None

from baselines.controllers import WaypointController
from baselines.policies import ObsDict, PlannerPolicy
from baselines.utils import (
    OBS_LAYOUT,
    agent_index_from_id,
    normalize,
    obs_grid,
    obs_to_target,
    predict_target_single,
)
from common.policy.context import PolicyContext

_NUMBA_AVAILABLE = njit is not None
_NUMBA_WARMED = False
logger = logging.getLogger(__name__)


class AStarGridPolicy(PlannerPolicy):
    """Локальный A* по рисковому гриду с памятью стен и посещённых клеток.

    Используется как лёгкий планировщик под частичной наблюдаемостью.
    """

    def __init__(
        self,
        alpha: float = 3.0,
        max_cost: float = 0.35,
        goal_radius: int = 7,
        goal_radius_control: float = 4.0,
        near_goal_speed_cap: float = 0.35,
        near_goal_damping: float = 0.5,
        near_goal_kp: float = 1.0,
        near_goal_frac: float = 0.2,
        near_goal_alpha: float = 1.0,
        near_goal_max_cost: float | None = None,
        stuck_steps: int = 10,
        stuck_dist_eps: float = 0.5,
        stuck_alpha: float = 0.6,
        stuck_max_cost: float = 0.7,
        escape_steps: int = 3,
        escape_turn_penalty: float = 0.15,
        allow_diagonal: bool = False,
        stop_risk_threshold: float = 0.05,
        risk_speed_scale: float = 0.7,
        risk_speed_floor: float = 0.2,
        safety_first: bool = True,
        safe_max_cost: float = 0.05,
        safe_alpha_mult: float = 1.2,
        risk_cost_mode: str = "hazard",
        heuristic_weight: float = 1.2,
        wall_threshold: float = 0.95,
        cell_cost_empty: float = 1.0,
        cell_cost_threat: float = 100.0,
        cell_cost_ref: float = 0.5,
        frontier_enabled: bool = True,
        frontier_alpha: float | None = None,
        frontier_min_risk: float = 0.0,
        frontier_oracle_weight: float = 0.5,
        los_enabled: bool = True,
        los_max_skip: int = 8,
        waypoint_lookahead: int = 2,
        path_turn_penalty: float = 0.15,
        plan_reuse: bool = True,
        replan_interval: int = 5,
        replan_target_shift: float = 2.0,
        replan_risk_threshold: float = 0.7,
        memory_threats_enabled: bool = True,
        memory_threat_threshold: float = 0.45,
        memory_threat_persist: int = 3,
        memory_threat_max_cells: int = 4000,
        memory_threat_prune_every: int = 200,
        cell_size: float = 1.0,
        steps_per_cell: float | None = None,
        memory_enabled: bool = True,
        memory_penalty: float = 10.0,
        memory_decay: float = 1.0,
        memory_max_cells: int = 500,
        memory_prune_every: int = 50,
        memory_walls_enabled: bool = True,
        memory_shared: bool = True,
        memory_wall_threshold: float = 0.9,
        memory_wall_max_cells: int = 4000,
        memory_wall_prune_every: int = 200,
        global_plan_enabled: bool = True,
        global_plan_cell_size: float | None = 2.0,
        global_plan_max_cells: int = 50000,
        global_plan_refresh: int = 10,
        intercept_enabled: bool = True,
        intercept_gain: float = 0.6,
        intercept_max_time: float = 3.0,
        numba_warmup: bool = True,
        **_,
    ):
        self.alpha = float(alpha)
        self.max_cost = float(max_cost)
        self.goal_radius = int(max(1, goal_radius))
        self.goal_radius_control = float(goal_radius_control)
        self.near_goal_speed_cap = float(near_goal_speed_cap)
        self.near_goal_damping = float(near_goal_damping)
        self.near_goal_kp = float(near_goal_kp)
        self.near_goal_frac = float(near_goal_frac)
        self.near_goal_alpha = float(near_goal_alpha)
        if near_goal_max_cost is None:
            self.near_goal_max_cost = self.max_cost
        else:
            self.near_goal_max_cost = float(near_goal_max_cost)
        self.stuck_steps = int(max(0, stuck_steps))
        self.stuck_dist_eps = float(stuck_dist_eps)
        self.stuck_alpha = float(stuck_alpha)
        self.stuck_max_cost = float(stuck_max_cost)
        self.escape_steps = int(max(0, escape_steps))
        self.escape_turn_penalty = float(escape_turn_penalty)
        self.allow_diagonal = bool(allow_diagonal)
        self.stop_risk_threshold = float(stop_risk_threshold)
        self.risk_speed_scale = float(risk_speed_scale)
        self.risk_speed_floor = float(risk_speed_floor)
        self.safety_first = bool(safety_first)
        self.safe_max_cost = float(safe_max_cost)
        self.safe_alpha_mult = float(safe_alpha_mult)
        self.risk_cost_mode = str(risk_cost_mode)
        self.heuristic_weight = float(heuristic_weight)
        self.wall_threshold = float(wall_threshold)
        self.cell_cost_empty = float(cell_cost_empty)
        self.cell_cost_threat = float(cell_cost_threat)
        self.cell_cost_ref = float(cell_cost_ref)
        self.frontier_enabled = bool(frontier_enabled)
        self.frontier_alpha = None if frontier_alpha is None else float(frontier_alpha)
        self.frontier_min_risk = float(frontier_min_risk)
        self.frontier_oracle_weight = float(frontier_oracle_weight)
        self.los_enabled = bool(los_enabled)
        self.los_max_skip = int(los_max_skip)
        self.waypoint_lookahead = int(max(1, waypoint_lookahead))
        self.path_turn_penalty = float(path_turn_penalty)
        self.plan_reuse = bool(plan_reuse)
        self.replan_interval = int(max(0, replan_interval))
        self.replan_target_shift = float(replan_target_shift)
        self.replan_risk_threshold = float(replan_risk_threshold)
        self.memory_threats_enabled = bool(memory_threats_enabled)
        self.memory_threat_threshold = float(memory_threat_threshold)
        self.memory_threat_persist = int(max(1, memory_threat_persist))
        self.memory_threat_max_cells = int(memory_threat_max_cells)
        self.memory_threat_prune_every = int(memory_threat_prune_every)
        self.cell_size = float(cell_size)
        self.steps_per_cell = None if steps_per_cell is None else float(steps_per_cell)
        self.memory_enabled = bool(memory_enabled)
        self.memory_penalty = float(memory_penalty)
        self.memory_decay = float(memory_decay)
        self.memory_max_cells = int(memory_max_cells)
        self.memory_prune_every = int(memory_prune_every)
        self.memory_walls_enabled = bool(memory_walls_enabled)
        self.memory_shared = bool(memory_shared)
        self.memory_wall_threshold = float(memory_wall_threshold)
        self.memory_wall_max_cells = int(memory_wall_max_cells)
        self.memory_wall_prune_every = int(memory_wall_prune_every)
        self.global_plan_enabled = bool(global_plan_enabled)
        self.global_plan_cell_size = None if global_plan_cell_size is None else float(global_plan_cell_size)
        self.global_plan_max_cells = int(global_plan_max_cells)
        self.global_plan_refresh = int(max(1, global_plan_refresh))
        self.intercept_enabled = bool(intercept_enabled)
        self.intercept_gain = float(intercept_gain)
        self.intercept_max_time = float(intercept_max_time)
        self._field_size = None
        self._max_speed = None
        self._dt = None
        self._controller = WaypointController(
            goal_radius_control=self.goal_radius_control,
            near_goal_speed_cap=self.near_goal_speed_cap,
            near_goal_damping=self.near_goal_damping,
            near_goal_kp=self.near_goal_kp,
            risk_speed_scale=self.risk_speed_scale,
            risk_speed_floor=self.risk_speed_floor,
        )
        self._agent_state = {}
        self._shared_wall_memory = OrderedDict()
        self._shared_wall_mask = None
        self._shared_wall_last = None
        self._shared_step = 0
        self._dijkstra_dist = None
        self._dijkstra_shape = None
        self._numba_scratch = None
        self._numba_scratch_shape = None
        self._border_cache = {}
        self._grid_offset_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        self._soft_failures: dict[str, int] = {}
        if numba_warmup:
            self._warmup_numba()

    def set_context(self, state: PolicyContext | None = None) -> None:
        if state is None:
            return
        self._field_size = float(state.field_size)
        self._max_speed = float(state.max_speed)
        self._dt = float(state.dt)

    def reset(self, seed: int | None = None) -> None:
        self._agent_state = {}
        self._shared_wall_memory = OrderedDict()
        self._shared_wall_mask = None
        self._shared_wall_last = None
        self._shared_step = 0
        self._dijkstra_dist = None
        self._dijkstra_shape = None
        self._numba_scratch = None
        self._numba_scratch_shape = None
        self._border_cache = {}
        self._grid_offset_cache = {}
        self._soft_failures = {}

    def _record_soft_failure(self, key: str, exc: Exception) -> None:
        count = int(self._soft_failures.get(key, 0)) + 1
        self._soft_failures[key] = count
        if count == 1:
            logger.warning("%s soft fallback [%s]: %s", self.__class__.__name__, key, exc)

    def _warmup_numba(self) -> None:
        global _NUMBA_WARMED
        if _NUMBA_WARMED or not _NUMBA_AVAILABLE:
            return
        try:
            grid = np.zeros((3, 3), dtype=np.float32)
            pen = np.zeros((3, 3), dtype=np.float32)
            mask = np.zeros((3, 3), dtype=np.uint8)
            _astar_numba(
                grid,
                1,
                1,
                1,
                1,
                float(self.alpha),
                bool(self.allow_diagonal),
                pen,
                False,
                mask,
                False,
                float(self.heuristic_weight),
                float(self.path_turn_penalty),
            )
            if "_dijkstra_risk_numba" in globals():
                _dijkstra_risk_numba(
                    grid,
                    1,
                    1,
                    mask,
                    False,
                    bool(self.allow_diagonal),
                )
        except Exception:
            return
        _NUMBA_WARMED = True

    def _default_agent_state(self) -> dict:
        return {
            "best_dist": None,
            "since_improve": 0,
            "escape_steps": 0,
            "escape_vec": None,
            "escape_last": None,
            "step": 0,
            "memory": OrderedDict(),
            "wall_memory": OrderedDict(),
            "wall_memory_mask": None,
            "wall_memory_last": None,
            "threat_memory": OrderedDict(),
            "threat_memory_count": None,
            "threat_memory_last": None,
            "global_path": None,
            "global_idx": 0,
            "global_target": None,
            "global_plan_step": 0,
            "current_path": None,
            "path_idx": 0,
            "plan_step": 0,
            "plan_target_pos": None,
            "plan_goal_cell": None,
        }

    def _ensure_plan_state(self, state: dict) -> None:
        if "current_path" not in state:
            state["current_path"] = None
        if "path_idx" not in state:
            state["path_idx"] = 0
        if "plan_step" not in state:
            state["plan_step"] = 0
        if "plan_target_pos" not in state:
            state["plan_target_pos"] = None
        if "plan_goal_cell" not in state:
            state["plan_goal_cell"] = None
        if "global_path" not in state:
            state["global_path"] = None
        if "global_idx" not in state:
            state["global_idx"] = 0
        if "global_target" not in state:
            state["global_target"] = None
        if "global_plan_step" not in state:
            state["global_plan_step"] = 0

    def plan(
        self,
        agent_id: str,
        obs: ObsDict,
        state: PolicyContext,
        info: dict | None = None,
    ) -> np.ndarray | tuple[np.ndarray, dict]:
        info = dict(info or {})
        if state is not None:
            self._field_size = float(state.field_size)
            self._max_speed = float(state.max_speed)
            self._dt = float(state.dt)
            info.setdefault("target_pos", state.target_pos)
            info.setdefault("target_vel", state.target_vel)
        idx = agent_index_from_id(agent_id, info)
        if idx is not None and state is not None and 0 <= idx < state.pos.shape[0]:
            info.setdefault("pos", state.pos[idx])
            info.setdefault("risk_p", float(state.risk_p[idx]))
            info.setdefault("in_goal", float(state.in_goal[idx]))
        vec = np.asarray(obs.get("vector", []), dtype=np.float32).reshape(-1)
        if vec.shape[0] < OBS_LAYOUT.grid_start:
            return np.zeros((2,), dtype=np.float32)
        agent_state = None
        if idx is not None:
            agent_state = self._agent_state.setdefault(
                idx,
                self._default_agent_state(),
            )
            agent_state["step"] = int(agent_state.get("step", 0)) + 1
            self._ensure_plan_state(agent_state)

        risk_p = float(info.get("risk_p", 0.0) or 0.0)
        in_goal = float(info.get("in_goal", 0.0) or 0.0) > 0.0
        # В цели и при низком риске останавливаемся, чтобы не дёргать траекторию.
        if self._controller.should_stop(in_goal, risk_p, self.stop_risk_threshold):
            return np.zeros((2,), dtype=np.float32)

        to_target = obs_to_target(obs)
        oracle_dir = None
        target_pos = info.get("target_pos")
        pos = info.get("pos")
        base_cell = None
        if pos is not None:
            try:
                base_cell = self._cell_from_pos(np.asarray(pos, dtype=np.float32))
            except Exception as exc:
                self._record_soft_failure("base_cell", exc)
                base_cell = None
        if target_pos is not None and pos is not None:
            try:
                target_arr = np.asarray(target_pos, dtype=np.float32)
                pos_arr = np.asarray(pos, dtype=np.float32)
                target_vel = info.get("target_vel")
                if target_vel is not None:
                    try:
                        # Сдвигаем цель вперёд по скорости, чтобы не запаздывать в погоне.
                        target_arr = predict_target_single(
                            target_arr,
                            target_vel,
                            pos=pos_arr,
                            max_speed=self._max_speed,
                            gain=self.intercept_gain,
                            max_time=self.intercept_max_time,
                            enabled=self.intercept_enabled,
                        )
                    except Exception as exc:
                        self._record_soft_failure("target_predict", exc)
                        target_arr = np.asarray(target_pos, dtype=np.float32)
                to_target = target_arr - pos_arr
            except Exception as exc:
                self._record_soft_failure("target_vector", exc)
                to_target = obs_to_target(obs)
        if state is not None and idx is not None:
            oracle_dirs = getattr(state, "oracle_dir", None)
            if oracle_dirs is not None and 0 <= idx < oracle_dirs.shape[0]:
                oracle_dir = np.asarray(oracle_dirs[idx], dtype=np.float32)
        # Последний блок наблюдения — локальная риск‑сетка, по ней строится A*.
        grid = obs_grid(obs)
        if grid is None:
            grid = np.zeros((1, 1), dtype=np.float32)
        center = int(grid.shape[0] // 2)
        start = (center, center)
        dist, dist_normed = _dist_from_info(info, to_target)
        dist_m = _maybe_denorm_dist(dist, dist_normed, self._field_size)
        near_goal = self._controller.near_goal_active(dist_m, dist_normed, self._field_size) or in_goal

        stuck = self._is_stuck(info, dist_m, dist_normed)
        if agent_state is not None and pos is not None and self.memory_enabled:
            try:
                self._memory_touch(agent_state, pos, base_cell=base_cell)
            except Exception as exc:
                self._record_soft_failure("memory_touch", exc)
        if agent_state is not None and pos is not None and (not stuck):
            fast_reuse = self._reuse_action(grid, info, agent_state, current_cell=base_cell)
            if fast_reuse is not None:
                return fast_reuse, {"to_target": to_target}

        if agent_state is not None and pos is not None and self.memory_walls_enabled:
            try:
                wall_state = agent_state
                if self.memory_shared:
                    # Общая память стен ускоряет разведку: один агент открывает проходы для всех.
                    wall_state = {"wall_memory": self._shared_wall_memory, "step": self._shared_step}
                    self._shared_step += 1
                self._memory_walls_update(wall_state, grid, info, base_cell=base_cell)
                self._memory_threats_update(agent_state, wall_state, grid, info, base_cell=base_cell)
                grid = self._apply_wall_memory(grid, info, wall_state, base_cell=base_cell)
            except Exception as exc:
                self._record_soft_failure("memory_update", exc)

        escape_action = None
        if stuck and (not near_goal) and self.escape_steps > 0:
            # Эвристика выхода из ловушки активируется только при длительном застое.
            escape_action = self._escape_action(info, grid, to_target, oracle_dir)

        alpha, max_cost = self._risk_params(dist_m, dist_normed, stuck)
        action = None

        global_action = None
        if (
            stuck
            and (not near_goal)
            and self.global_plan_enabled
            and info.get("pos") is not None
            and info.get("target_pos") is not None
        ):
            try:
                global_action = self._global_plan_action(info, agent_state)
            except Exception as exc:
                self._record_soft_failure("global_plan", exc)
                global_action = None

        if global_action is not None:
            action = global_action
            if agent_state is not None:
                agent_state["current_path"] = None
                agent_state["path_idx"] = 0
        elif escape_action is not None:
            action = escape_action
            if agent_state is not None:
                # Выход из ловушки сбрасывает старый план, чтобы не тащить его дальше.
                agent_state["current_path"] = None
                agent_state["path_idx"] = 0
        else:
            action = self._plan_or_reuse(
                grid,
                start,
                to_target,
                dist_normed,
                alpha,
                max_cost,
                info=info,
                state=agent_state,
                oracle_dir=oracle_dir,
                current_cell=base_cell,
            )

        if action is None:
            action = np.zeros((2,), dtype=np.float32)
        return action, {"to_target": to_target}

    def _plan_or_reuse(
        self,
        grid: np.ndarray,
        start,
        to_target: np.ndarray,
        dist_normed: bool,
        alpha: float,
        max_cost: float,
        *,
        info: dict,
        state: dict | None,
        oracle_dir: np.ndarray | None = None,
        current_cell: tuple[int, int] | None = None,
    ) -> np.ndarray:
        return self._plan_action(
            grid, start, to_target, dist_normed, alpha, max_cost, info, state, oracle_dir, base_cell=current_cell
        )

    def _plan_action(
        self,
        grid: np.ndarray,
        start,
        to_target: np.ndarray,
        dist_normed: bool,
        alpha: float,
        max_cost: float,
        info: dict,
        state: dict | None,
        oracle_dir: np.ndarray | None = None,
        base_cell: tuple[int, int] | None = None,
    ) -> np.ndarray:
        cost_grid, wall_mask = self._cached_cost_wall(grid, state)
        penalty_grid = self._memory_penalty_grid(grid.shape, state, info=info, base_cell=base_cell)
        if self.frontier_enabled:
            goal = self._frontier_goal(
                grid,
                start,
                to_target,
                dist_normed,
                alpha,
                penalty_grid,
                oracle_dir,
                cost_grid=cost_grid,
                wall_mask=wall_mask,
            )
        else:
            goal = self._project_goal(start, to_target, grid.shape[0])
        path = self._plan_path(
            grid,
            start,
            goal,
            alpha,
            max_cost,
            info=info,
            state=state,
            penalty_grid=penalty_grid,
            cost_grid=cost_grid,
            wall_mask=wall_mask,
        )

        if state is not None and info.get("pos") is not None:
            self._store_plan(state, path, info, grid.shape[0], base_cell=base_cell)

        if len(path) >= 2:
            lookahead = int(max(1, self.waypoint_lookahead))
            target_idx = int(min(lookahead, len(path) - 1))
            nxt = path[target_idx]
            if self.los_enabled:
                nxt = self._select_los_waypoint(path, start, wall_mask, max_idx=target_idx)
            vec = np.array([nxt[0] - start[0], nxt[1] - start[1]], dtype=np.float32)
            return normalize(vec)
        if np.linalg.norm(to_target) <= 1e-6:
            return np.zeros((2,), dtype=np.float32)
        return normalize(to_target)

    def _select_los_waypoint(self, path: list, start, wall_mask: np.ndarray | None, max_idx: int | None = None):
        if not self.los_enabled or wall_mask is None:
            return path[1 if max_idx is None else max(1, min(max_idx, len(path) - 1))]
        if len(path) <= 2:
            return path[1]
        if max_idx is None:
            max_idx = len(path) - 1
        max_idx = max(1, min(int(max_idx), len(path) - 1))
        checks = 0
        for i in range(max_idx, 0, -1):
            if self.los_max_skip > 0 and checks >= self.los_max_skip:
                break
            if self._los_visible(start, path[i], wall_mask):
                return path[i]
            checks += 1
        return path[min(1, max_idx)]

    def _select_los_from_global(
        self,
        path: list,
        path_idx: int,
        base_cell: tuple[int, int],
        wall_mask: np.ndarray,
        grid_size: int,
        *,
        max_idx: int | None = None,
    ) -> tuple[int, int] | None:
        if not self.los_enabled or wall_mask is None:
            return None
        center = int(grid_size // 2)
        start_local = (center, center)
        if max_idx is None:
            max_idx = len(path) - 1
        max_idx = max(path_idx + 1, min(int(max_idx), len(path) - 1))
        checks = 0
        for i in range(max_idx, path_idx, -1):
            if self.los_max_skip > 0 and checks >= self.los_max_skip:
                break
            local = self._local_from_global(path[i], base_cell, grid_size)
            if local is None:
                continue
            if self._los_visible(start_local, local, wall_mask):
                return path[i]
            checks += 1
        return None

    def _los_visible(self, start, end, wall_mask: np.ndarray) -> bool:
        x0, y0 = int(start[0]), int(start[1])
        x1, y1 = int(end[0]), int(end[1])
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        x, y = x0, y0
        while True:
            if (x, y) != (x0, y0):
                try:
                    if bool(wall_mask[y, x]):
                        return False
                except Exception:
                    return False
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        return True

    def _store_plan(
        self,
        state: dict,
        path: list,
        info: dict,
        grid_size: int,
        *,
        base_cell: tuple[int, int] | None = None,
    ) -> None:
        pos = info.get("pos")
        if pos is None:
            state["current_path"] = None
            state["path_idx"] = 0
            return
        if base_cell is None:
            base_cell = self._cell_from_pos(np.asarray(pos, dtype=np.float32))
        center = int(grid_size // 2)
        global_path = self._path_to_global(path, base_cell, center)
        state["current_path"] = global_path
        state["path_idx"] = 0
        state["plan_step"] = int(state.get("step", 0))
        target_pos = info.get("target_pos")
        state["plan_target_pos"] = None if target_pos is None else np.asarray(target_pos, dtype=np.float32)
        state["plan_goal_cell"] = global_path[-1] if global_path else None

    def _path_to_global(self, path: list, base_cell: tuple[int, int], center: int) -> list:
        if not path:
            return []
        out = []
        bx, by = base_cell
        for x, y in path:
            gx = bx + (int(x) - center)
            gy = by + (int(y) - center)
            out.append((gx, gy))
        return out

    def _advance_path_index(self, state: dict, current_cell: tuple[int, int]) -> None:
        path = state.get("current_path")
        if not path:
            state["path_idx"] = 0
            return
        path_idx = int(state.get("path_idx", 0))
        if path_idx < len(path) and path[path_idx] == current_cell:
            while path_idx + 1 < len(path) and path[path_idx + 1] == current_cell:
                path_idx += 1
        else:
            try:
                idx = path.index(current_cell)
            except ValueError:
                idx = None
            if idx is not None and idx >= path_idx:
                path_idx = idx
        state["path_idx"] = path_idx

    def _reuse_action(
        self,
        grid: np.ndarray,
        info: dict,
        state: dict | None,
        *,
        current_cell: tuple[int, int] | None = None,
    ) -> np.ndarray | None:
        if not self.plan_reuse or state is None:
            return None
        pos = info.get("pos")
        if pos is None:
            return None
        self._ensure_plan_state(state)
        if current_cell is None:
            current_cell = self._cell_from_pos(np.asarray(pos, dtype=np.float32))
        self._advance_path_index(state, current_cell)
        if self._should_replan(state, grid, current_cell, info):
            return None
        path = state.get("current_path")
        path_idx = int(state.get("path_idx", 0))
        if not path or path_idx >= len(path) - 1:
            return None

        lookahead = int(max(1, self.waypoint_lookahead))
        target_idx = int(min(path_idx + lookahead, len(path) - 1))
        next_cell = path[target_idx]
        wall_mask = None
        if self.los_enabled and target_idx > (path_idx + 1):
            if state.get("cache_grid") is grid:
                wall_mask = state.get("cache_wall_mask")
            if wall_mask is None:
                wall_mask = self._wall_mask(grid)
                state["cache_grid"] = grid
                state["cache_wall_mask"] = wall_mask
        if wall_mask is not None:
            los_cell = self._select_los_from_global(
                path,
                path_idx,
                current_cell,
                wall_mask,
                grid.shape[0],
                max_idx=target_idx,
            )
            if los_cell is not None:
                next_cell = los_cell
        return reuse_direction_to_cell(next_cell, pos, cell_size=self.cell_size)

    def _should_replan(self, state: dict, grid: np.ndarray, current_cell: tuple[int, int], info: dict) -> bool:
        path = state.get("current_path")
        path_idx = int(state.get("path_idx", 0))
        if not path or path_idx >= len(path) - 1:
            return True
        if self.replan_interval > 0:
            if (int(state.get("step", 0)) - int(state.get("plan_step", 0))) >= self.replan_interval:
                return True
        target_pos = info.get("target_pos")
        if target_pos is not None and self.replan_target_shift > 0.0:
            prev = state.get("plan_target_pos")
            if prev is not None:
                shift = float(
                    np.linalg.norm(np.asarray(target_pos, dtype=np.float32) - np.asarray(prev, dtype=np.float32))
                )
                if shift >= self.replan_target_shift:
                    return True
        if self.replan_risk_threshold > 0.0 and grid is not None and grid.size > 0:
            next_cell = path[path_idx + 1]
            local = self._local_from_global(next_cell, current_cell, grid.shape[0])
            if local is None:
                return True
            lx, ly = local
            try:
                risk_val = float(grid[ly, lx])
            except Exception:
                risk_val = 0.0
            if risk_val >= float(self.replan_risk_threshold):
                return True
            if float(risk_val) >= float(self.wall_threshold):
                return True
        return False

    def _local_from_global(
        self,
        cell: tuple[int, int],
        base_cell: tuple[int, int],
        grid_size: int,
    ) -> tuple[int, int] | None:
        center = int(grid_size // 2)
        gx, gy = cell
        bx, by = base_cell
        lx = int(gx - bx + center)
        ly = int(gy - by + center)
        if lx < 0 or ly < 0 or lx >= grid_size or ly >= grid_size:
            return None
        return (lx, ly)

    def _cell_center(self, cell: tuple[int, int]) -> np.ndarray:
        x, y = cell
        return np.array([(x + 0.5) * self.cell_size, (y + 0.5) * self.cell_size], dtype=np.float32)

    def _project_goal(self, start, to_target: np.ndarray, grid_size: int):
        if np.linalg.norm(to_target) <= 1e-6:
            return start
        direction = to_target / (np.linalg.norm(to_target) + 1e-6)
        radius = self.goal_radius
        gx = round(start[0] + direction[0] * radius)
        gy = round(start[1] + direction[1] * radius)
        max_idx = max(0, int(grid_size) - 1)
        gx = int(np.clip(gx, 0, max_idx))
        gy = int(np.clip(gy, 0, max_idx))
        return (gx, gy)

    def _risk_params(self, dist_m: float | None, dist_normed: bool, stuck: bool) -> tuple[float, float]:
        alpha = self.alpha
        max_cost = self.max_cost

        if dist_m is not None:
            if self._field_size is not None:
                near_goal_dist = self.near_goal_frac * self._field_size
            elif dist_normed:
                near_goal_dist = self.near_goal_frac
            else:
                near_goal_dist = self.near_goal_frac * 100.0
            if dist_m <= near_goal_dist:
                alpha *= self.near_goal_alpha
                max_cost = max(max_cost, self.near_goal_max_cost)

        if stuck:
            alpha *= self.stuck_alpha
            max_cost = max(max_cost, self.stuck_max_cost)

        return alpha, max_cost

    def _escape_action(
        self,
        info: dict,
        grid: np.ndarray,
        to_target: np.ndarray,
        oracle_dir: np.ndarray | None = None,
    ) -> np.ndarray | None:
        idx = agent_index_from_id(info.get("agent_id", ""), info)
        if idx is None:
            return None
        state = self._agent_state.setdefault(
            idx,
            {
                "best_dist": None,
                "since_improve": 0,
                "escape_steps": 0,
                "escape_vec": None,
                "escape_last": None,
                "threat_memory": {},
            },
        )

        if state.get("escape_steps", 0) > 0 and state.get("escape_vec") is not None:
            state["escape_steps"] -= 1
            return np.asarray(state["escape_vec"], dtype=np.float32)

        penalty_grid = self._memory_penalty_grid(grid.shape, state, info=info)
        prev = state.get("escape_last")
        vec = _escape_direction(
            grid,
            to_target,
            self.allow_diagonal,
            penalty_grid=penalty_grid,
            prev_vec=prev,
            turn_penalty=self.escape_turn_penalty,
            oracle_dir=oracle_dir,
        )
        if vec is None:
            return None
        state["escape_vec"] = vec
        state["escape_last"] = vec
        state["escape_steps"] = max(0, self.escape_steps - 1)
        return np.asarray(vec, dtype=np.float32)

    def _plan_path(
        self,
        grid: np.ndarray,
        start,
        goal,
        alpha: float,
        max_cost: float,
        info=None,
        state=None,
        penalty_grid: np.ndarray | None = None,
        cost_grid: np.ndarray | None = None,
        wall_mask: np.ndarray | None = None,
    ):
        cost_grid = self._cost_grid(grid) if cost_grid is None else cost_grid
        wall_mask = self._wall_mask(grid) if wall_mask is None else wall_mask
        if penalty_grid is None:
            penalty_grid = self._memory_penalty_grid(cost_grid.shape, state, info=info)
        eff_alpha = alpha
        if self.safety_first:
            # Без жёсткого отсечения усиливаем риск, чтобы путь избегал угроз, если есть обход.
            eff_alpha = max(0.0, alpha * self.safe_alpha_mult)
        scratch = self._get_numba_scratch(cost_grid.shape)
        return _astar(
            cost_grid,
            start,
            goal,
            alpha=eff_alpha,
            allow_diagonal=self.allow_diagonal,
            penalty_grid=penalty_grid,
            wall_mask=wall_mask,
            heuristic_weight=self.heuristic_weight,
            turn_penalty=self.path_turn_penalty,
            scratch=scratch,
        )

    def _global_plan_action(self, info: dict, state: dict | None) -> np.ndarray | None:
        if state is None:
            return None
        pos = info.get("pos")
        target_pos = info.get("target_pos")
        if pos is None or target_pos is None:
            return None
        if self._field_size is None:
            return None
        if not self.memory_walls_enabled:
            return None
        gp_cell = self.global_plan_cell_size or self.cell_size
        if gp_cell <= 0:
            return None

        size = int(max(1, np.ceil(float(self._field_size) / float(gp_cell))))
        if self.global_plan_max_cells > 0 and (size * size) > self.global_plan_max_cells:
            return None
        if self.memory_shared:
            wall_mask_full = self._shared_wall_mask
        else:
            wall_mask_full = state.get("wall_memory_mask")
        if wall_mask_full is None or not np.any(wall_mask_full):
            return None
        wall_mask = np.zeros((size, size), dtype=bool)
        ys, xs = np.nonzero(wall_mask_full)
        if ys.size == 0:
            return None
        wx = (xs.astype(np.float32) + 0.5) * float(self.cell_size)
        wy = (ys.astype(np.float32) + 0.5) * float(self.cell_size)
        cx = np.floor(wx / gp_cell).astype(np.int32)
        cy = np.floor(wy / gp_cell).astype(np.int32)
        valid = (cx >= 0) & (cx < size) & (cy >= 0) & (cy < size)
        if np.any(valid):
            wall_mask[cy[valid], cx[valid]] = True

        start_cell = (
            int(np.floor(float(pos[0]) / gp_cell)),
            int(np.floor(float(pos[1]) / gp_cell)),
        )
        target_cell = (
            int(np.floor(float(target_pos[0]) / gp_cell)),
            int(np.floor(float(target_pos[1]) / gp_cell)),
        )
        sx = int(np.clip(start_cell[0], 0, size - 1))
        sy = int(np.clip(start_cell[1], 0, size - 1))
        tx = int(np.clip(target_cell[0], 0, size - 1))
        ty = int(np.clip(target_cell[1], 0, size - 1))
        start_cell = (sx, sy)
        target_cell = (tx, ty)

        path = state.get("global_path")
        step = int(state.get("step", 0))
        refresh_due = (step - int(state.get("global_plan_step", 0))) >= self.global_plan_refresh
        if path is None or state.get("global_target") != target_cell or refresh_due:
            grid = np.zeros((size, size), dtype=np.float32)
            scratch = self._get_numba_scratch(grid.shape)
            path = _astar(
                grid,
                start_cell,
                target_cell,
                alpha=0.0,
                allow_diagonal=self.allow_diagonal,
                wall_mask=wall_mask,
                heuristic_weight=self.heuristic_weight,
                scratch=scratch,
            )
            state["global_path"] = path
            state["global_idx"] = 0
            state["global_target"] = target_cell
            state["global_plan_step"] = step

        if not path or len(path) < 2:
            return None

        try:
            idx = path.index(start_cell)
        except Exception:
            idx = int(state.get("global_idx", 0))
            if idx < 0 or idx >= len(path):
                # Если не нашли точного совпадения, берём ближайшую точку пути.
                best_i = 0
                best_d = float("inf")
                sx, sy = start_cell
                for i, cell in enumerate(path):
                    dx = float(cell[0] - sx)
                    dy = float(cell[1] - sy)
                    d = (dx * dx) + (dy * dy)
                    if d < best_d:
                        best_d = d
                        best_i = i
                idx = best_i
        if idx < len(path) - 1:
            next_cell = path[idx + 1]
            state["global_idx"] = idx + 1
        else:
            return None

        target = np.array([(next_cell[0] + 0.5) * gp_cell, (next_cell[1] + 0.5) * gp_cell], dtype=np.float32)
        vec = np.asarray(target, dtype=np.float32) - np.asarray(pos, dtype=np.float32)
        if np.linalg.norm(vec) <= 1e-6:
            return None
        return normalize(vec)

    def _steps_per_cell(self) -> float:
        if self.steps_per_cell is not None and self.steps_per_cell > 0.0:
            return self.steps_per_cell
        if self._max_speed and self._dt and self._max_speed > 0.0 and self._dt > 0.0:
            return max(self.cell_size / (self._max_speed * self._dt), 1.0)
        return 8.0

    def _cost_grid(self, grid: np.ndarray) -> np.ndarray:
        mode = self.risk_cost_mode
        if mode == "hazard":
            # Преобразуем вероятность в суммируемую стоимость: -log(1 - p).
            p = np.clip(grid, 0.0, 0.999)
            steps = self._steps_per_cell()
            hazard = -np.log1p(-p) * steps
            return hazard
        if mode == "cell":
            # Переводим риск в «цену клетки»: пустая ~1, угроза ~cell_cost_threat.
            ref = float(self.cell_cost_ref) if self.cell_cost_ref > 1e-6 else 0.5
            t = np.clip(np.asarray(grid, dtype=np.float32) / ref, 0.0, 1.0)
            base = self.cell_cost_empty
            hi = self.cell_cost_threat
            cost = base + t * (hi - base)
            # Возвращаем штраф сверх базовой стоимости шага.
            return np.maximum(cost - 1.0, 0.0).astype(np.float32)
        return np.asarray(grid, dtype=np.float32)

    def _wall_mask(self, grid: np.ndarray) -> np.ndarray:
        thr = float(self.wall_threshold)
        return np.asarray(grid, dtype=np.float32) >= thr

    def _cached_cost_wall(
        self, grid: np.ndarray, state: dict | None
    ) -> tuple[np.ndarray, np.ndarray]:
        if state is not None and state.get("cache_grid") is grid:
            cached_cost = state.get("cache_cost_grid")
            cached_mask = state.get("cache_wall_mask")
            if cached_cost is not None and cached_mask is not None:
                return cached_cost, cached_mask
        cost_grid = self._cost_grid(grid)
        wall_mask = self._wall_mask(grid)
        if state is not None:
            state["cache_grid"] = grid
            state["cache_cost_grid"] = cost_grid
            state["cache_wall_mask"] = wall_mask
        return cost_grid, wall_mask

    def _frontier_goal(
        self,
        grid: np.ndarray,
        start,
        to_target: np.ndarray,
        dist_normed: bool,
        alpha: float,
        penalty_grid: np.ndarray | None = None,
        oracle_dir: np.ndarray | None = None,
        *,
        cost_grid: np.ndarray | None = None,
        wall_mask: np.ndarray | None = None,
    ):
        if grid is None or grid.size == 0:
            return self._project_goal(start, to_target, grid.shape[0] if grid is not None else 1)
        if to_target is None or np.linalg.norm(to_target) <= 1e-6:
            return start
        if dist_normed and self._field_size is None:
            return self._project_goal(start, to_target, grid.shape[0])

        cost_grid = self._cost_grid(grid) if cost_grid is None else cost_grid
        if penalty_grid is not None:
            try:
                cost_grid = cost_grid + np.asarray(penalty_grid, dtype=np.float32)
            except Exception:
                pass
        wall_mask = self._wall_mask(grid) if wall_mask is None else wall_mask
        dist_buf = self._get_dijkstra_dist(cost_grid.shape)
        scratch = self._get_numba_scratch(cost_grid.shape)
        risk_costs = _dijkstra_risk(
            cost_grid,
            start,
            wall_mask,
            self.allow_diagonal,
            dist_buf=dist_buf,
            scratch=scratch,
        )

        to_target_m = _maybe_denorm_vec(to_target, dist_normed, self._field_size)
        if to_target_m is None:
            to_target_m = to_target
        scale = float(self.cell_size)
        frontier_alpha = alpha if self.frontier_alpha is None else float(self.frontier_alpha)
        oracle_weight = float(self.frontier_oracle_weight)
        oracle_norm = None
        if oracle_dir is not None and np.linalg.norm(oracle_dir) > 1e-6:
            oracle_norm = normalize(np.asarray(oracle_dir, dtype=np.float32))

        h, w = grid.shape
        border = self._border_cache.get((h, w))
        if border is None:
            xs_top = np.arange(w, dtype=np.int32)
            ys_top = np.zeros(w, dtype=np.int32)
            xs_bottom = np.arange(w, dtype=np.int32)
            ys_bottom = np.full(w, h - 1, dtype=np.int32)
            ys_mid = np.arange(1, h - 1, dtype=np.int32) if h > 2 else np.zeros(0, dtype=np.int32)
            xs_left = np.zeros(ys_mid.shape[0], dtype=np.int32)
            xs_right = np.full(ys_mid.shape[0], w - 1, dtype=np.int32)
            xs = np.concatenate([xs_top, xs_bottom, xs_left, xs_right])
            ys = np.concatenate([ys_top, ys_bottom, ys_mid, ys_mid])
            border = (xs, ys)
            self._border_cache[(h, w)] = border

        xs, ys = border
        if xs.size == 0:
            return self._project_goal(start, to_target, grid.shape[0])
        mask = ~wall_mask[ys, xs]
        if not np.any(mask):
            return self._project_goal(start, to_target, grid.shape[0])
        xs = xs[mask]
        ys = ys[mask]
        risk = risk_costs[ys, xs]
        valid = np.isfinite(risk) & (risk >= float(self.frontier_min_risk))
        if not np.any(valid):
            return self._project_goal(start, to_target, grid.shape[0])
        xs = xs[valid]
        ys = ys[valid]
        risk = risk[valid]

        off_x = (xs.astype(np.float32) - float(start[0])) * float(scale)
        off_y = (ys.astype(np.float32) - float(start[1])) * float(scale)
        to_target_arr = np.asarray(to_target_m, dtype=np.float32).reshape(2)
        diff_x = to_target_arr[0] - off_x
        diff_y = to_target_arr[1] - off_y
        dist_goal = np.sqrt((diff_x * diff_x) + (diff_y * diff_y))
        score = dist_goal + (frontier_alpha * risk)
        if oracle_norm is not None and oracle_weight > 0.0:
            norm = np.sqrt((off_x * off_x) + (off_y * off_y))
            norm = np.where(norm > 1e-6, norm, 1.0)
            align = ((off_x / norm) * oracle_norm[0]) + ((off_y / norm) * oracle_norm[1])
            score = score - (oracle_weight * align)
        order = np.lexsort((risk, score))
        best_idx = int(order[0])
        return (int(xs[best_idx]), int(ys[best_idx]))

    def _get_dijkstra_dist(self, shape: tuple[int, int]) -> np.ndarray | None:
        if shape is None:
            return None
        if self._dijkstra_dist is None or self._dijkstra_shape != shape:
            self._dijkstra_dist = np.full(shape, float("inf"), dtype=np.float32)
            self._dijkstra_shape = shape
        return self._dijkstra_dist

    def _get_numba_scratch(self, shape: tuple[int, int]) -> dict[str, np.ndarray] | None:
        if not _NUMBA_AVAILABLE or shape is None:
            return None
        if self._numba_scratch is None or self._numba_scratch_shape != shape:
            height, width = shape
            max_nodes = height * width * 8 + 1
            max_path = height * width + 1
            self._numba_scratch = {
                "dist": np.full(shape, float("inf"), dtype=np.float32),
                "mask": np.zeros(shape, dtype=np.uint8),
                "penalty": np.zeros(shape, dtype=np.float32),
                "g_score": np.full(shape, float("inf"), dtype=np.float32),
                "closed": np.zeros(shape, dtype=np.uint8),
                "came_x": np.full(shape, -1, dtype=np.int32),
                "came_y": np.full(shape, -1, dtype=np.int32),
                "heap_x": np.empty(max_nodes, dtype=np.int32),
                "heap_y": np.empty(max_nodes, dtype=np.int32),
                "heap_f": np.empty(max_nodes, dtype=np.float32),
                "path_x": np.empty(max_path, dtype=np.int32),
                "path_y": np.empty(max_path, dtype=np.int32),
            }
            self._numba_scratch_shape = shape
        return self._numba_scratch

    def _max_cost_threshold(self, max_cost: float) -> float:
        if self.risk_cost_mode != "hazard":
            return max_cost
        capped = float(np.clip(max_cost, 0.0, 0.999))
        return float(-np.log1p(-capped))

    def _memory_touch(self, state: dict, pos: np.ndarray, *, base_cell: tuple[int, int] | None = None) -> None:
        if not self.memory_enabled:
            return
        mem = state.get("memory")
        if mem is None:
            return
        cell = base_cell if base_cell is not None else self._cell_from_pos(pos)
        step = int(state.get("step", 0))
        score = self._memory_score(mem, cell, step)
        mem[cell] = (score + 1.0, step)
        if hasattr(mem, "move_to_end"):
            mem.move_to_end(cell)
        if self.memory_max_cells > 0 and len(mem) > self.memory_max_cells:
            if self.memory_prune_every > 0 and (step % self.memory_prune_every) == 0:
                self._prune_memory(mem)

    def _memory_penalty_grid(
        self,
        shape,
        state: dict | None,
        *,
        info: dict | None = None,
        base_cell: tuple[int, int] | None = None,
    ) -> np.ndarray | None:
        if not self.memory_enabled or state is None:
            return None
        mem = state.get("memory")
        if not mem:
            return None
        if base_cell is None:
            if info is None:
                return None
            pos = info.get("pos")
            if pos is None:
                return None
            try:
                base_cell = self._cell_from_pos(pos)
            except Exception:
                return None
        step = int(state.get("step", 0))
        items = list(mem.items())
        if not items:
            return None
        coords = np.asarray([cell for cell, _ in items], dtype=np.int32)
        scores = np.asarray([float(entry[0]) for _, entry in items], dtype=np.float32)
        last = np.asarray([int(entry[1]) for _, entry in items], dtype=np.int32)
        penalty, stale = memory_penalty_from_entries(
            coords,
            scores,
            last,
            base_cell=base_cell,
            step=step,
            shape=shape,
            memory_penalty=self.memory_penalty,
            memory_decay=self.memory_decay,
        )
        for cell in stale:
            mem.pop((int(cell[0]), int(cell[1])), None)
        return penalty

    def _grid_offsets(self, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        shape = grid.shape
        cached = self._grid_offset_cache.get(shape)
        if cached is not None:
            return cached
        center = int(shape[0] // 2)
        ys, xs = np.indices(shape)
        dx = (xs.astype(np.int32) - int(center))
        dy = (ys.astype(np.int32) - int(center))
        cached = (dx, dy)
        self._grid_offset_cache[shape] = cached
        return cached

    def _observed_global_cells(
        self,
        grid: np.ndarray,
        base_cell: tuple[int, int] | None,
        threshold: float,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        if base_cell is None:
            return None
        mask = np.asarray(grid, dtype=np.float32) >= float(threshold)
        if not np.any(mask):
            return None
        dx_grid, dy_grid = self._grid_offsets(grid)
        dx = dx_grid[mask]
        dy = dy_grid[mask]
        gx = base_cell[0] + dx
        gy = base_cell[1] + dy
        return gx.astype(np.int32, copy=False), gy.astype(np.int32, copy=False)

    def _clip_observed_cells(
        self,
        gx: np.ndarray,
        gy: np.ndarray,
        shape: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray] | None:
        valid = (gx >= 0) & (gx < shape[1]) & (gy >= 0) & (gy < shape[0])
        if not np.any(valid):
            return None
        return gx[valid], gy[valid]

    def _memory_walls_update(
        self,
        state: dict,
        grid: np.ndarray,
        info: dict,
        *,
        base_cell: tuple[int, int] | None = None,
    ) -> None:
        if not self.memory_walls_enabled:
            return
        if state is None:
            return
        if base_cell is None:
            pos = info.get("pos")
            if pos is None:
                return
            base_cell = self._cell_from_pos(np.asarray(pos, dtype=np.float32))
        step = int(state.get("step", 0))
        wall_mask, wall_last = self._ensure_wall_arrays(state)
        if wall_mask is None or wall_last is None:
            return
        coords = self._observed_global_cells(grid, base_cell, self.memory_wall_threshold)
        if coords is None:
            if self.memory_wall_max_cells > 0:
                if self.memory_wall_prune_every > 0 and (step % self.memory_wall_prune_every) == 0:
                    self._prune_mask_by_age(wall_mask, wall_last, self.memory_wall_max_cells)
            return
        clipped = self._clip_observed_cells(coords[0], coords[1], wall_mask.shape)
        if clipped is not None:
            gx, gy = clipped
            wall_mask[gy, gx] = 1.0
            wall_last[gy, gx] = step
        if self.memory_wall_max_cells > 0:
            if self.memory_wall_prune_every > 0 and (step % self.memory_wall_prune_every) == 0:
                self._prune_mask_by_age(wall_mask, wall_last, self.memory_wall_max_cells)

    def _memory_threats_update(
        self,
        state: dict,
        wall_state: dict,
        grid: np.ndarray,
        info: dict,
        *,
        base_cell: tuple[int, int] | None = None,
    ) -> None:
        if not self.memory_threats_enabled:
            return
        if state is None or wall_state is None:
            return
        if base_cell is None:
            pos = info.get("pos")
            if pos is None:
                return
            base_cell = self._cell_from_pos(np.asarray(pos, dtype=np.float32))
        step = int(state.get("step", 0))
        wall_mask, wall_last = self._ensure_wall_arrays(wall_state)
        threat_count, threat_last = self._ensure_threat_arrays(state)
        if wall_mask is None or wall_last is None or threat_count is None or threat_last is None:
            return
        coords = self._observed_global_cells(grid, base_cell, self.memory_threat_threshold)
        if coords is None:
            if self.memory_threat_max_cells > 0:
                if self.memory_threat_prune_every > 0 and (step % self.memory_threat_prune_every) == 0:
                    self._prune_threat_arrays(threat_count, threat_last, self.memory_threat_max_cells)
            return
        clipped = self._clip_observed_cells(coords[0], coords[1], threat_count.shape)
        if clipped is not None:
            gx, gy = clipped
            prev_last = threat_last[gy, gx]
            prev_count = threat_count[gy, gx]
            new_count = np.where(prev_last == (step - 1), prev_count + 1, 1)
            threat_count[gy, gx] = new_count
            threat_last[gy, gx] = step
            persist_mask = new_count >= self.memory_threat_persist
            if np.any(persist_mask):
                py = gy[persist_mask]
                px = gx[persist_mask]
                wall_mask[py, px] = 1.0
                wall_last[py, px] = step
        if self.memory_threat_max_cells > 0:
            if self.memory_threat_prune_every > 0 and (step % self.memory_threat_prune_every) == 0:
                self._prune_threat_arrays(threat_count, threat_last, self.memory_threat_max_cells)

    def _apply_wall_memory(
        self,
        grid: np.ndarray,
        info: dict,
        state: dict,
        *,
        base_cell: tuple[int, int] | None = None,
    ) -> np.ndarray:
        if not self.memory_walls_enabled:
            return grid
        if base_cell is None:
            pos = info.get("pos")
            if pos is None:
                return grid
            base_cell = self._cell_from_pos(np.asarray(pos, dtype=np.float32))
        center = int(grid.shape[0] // 2)
        wall_mask, _wall_last = self._ensure_wall_arrays(state)
        if wall_mask is None or not np.any(wall_mask):
            return grid

        x0 = int(base_cell[0] - center)
        y0 = int(base_cell[1] - center)
        x1 = x0 + grid.shape[1]
        y1 = y0 + grid.shape[0]
        gx0 = max(0, x0)
        gy0 = max(0, y0)
        gx1 = min(wall_mask.shape[1], x1)
        gy1 = min(wall_mask.shape[0], y1)
        if gx0 >= gx1 or gy0 >= gy1:
            return grid
        sx0 = gx0 - x0
        sy0 = gy0 - y0
        sx1 = sx0 + (gx1 - gx0)
        sy1 = sy0 + (gy1 - gy0)
        out = np.asarray(grid, dtype=np.float32)
        view = out[sy0:sy1, sx0:sx1]
        np.maximum(view, wall_mask[gy0:gy1, gx0:gx1], out=view)
        return out

    def _memory_score(self, mem: dict, cell: tuple[int, int], step: int) -> float:
        entry = mem.get(cell)
        if entry is None:
            return 0.0
        score, last = entry
        age = max(0, step - int(last))
        if age <= 0:
            return float(score)
        decay = float(self.memory_decay)
        if decay <= 0.0:
            return 0.0
        # Экспоненциальное затухание ускоряет забывание «старых» посещений.
        decayed = float(score) * (decay**age)
        if decayed < 1e-3:
            mem.pop(cell, None)
            return 0.0
        return decayed

    def _prune_memory(self, mem: dict) -> None:
        if not mem:
            return
        drop = max(1, int(len(mem) * 0.1))
        for _ in range(drop):
            if not mem:
                break
            self._pop_oldest(mem)

    def _prune_wall_memory(self, mem: dict, wall_mask: np.ndarray | None = None) -> None:
        if not mem:
            return
        drop = max(1, int(len(mem) * 0.1))
        for _ in range(drop):
            if not mem:
                break
            self._pop_oldest(mem, mask=wall_mask)

    def _prune_threat_memory(self, mem: dict) -> None:
        if not mem:
            return
        drop = max(1, int(len(mem) * 0.1))
        for _ in range(drop):
            if not mem:
                break
            self._pop_oldest(mem)

    def _pop_oldest(self, mem: dict, *, mask: np.ndarray | None = None) -> None:
        if not mem:
            return
        try:
            if isinstance(mem, OrderedDict):
                key, _ = mem.popitem(last=False)
                if mask is not None:
                    try:
                        x, y = key
                        if 0 <= int(y) < mask.shape[0] and 0 <= int(x) < mask.shape[1]:
                            mask[int(y), int(x)] = 0.0
                    except Exception:
                        pass
                return
        except Exception:
            pass
        key = next(iter(mem))
        mem.pop(key, None)
        if mask is not None:
            try:
                x, y = key
                if 0 <= int(y) < mask.shape[0] and 0 <= int(x) < mask.shape[1]:
                    mask[int(y), int(x)] = 0.0
            except Exception:
                pass

    def _wall_mask_shape(self) -> tuple[int, int] | None:
        if self._field_size is None:
            return None
        if self.cell_size <= 0:
            return None
        size = int(max(1, np.ceil(float(self._field_size) / float(self.cell_size))))
        return (size, size)

    def _ensure_wall_arrays(self, state: dict | None) -> tuple[np.ndarray | None, np.ndarray | None]:
        shape = self._wall_mask_shape()
        if shape is None:
            return None, None
        use_shared = bool(
            self.memory_shared and state is not None and state.get("wall_memory") is self._shared_wall_memory
        )
        if use_shared:
            if self._shared_wall_mask is None or self._shared_wall_mask.shape != shape:
                self._shared_wall_mask = np.zeros(shape, dtype=np.float32)
                self._shared_wall_last = np.zeros(shape, dtype=np.int32)
            return self._shared_wall_mask, self._shared_wall_last
        if state is None:
            return None, None
        mask = state.get("wall_memory_mask")
        last = state.get("wall_memory_last")
        if mask is None or mask.shape != shape:
            mask = np.zeros(shape, dtype=np.float32)
            state["wall_memory_mask"] = mask
        if last is None or last.shape != shape:
            last = np.zeros(shape, dtype=np.int32)
            state["wall_memory_last"] = last
        return mask, last

    def _ensure_threat_arrays(self, state: dict | None) -> tuple[np.ndarray | None, np.ndarray | None]:
        shape = self._wall_mask_shape()
        if shape is None or state is None:
            return None, None
        count = state.get("threat_memory_count")
        last = state.get("threat_memory_last")
        if count is None or count.shape != shape:
            count = np.zeros(shape, dtype=np.int16)
            state["threat_memory_count"] = count
        if last is None or last.shape != shape:
            last = np.zeros(shape, dtype=np.int32)
            state["threat_memory_last"] = last
        return count, last

    def _prune_mask_by_age(self, mask: np.ndarray, last: np.ndarray, max_cells: int) -> None:
        if max_cells <= 0:
            return
        active = mask > 0.0
        count = int(np.count_nonzero(active))
        if count <= max_cells:
            return
        drop = count - max_cells
        active_last = last[active]
        if active_last.size == 0:
            return
        if drop >= active_last.size:
            mask[active] = 0.0
            last[active] = 0
            return
        cutoff = np.partition(active_last, drop - 1)[drop - 1]
        drop_mask = active & (last <= cutoff)
        if np.any(drop_mask):
            mask[drop_mask] = 0.0
            last[drop_mask] = 0

    def _prune_threat_arrays(self, count: np.ndarray, last: np.ndarray, max_cells: int) -> None:
        if max_cells <= 0:
            return
        active = count > 0
        total = int(np.count_nonzero(active))
        if total <= max_cells:
            return
        drop = total - max_cells
        active_last = last[active]
        if active_last.size == 0:
            return
        if drop >= active_last.size:
            count[active] = 0
            last[active] = 0
            return
        cutoff = np.partition(active_last, drop - 1)[drop - 1]
        drop_mask = active & (last <= cutoff)
        if np.any(drop_mask):
            count[drop_mask] = 0
            last[drop_mask] = 0

    def _cell_from_pos(self, pos: np.ndarray) -> tuple[int, int]:
        p = np.asarray(pos, dtype=np.float32)
        return (int(np.floor(p[0] / self.cell_size)), int(np.floor(p[1] / self.cell_size)))

    def _is_stuck(self, info: dict, dist_m: float | None, dist_normed: bool) -> bool:
        if self.stuck_steps <= 0:
            return False
        idx = agent_index_from_id(info.get("agent_id", ""), info)
        if idx is None:
            return False
        if dist_m is None:
            return False
        if dist_normed and self._field_size is None:
            return False

        idx = int(idx)
        state = self._agent_state.setdefault(
            idx,
            {
                "best_dist": None,
                "since_improve": 0,
                "escape_steps": 0,
                "escape_vec": None,
            },
        )
        best = state["best_dist"]
        if best is None or dist_m < best - self.stuck_dist_eps:
            state["best_dist"] = dist_m
            state["since_improve"] = 0
            state["escape_steps"] = 0
            state["escape_vec"] = None
        else:
            state["since_improve"] += 1
        return state["since_improve"] >= self.stuck_steps


if _NUMBA_AVAILABLE:

    @njit(cache=True)
    def _heap_push(hx, hy, hf, size, x, y, f):
        if size >= hx.shape[0]:
            return -1
        i = size
        hx[i] = x
        hy[i] = y
        hf[i] = f
        size += 1
        while i > 0:
            p = (i - 1) // 2
            if hf[i] < hf[p]:
                tx = hx[p]
                ty = hy[p]
                tf = hf[p]
                hx[p] = hx[i]
                hy[p] = hy[i]
                hf[p] = hf[i]
                hx[i] = tx
                hy[i] = ty
                hf[i] = tf
                i = p
            else:
                break
        return size

    @njit(cache=True)
    def _heap_pop(hx, hy, hf, size):
        if size == 0:
            return -1, -1, 0.0, 0
        x = hx[0]
        y = hy[0]
        f = hf[0]
        size -= 1
        if size > 0:
            hx[0] = hx[size]
            hy[0] = hy[size]
            hf[0] = hf[size]
            i = 0
            while True:
                left = 2 * i + 1
                if left >= size:
                    break
                right = left + 1
                smallest = left
                if right < size and hf[right] < hf[left]:
                    smallest = right
                if hf[smallest] < hf[i]:
                    tx = hx[i]
                    ty = hy[i]
                    tf = hf[i]
                    hx[i] = hx[smallest]
                    hy[i] = hy[smallest]
                    hf[i] = hf[smallest]
                    hx[smallest] = tx
                    hy[smallest] = ty
                    hf[smallest] = tf
                    i = smallest
                else:
                    break
        return x, y, f, size

    @njit(cache=True)
    def _astar_numba(
        grid,
        start_x,
        start_y,
        goal_x,
        goal_y,
        alpha,
        allow_diagonal,
        penalty_grid,
        has_penalty,
        wall_mask,
        has_wall_mask,
        heuristic_weight,
        turn_penalty,
    ):
        height, width = grid.shape
        if start_x == goal_x and start_y == goal_y:
            path_x = np.empty(1, dtype=np.int32)
            path_y = np.empty(1, dtype=np.int32)
            path_x[0] = start_x
            path_y[0] = start_y
            return path_x, path_y, 1, False

        g_score = np.full((height, width), np.inf, dtype=np.float32)
        g_score[start_y, start_x] = 0.0
        closed = np.zeros((height, width), dtype=np.uint8)
        came_x = np.full((height, width), -1, dtype=np.int32)
        came_y = np.full((height, width), -1, dtype=np.int32)
        max_nodes = height * width * 8 + 1
        heap_x = np.empty(max_nodes, dtype=np.int32)
        heap_y = np.empty(max_nodes, dtype=np.int32)
        heap_f = np.empty(max_nodes, dtype=np.float32)
        heap_size = 0
        heap_size = _heap_push(heap_x, heap_y, heap_f, heap_size, start_x, start_y, 0.0)
        if heap_size < 0:
            path_x = np.empty(1, dtype=np.int32)
            path_y = np.empty(1, dtype=np.int32)
            path_x[0] = start_x
            path_y[0] = start_y
            return path_x, path_y, 1, True
        diag_cost = math.sqrt(2.0)
        move_count = 8 if allow_diagonal else 4
        hw = float(heuristic_weight) if heuristic_weight > 0.0 else 0.0

        while heap_size > 0:
            cx, cy, _f, heap_size = _heap_pop(heap_x, heap_y, heap_f, heap_size)
            if cx < 0:
                break
            if closed[cy, cx] == 1:
                continue
            if cx == goal_x and cy == goal_y:
                break
            closed[cy, cx] = 1
            prev_dx = 0
            prev_dy = 0
            prev_norm = 0.0
            px = came_x[cy, cx]
            py = came_y[cy, cx]
            if px >= 0:
                prev_dx = cx - px
                prev_dy = cy - py
                prev_norm = math.sqrt(float(prev_dx * prev_dx + prev_dy * prev_dy))
            for i in range(move_count):
                dx = MOVES_DX[i]
                dy = MOVES_DY[i]
                nx = cx + dx
                ny = cy + dy
                if nx < 0 or ny < 0 or nx >= width or ny >= height:
                    continue
                if has_wall_mask and wall_mask[ny, nx]:
                    continue
                cost_cell = float(grid[ny, nx])
                step_cost = diag_cost if (dx != 0 and dy != 0) else 1.0
                if turn_penalty > 0.0 and prev_norm > 0.0:
                    cur_norm = math.sqrt(float(dx * dx + dy * dy))
                    if cur_norm > 0.0:
                        cos = (prev_dx * dx + prev_dy * dy) / (prev_norm * cur_norm)
                        if cos < -1.0:
                            cos = -1.0
                        elif cos > 1.0:
                            cos = 1.0
                        step_cost += turn_penalty * (1.0 - cos)
                tentative = g_score[cy, cx] + step_cost + (alpha * cost_cell)
                if has_penalty:
                    tentative += float(penalty_grid[ny, nx])
                if tentative < g_score[ny, nx]:
                    g_score[ny, nx] = tentative
                    came_x[ny, nx] = cx
                    came_y[ny, nx] = cy
                    dxh = abs(nx - goal_x)
                    dyh = abs(ny - goal_y)
                    if allow_diagonal:
                        h = (dxh + dyh) + (diag_cost - 2.0) * min(dxh, dyh)
                    else:
                        h = dxh + dyh
                    f = tentative + hw * h
                    heap_size = _heap_push(heap_x, heap_y, heap_f, heap_size, nx, ny, f)
                    if heap_size < 0:
                        path_x = np.empty(1, dtype=np.int32)
                        path_y = np.empty(1, dtype=np.int32)
                        path_x[0] = start_x
                        path_y[0] = start_y
                        return path_x, path_y, 1, True

        if came_x[goal_y, goal_x] == -1:
            path_x = np.empty(1, dtype=np.int32)
            path_y = np.empty(1, dtype=np.int32)
            path_x[0] = start_x
            path_y[0] = start_y
            return path_x, path_y, 1, False

        max_path = height * width + 1
        path_x = np.empty(max_path, dtype=np.int32)
        path_y = np.empty(max_path, dtype=np.int32)
        length = 0
        cx = goal_x
        cy = goal_y
        while True:
            path_x[length] = cx
            path_y[length] = cy
            length += 1
            if cx == start_x and cy == start_y:
                break
            px = came_x[cy, cx]
            py = came_y[cy, cx]
            if px < 0:
                break
            cx = px
            cy = py
        # Разворачиваем путь на месте.
        for i in range(length // 2):
            j = length - 1 - i
            tx = path_x[i]
            ty = path_y[i]
            path_x[i] = path_x[j]
            path_y[i] = path_y[j]
            path_x[j] = tx
            path_y[j] = ty
        return path_x, path_y, length, False

    @njit(cache=True)
    def _astar_numba_reuse(
        grid,
        start_x,
        start_y,
        goal_x,
        goal_y,
        alpha,
        allow_diagonal,
        penalty_grid,
        has_penalty,
        wall_mask,
        has_wall_mask,
        heuristic_weight,
        turn_penalty,
        g_score,
        closed,
        came_x,
        came_y,
        heap_x,
        heap_y,
        heap_f,
        path_x,
        path_y,
    ):
        height, width = grid.shape
        if start_x == goal_x and start_y == goal_y:
            path_x[0] = start_x
            path_y[0] = start_y
            return 1, False

        g_score.fill(np.inf)
        g_score[start_y, start_x] = 0.0
        closed.fill(0)
        came_x.fill(-1)
        came_y.fill(-1)
        heap_size = 0
        heap_size = _heap_push(heap_x, heap_y, heap_f, heap_size, start_x, start_y, 0.0)
        if heap_size < 0:
            path_x[0] = start_x
            path_y[0] = start_y
            return 1, True

        diag_cost = math.sqrt(2.0)
        move_count = 8 if allow_diagonal else 4
        hw = float(heuristic_weight) if heuristic_weight > 0.0 else 0.0

        while heap_size > 0:
            cx, cy, _f, heap_size = _heap_pop(heap_x, heap_y, heap_f, heap_size)
            if cx < 0:
                break
            if closed[cy, cx] == 1:
                continue
            if cx == goal_x and cy == goal_y:
                break
            closed[cy, cx] = 1
            prev_dx = 0
            prev_dy = 0
            prev_norm = 0.0
            px = came_x[cy, cx]
            py = came_y[cy, cx]
            if px >= 0:
                prev_dx = cx - px
                prev_dy = cy - py
                prev_norm = math.sqrt(float(prev_dx * prev_dx + prev_dy * prev_dy))
            for i in range(move_count):
                dx = MOVES_DX[i]
                dy = MOVES_DY[i]
                nx = cx + dx
                ny = cy + dy
                if nx < 0 or ny < 0 or nx >= width or ny >= height:
                    continue
                if has_wall_mask and wall_mask[ny, nx]:
                    continue
                cost_cell = float(grid[ny, nx])
                step_cost = diag_cost if (dx != 0 and dy != 0) else 1.0
                if turn_penalty > 0.0 and prev_norm > 0.0:
                    cur_norm = math.sqrt(float(dx * dx + dy * dy))
                    if cur_norm > 0.0:
                        cos = (prev_dx * dx + prev_dy * dy) / (prev_norm * cur_norm)
                        if cos < -1.0:
                            cos = -1.0
                        elif cos > 1.0:
                            cos = 1.0
                        step_cost += turn_penalty * (1.0 - cos)
                tentative = g_score[cy, cx] + step_cost + (alpha * cost_cell)
                if has_penalty:
                    tentative += float(penalty_grid[ny, nx])
                if tentative < g_score[ny, nx]:
                    g_score[ny, nx] = tentative
                    came_x[ny, nx] = cx
                    came_y[ny, nx] = cy
                    dxh = abs(nx - goal_x)
                    dyh = abs(ny - goal_y)
                    if allow_diagonal:
                        h = (dxh + dyh) + (diag_cost - 2.0) * min(dxh, dyh)
                    else:
                        h = dxh + dyh
                    f = tentative + hw * h
                    heap_size = _heap_push(heap_x, heap_y, heap_f, heap_size, nx, ny, f)
                    if heap_size < 0:
                        path_x[0] = start_x
                        path_y[0] = start_y
                        return 1, True

        if came_x[goal_y, goal_x] == -1:
            path_x[0] = start_x
            path_y[0] = start_y
            return 1, False

        length = 0
        cx = goal_x
        cy = goal_y
        while True:
            if length >= path_x.shape[0]:
                path_x[0] = start_x
                path_y[0] = start_y
                return 1, True
            path_x[length] = cx
            path_y[length] = cy
            length += 1
            if cx == start_x and cy == start_y:
                break
            px = came_x[cy, cx]
            py = came_y[cy, cx]
            if px < 0:
                break
            cx = px
            cy = py
        for i in range(length // 2):
            j = length - 1 - i
            tx = path_x[i]
            ty = path_y[i]
            path_x[i] = path_x[j]
            path_y[i] = path_y[j]
            path_x[j] = tx
            path_y[j] = ty
        return length, False


def _astar_py(
    grid: np.ndarray,
    start,
    goal,
    alpha: float = 2.0,
    allow_diagonal: bool = True,
    penalty_grid: np.ndarray | None = None,
    wall_mask: np.ndarray | None = None,
    heuristic_weight: float = 1.0,
    turn_penalty: float = 0.0,
):
    if start == goal:
        return [start]

    def h(a, b):
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        if allow_diagonal:
            d2 = np.sqrt(2.0)
            return (dx + dy) + (d2 - 2.0) * min(dx, dy)
        return dx + dy

    open_set = []
    heapq.heappush(open_set, (0.0, start))
    came_from = {}
    g_score = {start: 0.0}
    closed = set()
    height, width = grid.shape

    moves = MOVES_DIAGONAL if allow_diagonal else MOVES_CARDINAL

    while open_set:
        _, current = heapq.heappop(open_set)
        if current in closed:
            continue
        if current == goal:
            break
        closed.add(current)

        prev = came_from.get(current)
        prev_dx = prev_dy = 0
        prev_norm = 0.0
        if prev is not None:
            prev_dx = current[0] - prev[0]
            prev_dy = current[1] - prev[1]
            prev_norm = np.sqrt(float(prev_dx * prev_dx + prev_dy * prev_dy))

        for dx, dy in moves:
            nx = current[0] + dx
            ny = current[1] + dy
            if nx < 0 or nx >= width or ny < 0 or ny >= height:
                continue
            if wall_mask is not None and bool(wall_mask[ny, nx]):
                continue
            cost_cell = float(grid[ny, nx])
            penalty = 0.0
            if penalty_grid is not None:
                try:
                    penalty = float(penalty_grid[ny, nx])
                except Exception:
                    penalty = 0.0
            step_cost = (np.sqrt(2.0) if (dx != 0 and dy != 0) else 1.0) + alpha * cost_cell
            if turn_penalty > 0.0 and prev_norm > 0.0:
                cur_norm = np.sqrt(float(dx * dx + dy * dy))
                if cur_norm > 0.0:
                    cos = (prev_dx * dx + prev_dy * dy) / (prev_norm * cur_norm)
                    cos = max(-1.0, min(1.0, float(cos)))
                    step_cost += float(turn_penalty) * (1.0 - cos)
            if penalty > 0.0:
                step_cost += penalty
            tentative = g_score[current] + step_cost
            neighbor = (nx, ny)
            if tentative < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative
                w = max(0.0, float(heuristic_weight))
                f = tentative + w * h(neighbor, goal)
                heapq.heappush(open_set, (f, neighbor))

    if goal not in came_from:
        return [start]

    path = [goal]
    cur = goal
    while cur in came_from:
        cur = came_from[cur]
        path.append(cur)
        if cur == start:
            break
    path.reverse()
    return path


def _astar(
    grid: np.ndarray,
    start,
    goal,
    alpha: float = 2.0,
    allow_diagonal: bool = True,
    penalty_grid: np.ndarray | None = None,
    wall_mask: np.ndarray | None = None,
    heuristic_weight: float = 1.0,
    turn_penalty: float = 0.0,
    scratch: dict[str, np.ndarray] | None = None,
):
    if not _NUMBA_AVAILABLE:
        return _astar_py(
            grid,
            start,
            goal,
            alpha=alpha,
            allow_diagonal=allow_diagonal,
            penalty_grid=penalty_grid,
            wall_mask=wall_mask,
            heuristic_weight=heuristic_weight,
            turn_penalty=turn_penalty,
        )
    height, width = grid.shape
    gx, gy = int(goal[0]), int(goal[1])
    sx, sy = int(start[0]), int(start[1])
    pen = None
    mask = None
    has_penalty = False
    has_mask = False
    if scratch is not None:
        pen = scratch["penalty"]
        mask = scratch["mask"]
        if penalty_grid is not None:
            np.copyto(pen, np.asarray(penalty_grid, dtype=np.float32), casting="unsafe")
            has_penalty = True
        if wall_mask is not None:
            np.copyto(mask, np.asarray(wall_mask, dtype=np.uint8), casting="unsafe")
            has_mask = True
    else:
        pen = np.zeros((height, width), dtype=np.float32)
        if penalty_grid is not None:
            pen = np.asarray(penalty_grid, dtype=np.float32)
            has_penalty = True
        mask = np.zeros((height, width), dtype=np.uint8)
        if wall_mask is not None:
            mask = np.asarray(wall_mask, dtype=np.uint8)
            has_mask = True
    try:
        if scratch is not None:
            length, overflow = _astar_numba_reuse(
                np.asarray(grid, dtype=np.float32),
                sx,
                sy,
                gx,
                gy,
                float(alpha),
                bool(allow_diagonal),
                pen,
                has_penalty,
                mask,
                has_mask,
                float(heuristic_weight),
                float(turn_penalty),
                scratch["g_score"],
                scratch["closed"],
                scratch["came_x"],
                scratch["came_y"],
                scratch["heap_x"],
                scratch["heap_y"],
                scratch["heap_f"],
                scratch["path_x"],
                scratch["path_y"],
            )
            path_x = scratch["path_x"]
            path_y = scratch["path_y"]
        else:
            path_x, path_y, length, overflow = _astar_numba(
                np.asarray(grid, dtype=np.float32),
                sx,
                sy,
                gx,
                gy,
                float(alpha),
                bool(allow_diagonal),
                pen,
                has_penalty,
                mask,
                has_mask,
                float(heuristic_weight),
                float(turn_penalty),
            )
        if overflow:
            raise RuntimeError("numba astar heap overflow")
    except Exception:
        return _astar_py(
            grid,
            start,
            goal,
            alpha=alpha,
            allow_diagonal=allow_diagonal,
            penalty_grid=penalty_grid,
            wall_mask=wall_mask,
            heuristic_weight=heuristic_weight,
            turn_penalty=turn_penalty,
        )
    if length <= 0:
        return [start]
    return [(int(path_x[i]), int(path_y[i])) for i in range(length)]


if _NUMBA_AVAILABLE:

    @njit(cache=True)
    def _dijkstra_risk_numba(
        grid,
        start_x,
        start_y,
        wall_mask,
        has_wall_mask,
        allow_diagonal,
    ):
        height, width = grid.shape
        dist = np.full((height, width), np.inf, dtype=np.float32)
        dist[start_y, start_x] = 0.0
        max_nodes = height * width * 8 + 1
        heap_x = np.empty(max_nodes, dtype=np.int32)
        heap_y = np.empty(max_nodes, dtype=np.int32)
        heap_f = np.empty(max_nodes, dtype=np.float32)
        heap_size = 0
        heap_size = _heap_push(heap_x, heap_y, heap_f, heap_size, start_x, start_y, 0.0)
        if heap_size < 0:
            return dist, True
        move_count = 8 if allow_diagonal else 4

        while heap_size > 0:
            cx, cy, cost, heap_size = _heap_pop(heap_x, heap_y, heap_f, heap_size)
            if cx < 0:
                break
            if cost > dist[cy, cx]:
                continue
            for i in range(move_count):
                dx = MOVES_DX[i]
                dy = MOVES_DY[i]
                nx = cx + dx
                ny = cy + dy
                if nx < 0 or nx >= width or ny < 0 or ny >= height:
                    continue
                if has_wall_mask and wall_mask[ny, nx]:
                    continue
                step_cost = float(grid[ny, nx])
                new_cost = cost + step_cost
                if new_cost < dist[ny, nx]:
                    dist[ny, nx] = new_cost
                    heap_size = _heap_push(heap_x, heap_y, heap_f, heap_size, nx, ny, new_cost)
                    if heap_size < 0:
                        return dist, True
        return dist, False

    @njit(cache=True)
    def _dijkstra_risk_numba_reuse(
        grid,
        start_x,
        start_y,
        wall_mask,
        has_wall_mask,
        allow_diagonal,
        dist,
        heap_x,
        heap_y,
        heap_f,
    ):
        height, width = grid.shape
        dist.fill(np.inf)
        dist[start_y, start_x] = 0.0
        heap_size = 0
        heap_size = _heap_push(heap_x, heap_y, heap_f, heap_size, start_x, start_y, 0.0)
        if heap_size < 0:
            return True
        move_count = 8 if allow_diagonal else 4

        while heap_size > 0:
            cx, cy, cost, heap_size = _heap_pop(heap_x, heap_y, heap_f, heap_size)
            if cx < 0:
                break
            if cost > dist[cy, cx]:
                continue
            for i in range(move_count):
                dx = MOVES_DX[i]
                dy = MOVES_DY[i]
                nx = cx + dx
                ny = cy + dy
                if nx < 0 or nx >= width or ny < 0 or ny >= height:
                    continue
                if has_wall_mask and wall_mask[ny, nx]:
                    continue
                step_cost = float(grid[ny, nx])
                new_cost = cost + step_cost
                if new_cost < dist[ny, nx]:
                    dist[ny, nx] = new_cost
                    heap_size = _heap_push(heap_x, heap_y, heap_f, heap_size, nx, ny, new_cost)
                    if heap_size < 0:
                        return True
        return False


def _dijkstra_risk_py(
    grid: np.ndarray,
    start,
    wall_mask: np.ndarray | None,
    allow_diagonal: bool,
    dist_buf: np.ndarray | None = None,
):
    height, width = grid.shape
    dist = None
    if dist_buf is not None and dist_buf.shape == (height, width):
        dist = dist_buf
        dist.fill(float("inf"))
    if dist is None:
        dist = np.full((height, width), float("inf"), dtype=np.float32)
    dist[start[1], start[0]] = 0.0
    heap = [(0.0, start)]
    moves = MOVES_DIAGONAL if allow_diagonal else MOVES_CARDINAL

    while heap:
        cost, (cx, cy) = heapq.heappop(heap)
        if cost > dist[cy, cx]:
            continue
        for dx, dy in moves:
            nx = cx + dx
            ny = cy + dy
            if nx < 0 or nx >= width or ny < 0 or ny >= height:
                continue
            if wall_mask is not None and bool(wall_mask[ny, nx]):
                continue
            step_cost = float(grid[ny, nx])
            new_cost = cost + step_cost
            if new_cost < dist[ny, nx]:
                dist[ny, nx] = new_cost
                heapq.heappush(heap, (new_cost, (nx, ny)))
    return dist


def _dijkstra_risk(
    grid: np.ndarray,
    start,
    wall_mask: np.ndarray | None,
    allow_diagonal: bool,
    dist_buf: np.ndarray | None = None,
    scratch: dict[str, np.ndarray] | None = None,
):
    if not _NUMBA_AVAILABLE:
        return _dijkstra_risk_py(grid, start, wall_mask, allow_diagonal, dist_buf=dist_buf)
    height, width = grid.shape
    has_mask = False
    if scratch is not None:
        mask = scratch["mask"]
        if wall_mask is not None:
            np.copyto(mask, np.asarray(wall_mask, dtype=np.uint8), casting="unsafe")
            has_mask = True
    else:
        mask = np.zeros((height, width), dtype=np.uint8)
        if wall_mask is not None:
            mask = np.asarray(wall_mask, dtype=np.uint8)
            has_mask = True
    try:
        if scratch is not None:
            dist = scratch["dist"]
            overflow = _dijkstra_risk_numba_reuse(
                np.asarray(grid, dtype=np.float32),
                int(start[0]),
                int(start[1]),
                mask,
                has_mask,
                bool(allow_diagonal),
                dist,
                scratch["heap_x"],
                scratch["heap_y"],
                scratch["heap_f"],
            )
        else:
            dist, overflow = _dijkstra_risk_numba(
                np.asarray(grid, dtype=np.float32),
                int(start[0]),
                int(start[1]),
                mask,
                has_mask,
                bool(allow_diagonal),
            )
        if overflow:
            raise RuntimeError("numba dijkstra heap overflow")
    except Exception:
        return _dijkstra_risk_py(grid, start, wall_mask, allow_diagonal, dist_buf=dist_buf)
    if dist_buf is not None and dist_buf.shape == dist.shape:
        dist_buf[:] = dist
        return dist_buf
    return dist
