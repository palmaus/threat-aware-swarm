"""Менеджер оракула кратчайшего пути и кэширования."""

from __future__ import annotations

import numpy as np

from env.oracles.grid_oracle import (
    build_distance_field,
    build_occupancy_grid,
    build_risk_grid,
    path_from_distance_field,
)
from env.utils.geometry import parse_circle, parse_wall_rect


class OracleManager:
    def __init__(self, env) -> None:
        self._env = env
        self._oracle_walls: list = []
        self.optimal_len = np.full(self._env.config.n_agents, float("nan"), dtype=np.float32)
        self._path: list = []
        self._path_cache_key = None
        self.grid = None
        self.cache_key = None
        self.risk_grid = None
        self.risk_cache_key = None
        self.risk_grad = None
        self.risk_grad_cache_key = None
        self.dist_field = None
        self.dist_cache_key = None
        self.dist_grad = None
        self.dist_grad_cache_key = None
        self._last_target_pos: np.ndarray | None = None

    def set_walls(self, rects: list) -> None:
        self._oracle_walls = list(rects or [])

    def optimality_gap(self, idx: int) -> float:
        opt = float(self.optimal_len[idx])
        if not np.isfinite(opt):
            return float("nan")
        if opt <= 1e-6:
            return 1.0
        path_len = self._path_lengths()
        return float(path_len[idx] / opt)

    def close(self) -> None:
        return

    def _path_lengths(self) -> np.ndarray:
        getter = getattr(self._env, "get_path_lengths", None)
        if callable(getter):
            return np.asarray(getter(copy=False), dtype=np.float32)
        value = getattr(self._env, "_path_len", None)
        if value is None:
            value = np.zeros(int(self._env.config.n_agents), dtype=np.float32)
        return np.asarray(value, dtype=np.float32)

    def _time_step(self) -> int:
        getter = getattr(self._env, "get_time_step", None)
        if callable(getter):
            return int(getter())
        sim = getattr(self._env, "sim", None)
        return int(getattr(sim, "time_step", 0))

    def _agent_positions(self) -> np.ndarray:
        getter = getattr(self._env, "get_agent_positions", None)
        if callable(getter):
            return np.asarray(getter(copy=True), dtype=np.float32)
        sim = getattr(self._env, "sim", None)
        fallback = np.zeros((int(self._env.config.n_agents), 2), dtype=np.float32)
        return np.asarray(getattr(sim, "agents_pos", fallback), dtype=np.float32).copy()

    def _threats(self) -> tuple:
        getter = getattr(self._env, "get_threats", None)
        if callable(getter):
            return tuple(getter())
        sim = getattr(self._env, "sim", None)
        return tuple(getattr(sim, "threats", ()) or ())

    @property
    def path(self) -> list:
        return self.get_path()

    def compute_lengths(self, *, clear_state: bool = False) -> None:
        if not self._env.oracle_enabled:
            self.optimal_len[:] = float("nan")
            self._path = []
            self._path_cache_key = None
            self.grid = None
            self.cache_key = None
            self.dist_field = None
            self.dist_cache_key = None
            self.dist_grad = None
            self.dist_grad_cache_key = None
            return
        interval = int(getattr(self._env.config, "oracle_update_interval", 1))
        if interval > 1 and (self._time_step() % interval) != 0:
            return
        if clear_state:
            self.optimal_len[:] = float("nan")
            self._path = []
            self._path_cache_key = None
            self._last_target_pos = None
        obstacles = self._oracle_obstacles()
        grid = None
        risk_grid = None
        inflation = float(self._env.config.agent_radius) + float(
            getattr(self._env.config, "oracle_inflation_buffer", 0.0)
        )
        target = np.asarray(self._env.target_pos, dtype=np.float32)
        cell_size = float(self._env.oracle_cell_size)
        if obstacles:
            cache_key = self._oracle_cache_key_for(obstacles)
            if cache_key == self.cache_key and self.grid is not None:
                grid = self.grid
            else:
                grid = build_occupancy_grid(
                    self._env.config.field_size,
                    self._env.oracle_cell_size,
                    inflation_radius=inflation,
                    walls=obstacles,
                )
                self.grid = grid
                self.cache_key = cache_key
        else:
            self.grid = None
            self.cache_key = None

        risk_weight = float(getattr(self._env.config, "oracle_risk_weight", 0.0))
        if risk_weight > 0.0:
            sources = self._risk_sources()
            if sources:
                risk_key = self._risk_cache_key_for(sources)
                if risk_key == self.risk_cache_key and self.risk_grid is not None:
                    risk_grid = self.risk_grid
                else:
                    risk_grid = build_risk_grid(
                        self._env.config.field_size,
                        self._env.oracle_cell_size,
                        threats=sources,
                        inflation_radius=inflation,
                    )
                    self.risk_grid = risk_grid
                    self.risk_cache_key = risk_key
                if risk_grid is not None:
                    if self.risk_grad_cache_key != risk_key:
                        self.risk_grad = self._build_dist_grad(risk_grid)
                        self.risk_grad_cache_key = risk_key
                else:
                    self.risk_grad = None
                    self.risk_grad_cache_key = None
            else:
                self.risk_grid = None
                self.risk_cache_key = None
                self.risk_grad = None
                self.risk_grad_cache_key = None
        else:
            self.risk_grid = None
            self.risk_cache_key = None
            self.risk_grad = None
            self.risk_grad_cache_key = None
        if self._env.config.n_agents > 0:
            field_size = float(self._env.config.field_size)
            dist_key = None
            if grid is not None:
                height, width = grid.shape
                gx = int(np.clip(np.floor(target[0] / cell_size), 0, width - 1))
                gy = int(np.clip(np.floor(target[1] / cell_size), 0, height - 1))
                dist_key = (self.cache_key, self.risk_cache_key, gx, gy)
            use_cached = False
            if (
                dist_key is not None
                and dist_key == self.dist_cache_key
                and self.dist_field is not None
                and self._last_target_pos is not None
            ):
                shift = float(np.linalg.norm(target - self._last_target_pos))
                use_cached = shift < cell_size
            dist_grad = None
            if use_cached:
                dist_field = self.dist_field
                if self.dist_grad_cache_key == dist_key:
                    dist_grad = self.dist_grad
            else:
                dist_field = build_distance_field(
                    target,
                    field_size,
                    cell_size=cell_size,
                    inflation_radius=inflation,
                    walls=obstacles,
                    allow_diagonal=True,
                    grid=grid,
                    risk_grid=risk_grid,
                    risk_weight=risk_weight,
                )
                self.dist_field = dist_field
                self.dist_cache_key = dist_key
                dist_grad = self._build_dist_grad(dist_field)
                self.dist_grad = dist_grad
                self.dist_grad_cache_key = dist_key
            if dist_grad is None and self.dist_grad_cache_key != dist_key:
                self.dist_grad = None
                self.dist_grad_cache_key = None

            agents_pos = self._agent_positions()
            if dist_field is None:
                diff = target[None, :] - agents_pos
                self.optimal_len[:] = np.linalg.norm(diff, axis=1).astype(np.float32)
            else:
                height, width = dist_field.shape
                gx = np.clip(np.floor(agents_pos[:, 0] / cell_size), 0, width - 1).astype(np.int32)
                gy = np.clip(np.floor(agents_pos[:, 1] / cell_size), 0, height - 1).astype(np.int32)
                vals = np.asarray(dist_field[gy, gx], dtype=np.float32)
                vals[~np.isfinite(vals)] = np.float32(np.nan)
                self.optimal_len[:] = vals
            self._path = []
            self._path_cache_key = self._make_path_cache_key(target)
        self._last_target_pos = np.asarray(target, dtype=np.float32)

    def get_path(self) -> list:
        if not self._env.oracle_enabled or self._env.config.n_agents <= 0:
            return []
        key = self._make_path_cache_key(np.asarray(self._env.target_pos, dtype=np.float32))
        if self._path_cache_key == key and self._path:
            return self._path

        target = np.asarray(self._env.target_pos, dtype=np.float32)
        start = np.asarray(self._agent_positions()[0], dtype=np.float32)
        dist_field = self.dist_field
        if dist_field is None:
            self._path = [tuple(start.tolist()), tuple(target.tolist())]
        else:
            self._path = path_from_distance_field(
                start,
                target,
                dist_field,
                float(self._env.config.field_size),
                float(self._env.oracle_cell_size),
                allow_diagonal=True,
            )
        self._path_cache_key = key
        return self._path

    def direction_to_goal(self, pos: np.ndarray) -> np.ndarray:
        target = np.asarray(self._env.target_pos, dtype=np.float32)
        pos = np.asarray(pos, dtype=np.float32)
        if pos.shape != (2,):
            return np.zeros((2,), dtype=np.float32)
        grad = self.dist_grad
        if grad is None:
            vec = target - pos
            norm = float(np.linalg.norm(vec))
            if norm <= 1e-6:
                return np.zeros((2,), dtype=np.float32)
            return (vec / norm).astype(np.float32)
        cell_size = float(self._env.oracle_cell_size)
        height, width = grad.shape[:2]
        gx = int(np.clip(np.floor(pos[0] / cell_size), 0, width - 1))
        gy = int(np.clip(np.floor(pos[1] / cell_size), 0, height - 1))
        vec = grad[gy, gx].astype(np.float32)
        if not np.all(np.isfinite(vec)):
            return np.zeros((2,), dtype=np.float32)
        return vec

    def direction_to_goal_batch(self, pos: np.ndarray) -> np.ndarray:
        pos = np.asarray(pos, dtype=np.float32)
        if pos.ndim != 2 or pos.shape[1] != 2:
            return np.zeros((pos.shape[0], 2), dtype=np.float32)
        target = np.asarray(self._env.target_pos, dtype=np.float32)
        grad = self.dist_grad
        if grad is None:
            vec = target[None, :] - pos
            norm = np.linalg.norm(vec, axis=1, keepdims=True)
            out = np.zeros_like(vec, dtype=np.float32)
            mask = norm[:, 0] > 1e-6
            if np.any(mask):
                out[mask] = (vec[mask] / norm[mask]).astype(np.float32)
            return out
        cell_size = float(self._env.oracle_cell_size)
        height, width = grad.shape[:2]
        gx = np.clip(np.floor(pos[:, 0] / cell_size), 0, width - 1).astype(np.int32)
        gy = np.clip(np.floor(pos[:, 1] / cell_size), 0, height - 1).astype(np.int32)
        out = grad[gy, gx].astype(np.float32)
        invalid = ~np.isfinite(out).all(axis=1)
        if np.any(invalid):
            out[invalid] = 0.0
        return out

    def _oracle_obstacles(self) -> list:
        obstacles = list(self._oracle_walls)
        for t in self._threats():
            if bool(getattr(t, "oracle_block", False)):
                obstacles.append(t)
        return obstacles

    def _risk_sources(self) -> list:
        sources = []
        for t in self._threats():
            if bool(getattr(t, "is_dynamic", False)):
                continue
            if bool(getattr(t, "oracle_block", False)):
                continue
            try:
                intensity = float(getattr(t, "intensity", 0.0))
            except Exception:
                intensity = 0.0
            if intensity <= 0.0:
                continue
            sources.append(t)
        return sources

    def _risk_snapshot(self) -> list:
        sources = self._risk_sources()
        if not sources:
            return []
        snapshot: list = []
        for obj in sources:
            circle = parse_circle(obj)
            if circle is None:
                continue
            cx, cy, r = circle
            try:
                intensity = float(getattr(obj, "intensity", 0.0))
            except Exception:
                intensity = 0.0
            if intensity <= 0.0:
                continue
            snapshot.append({"pos": (float(cx), float(cy)), "radius": float(r), "intensity": float(intensity)})
        return snapshot

    def _oracle_snapshot(self) -> list:
        obstacles = self._oracle_obstacles()
        if not obstacles:
            return []
        snapshot: list = []
        for obj in obstacles:
            rect = parse_wall_rect(obj)
            if rect is not None:
                x1, y1, x2, y2 = rect
                snapshot.append((float(x1), float(y1), float(x2), float(y2)))
                continue
            circle = parse_circle(obj)
            if circle is not None:
                cx, cy, r = circle
                snapshot.append({"pos": (float(cx), float(cy)), "radius": float(r)})
                continue
            if isinstance(obj, dict):
                snapshot.append(dict(obj))
        return snapshot

    def _oracle_cache_key_for(self, obstacles: list) -> tuple:
        items = []
        for obj in obstacles:
            rect = parse_wall_rect(obj)
            if rect is not None:
                x1, y1, x2, y2 = rect
                items.append(("rect", float(x1), float(y1), float(x2), float(y2)))
                continue
            circle = parse_circle(obj)
            if circle is not None:
                cx, cy, r = circle
                items.append(("circle", float(cx), float(cy), float(r)))
                continue
            if isinstance(obj, dict):
                if all(k in obj for k in ("x1", "y1", "x2", "y2")):
                    items.append(("rect", float(obj["x1"]), float(obj["y1"]), float(obj["x2"]), float(obj["y2"])))
                elif "radius" in obj:
                    if "pos" in obj:
                        x, y = obj["pos"]
                    elif "center" in obj:
                        x, y = obj["center"]
                    else:
                        x, y = obj.get("x", 0.0), obj.get("y", 0.0)
                    items.append(("circle", float(x), float(y), float(obj["radius"])))
        return (
            float(self._env.config.field_size),
            float(self._env.oracle_cell_size),
            float(self._env.config.agent_radius) + float(getattr(self._env.config, "oracle_inflation_buffer", 0.0)),
            tuple(items),
        )

    def _build_dist_grad(self, dist_field: np.ndarray | None) -> np.ndarray | None:
        if dist_field is None or dist_field.size == 0:
            return None
        finite = np.isfinite(dist_field)
        if not np.any(finite):
            return None
        filled = np.array(dist_field, dtype=np.float32, copy=True)
        max_val = float(np.max(filled[finite]))
        filled[~finite] = max_val + 1.0
        grad_y, grad_x = np.gradient(filled)
        grad_x = -grad_x
        grad_y = -grad_y
        norm = np.sqrt(grad_x * grad_x + grad_y * grad_y) + 1e-6
        grad_x = grad_x / norm
        grad_y = grad_y / norm
        grad = np.stack([grad_x, grad_y], axis=-1).astype(np.float32)
        grad[~finite] = 0.0
        return grad

    def _risk_cache_key_for(self, sources: list) -> tuple:
        items = []
        for obj in sources:
            circle = parse_circle(obj)
            if circle is not None:
                cx, cy, r = circle
                try:
                    intensity = float(getattr(obj, "intensity", 0.0))
                except Exception:
                    intensity = 0.0
                items.append(("circle", float(cx), float(cy), float(r), float(intensity)))
        return (
            float(self._env.config.field_size),
            float(self._env.oracle_cell_size),
            float(self._env.config.agent_radius) + float(getattr(self._env.config, "oracle_inflation_buffer", 0.0)),
            tuple(items),
        )

    def _make_path_cache_key(self, target: np.ndarray) -> tuple:
        if self._env.config.n_agents <= 0:
            return (self.dist_cache_key, None)
        start = np.asarray(self._agent_positions()[0], dtype=np.float32)
        cell_size = float(self._env.oracle_cell_size)
        field_size = float(self._env.config.field_size)
        if self.dist_field is not None:
            height, width = self.dist_field.shape
            sx = int(np.clip(np.floor(start[0] / cell_size), 0, width - 1))
            sy = int(np.clip(np.floor(start[1] / cell_size), 0, height - 1))
            tx = int(np.clip(np.floor(target[0] / cell_size), 0, width - 1))
            ty = int(np.clip(np.floor(target[1] / cell_size), 0, height - 1))
            start_key = ("grid", sx, sy)
            target_key = ("grid", tx, ty)
        else:
            start_key = ("pos", round(float(start[0]), 4), round(float(start[1]), 4))
            target_key = (
                "pos",
                round(float(np.clip(target[0], 0.0, field_size)), 4),
                round(float(np.clip(target[1], 0.0, field_size)), 4),
            )
        return (self.dist_cache_key, start_key, target_key)
