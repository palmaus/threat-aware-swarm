from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from env.state import SimState


class ObservationBuilder:
    def __init__(
        self,
        field_size: float,
        max_speed: float,
        grid_width: int = 41,
        env_schema_version: str = "obs@1694:v5",
        grid_res: float = 1.0,
        agent_radius: float = 0.0,
        agent_grid_weight: float = 1.0,
        agent_blob_radius: int = 1,
        wall_value: float = 1.0,
        threat_base: float = 0.7,
        threat_range: float = 0.1,
        agent_value: float = 0.3,
        self_value: float = 0.1,
        obs_noise_target: float = 0.0,
        obs_noise_vel: float = 0.0,
        obs_noise_grid: float = 0.0,
        rng: np.random.Generator | None = None,
    ):
        self.field_size = float(field_size)
        self.max_speed = float(max_speed)
        self.grid_width = int(grid_width)
        self.env_schema_version = env_schema_version
        self.grid_res = float(grid_res)
        self.agent_radius = float(agent_radius)
        self.agent_grid_weight = float(agent_grid_weight)
        self.agent_blob_radius = int(agent_blob_radius)
        self.wall_value = float(wall_value)
        self.threat_base = float(threat_base)
        self.threat_range = float(threat_range)
        self.agent_value = float(agent_value)
        self.self_value = float(self_value)
        self.obs_noise_target = float(obs_noise_target)
        self.obs_noise_vel = float(obs_noise_vel)
        self.obs_noise_grid = float(obs_noise_grid)
        self._rng = rng if rng is not None else np.random.default_rng()
        self._offset_x, self._offset_y = self._build_offsets(self.grid_width, self.grid_res)
        self._abs_x = np.empty_like(self._offset_x)
        self._abs_y = np.empty_like(self._offset_y)
        self._center = self.grid_width // 2
        self._half_range = (self._center + 1) * self.grid_res
        self._global_w = None
        self._global_x = None
        self._global_y = None
        self._static_grid = None
        self._threat_grid = None
        self._threat_timestep = None
        self._static_windows = None
        self._threat_windows = None
        self._blob_dy = None
        self._blob_dx = None
        r = max(0, int(self.agent_blob_radius))
        offsets = np.arange(-r, r + 1, dtype=int)
        dy, dx = np.meshgrid(offsets, offsets, indexing="ij")
        self._blob_dy = dy.reshape(-1)
        self._blob_dx = dx.reshape(-1)
        self._registry = {
            "obs@1694:v5": self._build_v5,
        }
        self._registry_batch = {
            "obs@1694:v5": self._build_v5_batch,
        }

    def build(self, state: SimState, idx: int) -> dict[str, np.ndarray]:
        fn = self._registry.get(self.env_schema_version)
        if fn is None:
            raise ValueError(f"Неизвестная версия env_schema: {self.env_schema_version}")
        return fn(state, idx)

    def build_all(self, state: SimState, idxs: list[int]) -> list[dict[str, np.ndarray]]:
        fn = self._registry_batch.get(self.env_schema_version)
        if fn is None:
            return [self.build(state, idx) for idx in idxs]
        return fn(state, idxs)

    def set_rng(self, rng: np.random.Generator | None) -> None:
        if rng is None:
            return
        self._rng = rng

    def set_noise(
        self,
        *,
        obs_noise_target: float | None = None,
        obs_noise_vel: float | None = None,
        obs_noise_grid: float | None = None,
    ) -> None:
        if obs_noise_target is not None:
            self.obs_noise_target = float(obs_noise_target)
        if obs_noise_vel is not None:
            self.obs_noise_vel = float(obs_noise_vel)
        if obs_noise_grid is not None:
            self.obs_noise_grid = float(obs_noise_grid)

    def reset_cache(self) -> None:
        self._static_grid = None
        self._threat_grid = None
        self._threat_timestep = None
        self._static_windows = None
        self._threat_windows = None

    def _ensure_global_grid(self) -> None:
        width = int(np.ceil(self.field_size / self.grid_res)) + 1
        if self._global_w == width and self._global_x is not None and self._global_y is not None:
            return
        xs = (np.arange(width, dtype=np.float32) * float(self.grid_res))
        ys = (np.arange(width, dtype=np.float32) * float(self.grid_res))
        grid_x, grid_y = np.meshgrid(xs, ys)
        self._global_w = width
        self._global_x = grid_x.astype(np.float32)
        self._global_y = grid_y.astype(np.float32)
        self._static_grid = None
        self._threat_grid = None
        self._threat_timestep = None
        self._static_windows = None
        self._threat_windows = None

    def _build_static_grid(self, state: SimState) -> np.ndarray:
        self._ensure_global_grid()
        width = int(self._global_w)
        grid = np.zeros((width, width), dtype=np.float32)
        grid_x = self._global_x
        grid_y = self._global_y

        rects = state.static_walls or []
        if rects:
            rect_arr = None
            try:
                rect_arr = np.asarray(rects, dtype=np.float32)
            except Exception:
                rect_arr = None
            if rect_arr is not None and rect_arr.ndim == 2 and rect_arr.shape[1] >= 4:
                for x1, y1, x2, y2 in rect_arr[:, :4]:
                    mask = (grid_x >= x1) & (grid_x <= x2) & (grid_y >= y1) & (grid_y <= y2)
                    if np.any(mask):
                        grid[mask] = float(self.wall_value)
            else:
                for rect in rects:
                    try:
                        x1, y1, x2, y2 = rect
                    except Exception:
                        continue
                    mask = (grid_x >= float(x1)) & (grid_x <= float(x2)) & (grid_y >= float(y1)) & (grid_y <= float(y2))
                    if np.any(mask):
                        grid[mask] = float(self.wall_value)

        circles = getattr(state, "static_circles", []) or []
        if circles:
            try:
                circ_arr = np.asarray(circles, dtype=np.float32)
            except Exception:
                circ_arr = None
            if circ_arr is not None and circ_arr.ndim == 2 and circ_arr.shape[1] >= 3:
                for cx, cy, r in circ_arr[:, :3]:
                    dx = grid_x - cx
                    dy = grid_y - cy
                    mask = (dx * dx + dy * dy) <= (r * r)
                    if np.any(mask):
                        grid[mask] = float(self.wall_value)
            else:
                for circ in circles:
                    try:
                        cx, cy, r = circ
                    except Exception:
                        continue
                    r = float(r)
                    dx = grid_x - float(cx)
                    dy = grid_y - float(cy)
                    mask = (dx * dx + dy * dy) <= (r * r)
                    if np.any(mask):
                        grid[mask] = float(self.wall_value)

        inflate = float(self.agent_radius)
        border_mask = (
            (grid_x <= inflate)
            | (grid_x >= (self.field_size - inflate))
            | (grid_y <= inflate)
            | (grid_y >= (self.field_size - inflate))
        )
        grid[border_mask] = float(self.wall_value)
        return grid.astype(np.float32)

    def _build_static_windows(self, state: SimState) -> np.ndarray:
        base_grid = self._build_static_grid(state)
        grid = base_grid
        pad = int(self._center)
        if pad > 0:
            grid = np.pad(grid, pad_width=pad, mode="constant", constant_values=float(self.wall_value))
        windows = sliding_window_view(grid, (self.grid_width, self.grid_width))
        self._static_grid = base_grid
        self._static_windows = windows
        return windows

    def _build_threat_grid(self, state: SimState) -> np.ndarray:
        self._ensure_global_grid()
        width = int(self._global_w)
        if not (state.threats or []):
            return np.zeros((width, width), dtype=np.float32)
        grid_x = self._global_x
        grid_y = self._global_y
        threats = list(state.threats or [])
        if not threats:
            return np.zeros((width, width), dtype=np.float32)
        t_pos = np.asarray([t.position for t in threats], dtype=np.float32)
        t_rad = np.asarray([t.radius for t in threats], dtype=np.float32)
        t_int = np.asarray([getattr(t, "intensity", 0.0) for t in threats], dtype=np.float32)
        t_val = self.threat_base + self.threat_range * np.clip(t_int, 0.0, 1.0)
        dx = grid_x[None, :, :] - t_pos[:, 0][:, None, None]
        dy = grid_y[None, :, :] - t_pos[:, 1][:, None, None]
        mask = (dx * dx + dy * dy) <= (t_rad[:, None, None] ** 2)
        if not np.any(mask):
            return np.zeros((width, width), dtype=np.float32)
        layer = mask.astype(np.float32) * t_val[:, None, None]
        return layer.max(axis=0).astype(np.float32)

    def _build_threat_windows(self, state: SimState) -> np.ndarray:
        base_grid = self._build_threat_grid(state)
        grid = base_grid
        pad = int(self._center)
        if pad > 0:
            grid = np.pad(grid, pad_width=pad, mode="constant", constant_values=0.0)
        windows = sliding_window_view(grid, (self.grid_width, self.grid_width))
        self._threat_grid = base_grid
        self._threat_windows = windows
        return windows

    def _slice_global_grid(self, grid: np.ndarray, gx: int, gy: int, *, fill: float) -> np.ndarray:
        out = np.full((self.grid_width, self.grid_width), float(fill), dtype=np.float32)
        width = grid.shape[1]
        height = grid.shape[0]
        x0 = int(gx) - self._center
        y0 = int(gy) - self._center
        x1 = x0 + self.grid_width
        y1 = y0 + self.grid_width

        sx0 = max(0, x0)
        sy0 = max(0, y0)
        sx1 = min(width, x1)
        sy1 = min(height, y1)
        if sx0 >= sx1 or sy0 >= sy1:
            return out

        dx0 = sx0 - x0
        dy0 = sy0 - y0
        dx1 = dx0 + (sx1 - sx0)
        dy1 = dy0 + (sy1 - sy0)
        out[dy0:dy1, dx0:dx1] = grid[sy0:sy1, sx0:sx1]
        return out

    def _build_v5(self, state: SimState, idx: int) -> dict[str, np.ndarray]:
        if not state.alive[idx]:
            return {
                "vector": np.zeros((self.vector_dim,), dtype=np.float32),
                "grid": np.zeros((1, self.grid_width, self.grid_width), dtype=np.float32),
            }

        me_pos = state.pos[idx]
        me_vel = state.vel[idx]
        to_target = (state.target_pos - me_pos) / max(self.field_size, 1e-6)
        norm_vel = me_vel / max(self.max_speed, 1e-6)
        walls = state.walls[idx]
        last_action = np.asarray(state.last_action[idx], dtype=np.float32).copy()
        measured_accel = np.asarray(state.measured_accel[idx], dtype=np.float32).copy()
        energy_level = float(state.energy_level[idx]) if state.energy_level is not None else 0.0
        max_accel = float(getattr(state, "max_accel", 0.0))
        if max_accel > 0.0:
            measured_accel = measured_accel / max(max_accel, 1e-6)

        if self.obs_noise_target > 0.0:
            to_target = to_target + self._rng.normal(0.0, self.obs_noise_target, size=2).astype(np.float32)
        if self.obs_noise_vel > 0.0:
            norm_vel = norm_vel + self._rng.normal(0.0, self.obs_noise_vel, size=2).astype(np.float32)
            measured_accel = measured_accel + self._rng.normal(0.0, self.obs_noise_vel, size=2).astype(np.float32)
        np.clip(to_target, -1.0, 1.0, out=to_target)
        np.clip(norm_vel, -1.0, 1.0, out=norm_vel)
        np.clip(last_action, -1.0, 1.0, out=last_action)
        np.clip(measured_accel, -1.0, 1.0, out=measured_accel)
        energy_level = float(np.clip(energy_level, 0.0, 1.0))

        grid = np.zeros((self.grid_width, self.grid_width), dtype=np.float32)
        grid_occ = None
        center = self._center
        res = self.grid_res
        half_range = self._half_range
        abs_x = self._abs_x
        abs_y = self._abs_y
        np.add(self._offset_x, float(me_pos[0]), out=abs_x)
        np.add(self._offset_y, float(me_pos[1]), out=abs_y)

        threats = list(state.threats or [])
        if threats:
            t_pos = np.asarray([t.position for t in threats], dtype=np.float32)
            t_rad = np.asarray([t.radius for t in threats], dtype=np.float32)
            t_int = np.asarray([getattr(t, "intensity", 0.0) for t in threats], dtype=np.float32)
            rel = t_pos - np.asarray(me_pos, dtype=np.float32)[None, :]
            in_range = (np.abs(rel[:, 0]) <= (half_range + t_rad)) & (np.abs(rel[:, 1]) <= (half_range + t_rad))
            if np.any(in_range):
                t_pos = t_pos[in_range]
                t_rad = t_rad[in_range]
                t_int = t_int[in_range]
                t_val = self.threat_base + self.threat_range * np.clip(t_int, 0.0, 1.0)
                dx = abs_x[None, :, :] - t_pos[:, 0][:, None, None]
                dy = abs_y[None, :, :] - t_pos[:, 1][:, None, None]
                mask = (dx * dx + dy * dy) <= (t_rad[:, None, None] ** 2)
                if np.any(mask):
                    layer = mask.astype(np.float32) * t_val[:, None, None]
                    np.maximum(grid, layer.max(axis=0), out=grid)

        rects = state.static_walls or []
        if rects:
            rect_arr = None
            try:
                rect_arr = np.asarray(rects, dtype=np.float32)
            except Exception:
                rect_arr = None
            if rect_arr is not None and rect_arr.ndim == 2 and rect_arr.shape[1] >= 4:
                x1 = rect_arr[:, 0]
                y1 = rect_arr[:, 1]
                x2 = rect_arr[:, 2]
                y2 = rect_arr[:, 3]
                inflate = 0.0
                if inflate != 0.0:
                    x1 = x1 - inflate
                    y1 = y1 - inflate
                    x2 = x2 + inflate
                    y2 = y2 + inflate
                in_range = (x2 >= (me_pos[0] - half_range)) & (x1 <= (me_pos[0] + half_range))
                in_range &= (y2 >= (me_pos[1] - half_range)) & (y1 <= (me_pos[1] + half_range))
                if np.any(in_range):
                    x1 = x1[in_range]
                    y1 = y1[in_range]
                    x2 = x2[in_range]
                    y2 = y2[in_range]
                    mask = (
                        (abs_x[None, :, :] >= x1[:, None, None])
                        & (abs_x[None, :, :] <= x2[:, None, None])
                        & (abs_y[None, :, :] >= y1[:, None, None])
                        & (abs_y[None, :, :] <= y2[:, None, None])
                    )
                    if np.any(mask):
                        grid[np.any(mask, axis=0)] = float(self.wall_value)
            else:
                for rect in rects:
                    x1, y1, x2, y2 = rect
                    inflate = 0.0
                    x1 -= inflate
                    y1 -= inflate
                    x2 += inflate
                    y2 += inflate
                    if (x2 < (me_pos[0] - half_range)) or (x1 > (me_pos[0] + half_range)):
                        continue
                    if (y2 < (me_pos[1] - half_range)) or (y1 > (me_pos[1] + half_range)):
                        continue
                    mask = (abs_x >= float(x1)) & (abs_x <= float(x2)) & (abs_y >= float(y1)) & (abs_y <= float(y2))
                    if np.any(mask):
                        grid[mask] = float(self.wall_value)

        circles = getattr(state, "static_circles", []) or []
        if circles:
            try:
                circ_arr = np.asarray(circles, dtype=np.float32)
            except Exception:
                circ_arr = None
            if circ_arr is not None and circ_arr.ndim == 2 and circ_arr.shape[1] >= 3:
                cx = circ_arr[:, 0]
                cy = circ_arr[:, 1]
                r = circ_arr[:, 2]
                in_range = (cx + r >= (me_pos[0] - half_range)) & (cx - r <= (me_pos[0] + half_range))
                in_range &= (cy + r >= (me_pos[1] - half_range)) & (cy - r <= (me_pos[1] + half_range))
                if np.any(in_range):
                    cx = cx[in_range]
                    cy = cy[in_range]
                    r = r[in_range]
                    dx = abs_x[None, :, :] - cx[:, None, None]
                    dy = abs_y[None, :, :] - cy[:, None, None]
                    mask = (dx * dx + dy * dy) <= (r[:, None, None] ** 2)
                    if np.any(mask):
                        grid[np.any(mask, axis=0)] = float(self.wall_value)
            else:
                for circ in circles:
                    try:
                        cx, cy, r = circ
                    except Exception:
                        continue
                    r = float(r)
                    if (cx + r) < (me_pos[0] - half_range) or (cx - r) > (me_pos[0] + half_range):
                        continue
                    if (cy + r) < (me_pos[1] - half_range) or (cy - r) > (me_pos[1] + half_range):
                        continue
                    dx = abs_x - float(cx)
                    dy = abs_y - float(cy)
                    mask = (dx * dx + dy * dy) <= float(r * r)
                    if np.any(mask):
                        grid[mask] = float(self.wall_value)

        if self.agent_grid_weight > 0.0:
            alive = np.asarray(state.alive, dtype=bool)
            finished = np.asarray(state.finished, dtype=bool)
            mask = alive & (~finished)
            if 0 <= idx < mask.shape[0]:
                mask[idx] = False
            if np.any(mask):
                rel = state.pos[mask] - me_pos
                in_range = (np.abs(rel[:, 0]) <= half_range) & (np.abs(rel[:, 1]) <= half_range)
                if np.any(in_range):
                    rel = rel[in_range]
                    cx = center + (rel[:, 0] / res).astype(int)
                    cy = center + (rel[:, 1] / res).astype(int)
                    valid = (
                        (cx >= 0)
                        & (cx < self.grid_width)
                        & (cy >= 0)
                        & (cy < self.grid_width)
                    )
                    cx = cx[valid]
                    cy = cy[valid]
                    if cx.size > 0:
                        if grid_occ is None:
                            grid_occ = np.zeros((self.grid_width, self.grid_width), dtype=np.float32)
                        dy = self._blob_dy
                        dx = self._blob_dx
                        yy = cy[:, None] + dy[None, :]
                        xx = cx[:, None] + dx[None, :]
                        yy = yy.reshape(-1)
                        xx = xx.reshape(-1)
                        valid2 = (
                            (yy >= 0)
                            & (yy < self.grid_width)
                            & (xx >= 0)
                            & (xx < self.grid_width)
                        )
                        if np.any(valid2):
                            grid_occ[yy[valid2], xx[valid2]] = 1.0

        if self.agent_grid_weight > 0.0 and grid_occ is not None:
            np.maximum(grid, (self.agent_value * self.agent_grid_weight) * grid_occ, out=grid)

        # Рисуем границы мира на гриде, чтобы локальные планировщики
        # не "видели" за пределы карты.
        inflate = float(self.agent_radius)
        border_mask = (
            (abs_x <= inflate)
            | (abs_x >= (self.field_size - inflate))
            | (abs_y <= inflate)
            | (abs_y >= (self.field_size - inflate))
        )
        grid[border_mask] = float(self.wall_value)

        if self.obs_noise_grid > 0.0:
            grid = grid + self._rng.normal(0.0, self.obs_noise_grid, size=grid.shape).astype(np.float32)
        grid[center, center] = max(float(grid[center, center]), self.self_value)
        np.clip(grid, 0.0, 1.0, out=grid)

        vector = np.concatenate(
            [to_target, norm_vel, walls, last_action, measured_accel, np.asarray([energy_level], dtype=np.float32)]
        ).astype(np.float32)
        return {"vector": vector, "grid": grid[None, ...].astype(np.float32)}

    def _build_v5_batch(self, state: SimState, idxs: list[int]) -> list[dict[str, np.ndarray]]:
        if not idxs:
            return []
        if self.agent_grid_weight > 0.0:
            return [self._build_v5(state, idx) for idx in idxs]

        idx_arr = np.asarray(idxs, dtype=int)
        alive_mask = np.asarray(state.alive, dtype=bool)[idx_arr]
        active_idx = idx_arr[alive_mask]
        out: list[dict[str, np.ndarray]] = [
            {
                "vector": np.zeros((self.vector_dim,), dtype=np.float32),
                "grid": np.zeros((1, self.grid_width, self.grid_width), dtype=np.float32),
            }
            for _ in idxs
        ]
        if active_idx.size == 0:
            return out

        me_pos = np.asarray(state.pos, dtype=np.float32)[active_idx]
        me_vel = np.asarray(state.vel, dtype=np.float32)[active_idx]
        to_target = (np.asarray(state.target_pos, dtype=np.float32)[None, :] - me_pos) / max(self.field_size, 1e-6)
        norm_vel = me_vel / max(self.max_speed, 1e-6)
        walls = np.asarray(state.walls, dtype=np.float32)[active_idx]
        last_action = np.asarray(state.last_action, dtype=np.float32)[active_idx]
        measured_accel = np.asarray(state.measured_accel, dtype=np.float32)[active_idx]
        energy_level = (
            np.asarray(state.energy_level, dtype=np.float32)[active_idx]
            if state.energy_level is not None
            else np.zeros((active_idx.size,), dtype=np.float32)
        )
        max_accel = float(getattr(state, "max_accel", 0.0))
        if max_accel > 0.0:
            measured_accel = measured_accel / max(max_accel, 1e-6)

        if self.obs_noise_target > 0.0:
            to_target = to_target + self._rng.normal(0.0, self.obs_noise_target, size=to_target.shape).astype(
                np.float32
            )
        if self.obs_noise_vel > 0.0:
            norm_vel = norm_vel + self._rng.normal(0.0, self.obs_noise_vel, size=norm_vel.shape).astype(np.float32)
            measured_accel = measured_accel + self._rng.normal(0.0, self.obs_noise_vel, size=measured_accel.shape).astype(
                np.float32
            )
        np.clip(to_target, -1.0, 1.0, out=to_target)
        np.clip(norm_vel, -1.0, 1.0, out=norm_vel)
        np.clip(last_action, -1.0, 1.0, out=last_action)
        np.clip(measured_accel, -1.0, 1.0, out=measured_accel)
        np.clip(energy_level, 0.0, 1.0, out=energy_level)

        n = active_idx.size
        grid = np.zeros((n, self.grid_width, self.grid_width), dtype=np.float32)
        center = self._center
        half_range = self._half_range
        abs_x = self._offset_x[None, :, :] + me_pos[:, 0][:, None, None]
        abs_y = self._offset_y[None, :, :] + me_pos[:, 1][:, None, None]

        self._paint_threats_batch(grid, abs_x, abs_y, me_pos, state.threats or [], half_range)
        self._paint_rects_batch(grid, abs_x, abs_y, me_pos, state.static_walls or [], half_range)
        self._paint_circles_batch(grid, abs_x, abs_y, me_pos, getattr(state, "static_circles", []) or [], half_range)

        inflate = float(self.agent_radius)
        border_mask = (
            (abs_x <= inflate)
            | (abs_x >= (self.field_size - inflate))
            | (abs_y <= inflate)
            | (abs_y >= (self.field_size - inflate))
        )
        grid[border_mask] = float(self.wall_value)

        if self.obs_noise_grid > 0.0:
            grid = grid + self._rng.normal(0.0, self.obs_noise_grid, size=grid.shape).astype(np.float32)
        grid[:, center, center] = np.maximum(grid[:, center, center], self.self_value)
        np.clip(grid, 0.0, 1.0, out=grid)

        vector = np.concatenate(
            [to_target, norm_vel, walls, last_action, measured_accel, energy_level[:, None]], axis=1
        ).astype(np.float32)

        active_map = {int(idx): j for j, idx in enumerate(active_idx.tolist())}
        for i, idx in enumerate(idx_arr):
            if not alive_mask[i]:
                continue
            a_idx = active_map.get(int(idx))
            if a_idx is None:
                continue
            out[i] = {"vector": vector[a_idx], "grid": grid[a_idx][None, ...].astype(np.float32)}
        return out

    def _paint_threats_batch(
        self,
        grid: np.ndarray,
        abs_x: np.ndarray,
        abs_y: np.ndarray,
        me_pos: np.ndarray,
        threats,
        half_range: float,
    ) -> None:
        if not threats:
            return
        t_pos = np.asarray([t.position for t in threats], dtype=np.float32)
        t_rad = np.asarray([t.radius for t in threats], dtype=np.float32)
        t_int = np.asarray([getattr(t, "intensity", 0.0) for t in threats], dtype=np.float32)
        rel = t_pos[None, :, :] - me_pos[:, None, :]
        in_range = (np.abs(rel[:, :, 0]) <= (half_range + t_rad[None, :])) & (
            np.abs(rel[:, :, 1]) <= (half_range + t_rad[None, :])
        )
        if not np.any(in_range):
            return
        t_val = self.threat_base + self.threat_range * np.clip(t_int, 0.0, 1.0)
        for threat_idx in range(t_pos.shape[0]):
            active = np.flatnonzero(in_range[:, threat_idx])
            if active.size == 0:
                continue
            tx = float(t_pos[threat_idx, 0])
            ty = float(t_pos[threat_idx, 1])
            tr2 = float(t_rad[threat_idx] * t_rad[threat_idx])
            value = float(t_val[threat_idx])
            for agent_idx in active:
                plane = grid[int(agent_idx)]
                dx = abs_x[int(agent_idx)] - tx
                dy = abs_y[int(agent_idx)] - ty
                mask = (dx * dx + dy * dy) <= tr2
                if np.any(mask):
                    plane[mask] = np.maximum(plane[mask], value)

    def _paint_rects_batch(
        self,
        grid: np.ndarray,
        abs_x: np.ndarray,
        abs_y: np.ndarray,
        me_pos: np.ndarray,
        rects,
        half_range: float,
    ) -> None:
        if not rects:
            return
        wall_value = float(self.wall_value)
        for rect in rects:
            try:
                x1, y1, x2, y2 = [float(v) for v in rect[:4]]
            except Exception:
                continue
            active = np.flatnonzero(
                (x2 >= (me_pos[:, 0] - half_range))
                & (x1 <= (me_pos[:, 0] + half_range))
                & (y2 >= (me_pos[:, 1] - half_range))
                & (y1 <= (me_pos[:, 1] + half_range))
            )
            if active.size == 0:
                continue
            for agent_idx in active:
                plane = grid[int(agent_idx)]
                mask = (
                    (abs_x[int(agent_idx)] >= x1)
                    & (abs_x[int(agent_idx)] <= x2)
                    & (abs_y[int(agent_idx)] >= y1)
                    & (abs_y[int(agent_idx)] <= y2)
                )
                if np.any(mask):
                    plane[mask] = wall_value

    def _paint_circles_batch(
        self,
        grid: np.ndarray,
        abs_x: np.ndarray,
        abs_y: np.ndarray,
        me_pos: np.ndarray,
        circles,
        half_range: float,
    ) -> None:
        if not circles:
            return
        wall_value = float(self.wall_value)
        for circ in circles:
            try:
                cx, cy, r = float(circ[0]), float(circ[1]), float(circ[2])
            except Exception:
                continue
            active = np.flatnonzero(
                (cx + r >= (me_pos[:, 0] - half_range))
                & (cx - r <= (me_pos[:, 0] + half_range))
                & (cy + r >= (me_pos[:, 1] - half_range))
                & (cy - r <= (me_pos[:, 1] + half_range))
            )
            if active.size == 0:
                continue
            r2 = float(r * r)
            for agent_idx in active:
                plane = grid[int(agent_idx)]
                dx = abs_x[int(agent_idx)] - cx
                dy = abs_y[int(agent_idx)] - cy
                mask = (dx * dx + dy * dy) <= r2
                if np.any(mask):
                    plane[mask] = wall_value

    @property
    def obs_dim(self) -> int:
        return self.vector_dim + (self.grid_width**2)

    @property
    def vector_dim(self) -> int:
        return 2 + 2 + 4 + 2 + 2 + 1

    @staticmethod
    def _build_offsets(grid_width: int, grid_res: float) -> tuple[np.ndarray, np.ndarray]:
        center = grid_width // 2
        idx = np.arange(grid_width, dtype=np.float32)
        x_idx, y_idx = np.meshgrid(idx, idx)
        offset_x = (x_idx - float(center)) * float(grid_res)
        offset_y = (y_idx - float(center)) * float(grid_res)
        return offset_x, offset_y
