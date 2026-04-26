"""Логика безопасного спавна и проверки позиций."""

from __future__ import annotations

import numpy as np


class SpawnController:
    def __init__(self, env, rng: np.random.Generator | None = None) -> None:
        self._env = env
        self._rng = rng if rng is not None else np.random.default_rng()
        self._mask_cache: dict[tuple, np.ndarray] = {}

    def set_rng(self, rng: np.random.Generator | None) -> None:
        if rng is None:
            return
        self._rng = rng

    def sample_safe_pos(
        self,
        x_range: tuple[float, float],
        y_range: tuple[float, float],
        *,
        margin: float,
        existing_positions: np.ndarray | None = None,
        min_agent_dist: float = 0.0,
        max_attempts: int = 100,
    ) -> np.ndarray:
        field = float(self._env.config.field_size)
        margin = float(max(0.0, margin))
        res = float(getattr(self._env.config, "grid_res", 1.0))
        coords = self._get_safe_coords(margin=margin, res=res)
        coords = self._filter_coords_range(coords, x_range, y_range)
        if coords.size == 0:
            coords = self._get_safe_coords(margin=margin, res=res)
        if coords.size == 0:
            raise RuntimeError("Failed to sample safe spawn position")
        pos = self._sample_from_coords(
            coords,
            res=res,
            margin=margin,
            field=field,
            max_attempts=max_attempts,
            existing_positions=existing_positions,
            min_agent_dist=min_agent_dist,
        )
        return pos

    def sample_safe_pos_with_min_dist(
        self,
        x_range: tuple[float, float],
        y_range: tuple[float, float],
        *,
        margin: float,
        target_pos: np.ndarray,
        min_dist: float,
        existing_positions: np.ndarray | None = None,
        min_agent_dist: float = 0.0,
        max_attempts: int = 50,
        relax_attempts: int = 10,
        relax_factor: float = 0.9,
        relax_rounds: int = 5,
    ) -> np.ndarray:
        """Сэмплирует точку с ограничением по дистанции до цели.

        Если подходящих позиций нет, дистанция постепенно ослабляется.
        """
        min_dist = float(max(0.0, min_dist))
        if min_dist <= 0.0:
            return self.sample_safe_pos(
                x_range,
                y_range,
                margin=margin,
                existing_positions=existing_positions,
                min_agent_dist=min_agent_dist,
                max_attempts=max_attempts,
            )

        field = float(self._env.config.field_size)
        margin = float(max(0.0, margin))
        res = float(getattr(self._env.config, "grid_res", 1.0))
        coords = self._get_safe_coords(margin=margin, res=res)
        coords = self._filter_coords_range(coords, x_range, y_range)
        cur_min = float(min_dist)
        for _ in range(max(1, relax_rounds)):
            if coords.size > 0:
                d2 = np.sum((coords - target_pos[None, :]) ** 2, axis=1)
                mask = d2 >= float(cur_min * cur_min)
                filtered = coords[mask]
                if filtered.size > 0:
                    return self._sample_from_coords(
                        filtered,
                        res=res,
                        margin=margin,
                        field=field,
                        max_attempts=max(max_attempts, relax_attempts),
                        existing_positions=existing_positions,
                        min_agent_dist=min_agent_dist,
                        target_pos=target_pos,
                        min_target_dist=cur_min,
                    )
            cur_min *= float(relax_factor)
        # Последняя попытка — общий спавн без ограничения, но безопасный.
        coords = self._get_safe_coords(margin=margin, res=res)
        if coords.size == 0:
            raise RuntimeError("Failed to sample safe spawn position")
        return self._sample_from_coords(
            coords,
            res=res,
            margin=margin,
            field=field,
            max_attempts=max_attempts,
            existing_positions=existing_positions,
            min_agent_dist=min_agent_dist,
        )

    def sample_safe_threat_pos(
        self,
        x_range: tuple[float, float],
        y_range: tuple[float, float],
        *,
        margin: float,
        min_agent_dist: float,
        target_pos: np.ndarray | None = None,
        min_target_dist: float = 0.0,
        max_attempts: int = 200,
    ) -> np.ndarray:
        field = float(self._env.config.field_size)
        res = float(getattr(self._env.config, "grid_res", 1.0))
        max_margin = max(0.0, (field - max(res, 1e-6)) * 0.5)
        margin = float(min(max(0.0, margin), max_margin))
        min_agent_dist = float(max(0.0, min_agent_dist))
        min_target_dist = float(max(0.0, min_target_dist))
        margins = [margin, margin * 0.75, margin * 0.5, margin * 0.25, 0.0]
        agents_pos = np.asarray(self._env.get_agent_positions(copy=True), dtype=np.float32)
        target = None
        if target_pos is not None:
            target_arr = np.asarray(target_pos, dtype=np.float32).reshape(-1)
            if target_arr.size >= 2:
                target = target_arr[:2]
        agent_dists = (
            min_agent_dist,
            min_agent_dist * 0.75,
            min_agent_dist * 0.5,
            min_agent_dist * 0.25,
            0.0,
        )
        target_dists = (
            min_target_dist,
            min_target_dist * 0.75,
            min_target_dist * 0.5,
            min_target_dist * 0.25,
            0.0,
        )
        for cur_margin in dict.fromkeys(float(m) for m in margins):
            coords = self._get_safe_coords(margin=cur_margin, res=res)
            ranged = self._filter_coords_range(coords, x_range, y_range)
            if ranged.size > 0:
                coords = ranged
            if coords.size == 0:
                continue
            for target_dist in dict.fromkeys(float(d) for d in target_dists):
                target_filtered = coords
                if target is not None and target_dist > 0.0:
                    diff = target_filtered - target[None, :]
                    d2 = np.sum(diff * diff, axis=1)
                    target_filtered = target_filtered[d2 >= float(target_dist * target_dist)]
                    if target_filtered.size == 0:
                        continue
                if min_agent_dist > 0.0 and agents_pos.size > 0:
                    for dist in dict.fromkeys(float(d) for d in agent_dists):
                        if dist <= 0.0:
                            return self._sample_from_coords(
                                target_filtered,
                                res=res,
                                margin=cur_margin,
                                field=field,
                                max_attempts=max_attempts,
                                target_pos=target,
                                min_target_dist=target_dist,
                            )
                        diff = target_filtered[:, None, :] - agents_pos[None, :, :]
                        d2 = np.sum(diff * diff, axis=2)
                        mask = np.min(d2, axis=1) >= float(dist * dist)
                        filtered = target_filtered[mask]
                        if filtered.size > 0:
                            return self._sample_from_coords(
                                filtered,
                                res=res,
                                margin=cur_margin,
                                field=field,
                                max_attempts=max_attempts,
                                existing_positions=agents_pos,
                                min_agent_dist=dist,
                                target_pos=target,
                                min_target_dist=target_dist,
                            )
                else:
                    return self._sample_from_coords(
                        target_filtered,
                        res=res,
                        margin=cur_margin,
                        field=field,
                        max_attempts=max_attempts,
                        target_pos=target,
                        min_target_dist=target_dist,
                    )
        raise RuntimeError("Failed to sample safe threat position")

    def sample_safe_cluster(
        self,
        center: np.ndarray,
        *,
        n: int | None = None,
        sigma: float = 2.0,
        existing_positions: np.ndarray | None = None,
    ) -> np.ndarray:
        n = int(self._env.config.n_agents if n is None else n)
        out = np.zeros((n, 2), dtype=np.float32)
        margin = float(self._env.config.agent_radius) * 1.5
        field = float(self._env.config.field_size)
        min_agent_dist = float(max(2.0 * float(self._env.config.agent_radius), 1e-6))
        base_existing = None
        if existing_positions is not None:
            base_existing = np.asarray(existing_positions, dtype=np.float32).reshape(-1, 2)
        for i in range(n):
            placed = False
            if base_existing is not None and base_existing.size > 0:
                existing = np.vstack([base_existing, out[:i]]) if i > 0 else base_existing
            else:
                existing = out[:i] if i > 0 else None
            for _ in range(50):
                pos = center + self._rng.normal(0.0, float(sigma), 2).astype(np.float32)
                pos = np.clip(pos, margin, field - margin)
                if self._is_candidate_valid(
                    pos,
                    margin=margin,
                    field=field,
                    existing_positions=existing,
                    min_agent_dist=min_agent_dist,
                ):
                    out[i] = pos
                    placed = True
                    break
            if not placed:
                for dist in (min_agent_dist, min_agent_dist * 0.75, min_agent_dist * 0.5, 0.0):
                    try:
                        out[i] = self.sample_safe_pos(
                            (margin, field - margin),
                            (margin, field - margin),
                            margin=margin,
                            existing_positions=existing,
                            min_agent_dist=dist,
                        )
                        placed = True
                        break
                    except RuntimeError:
                        continue
            if not placed:
                raise RuntimeError("Failed to sample non-overlapping safe cluster position")
        return out

    def _get_safe_coords(self, *, margin: float, res: float) -> np.ndarray:
        key = self._mask_key(margin, res)
        coords = self._mask_cache.get(key)
        if coords is not None:
            return coords
        field = float(self._env.config.field_size)
        res = float(res)
        if res <= 0.0:
            return np.zeros((0, 2), dtype=np.float32)
        xs = np.arange(res / 2.0, field, res, dtype=np.float32)
        ys = np.arange(res / 2.0, field, res, dtype=np.float32)
        if xs.size == 0 or ys.size == 0:
            return np.zeros((0, 2), dtype=np.float32)
        grid_x, grid_y = np.meshgrid(xs, ys)
        mask = (grid_x >= margin) & (grid_x <= (field - margin)) & (grid_y >= margin) & (grid_y <= (field - margin))
        for x1, y1, x2, y2 in self._env.get_static_walls():
            mask &= ~(
                (grid_x >= (x1 - margin))
                & (grid_x <= (x2 + margin))
                & (grid_y >= (y1 - margin))
                & (grid_y <= (y2 + margin))
            )
        for t in self._env.get_threats():
            dx = grid_x - float(t.position[0])
            dy = grid_y - float(t.position[1])
            rr = float(t.radius + margin)
            mask &= (dx * dx + dy * dy) > (rr * rr)
        for c_pos, c_rad in self._env.get_static_circles():
            dx = grid_x - float(c_pos[0])
            dy = grid_y - float(c_pos[1])
            rr = float(c_rad + margin)
            mask &= (dx * dx + dy * dy) > (rr * rr)
        coords = np.stack([grid_x[mask], grid_y[mask]], axis=1).astype(np.float32)
        self._mask_cache[key] = coords
        return coords

    def _mask_key(self, margin: float, res: float) -> tuple:
        walls_key = tuple((float(x1), float(y1), float(x2), float(y2)) for x1, y1, x2, y2 in self._env.get_static_walls())
        threats_key = []
        for t in self._env.get_threats():
            threats_key.append((float(t.position[0]), float(t.position[1]), float(t.radius)))
        circles_key = []
        for c_pos, c_rad in self._env.get_static_circles():
            circles_key.append((float(c_pos[0]), float(c_pos[1]), float(c_rad)))
        return (
            float(self._env.config.field_size),
            float(margin),
            float(res),
            walls_key,
            tuple(threats_key),
            tuple(circles_key),
        )

    @staticmethod
    def _filter_coords_range(
        coords: np.ndarray, x_range: tuple[float, float], y_range: tuple[float, float]
    ) -> np.ndarray:
        if coords.size == 0:
            return coords
        x1, x2 = float(x_range[0]), float(x_range[1])
        y1, y2 = float(y_range[0]), float(y_range[1])
        mask = (coords[:, 0] >= x1) & (coords[:, 0] <= x2) & (coords[:, 1] >= y1) & (coords[:, 1] <= y2)
        return coords[mask]

    def _sample_from_coords(
        self,
        coords: np.ndarray,
        *,
        res: float,
        margin: float,
        field: float,
        max_attempts: int = 100,
        existing_positions: np.ndarray | None = None,
        min_agent_dist: float = 0.0,
        target_pos: np.ndarray | None = None,
        min_target_dist: float = 0.0,
    ) -> np.ndarray:
        if coords.size == 0:
            raise RuntimeError("Failed to sample safe spawn position")
        max_attempts = max(int(max_attempts), 1)
        for _ in range(max_attempts):
            idx = int(self._rng.integers(0, coords.shape[0]))
            base = coords[idx].astype(np.float32)
            jitter = self._rng.uniform(-0.5 * res, 0.5 * res, size=2).astype(np.float32)
            for candidate in (base + jitter, base):
                pos = np.clip(candidate, margin, field - margin).astype(np.float32)
                if self._is_candidate_valid(
                    pos,
                    margin=margin,
                    field=field,
                    existing_positions=existing_positions,
                    min_agent_dist=min_agent_dist,
                    target_pos=target_pos,
                    min_target_dist=min_target_dist,
                ):
                    return pos
        raise RuntimeError("Failed to sample validated spawn position")

    def _is_candidate_valid(
        self,
        pos: np.ndarray,
        *,
        margin: float,
        field: float,
        existing_positions: np.ndarray | None = None,
        min_agent_dist: float = 0.0,
        target_pos: np.ndarray | None = None,
        min_target_dist: float = 0.0,
    ) -> bool:
        if not self.is_safe_pos(pos, margin, field):
            return False
        if target_pos is not None and float(min_target_dist) > 0.0:
            diff = np.asarray(pos, dtype=np.float32) - np.asarray(target_pos, dtype=np.float32)
            if float(diff[0] * diff[0] + diff[1] * diff[1]) < float(min_target_dist * min_target_dist):
                return False
        if existing_positions is None:
            return True
        existing = np.asarray(existing_positions, dtype=np.float32)
        if existing.size == 0 or float(min_agent_dist) <= 0.0:
            return True
        diff = existing - np.asarray(pos, dtype=np.float32)[None, :]
        d2 = np.sum(diff * diff, axis=1)
        return bool(np.all(d2 >= float(min_agent_dist * min_agent_dist)))

    def is_safe_pos(self, pos: np.ndarray, margin: float, field: float) -> bool:
        x = float(pos[0])
        y = float(pos[1])
        if x < margin or y < margin or x > (field - margin) or y > (field - margin):
            return False
        for x1, y1, x2, y2 in self._env.get_static_walls():
            if (x1 - margin) <= x <= (x2 + margin) and (y1 - margin) <= y <= (y2 + margin):
                return False
        for t in self._env.get_threats():
            diff = np.asarray(pos, dtype=np.float32) - np.asarray(t.position, dtype=np.float32)
            r = float(t.radius) + float(margin)
            if float(diff[0] * diff[0] + diff[1] * diff[1]) <= (r * r):
                return False
        for c_pos, c_rad in self._env.get_static_circles():
            diff = np.asarray(pos, dtype=np.float32) - np.asarray(c_pos, dtype=np.float32)
            r = float(c_rad) + float(margin)
            if float(diff[0] * diff[0] + diff[1] * diff[1]) <= (r * r):
                return False
        return True
