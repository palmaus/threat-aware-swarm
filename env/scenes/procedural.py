from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class ForestConfig:
    enabled: bool = False
    count: int = 0
    radius_min: float = 1.0
    radius_max: float = 2.0
    min_dist: float = 3.0
    region_min: tuple[float, float] = (0.0, 0.0)
    region_max: tuple[float, float] = (100.0, 100.0)


def poisson_disk_sample(
    rng: np.random.Generator,
    *,
    region_min: tuple[float, float],
    region_max: tuple[float, float],
    radius: float,
    k: int = 30,
    target_count: int | None = None,
) -> list[tuple[float, float]]:
    """Алгоритм Бридсона для Poisson Disk Sampling (2D)."""
    xmin, ymin = region_min
    xmax, ymax = region_max
    width = max(0.0, float(xmax - xmin))
    height = max(0.0, float(ymax - ymin))
    if width <= 0.0 or height <= 0.0 or radius <= 0.0:
        return []

    cell_size = radius / math.sqrt(2.0)
    grid_w = math.ceil(width / cell_size)
    grid_h = math.ceil(height / cell_size)
    grid = -np.ones((grid_h, grid_w), dtype=np.int32)

    def _grid_coords(p: np.ndarray) -> tuple[int, int]:
        return int((p[0] - xmin) / cell_size), int((p[1] - ymin) / cell_size)

    def _in_bounds(p: np.ndarray) -> bool:
        return xmin <= p[0] <= xmax and ymin <= p[1] <= ymax

    def _fits(p: np.ndarray, pts: list[np.ndarray]) -> bool:
        gx, gy = _grid_coords(p)
        r2 = radius * radius
        for oy in range(max(0, gy - 2), min(grid_h, gy + 3)):
            for ox in range(max(0, gx - 2), min(grid_w, gx + 3)):
                idx = grid[oy, ox]
                if idx < 0:
                    continue
                diff = pts[idx] - p
                if float(diff[0] * diff[0] + diff[1] * diff[1]) < r2:
                    return False
        return True

    pts: list[np.ndarray] = []
    active: list[int] = []

    init = np.array([rng.uniform(xmin, xmax), rng.uniform(ymin, ymax)], dtype=np.float32)
    pts.append(init)
    gx, gy = _grid_coords(init)
    grid[gy, gx] = 0
    active.append(0)

    while active:
        if target_count is not None and len(pts) >= target_count:
            break
        idx = int(rng.choice(active))
        base = pts[idx]
        found = False
        for _ in range(int(k)):
            ang = float(rng.uniform(0.0, 2.0 * math.pi))
            rad = float(rng.uniform(radius, 2.0 * radius))
            cand = base + np.array([math.cos(ang), math.sin(ang)], dtype=np.float32) * rad
            if not _in_bounds(cand):
                continue
            if _fits(cand, pts):
                pts.append(cand.astype(np.float32))
                gx, gy = _grid_coords(cand)
                grid[gy, gx] = len(pts) - 1
                active.append(len(pts) - 1)
                found = True
                break
        if not found:
            active.remove(idx)

    return [(float(p[0]), float(p[1])) for p in pts]


def generate_forest(
    rng: np.random.Generator,
    *,
    config: ForestConfig,
) -> list[tuple[np.ndarray, float]]:
    if not config.enabled or int(config.count) <= 0:
        return []
    count = int(config.count)
    min_dist = float(max(0.0, config.min_dist))
    r_min = float(max(0.1, config.radius_min))
    r_max = float(max(r_min, config.radius_max))
    centers = poisson_disk_sample(
        rng,
        region_min=config.region_min,
        region_max=config.region_max,
        radius=min_dist,
        target_count=count,
    )
    obstacles = []
    for c in centers[:count]:
        radius = float(rng.uniform(r_min, r_max))
        obstacles.append((np.asarray(c, dtype=np.float32), radius))
    return obstacles
