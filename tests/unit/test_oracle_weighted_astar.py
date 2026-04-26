"""Проверка risk-aware A* в оракуле."""

from __future__ import annotations

import numpy as np

from env.oracles.grid_oracle import build_risk_grid, shortest_path


def _path_length(path: list[tuple[float, float]]) -> float:
    if len(path) < 2:
        return 0.0
    pts = [np.asarray(p, dtype=np.float32) for p in path]
    return float(sum(np.linalg.norm(pts[i + 1] - pts[i]) for i in range(len(pts) - 1)))


def test_weighted_astar_avoids_static_risk():
    field_size = 10.0
    cell_size = 1.0
    start = np.array([1.0, 5.0], dtype=np.float32)
    goal = np.array([9.0, 5.0], dtype=np.float32)

    threat = {"pos": [5.0, 5.0], "radius": 2.5, "intensity": 1.0}
    risk_grid = build_risk_grid(field_size, cell_size, threats=[threat])
    assert risk_grid is not None

    direct_len = float(np.linalg.norm(goal - start))
    path = shortest_path(
        start,
        goal,
        field_size,
        cell_size=cell_size,
        allow_diagonal=True,
        risk_grid=risk_grid,
        risk_weight=50.0,
        heuristic_weight=1.0,
    )
    assert path, "Ожидается путь при наличии risk-grid."
    risk_len = _path_length(path)
    assert risk_len > (direct_len + 0.5)
