"""Оракул кратчайшего пути по статической геометрии без учёта динамики."""

from __future__ import annotations

import heapq
import math
from collections.abc import Iterable

import numpy as np

from env.utils.geometry import parse_circle, parse_wall_rect

try:  # Опциональное ускорение через Numba.
    from numba import njit
except Exception:  # pragma: no cover - опциональная зависимость
    njit = None

_NUMBA_AVAILABLE = njit is not None

# Массивы для Numba (фиксированный порядок, сначала 4 кардинальных, затем диагонали).
MOVES_DX = np.array([1, -1, 0, 0, 1, 1, -1, -1], dtype=np.int8)
MOVES_DY = np.array([0, 0, 1, -1, 1, -1, 1, -1], dtype=np.int8)


def shortest_path_length(
    start: np.ndarray,
    goal: np.ndarray,
    field_size: float,
    *,
    cell_size: float = 1.0,
    inflation_radius: float = 0.0,
    walls: Iterable[dict] | None = None,
    allow_diagonal: bool = True,
    grid: np.ndarray | None = None,
    risk_grid: np.ndarray | None = None,
    risk_weight: float = 0.0,
    heuristic_weight: float = 1.0,
) -> float:
    start = np.asarray(start, dtype=np.float32)
    goal = np.asarray(goal, dtype=np.float32)
    if start.shape != (2,) or goal.shape != (2,):
        return float("nan")

    if grid is None:
        walls_list = list(walls or [])
        if not walls_list and risk_grid is None:
            # Без препятствий возвращаем евклидову длину, чтобы не запускать A*.
            return float(np.linalg.norm(goal - start))
        if walls_list:
            grid = _build_occupancy(field_size, cell_size, walls_list, inflation_radius)
        elif risk_grid is not None:
            grid = np.zeros_like(risk_grid, dtype=bool)
    if grid is None or grid.size == 0:
        return float(np.linalg.norm(goal - start))
    return _astar_length(
        start,
        goal,
        grid,
        field_size,
        cell_size,
        allow_diagonal=allow_diagonal,
        risk_grid=risk_grid,
        risk_weight=risk_weight,
        heuristic_weight=heuristic_weight,
    )


def shortest_path(
    start: np.ndarray,
    goal: np.ndarray,
    field_size: float,
    *,
    cell_size: float = 1.0,
    inflation_radius: float = 0.0,
    walls: Iterable[dict] | Iterable[tuple[float, float, float, float]] | None = None,
    allow_diagonal: bool = True,
    grid: np.ndarray | None = None,
    risk_grid: np.ndarray | None = None,
    risk_weight: float = 0.0,
    heuristic_weight: float = 1.0,
) -> list[tuple[float, float]]:
    start = np.asarray(start, dtype=np.float32)
    goal = np.asarray(goal, dtype=np.float32)
    if start.shape != (2,) or goal.shape != (2,):
        return []

    if grid is None:
        walls_list = list(walls or [])
        if not walls_list and risk_grid is None:
            # Без препятствий достаточно прямого отрезка от старта к цели.
            return [tuple(start.tolist()), tuple(goal.tolist())]
        if walls_list:
            grid = _build_occupancy(field_size, cell_size, walls_list, inflation_radius)
        elif risk_grid is not None:
            grid = np.zeros_like(risk_grid, dtype=bool)
    if grid is None or grid.size == 0:
        return [tuple(start.tolist()), tuple(goal.tolist())]

    path = _astar_path(
        start,
        goal,
        grid,
        field_size,
        cell_size,
        allow_diagonal=allow_diagonal,
        risk_grid=risk_grid,
        risk_weight=risk_weight,
        heuristic_weight=heuristic_weight,
    )
    return path


def build_occupancy_grid(
    field_size: float,
    cell_size: float,
    *,
    inflation_radius: float = 0.0,
    walls: Iterable[dict] | Iterable[tuple[float, float, float, float]] | None = None,
) -> np.ndarray | None:
    walls_list = list(walls or [])
    if not walls_list:
        return None
    return _build_occupancy(field_size, cell_size, walls_list, inflation_radius)


def build_risk_grid(
    field_size: float,
    cell_size: float,
    *,
    threats: Iterable[object] | None = None,
    inflation_radius: float = 0.0,
) -> np.ndarray | None:
    threats_list = list(threats or [])
    if not threats_list:
        return None
    size = float(field_size)
    step = float(cell_size)
    if size <= 0.0 or step <= 0.0:
        return None
    width = int(np.ceil(size / step)) + 1
    grid = np.zeros((width, width), dtype=np.float32)
    height = grid.shape[0]
    width = grid.shape[1]
    inflate = float(max(0.0, inflation_radius))
    filled = False
    for obj in threats_list:
        circle = parse_circle(obj)
        if circle is None:
            continue
        cx, cy, r = circle
        intensity = _extract_intensity(obj)
        if intensity <= 0.0:
            continue
        r = float(max(0.0, r + inflate))
        if r <= 0.0:
            continue
        if (cx + r) < 0.0 or (cx - r) > size or (cy + r) < 0.0 or (cy - r) > size:
            continue
        x1 = float(np.clip(cx - r, 0.0, size))
        y1 = float(np.clip(cy - r, 0.0, size))
        x2 = float(np.clip(cx + r, 0.0, size))
        y2 = float(np.clip(cy + r, 0.0, size))
        ix1 = max(0, int(np.floor(x1 / step)))
        iy1 = max(0, int(np.floor(y1 / step)))
        ix2 = min(width - 1, int(np.ceil(x2 / step)))
        iy2 = min(height - 1, int(np.ceil(y2 / step)))
        rr = r * r
        for iy in range(iy1, iy2 + 1):
            wy = (iy * step) + (step / 2.0)
            dy = wy - cy
            dy2 = dy * dy
            if dy2 > rr:
                continue
            for ix in range(ix1, ix2 + 1):
                wx = (ix * step) + (step / 2.0)
                dx = wx - cx
                dist2 = dx * dx + dy2
                if dist2 > rr:
                    continue
                dist = float(np.sqrt(dist2))
                # Линейное затухание риска от центра угрозы.
                val = float(intensity) * max(0.0, 1.0 - (dist / max(r, 1e-6)))
                if val > grid[iy, ix]:
                    grid[iy, ix] = val
                    filled = True
    if not filled:
        return None
    return np.clip(grid, 0.0, 1.0)


def build_distance_field(
    goal: np.ndarray,
    field_size: float,
    *,
    cell_size: float = 1.0,
    inflation_radius: float = 0.0,
    walls: Iterable[dict] | Iterable[tuple[float, float, float, float]] | None = None,
    allow_diagonal: bool = True,
    grid: np.ndarray | None = None,
    risk_grid: np.ndarray | None = None,
    risk_weight: float = 0.0,
) -> np.ndarray | None:
    goal = np.asarray(goal, dtype=np.float32)
    if goal.shape != (2,):
        return None
    if grid is None:
        walls_list = list(walls or [])
        if not walls_list and risk_grid is None:
            return None
        if walls_list:
            grid = _build_occupancy(field_size, cell_size, walls_list, inflation_radius)
        elif risk_grid is not None:
            grid = np.zeros_like(risk_grid, dtype=bool)
    if grid is None or grid.size == 0:
        return None
    height, width = grid.shape
    gx, gy = _world_to_grid(goal, field_size, cell_size, width, height)
    if grid[gy, gx]:
        return np.full((height, width), np.inf, dtype=np.float32)

    dist = np.full((height, width), np.inf, dtype=np.float64)
    dist[gy, gx] = 0.0
    if allow_diagonal:
        moves = list(zip(MOVES_DX, MOVES_DY))
        costs = [1.0, 1.0, 1.0, 1.0, math.sqrt(2.0), math.sqrt(2.0), math.sqrt(2.0), math.sqrt(2.0)]
    else:
        moves = list(zip(MOVES_DX[:4], MOVES_DY[:4]))
        costs = [1.0, 1.0, 1.0, 1.0]
    heap = [(0.0, gx, gy)]
    while heap:
        cur_dist, x, y = heapq.heappop(heap)
        if cur_dist > dist[y, x] + 1e-9:
            continue
        for (dx, dy), base_cost in zip(moves, costs):
            nx = x + int(dx)
            ny = y + int(dy)
            if nx < 0 or ny < 0 or nx >= width or ny >= height:
                continue
            if grid[ny, nx]:
                continue
            step_cost = base_cost * float(cell_size)
            if risk_grid is not None and risk_weight > 0.0:
                step_cost += float(risk_weight) * float(risk_grid[ny, nx])
            nd = cur_dist + step_cost
            if nd < dist[ny, nx]:
                dist[ny, nx] = nd
                heapq.heappush(heap, (nd, nx, ny))
    return dist


def path_from_distance_field(
    start: np.ndarray,
    goal: np.ndarray,
    dist: np.ndarray | None,
    field_size: float,
    cell_size: float,
    *,
    allow_diagonal: bool = True,
) -> list[tuple[float, float]]:
    start = np.asarray(start, dtype=np.float32)
    goal = np.asarray(goal, dtype=np.float32)
    if start.shape != (2,) or goal.shape != (2,):
        return []
    if dist is None or dist.size == 0:
        return [tuple(start.tolist()), tuple(goal.tolist())]
    height, width = dist.shape
    sx, sy = _world_to_grid(start, field_size, cell_size, width, height)
    gx, gy = _world_to_grid(goal, field_size, cell_size, width, height)
    if not np.isfinite(dist[sy, sx]):
        return []
    moves = list(zip(MOVES_DX, MOVES_DY)) if allow_diagonal else list(zip(MOVES_DX[:4], MOVES_DY[:4]))
    path = [(sx, sy)]
    cur_x, cur_y = sx, sy
    max_steps = width * height
    for _ in range(max_steps):
        if cur_x == gx and cur_y == gy:
            break
        best = (cur_x, cur_y)
        best_val = float(dist[cur_y, cur_x])
        for dx, dy in moves:
            nx = cur_x + int(dx)
            ny = cur_y + int(dy)
            if nx < 0 or ny < 0 or nx >= width or ny >= height:
                continue
            val = float(dist[ny, nx])
            if val < best_val:
                best_val = val
                best = (nx, ny)
        if best == (cur_x, cur_y):
            break
        cur_x, cur_y = best
        path.append(best)
    return [_grid_to_world(p[0], p[1], field_size, cell_size) for p in path]


def _build_occupancy(
    field_size: float, cell_size: float, walls: Iterable[dict], inflation_radius: float
) -> np.ndarray | None:
    size = float(field_size)
    step = float(cell_size)
    if size <= 0.0 or step <= 0.0:
        return None
    width = int(np.ceil(size / step)) + 1
    grid = np.zeros((width, width), dtype=bool)
    height = grid.shape[0]
    width = grid.shape[1]
    # Инфляция моделирует радиус агента и строит пространство конфигураций (C‑space) для безопасного планирования.
    inflate = float(max(0.0, inflation_radius))
    for w in walls:
        rect = _parse_wall_rect(w)
        if rect is not None:
            x1, y1, x2, y2 = rect
            if inflate > 0.0:
                x1 -= inflate
                y1 -= inflate
                x2 += inflate
                y2 += inflate
            x1 = float(np.clip(x1, 0.0, size))
            y1 = float(np.clip(y1, 0.0, size))
            x2 = float(np.clip(x2, 0.0, size))
            y2 = float(np.clip(y2, 0.0, size))
            if x2 <= x1 or y2 <= y1:
                continue
            ix1 = max(0, int(np.floor(x1 / step)))
            iy1 = max(0, int(np.floor(y1 / step)))
            ix2 = min(width - 1, int(np.ceil(x2 / step)))
            iy2 = min(height - 1, int(np.ceil(y2 / step)))
            grid[iy1 : iy2 + 1, ix1 : ix2 + 1] = True
            continue

        circle = _parse_circle(w)
        if circle is None:
            continue
        cx, cy, r = circle
        r = float(max(0.0, r + inflate))
        if r <= 0.0:
            continue
        if (cx + r) < 0.0 or (cx - r) > size or (cy + r) < 0.0 or (cy - r) > size:
            continue
        x1 = float(np.clip(cx - r, 0.0, size))
        y1 = float(np.clip(cy - r, 0.0, size))
        x2 = float(np.clip(cx + r, 0.0, size))
        y2 = float(np.clip(cy + r, 0.0, size))
        ix1 = max(0, int(np.floor(x1 / step)))
        iy1 = max(0, int(np.floor(y1 / step)))
        ix2 = min(width - 1, int(np.ceil(x2 / step)))
        iy2 = min(height - 1, int(np.ceil(y2 / step)))
        rr = r * r
        for iy in range(iy1, iy2 + 1):
            wy = (iy * step) + (step / 2.0)
            dy = wy - cy
            dy2 = dy * dy
            if dy2 > rr:
                continue
            for ix in range(ix1, ix2 + 1):
                wx = (ix * step) + (step / 2.0)
                dx = wx - cx
                if (dx * dx + dy2) <= rr:
                    grid[iy, ix] = True
    return grid


def _parse_wall_rect(wall: dict) -> tuple[float, float, float, float] | None:
    return parse_wall_rect(wall)


def _parse_circle(obj) -> tuple[float, float, float] | None:
    return parse_circle(obj)


def _extract_intensity(obj) -> float:
    if isinstance(obj, dict):
        try:
            return float(obj.get("intensity", 0.0))
        except Exception:
            return 0.0
    if hasattr(obj, "intensity"):
        try:
            return float(obj.intensity)
        except Exception:
            return 0.0
    return 0.0


if _NUMBA_AVAILABLE:

    @njit(cache=True)
    def _heap_push(hx, hy, hf, size, x, y, f):
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
    def _astar_length_numba(
        grid,
        risk_grid,
        has_risk,
        risk_weight,
        heuristic_weight,
        allow_diagonal,
        cell_size,
        sx,
        sy,
        gx,
        gy,
    ):
        height, width = grid.shape
        if grid[sy, sx] or grid[gy, gx]:
            return float("nan")
        diag = math.sqrt(2.0) * cell_size
        ortho = cell_size
        g_cost = np.full((height, width), np.inf, dtype=np.float32)
        g_dist = np.full((height, width), np.inf, dtype=np.float32)
        closed = np.zeros((height, width), dtype=np.uint8)
        g_cost[sy, sx] = 0.0
        g_dist[sy, sx] = 0.0
        max_nodes = height * width + 1
        heap_x = np.empty(max_nodes, dtype=np.int32)
        heap_y = np.empty(max_nodes, dtype=np.int32)
        heap_f = np.empty(max_nodes, dtype=np.float32)
        heap_size = 0
        heap_size = _heap_push(heap_x, heap_y, heap_f, heap_size, sx, sy, 0.0)
        move_count = 8 if allow_diagonal else 4
        hw = float(heuristic_weight) if heuristic_weight > 0.0 else 0.0

        while heap_size > 0:
            x, y, _f, heap_size = _heap_pop(heap_x, heap_y, heap_f, heap_size)
            if x < 0:
                break
            if closed[y, x] == 1:
                continue
            if x == gx and y == gy:
                return float(g_dist[y, x])
            closed[y, x] = 1
            for i in range(move_count):
                dx = MOVES_DX[i]
                dy = MOVES_DY[i]
                nx = x + dx
                ny = y + dy
                if nx < 0 or ny < 0 or nx >= width or ny >= height:
                    continue
                if grid[ny, nx]:
                    continue
                step_cost = diag if (dx != 0 and dy != 0) else ortho
                risk = float(risk_grid[ny, nx]) if has_risk else 0.0
                step_cost_weighted = float(step_cost * (1.0 + (risk_weight * risk)))
                tentative = float(g_cost[y, x] + step_cost_weighted)
                if tentative < g_cost[ny, nx]:
                    g_cost[ny, nx] = tentative
                    g_dist[ny, nx] = float(g_dist[y, x] + step_cost)
                    dxh = float(nx - gx)
                    dyh = float(ny - gy)
                    h = math.sqrt(dxh * dxh + dyh * dyh) * cell_size
                    heap_size = _heap_push(
                        heap_x,
                        heap_y,
                        heap_f,
                        heap_size,
                        nx,
                        ny,
                        float(tentative + (hw * h)),
                    )
        return float("nan")

    @njit(cache=True)
    def _astar_path_numba(
        grid,
        risk_grid,
        has_risk,
        risk_weight,
        heuristic_weight,
        allow_diagonal,
        cell_size,
        sx,
        sy,
        gx,
        gy,
    ):
        height, width = grid.shape
        if grid[sy, sx] or grid[gy, gx]:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32), 0
        diag = math.sqrt(2.0) * cell_size
        ortho = cell_size
        g_cost = np.full((height, width), np.inf, dtype=np.float32)
        closed = np.zeros((height, width), dtype=np.uint8)
        came_x = np.full((height, width), -1, dtype=np.int32)
        came_y = np.full((height, width), -1, dtype=np.int32)
        g_cost[sy, sx] = 0.0
        max_nodes = height * width + 1
        heap_x = np.empty(max_nodes, dtype=np.int32)
        heap_y = np.empty(max_nodes, dtype=np.int32)
        heap_f = np.empty(max_nodes, dtype=np.float32)
        heap_size = 0
        heap_size = _heap_push(heap_x, heap_y, heap_f, heap_size, sx, sy, 0.0)
        move_count = 8 if allow_diagonal else 4
        hw = float(heuristic_weight) if heuristic_weight > 0.0 else 0.0

        while heap_size > 0:
            x, y, _f, heap_size = _heap_pop(heap_x, heap_y, heap_f, heap_size)
            if x < 0:
                break
            if closed[y, x] == 1:
                continue
            if x == gx and y == gy:
                break
            closed[y, x] = 1
            for i in range(move_count):
                dx = MOVES_DX[i]
                dy = MOVES_DY[i]
                nx = x + dx
                ny = y + dy
                if nx < 0 or ny < 0 or nx >= width or ny >= height:
                    continue
                if grid[ny, nx]:
                    continue
                step_cost = diag if (dx != 0 and dy != 0) else ortho
                risk = float(risk_grid[ny, nx]) if has_risk else 0.0
                step_cost_weighted = float(step_cost * (1.0 + (risk_weight * risk)))
                tentative = float(g_cost[y, x] + step_cost_weighted)
                if tentative < g_cost[ny, nx]:
                    g_cost[ny, nx] = tentative
                    came_x[ny, nx] = x
                    came_y[ny, nx] = y
                    dxh = float(nx - gx)
                    dyh = float(ny - gy)
                    h = math.sqrt(dxh * dxh + dyh * dyh) * cell_size
                    heap_size = _heap_push(
                        heap_x,
                        heap_y,
                        heap_f,
                        heap_size,
                        nx,
                        ny,
                        float(tentative + (hw * h)),
                    )

        if sx == gx and sy == gy:
            path_x = np.empty(1, dtype=np.int32)
            path_y = np.empty(1, dtype=np.int32)
            path_x[0] = sx
            path_y[0] = sy
            return path_x, path_y, 1
        if came_x[gy, gx] == -1:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32), 0

        max_path = height * width + 1
        path_x = np.empty(max_path, dtype=np.int32)
        path_y = np.empty(max_path, dtype=np.int32)
        length = 0
        cx = gx
        cy = gy
        while True:
            path_x[length] = cx
            path_y[length] = cy
            length += 1
            if cx == sx and cy == sy:
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
        return path_x, path_y, length


def _astar_length_py(
    start: np.ndarray,
    goal: np.ndarray,
    grid: np.ndarray,
    field_size: float,
    cell_size: float,
    *,
    allow_diagonal: bool,
    risk_grid: np.ndarray | None = None,
    risk_weight: float = 0.0,
    heuristic_weight: float = 1.0,
) -> float:
    height, width = grid.shape
    sx, sy = _world_to_grid(start, field_size, cell_size, width, height)
    gx, gy = _world_to_grid(goal, field_size, cell_size, width, height)
    # Старт или цель внутри препятствия делают маршрут неопределённым.
    if grid[sy, sx] or grid[gy, gx]:
        return float("nan")

    diag = float(np.sqrt(2.0) * cell_size)
    ortho = float(cell_size)
    if allow_diagonal:
        moves = (
            (1, 0, ortho),
            (-1, 0, ortho),
            (0, 1, ortho),
            (0, -1, ortho),
            (1, 1, diag),
            (1, -1, diag),
            (-1, 1, diag),
            (-1, -1, diag),
        )
    else:
        moves = ((1, 0, ortho), (-1, 0, ortho), (0, 1, ortho), (0, -1, ortho))

    g_cost = np.full((height, width), np.inf, dtype=np.float32)
    g_dist = np.full((height, width), np.inf, dtype=np.float32)
    g_cost[sy, sx] = 0.0
    g_dist[sy, sx] = 0.0
    closed = np.zeros((height, width), dtype=bool)
    heap: list[tuple[float, tuple[int, int]]] = []
    heapq.heappush(heap, (0.0, (sx, sy)))

    while heap:
        _, (x, y) = heapq.heappop(heap)
        if closed[y, x]:
            continue
        if x == gx and y == gy:
            return float(g_dist[y, x])
        closed[y, x] = True
        for dx, dy, step_cost in moves:
            nx = x + dx
            ny = y + dy
            if nx < 0 or ny < 0 or nx >= width or ny >= height:
                continue
            if grid[ny, nx]:
                continue
            risk = float(risk_grid[ny, nx]) if risk_grid is not None else 0.0
            step_cost_weighted = float(step_cost * (1.0 + (risk_weight * risk)))
            tentative = float(g_cost[y, x] + step_cost_weighted)
            if tentative < g_cost[ny, nx]:
                g_cost[ny, nx] = tentative
                g_dist[ny, nx] = float(g_dist[y, x] + step_cost)
                h = _heuristic(nx, ny, gx, gy, cell_size)
                heapq.heappush(heap, (tentative + (heuristic_weight * h), (nx, ny)))
    return float("nan")


def _astar_length(
    start: np.ndarray,
    goal: np.ndarray,
    grid: np.ndarray,
    field_size: float,
    cell_size: float,
    *,
    allow_diagonal: bool,
    risk_grid: np.ndarray | None = None,
    risk_weight: float = 0.0,
    heuristic_weight: float = 1.0,
) -> float:
    if not _NUMBA_AVAILABLE:
        return _astar_length_py(
            start,
            goal,
            grid,
            field_size,
            cell_size,
            allow_diagonal=allow_diagonal,
            risk_grid=risk_grid,
            risk_weight=risk_weight,
            heuristic_weight=heuristic_weight,
        )
    height, width = grid.shape
    sx, sy = _world_to_grid(start, field_size, cell_size, width, height)
    gx, gy = _world_to_grid(goal, field_size, cell_size, width, height)
    rg = np.zeros((height, width), dtype=np.float32)
    has_risk = False
    if risk_grid is not None:
        rg = np.asarray(risk_grid, dtype=np.float32)
        has_risk = True
    try:
        return float(
            _astar_length_numba(
                np.asarray(grid, dtype=np.bool_),
                rg,
                has_risk,
                float(risk_weight),
                float(heuristic_weight),
                bool(allow_diagonal),
                float(cell_size),
                int(sx),
                int(sy),
                int(gx),
                int(gy),
            ),
        )
    except Exception:
        return _astar_length_py(
            start,
            goal,
            grid,
            field_size,
            cell_size,
            allow_diagonal=allow_diagonal,
            risk_grid=risk_grid,
            risk_weight=risk_weight,
            heuristic_weight=heuristic_weight,
        )


def _astar_path_py(
    start: np.ndarray,
    goal: np.ndarray,
    grid: np.ndarray,
    field_size: float,
    cell_size: float,
    *,
    allow_diagonal: bool,
    risk_grid: np.ndarray | None = None,
    risk_weight: float = 0.0,
    heuristic_weight: float = 1.0,
) -> list[tuple[float, float]]:
    height, width = grid.shape
    sx, sy = _world_to_grid(start, field_size, cell_size, width, height)
    gx, gy = _world_to_grid(goal, field_size, cell_size, width, height)
    # Если старт или цель недоступны, возвращаем пустой путь.
    if grid[sy, sx] or grid[gy, gx]:
        return []

    diag = float(np.sqrt(2.0) * cell_size)
    ortho = float(cell_size)
    if allow_diagonal:
        moves = (
            (1, 0, ortho),
            (-1, 0, ortho),
            (0, 1, ortho),
            (0, -1, ortho),
            (1, 1, diag),
            (1, -1, diag),
            (-1, 1, diag),
            (-1, -1, diag),
        )
    else:
        moves = ((1, 0, ortho), (-1, 0, ortho), (0, 1, ortho), (0, -1, ortho))

    g_cost = np.full((height, width), np.inf, dtype=np.float32)
    g_cost[sy, sx] = 0.0
    closed = np.zeros((height, width), dtype=bool)
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    heap: list[tuple[float, tuple[int, int]]] = []
    heapq.heappush(heap, (0.0, (sx, sy)))

    while heap:
        _, (x, y) = heapq.heappop(heap)
        if closed[y, x]:
            continue
        if x == gx and y == gy:
            return _reconstruct_path(came_from, (sx, sy), (gx, gy), field_size, cell_size)
        closed[y, x] = True
        for dx, dy, step_cost in moves:
            nx = x + dx
            ny = y + dy
            if nx < 0 or ny < 0 or nx >= width or ny >= height:
                continue
            if grid[ny, nx]:
                continue
            risk = float(risk_grid[ny, nx]) if risk_grid is not None else 0.0
            step_cost_weighted = float(step_cost * (1.0 + (risk_weight * risk)))
            tentative = float(g_cost[y, x] + step_cost_weighted)
            if tentative < g_cost[ny, nx]:
                g_cost[ny, nx] = tentative
                came_from[(nx, ny)] = (x, y)
                h = _heuristic(nx, ny, gx, gy, cell_size)
                heapq.heappush(heap, (tentative + (heuristic_weight * h), (nx, ny)))
    return []


def _astar_path(
    start: np.ndarray,
    goal: np.ndarray,
    grid: np.ndarray,
    field_size: float,
    cell_size: float,
    *,
    allow_diagonal: bool,
    risk_grid: np.ndarray | None = None,
    risk_weight: float = 0.0,
    heuristic_weight: float = 1.0,
) -> list[tuple[float, float]]:
    if not _NUMBA_AVAILABLE:
        return _astar_path_py(
            start,
            goal,
            grid,
            field_size,
            cell_size,
            allow_diagonal=allow_diagonal,
            risk_grid=risk_grid,
            risk_weight=risk_weight,
            heuristic_weight=heuristic_weight,
        )
    height, width = grid.shape
    sx, sy = _world_to_grid(start, field_size, cell_size, width, height)
    gx, gy = _world_to_grid(goal, field_size, cell_size, width, height)
    rg = np.zeros((height, width), dtype=np.float32)
    has_risk = False
    if risk_grid is not None:
        rg = np.asarray(risk_grid, dtype=np.float32)
        has_risk = True
    try:
        path_x, path_y, length = _astar_path_numba(
            np.asarray(grid, dtype=np.bool_),
            rg,
            has_risk,
            float(risk_weight),
            float(heuristic_weight),
            bool(allow_diagonal),
            float(cell_size),
            int(sx),
            int(sy),
            int(gx),
            int(gy),
        )
    except Exception:
        return _astar_path_py(
            start,
            goal,
            grid,
            field_size,
            cell_size,
            allow_diagonal=allow_diagonal,
            risk_grid=risk_grid,
            risk_weight=risk_weight,
            heuristic_weight=heuristic_weight,
        )
    if length <= 0:
        return []
    return [_grid_to_world(int(path_x[i]), int(path_y[i]), field_size, cell_size) for i in range(length)]


def _reconstruct_path(
    came_from: dict[tuple[int, int], tuple[int, int]],
    start: tuple[int, int],
    goal: tuple[int, int],
    field_size: float,
    cell_size: float,
) -> list[tuple[float, float]]:
    cur = goal
    path = [cur]
    while cur != start:
        cur = came_from.get(cur)
        if cur is None:
            break
        path.append(cur)
    path.reverse()
    return [_grid_to_world(p[0], p[1], field_size, cell_size) for p in path]


def _grid_to_world(x: int, y: int, field_size: float, cell_size: float) -> tuple[float, float]:
    px = float(x * cell_size + cell_size / 2.0)
    py = float(y * cell_size + cell_size / 2.0)
    px = float(np.clip(px, 0.0, field_size))
    py = float(np.clip(py, 0.0, field_size))
    return (px, py)


def _heuristic(x: int, y: int, gx: int, gy: int, cell_size: float) -> float:
    dx = float(x - gx)
    dy = float(y - gy)
    return float(np.hypot(dx, dy) * cell_size)


def _world_to_grid(
    pos: np.ndarray,
    field_size: float,
    cell_size: float,
    width: int,
    height: int,
) -> tuple[int, int]:
    x = float(np.clip(pos[0], 0.0, field_size))
    y = float(np.clip(pos[1], 0.0, field_size))
    gx = int(np.clip(np.floor(x / cell_size), 0, width - 1))
    gy = int(np.clip(np.floor(y / cell_size), 0, height - 1))
    return gx, gy
