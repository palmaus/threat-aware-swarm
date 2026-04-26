from __future__ import annotations

import numpy as np

from baselines.utils import normalize

MOVES_CARDINAL = (
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
)
MOVES_DIAGONAL = (
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
    (1, 1),
    (1, -1),
    (-1, 1),
    (-1, -1),
)
MOVES_DX = np.array([1, -1, 0, 0, 1, 1, -1, -1], dtype=np.int8)
MOVES_DY = np.array([0, 0, 1, -1, 1, -1, 1, -1], dtype=np.int8)


def escape_direction(
    grid: np.ndarray,
    to_target: np.ndarray,
    allow_diagonal: bool,
    penalty_grid: np.ndarray | None = None,
    prev_vec: np.ndarray | None = None,
    turn_penalty: float = 0.15,
    oracle_dir: np.ndarray | None = None,
) -> np.ndarray | None:
    if grid is None or grid.size == 0:
        if to_target is None or np.linalg.norm(to_target) <= 1e-6:
            return None
        vec = np.array([-to_target[1], to_target[0]], dtype=np.float32)
        return normalize(vec)

    center = grid.shape[0] // 2
    moves = MOVES_DIAGONAL if allow_diagonal else MOVES_CARDINAL
    target_dir = normalize(to_target) if to_target is not None else np.zeros(2, dtype=np.float32)
    oracle_valid = oracle_dir is not None and np.linalg.norm(oracle_dir) > 1e-6
    if oracle_valid:
        target_dir = normalize(oracle_dir)
    prev_dir = normalize(prev_vec) if prev_vec is not None else None
    best = None
    best_score = float("inf")
    for dx, dy in moves:
        nx = center + dx
        ny = center + dy
        if nx < 0 or ny < 0 or nx >= grid.shape[1] or ny >= grid.shape[0]:
            continue
        cost = float(grid[ny, nx])
        if penalty_grid is not None:
            try:
                cost += float(penalty_grid[ny, nx])
            except Exception:
                pass
        move_vec = normalize(np.array([dx, dy], dtype=np.float32))
        dot_prod = float(np.dot(move_vec, target_dir))
        if oracle_valid:
            align = 1.0 - float(np.clip(dot_prod, -1.0, 1.0))
            score = cost + 2.0 * align
        else:
            align = abs(dot_prod)
            score = cost + 2.0 * align
        if prev_dir is not None:
            change = 1.0 - float(np.dot(move_vec, prev_dir))
            score += float(turn_penalty) * change
        if score < best_score:
            best_score = score
            best = move_vec
    return best


def dist_from_info(info: dict, to_target: np.ndarray) -> tuple[float | None, bool]:
    if "dist" in info:
        try:
            return float(info["dist"]), False
        except Exception:
            pass
    pos = info.get("pos")
    target = info.get("target_pos")
    if pos is not None and target is not None:
        try:
            return float(
                np.linalg.norm(np.asarray(target, dtype=np.float32) - np.asarray(pos, dtype=np.float32))
            ), False
        except Exception:
            pass
    if to_target is not None:
        return float(np.linalg.norm(to_target)), True
    return None, False


def maybe_denorm_dist(dist: float | None, dist_normed: bool, field_size: float | None) -> float | None:
    if dist is None:
        return None
    if dist_normed and field_size is not None:
        return float(dist * field_size)
    return float(dist)


def maybe_denorm_vec(vec: np.ndarray, dist_normed: bool, field_size: float | None) -> np.ndarray | None:
    if vec is None:
        return None
    if dist_normed and field_size is not None:
        return np.asarray(vec, dtype=np.float32) * float(field_size)
    return np.asarray(vec, dtype=np.float32)


def memory_penalty_from_entries(
    coords: np.ndarray,
    scores: np.ndarray,
    last: np.ndarray,
    *,
    base_cell: tuple[int, int],
    step: int,
    shape: tuple[int, int],
    memory_penalty: float,
    memory_decay: float,
) -> tuple[np.ndarray | None, np.ndarray]:
    if coords.size == 0 or scores.size == 0 or last.size == 0:
        return None, np.zeros((0, 2), dtype=np.int32)
    decay = float(memory_decay)
    if decay <= 0.0:
        return None, np.asarray(coords, dtype=np.int32)
    age = np.maximum(0, int(step) - np.asarray(last, dtype=np.int32))
    decayed = np.asarray(scores, dtype=np.float32) * np.power(decay, age, dtype=np.float32)
    stale_mask = decayed < 1e-3
    active_mask = ~stale_mask
    stale = np.asarray(coords[stale_mask], dtype=np.int32)
    if not np.any(active_mask):
        return None, stale
    active_coords = np.asarray(coords[active_mask], dtype=np.int32)
    active_scores = np.asarray(decayed[active_mask], dtype=np.float32)
    center = int(shape[0] // 2)
    dx = active_coords[:, 0] - int(base_cell[0])
    dy = active_coords[:, 1] - int(base_cell[1])
    valid = (np.abs(dx) <= center) & (np.abs(dy) <= center)
    if not np.any(valid):
        return None, stale
    penalty = np.zeros(shape, dtype=np.float32)
    xs = center + dx[valid]
    ys = center + dy[valid]
    penalty[ys, xs] = float(memory_penalty) * active_scores[valid]
    return penalty, stale


def reuse_direction_to_cell(
    next_cell: tuple[int, int],
    pos: np.ndarray,
    *,
    cell_size: float,
) -> np.ndarray:
    pos_arr = np.asarray(pos, dtype=np.float32)
    vec = np.array(
        [
            (float(next_cell[0]) + 0.5) * float(cell_size) - float(pos_arr[0]),
            (float(next_cell[1]) + 0.5) * float(cell_size) - float(pos_arr[1]),
        ],
        dtype=np.float32,
    )
    if float(vec[0] * vec[0] + vec[1] * vec[1]) <= 1e-12:
        return np.zeros((2,), dtype=np.float32)
    return normalize(vec)
