"""Public wall-collision helpers shared by physics and planners."""

from __future__ import annotations

import numpy as np


def resolve_wall_slide(
    pos: np.ndarray,
    vel: np.ndarray,
    dt: float,
    walls: list[tuple[float, float, float, float]],
    radius: float,
    friction: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    v = np.asarray(vel, dtype=np.float32).copy()
    if not walls or (v[0] == 0.0 and v[1] == 0.0):
        return np.asarray(pos, dtype=np.float32) + (v * dt), v, 0.0

    hit_speed = 0.0
    cur_pos = np.asarray(pos, dtype=np.float32)
    for _ in range(2):
        next_pos = cur_pos + (v * dt)
        hit_any = False
        for rect in walls:
            normal = circle_rect_normal(next_pos, rect, radius)
            if normal is None:
                continue
            hit_any = True
            ncomp = float(np.dot(v, normal))
            if ncomp < 0.0:
                v = v - (normal * ncomp)
                if friction > 0.0:
                    v = v * (1.0 - friction)
                hit_speed = max(hit_speed, -ncomp)
        if not hit_any:
            break
    next_pos = cur_pos + (v * dt)
    if circle_hits_any(next_pos, walls, radius):
        return cur_pos, np.zeros_like(v), float(hit_speed)
    return next_pos, v, float(hit_speed)


def resolve_wall_slide_batch(
    pos: np.ndarray,
    vel: np.ndarray,
    dt: float,
    walls: list[tuple[float, float, float, float]],
    radius: float,
    friction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pos = np.asarray(pos, dtype=np.float32)
    vel = np.asarray(vel, dtype=np.float32)
    n = pos.shape[0]
    impacts = np.zeros(n, dtype=np.float32)
    if n == 0:
        return pos, vel, impacts
    if not walls:
        return pos + (vel * dt), vel, impacts

    speeds = np.linalg.norm(vel, axis=1)
    moving_mask = speeds > 0.0
    if not np.any(moving_mask):
        return pos + (vel * dt), vel, impacts

    idx = np.where(moving_mask)[0]
    sub_pos = pos[idx].copy()
    sub_vel = vel[idx].copy()
    sub_pos, sub_vel, sub_imp = resolve_wall_slide_batch_inner(sub_pos, sub_vel, dt, walls, radius, friction)
    out_pos = pos + (vel * dt)
    out_vel = vel.copy()
    out_pos[idx] = sub_pos
    out_vel[idx] = sub_vel
    impacts[idx] = sub_imp
    return out_pos, out_vel, impacts


def resolve_wall_slide_batch_inner(
    pos: np.ndarray,
    vel: np.ndarray,
    dt: float,
    walls: list[tuple[float, float, float, float]],
    radius: float,
    friction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    v = np.asarray(vel, dtype=np.float32).copy()
    cur_pos = np.asarray(pos, dtype=np.float32).copy()
    hit_speed = np.zeros(cur_pos.shape[0], dtype=np.float32)
    if not walls:
        return cur_pos + (v * dt), v, hit_speed

    for _ in range(2):
        next_pos = cur_pos + (v * dt)
        hit_any = np.zeros(cur_pos.shape[0], dtype=bool)
        for rect in walls:
            normals, hit_mask = circle_rect_normals_batch(next_pos, rect, radius)
            if not np.any(hit_mask):
                continue
            hit_any |= hit_mask
            ncomp = np.sum(v * normals, axis=1)
            hit_dir = hit_mask & (ncomp < 0.0)
            if np.any(hit_dir):
                v[hit_dir] = v[hit_dir] - (normals[hit_dir] * ncomp[hit_dir][:, None])
                if friction > 0.0:
                    v[hit_dir] = v[hit_dir] * (1.0 - friction)
                hit_speed[hit_dir] = np.maximum(hit_speed[hit_dir], (-ncomp[hit_dir]).astype(np.float32))
        if not np.any(hit_any):
            break

    next_pos = cur_pos + (v * dt)
    blocked = circle_hits_any_batch(next_pos, walls, radius)
    if np.any(blocked):
        next_pos[blocked] = cur_pos[blocked]
        v[blocked] = 0.0
    return next_pos, v, hit_speed


def circle_hits_any_batch(
    centers: np.ndarray,
    walls: list[tuple[float, float, float, float]],
    radius: float,
) -> np.ndarray:
    if centers.size == 0 or not walls:
        return np.zeros((centers.shape[0],), dtype=bool)
    hit = np.zeros((centers.shape[0],), dtype=bool)
    for rect in walls:
        _, mask = circle_rect_normals_batch(centers, rect, radius)
        hit |= mask
    return hit


def circle_rect_normals_batch(
    centers: np.ndarray,
    rect: tuple[float, float, float, float],
    radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    cx = centers[:, 0]
    cy = centers[:, 1]
    x1, y1, x2, y2 = rect
    closest_x = np.clip(cx, x1, x2)
    closest_y = np.clip(cy, y1, y2)
    dx = cx - closest_x
    dy = cy - closest_y
    dist2 = (dx * dx) + (dy * dy)
    r2 = float(radius) * float(radius)
    hit = dist2 <= r2
    normals = np.zeros_like(centers, dtype=np.float32)

    mask = hit & (dist2 > 1e-8)
    if np.any(mask):
        inv = 1.0 / np.sqrt(dist2[mask])
        normals[mask, 0] = dx[mask] * inv
        normals[mask, 1] = dy[mask] * inv

    inside = hit & ~mask
    if np.any(inside):
        idx = np.where(inside)[0]
        left = np.abs(cx[idx] - float(x1))
        right = np.abs(float(x2) - cx[idx])
        bottom = np.abs(cy[idx] - float(y1))
        top = np.abs(float(y2) - cy[idx])
        dists = np.stack([left, right, bottom, top], axis=1)
        arg = np.argmin(dists, axis=1)
        normals[idx[arg == 0]] = np.array([1.0, 0.0], dtype=np.float32)
        normals[idx[arg == 1]] = np.array([-1.0, 0.0], dtype=np.float32)
        normals[idx[arg == 2]] = np.array([0.0, 1.0], dtype=np.float32)
        normals[idx[arg == 3]] = np.array([0.0, -1.0], dtype=np.float32)

    return normals, hit


def circle_hits_any(center: np.ndarray, walls: list[tuple[float, float, float, float]], radius: float) -> bool:
    for rect in walls:
        if circle_rect_normal(center, rect, radius) is not None:
            return True
    return False


def circle_rect_normal(
    center: np.ndarray,
    rect: tuple[float, float, float, float],
    radius: float,
) -> np.ndarray | None:
    cx = float(center[0])
    cy = float(center[1])
    x1, y1, x2, y2 = rect
    closest_x = float(np.clip(cx, x1, x2))
    closest_y = float(np.clip(cy, y1, y2))
    dx = cx - closest_x
    dy = cy - closest_y
    dist2 = (dx * dx) + (dy * dy)
    r2 = float(radius) * float(radius)
    if dist2 > r2:
        return None
    if dist2 > 1e-8:
        inv = 1.0 / float(np.sqrt(dist2))
        return np.array([dx * inv, dy * inv], dtype=np.float32)
    left = abs(cx - float(x1))
    right = abs(float(x2) - cx)
    bottom = abs(cy - float(y1))
    top = abs(float(y2) - cy)
    m = min(left, right, bottom, top)
    if m == left:
        return np.array([1.0, 0.0], dtype=np.float32)
    if m == right:
        return np.array([-1.0, 0.0], dtype=np.float32)
    if m == bottom:
        return np.array([0.0, 1.0], dtype=np.float32)
    return np.array([0.0, -1.0], dtype=np.float32)
