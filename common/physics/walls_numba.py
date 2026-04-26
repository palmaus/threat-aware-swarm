"""Optional Numba kernels matching :mod:`common.physics.walls` semantics."""

from __future__ import annotations

import math

try:  # pragma: no cover - optional acceleration
    from numba import njit
except Exception:  # pragma: no cover
    njit = None

NUMBA_AVAILABLE = njit is not None

if NUMBA_AVAILABLE:

    @njit(cache=True, fastmath=True)
    def circle_rect_normal_numba(cx, cy, x1, y1, x2, y2, radius):
        closest_x = x1 if cx < x1 else (x2 if cx > x2 else cx)
        closest_y = y1 if cy < y1 else (y2 if cy > y2 else cy)
        dx = cx - closest_x
        dy = cy - closest_y
        dist2 = (dx * dx) + (dy * dy)
        r2 = radius * radius
        if dist2 > r2:
            return False, 0.0, 0.0
        if dist2 > 1e-8:
            inv = 1.0 / math.sqrt(dist2)
            return True, dx * inv, dy * inv
        left = abs(cx - x1)
        right = abs(x2 - cx)
        bottom = abs(cy - y1)
        top = abs(y2 - cy)
        nx = 1.0
        ny = 0.0
        m = left
        if right < m:
            m = right
            nx = -1.0
            ny = 0.0
        if bottom < m:
            m = bottom
            nx = 0.0
            ny = 1.0
        if top < m:
            nx = 0.0
            ny = -1.0
        return True, nx, ny

    @njit(cache=True, fastmath=True)
    def circle_hits_any_numba(cx, cy, walls, radius):
        for i in range(walls.shape[0]):
            x1 = walls[i, 0]
            y1 = walls[i, 1]
            x2 = walls[i, 2]
            y2 = walls[i, 3]
            hit, _, _ = circle_rect_normal_numba(cx, cy, x1, y1, x2, y2, radius)
            if hit:
                return True
        return False

    @njit(cache=True, fastmath=True)
    def resolve_wall_slide_numba(px, py, vx, vy, dt, walls, radius, friction):
        hit_speed = 0.0
        for _ in range(2):
            next_x = px + vx * dt
            next_y = py + vy * dt
            hit_any = False
            for i in range(walls.shape[0]):
                x1 = walls[i, 0]
                y1 = walls[i, 1]
                x2 = walls[i, 2]
                y2 = walls[i, 3]
                hit, nx, ny = circle_rect_normal_numba(next_x, next_y, x1, y1, x2, y2, radius)
                if hit:
                    hit_any = True
                    ncomp = (vx * nx) + (vy * ny)
                    if ncomp < 0.0:
                        vx = vx - (nx * ncomp)
                        vy = vy - (ny * ncomp)
                        if friction > 0.0:
                            vx *= 1.0 - friction
                            vy *= 1.0 - friction
                        hs = -ncomp
                        if hs > hit_speed:
                            hit_speed = hs
            if not hit_any:
                break
        next_x = px + vx * dt
        next_y = py + vy * dt
        if circle_hits_any_numba(next_x, next_y, walls, radius):
            return px, py, 0.0, 0.0, hit_speed
        return next_x, next_y, vx, vy, hit_speed

else:
    circle_rect_normal_numba = None
    circle_hits_any_numba = None
    resolve_wall_slide_numba = None
