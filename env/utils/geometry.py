"""Геометрические утилиты для сцен, стен и препятствий."""

from __future__ import annotations

from collections.abc import Iterable


def parse_wall_rect(wall) -> tuple[float, float, float, float] | None:
    if isinstance(wall, (list, tuple)) and len(wall) == 4:
        x1, y1, x2, y2 = wall
        return float(x1), float(y1), float(x2), float(y2)
    if not isinstance(wall, dict):
        return None
    if all(k in wall for k in ("x1", "y1", "x2", "y2")):
        return float(wall["x1"]), float(wall["y1"]), float(wall["x2"]), float(wall["y2"])
    if all(k in wall for k in ("x", "y", "w", "h")):
        x = float(wall["x"])
        y = float(wall["y"])
        w = float(wall["w"])
        h = float(wall["h"])
        return x, y, x + w, y + h
    if all(k in wall for k in ("center", "size")):
        cx, cy = wall["center"]
        sx, sy = wall["size"]
        cx = float(cx)
        cy = float(cy)
        sx = float(sx)
        sy = float(sy)
        return cx - sx / 2.0, cy - sy / 2.0, cx + sx / 2.0, cy + sy / 2.0
    return None


def normalize_walls(walls: Iterable) -> list[tuple[float, float, float, float]]:
    rects: list[tuple[float, float, float, float]] = []
    for w in walls or []:
        rect = parse_wall_rect(w)
        if rect is None:
            continue
        x1, y1, x2, y2 = rect
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        rects.append((float(x1), float(y1), float(x2), float(y2)))
    return rects


def parse_circle(obj) -> tuple[float, float, float] | None:
    if isinstance(obj, dict):
        if "radius" in obj:
            if "pos" in obj:
                cx, cy = obj["pos"]
                return float(cx), float(cy), float(obj["radius"])
            if "center" in obj:
                cx, cy = obj["center"]
                return float(cx), float(cy), float(obj["radius"])
            if "x" in obj and "y" in obj:
                return float(obj["x"]), float(obj["y"]), float(obj["radius"])
    if hasattr(obj, "position") and hasattr(obj, "radius"):
        try:
            pos = obj.position
            return float(pos[0]), float(pos[1]), float(obj.radius)
        except Exception:
            return None
    return None
