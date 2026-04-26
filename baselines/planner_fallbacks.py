"""Composition helpers for planner-to-planner fallback policies."""

from __future__ import annotations

from typing import Any


def create_astar_fallback(**kwargs: Any):
    """Build the A* fallback lazily without coupling MPC to A* internals."""

    from baselines.astar_grid import AStarGridPolicy

    return AStarGridPolicy(**kwargs)
