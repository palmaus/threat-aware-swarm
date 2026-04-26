"""Смоук‑тест визуализатора в безэкранном режиме."""

import os

import pytest


def test_visualizer_smoke():
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    _pygame = pytest.importorskip("pygame")

    from env.config import EnvConfig
    from env.pz_env import SwarmPZEnv
    from env.visualizer import SwarmVisualizer

    env = SwarmPZEnv(EnvConfig(), max_steps=10, goal_radius=3.0)
    env.reset(seed=0)

    viz = SwarmVisualizer(env.config)
    ok = viz.render(env.sim, target_pos=env.target_pos, goal_radius=env.goal_radius)
    assert ok is not None
