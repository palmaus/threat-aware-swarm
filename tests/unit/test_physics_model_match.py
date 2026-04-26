"""Сравнение динамики PhysicsCore и общей модели."""

from __future__ import annotations

import numpy as np

from common.physics_model import apply_accel_dynamics_vel, compute_drag_factor
from env.config import EnvConfig
from env.physics.core import PhysicsCore


def test_physics_core_matches_model_step():
    cfg = EnvConfig(n_agents=1)
    cfg.dt = 0.1
    cfg.max_speed = 3.0
    cfg.physics.mass = 2.0
    cfg.physics.max_thrust = 4.0
    cfg.physics.drag_coeff = 0.2
    cfg.wind.enabled = False
    cfg.battery.capacity = 0.0
    core = PhysicsCore(cfg)
    core.walls = []
    core.circle_obstacles_pos = np.zeros((0, 2), dtype=np.float32)
    core.circle_obstacles_radius = np.zeros((0,), dtype=np.float32)

    pos0 = np.array([[1.0, 2.0]], dtype=np.float32)
    vel0 = np.array([[0.5, -0.25]], dtype=np.float32)
    core.agents_pos[:] = pos0
    core.agents_vel[:] = vel0

    thrust = np.array([[3.0, 0.0]], dtype=np.float32)
    drag = compute_drag_factor(cfg.physics.drag_coeff, cfg.physics.mass, cfg.dt)
    accel = thrust / cfg.physics.mass
    expected_vel = apply_accel_dynamics_vel(
        vel0,
        accel,
        cfg.dt,
        drag=drag,
        max_speed=cfg.max_speed,
        wind=None,
    )
    expected_pos = pos0 + expected_vel * cfg.dt

    core.step(thrust)

    np.testing.assert_allclose(core.agents_vel, expected_vel, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(core.agents_pos, expected_pos, atol=1e-6, rtol=1e-6)


def test_apply_accel_dynamics_vel_clips_batch_speed_without_index_error():
    vel = np.array([[2.35, 4.11], [2.35, 4.11]], dtype=np.float32)
    accel = np.array([[4.0, 6.92], [4.0, 6.92]], dtype=np.float32)

    out = apply_accel_dynamics_vel(
        vel,
        accel,
        0.1,
        drag=0.0,
        max_speed=5.0,
        wind=None,
    )

    assert out.shape == vel.shape
    speeds = np.linalg.norm(out, axis=1)
    assert np.all(speeds <= 5.0 + 1e-6)
