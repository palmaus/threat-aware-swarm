"""Проверка штрафа за время в награде."""

from __future__ import annotations

import numpy as np

from env.rewards.pipeline import RewardFn
from env.rewards.rewarder import RewardConfig
from env.state import SimState


def _state(timestep: int) -> SimState:
    pos = np.array([[0.0, 0.0]], dtype=np.float32)
    vel = np.zeros_like(pos)
    alive = np.array([True])
    target = np.array([10.0, 0.0], dtype=np.float32)
    dists = np.array([10.0], dtype=np.float32)
    in_goal = np.array([False])
    return SimState(
        pos=pos,
        vel=vel,
        alive=alive,
        target_pos=target,
        target_vel=np.zeros(2, dtype=np.float32),
        timestep=timestep,
        threats=[],
        dists=dists,
        in_goal=in_goal,
        in_goal_steps=np.array([0], dtype=np.int32),
        finished=np.array([False]),
        newly_finished=np.array([False]),
        risk_p=np.array([0.0], dtype=np.float32),
        min_neighbor_dist=np.array([10.0], dtype=np.float32),
        last_action=np.zeros((1, 2), dtype=np.float32),
        walls=np.zeros((1, 4), dtype=np.float32),
        oracle_dir=None,
        static_walls=[],
        static_circles=[],
        collision_speed=np.zeros((1,), dtype=np.float32),
        measured_accel=np.zeros((1, 2), dtype=np.float32),
        energy=np.array([100.0], dtype=np.float32),
        energy_level=np.array([1.0], dtype=np.float32),
        agent_state=np.array([0], dtype=np.int8),
        control_mode="waypoint",
        field_size=100.0,
        max_speed=5.0,
        max_accel=4.0,
        max_thrust=8.0,
        mass=1.0,
        drag_coeff=0.0,
        dt=0.1,
        drag=0.0,
        grid_res=1.0,
        agent_radius=0.5,
        wall_friction=0.0,
        decision_step=0,
    )


def test_time_penalty_component():
    cfg = RewardConfig(w_time=1.0)
    reward_fn = RewardFn(cfg, field_size=100.0, goal_radius=3.0)
    prev = _state(0)
    cur = _state(1)
    total, parts = reward_fn(prev, cur, 0)
    assert parts.get("rew_time", 0.0) < 0.0
    assert total < 0.0
