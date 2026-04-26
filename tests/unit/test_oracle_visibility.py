"""Проверка видимости оракула в контексте."""

import numpy as np

from common.context import context_from_state
from env.state import SimState, public_state_from_state


def _make_state() -> SimState:
    pos = np.array([[10.0, 10.0]], dtype=np.float32)
    vel = np.array([[1.0, 0.0]], dtype=np.float32)
    alive = np.array([True])
    target = np.array([20.0, 20.0], dtype=np.float32)
    target_vel = np.zeros(2, dtype=np.float32)
    dists = np.linalg.norm(pos - target, axis=1).astype(np.float32)
    in_goal = np.array([False])
    in_goal_steps = np.array([0], dtype=np.int32)
    finished = np.array([False])
    newly_finished = np.array([False])
    risk_p = np.array([0.0], dtype=np.float32)
    min_neighbor = np.array([5.0], dtype=np.float32)
    last_action = np.zeros((1, 2), dtype=np.float32)
    walls = np.array([[1.0, 1.0, 1.0, 1.0]], dtype=np.float32)
    measured_accel = np.zeros_like(vel, dtype=np.float32)
    energy = np.array([100.0], dtype=np.float32)
    energy_level = np.array([1.0], dtype=np.float32)
    agent_state = np.array([0], dtype=np.int8)
    return SimState(
        pos=pos,
        vel=vel,
        alive=alive,
        target_pos=target,
        target_vel=target_vel,
        timestep=1,
        threats=[],
        dists=dists,
        in_goal=in_goal,
        in_goal_steps=in_goal_steps,
        finished=finished,
        newly_finished=newly_finished,
        risk_p=risk_p,
        min_neighbor_dist=min_neighbor,
        last_action=last_action,
        walls=walls,
        oracle_dir=np.array([[1.0, 0.0]], dtype=np.float32),
        static_walls=[],
        static_circles=[],
        collision_speed=np.zeros_like(dists),
        measured_accel=measured_accel,
        energy=energy,
        energy_level=energy_level,
        agent_state=agent_state,
        field_size=100.0,
        control_mode="waypoint",
        max_speed=10.0,
        max_accel=5.0,
        max_thrust=5.0,
        mass=1.0,
        drag_coeff=0.0,
        dt=0.1,
        drag=0.0,
        grid_res=1.0,
        agent_radius=0.5,
        wall_friction=0.0,
        decision_step=1,
    )


def test_oracle_visibility_context():
    state = public_state_from_state(_make_state(), include_oracle=True)
    ctx_hidden = context_from_state(state, include_oracle=False)
    assert ctx_hidden.oracle_dir is None
    ctx_visible = context_from_state(state, include_oracle=True)
    assert ctx_visible.oracle_dir is not None


def test_context_rejects_simstate():
    with np.testing.assert_raises(TypeError):
        context_from_state(_make_state(), include_oracle=True)
