from __future__ import annotations

import numpy as np

from common.contracts import maybe_validate_reset
from env.state import PublicState, SimState


def _make_state() -> tuple[SimState, PublicState]:
    pos = np.zeros((1, 2), dtype=np.float32)
    vel = np.zeros_like(pos)
    alive = np.array([True])
    state = SimState(
        pos=pos,
        vel=vel,
        alive=alive,
        target_pos=np.array([1.0, 1.0], dtype=np.float32),
        target_vel=np.zeros((2,), dtype=np.float32),
        timestep=0,
        threats=[],
        dists=np.array([0.0], dtype=np.float32),
        in_goal=np.array([False]),
        in_goal_steps=np.array([0], dtype=np.int32),
        finished=np.array([False]),
        newly_finished=np.array([False]),
        risk_p=np.array([0.0], dtype=np.float32),
        min_neighbor_dist=np.array([np.inf], dtype=np.float32),
        last_action=np.zeros((1, 2), dtype=np.float32),
        walls=np.zeros((1, 4), dtype=np.float32),
        oracle_dir=None,
        static_walls=[],
        static_circles=[],
        collision_speed=np.zeros((1,), dtype=np.float32),
        measured_accel=np.zeros_like(pos, dtype=np.float32),
        energy=np.array([1.0], dtype=np.float32),
        energy_level=np.array([1.0], dtype=np.float32),
        agent_state=np.array([0], dtype=np.int8),
        field_size=10.0,
        control_mode="waypoint",
        max_speed=1.0,
        max_accel=1.0,
        max_thrust=1.0,
        mass=1.0,
        drag_coeff=0.0,
        dt=0.1,
        drag=0.0,
        grid_res=1.0,
        agent_radius=0.5,
        wall_friction=0.0,
    )
    public_state = PublicState(
        pos=pos,
        vel=vel,
        alive=alive,
        agent_state=np.array([0], dtype=np.int8),
        target_pos=np.array([1.0, 1.0], dtype=np.float32),
        target_vel=np.zeros((2,), dtype=np.float32),
        threats=[],
        dists=np.array([0.0], dtype=np.float32),
        in_goal=np.array([False]),
        risk_p=np.array([0.0], dtype=np.float32),
        oracle_dir=None,
        static_walls=[],
        static_circles=[],
        field_size=10.0,
        max_speed=1.0,
        max_accel=1.0,
        max_thrust=1.0,
        mass=1.0,
        drag_coeff=0.0,
        dt=0.1,
        drag=0.0,
        grid_res=1.0,
        agent_radius=0.5,
        wall_friction=0.0,
        energy_level=np.array([1.0], dtype=np.float32),
        measured_accel=np.zeros_like(pos, dtype=np.float32),
        timestep=0,
        decision_step=0,
        control_mode="waypoint",
    )
    return state, public_state


def test_contract_guardrails_trigger(monkeypatch):
    state, public_state = _make_state()
    obs = {"drone_0": {"vector": np.zeros((4,), dtype=np.float32), "grid": None}}
    monkeypatch.setenv("TA_STRICT_DEBUG", "1")
    try:
        maybe_validate_reset(state=state, public_state=public_state, observations=obs, grid_width=41)
    except ValueError as exc:
        assert "vector" in str(exc)
