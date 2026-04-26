"""Проверка кэша наблюдений для Dict-формата."""

import numpy as np

from env.config import EnvConfig
from env.observations.observer import SwarmObserver
from env.state import SimState


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
        oracle_dir=None,
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


def test_observer_cache_dict():
    cfg = EnvConfig(n_agents=1)
    observer = SwarmObserver(cfg, rng=np.random.default_rng(0))
    state = _make_state()
    obs_a = observer.build(state, 0)
    obs_b = observer.build(state, 0)
    assert obs_a is obs_b
    obs_map_a = observer.build_all(state, ["drone_0"])
    obs_map_b = observer.build_all(state, ["drone_0"])
    assert obs_map_a is obs_map_b
