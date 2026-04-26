"""Проверка компонентной награды."""

import numpy as np

from env.rewards.pipeline import RewardFn
from env.rewards.rewarder import RewardConfig
from env.state import SimState


def _make_state(*, dist: float) -> SimState:
    pos = np.array([[10.0, 10.0]], dtype=np.float32)
    vel = np.array([[1.0, 0.0]], dtype=np.float32)
    alive = np.array([True])
    target = np.array([20.0, 20.0], dtype=np.float32)
    target_vel = np.zeros(2, dtype=np.float32)
    dists = np.array([float(dist)], dtype=np.float32)
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


def test_reward_pipeline_components():
    prev_state = _make_state(dist=10.0)
    cur_state = _make_state(dist=9.0)
    cfg = RewardConfig(components=("progress",))
    reward_fn = RewardFn(cfg, field_size=100.0, goal_radius=3.0)
    rew, parts = reward_fn(prev_state, cur_state, 0)
    assert parts["rew_progress"] > 0.0
    assert parts["rew_risk"] == 0.0
    assert np.isclose(rew, parts["rew_progress"])


def test_finish_bonus_paid_on_newly_finished_step():
    prev_state = _make_state(dist=2.0)
    cur_state = _make_state(dist=1.0)
    cur_state.in_goal[:] = True
    cur_state.finished[:] = True
    cur_state.newly_finished[:] = True
    cfg = RewardConfig(components=("finish_bonus",), w_finish_bonus=123.0)
    reward_fn = RewardFn(cfg, field_size=100.0, goal_radius=3.0)

    rew, parts = reward_fn(prev_state, cur_state, 0)
    batch_rewards, batch_parts = reward_fn.compute_all(prev_state, cur_state)

    assert rew == 123.0
    assert parts["rew_finish_bonus"] == 123.0
    assert batch_rewards[0] == 123.0
    assert batch_parts["rew_finish_bonus"][0] == 123.0
