"""Проверка сборщиков наблюдений и базовых метрик."""

import numpy as np

from env.observations.builder import ObservationBuilder
from env.rewards.metrics import MetricsFn
from env.rewards.pipeline import RewardFn
from env.rewards.rewarder import RewardConfig
from env.state import SimState


class DummyThreat:
    def __init__(self, position, radius, intensity):
        self.position = np.asarray(position, dtype=np.float32)
        self.radius = float(radius)
        self.intensity = float(intensity)


def make_state():
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
    threats = [DummyThreat([30.0, 30.0], 5.0, 0.1)]
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
        timestep=0,
        threats=threats,
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
    )


def test_observation_builder_shape():
    state = make_state()
    builder = ObservationBuilder(field_size=100.0, max_speed=10.0, grid_width=41, grid_res=1.0)
    obs = builder.build(state, 0)
    assert isinstance(obs, dict)
    assert obs["vector"].shape == (13,)
    assert obs["grid"].shape == (1, 41, 41)
    assert obs["vector"].dtype == np.float32
    assert obs["grid"].dtype == np.float32
    assert np.all(np.isfinite(obs["vector"]))
    assert np.all(np.isfinite(obs["grid"]))


def test_observation_builder_does_not_mutate_last_action():
    state = make_state()
    state.last_action[:] = np.array([[2.0, -2.0]], dtype=np.float32)
    original = state.last_action.copy()
    builder = ObservationBuilder(field_size=100.0, max_speed=10.0, grid_width=41, grid_res=1.0)

    obs = builder.build(state, 0)

    np.testing.assert_allclose(state.last_action, original)
    np.testing.assert_allclose(obs["vector"][8:10], np.array([1.0, -1.0], dtype=np.float32))


def test_observation_builder_marks_world_border():
    pos = np.array([[1.0, 1.0]], dtype=np.float32)
    vel = np.zeros_like(pos)
    alive = np.array([True])
    target = np.array([50.0, 50.0], dtype=np.float32)
    target_vel = np.zeros(2, dtype=np.float32)
    dists = np.linalg.norm(pos - target, axis=1).astype(np.float32)
    in_goal = np.array([False])
    in_goal_steps = np.array([0], dtype=np.int32)
    finished = np.array([False])
    newly_finished = np.array([False])
    risk_p = np.array([0.0], dtype=np.float32)
    min_neighbor = np.array([5.0], dtype=np.float32)
    last_action = np.zeros((1, 2), dtype=np.float32)
    walls = np.zeros((1, 4), dtype=np.float32)
    threats = []
    measured_accel = np.zeros_like(vel, dtype=np.float32)
    energy = np.array([100.0], dtype=np.float32)
    energy_level = np.array([1.0], dtype=np.float32)
    agent_state = np.array([0], dtype=np.int8)
    state = SimState(
        pos=pos,
        vel=vel,
        alive=alive,
        target_pos=target,
        target_vel=target_vel,
        timestep=0,
        threats=threats,
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
    )
    builder = ObservationBuilder(field_size=100.0, max_speed=10.0, grid_width=41, grid_res=1.0, agent_radius=0.5)
    obs = builder.build(state, 0)
    grid = obs["grid"][0]
    center = builder.grid_width // 2
    res = builder.grid_res
    y_idx, x_idx = np.indices((builder.grid_width, builder.grid_width))
    abs_x = pos[0, 0] + (x_idx - center) * res
    abs_y = pos[0, 1] + (y_idx - center) * res
    inflate = float(builder.agent_radius)
    border_mask = (
        (abs_x <= inflate)
        | (abs_x >= (builder.field_size - inflate))
        | (abs_y <= inflate)
        | (abs_y >= (builder.field_size - inflate))
    )
    assert border_mask.any()
    assert np.allclose(grid[border_mask], builder.wall_value)
    assert np.isclose(grid[center, center], builder.self_value)


def test_observation_builder_batch_matches_single_for_static_and_threat_layers():
    state = make_state()
    state.pos = np.array([[10.0, 10.0], [18.0, 14.0]], dtype=np.float32)
    state.vel = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    state.alive = np.array([True, True])
    state.dists = np.linalg.norm(state.pos - state.target_pos[None, :], axis=1).astype(np.float32)
    state.in_goal = np.array([False, False])
    state.in_goal_steps = np.zeros((2,), dtype=np.int32)
    state.finished = np.array([False, False])
    state.newly_finished = np.array([False, False])
    state.risk_p = np.array([0.0, 0.0], dtype=np.float32)
    state.min_neighbor_dist = np.array([5.0, 5.0], dtype=np.float32)
    state.last_action = np.zeros((2, 2), dtype=np.float32)
    state.walls = np.array([[1.0, 1.0, 1.0, 1.0], [0.8, 0.9, 1.0, 0.7]], dtype=np.float32)
    state.measured_accel = np.zeros((2, 2), dtype=np.float32)
    state.energy = np.array([100.0, 75.0], dtype=np.float32)
    state.energy_level = np.array([1.0, 0.75], dtype=np.float32)
    state.agent_state = np.array([0, 0], dtype=np.int8)
    state.threats = [
        DummyThreat([30.0, 30.0], 5.0, 0.1),
        DummyThreat([16.0, 16.0], 4.0, 0.9),
    ]
    state.static_walls = [(12.0, 12.0, 15.0, 17.0)]
    state.static_circles = [(22.0, 18.0, 3.0)]

    builder = ObservationBuilder(
        field_size=100.0,
        max_speed=10.0,
        grid_width=41,
        grid_res=1.0,
        agent_radius=0.5,
        agent_grid_weight=0.0,
    )
    single = [builder.build(state, i) for i in range(2)]
    batch = builder.build_all(state, [0, 1])

    for s_obs, b_obs in zip(single, batch):
        np.testing.assert_allclose(s_obs["vector"], b_obs["vector"])
        np.testing.assert_allclose(s_obs["grid"], b_obs["grid"])


def test_reward_fn_returns_float():
    prev = make_state()
    cur = make_state()
    reward_fn = RewardFn(RewardConfig(), field_size=100.0, goal_radius=3.0)
    r = reward_fn(prev, cur, 0)
    assert isinstance(r, tuple)
    total, parts = r
    assert isinstance(total, float)
    assert np.isfinite(total)
    assert isinstance(parts, dict)


def test_metrics_fn_keys():
    state = make_state()
    metrics = MetricsFn()(state, 0)
    for key in [
        "dist",
        "alive",
        "in_goal",
        "finished",
        "risk_p",
        "min_neighbor_dist",
        "finished_alive",
        "newly_finished",
        "in_goal_steps",
        "target_vel",
    ]:
        assert key in metrics
