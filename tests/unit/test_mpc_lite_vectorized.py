import numpy as np
import pytest

import baselines.mpc_lite as mpc_lite
from baselines.mpc_lite import MPCLitePolicy
from env.state import SimState


class DummyThreat:
    def __init__(self, position, radius, intensity=1.0):
        self.position = np.asarray(position, dtype=np.float32)
        self.radius = float(radius)
        self.intensity = float(intensity)


class DummyConfig:
    def __init__(self, dt=0.1, agent_radius=0.5, wall_friction=0.0):
        self.dt = float(dt)
        self.agent_radius = float(agent_radius)
        self.wall_friction = float(wall_friction)


class DummySim:
    def __init__(self, pos, active, threats=None, walls=None):
        self.agents_pos = np.asarray(pos, dtype=np.float32)
        self.agents_active = np.asarray(active, dtype=bool)
        self.threats = threats or []
        self.walls = walls or []
        self.config = DummyConfig()


def _make_state(sim: DummySim, target: np.ndarray) -> SimState:
    pos = np.asarray(sim.agents_pos, dtype=np.float32)
    vel = np.zeros_like(pos, dtype=np.float32)
    alive = np.asarray(sim.agents_active, dtype=bool)
    target = np.asarray(target, dtype=np.float32)
    n = pos.shape[0]
    dists = np.linalg.norm(pos - target[None, :], axis=1).astype(np.float32)
    in_goal = np.zeros((n,), dtype=bool)
    in_goal_steps = np.zeros((n,), dtype=np.int32)
    finished = np.zeros((n,), dtype=bool)
    newly_finished = np.zeros((n,), dtype=bool)
    risk_p = np.zeros((n,), dtype=np.float32)
    min_neighbor = np.full((n,), np.inf, dtype=np.float32)
    last_action = np.zeros((n, 2), dtype=np.float32)
    walls = np.zeros((n, 4), dtype=np.float32)
    measured_accel = np.zeros_like(pos, dtype=np.float32)
    energy = np.full((n,), 100.0, dtype=np.float32)
    energy_level = np.ones((n,), dtype=np.float32)
    agent_state = np.zeros((n,), dtype=np.int8)
    return SimState(
        pos=pos,
        vel=vel,
        alive=alive,
        target_pos=target,
        target_vel=np.zeros((2,), dtype=np.float32),
        timestep=0,
        threats=sim.threats,
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
        static_walls=list(sim.walls),
        static_circles=[],
        collision_speed=np.zeros((n,), dtype=np.float32),
        measured_accel=measured_accel,
        energy=energy,
        energy_level=energy_level,
        agent_state=agent_state,
        field_size=100.0,
        control_mode="waypoint",
        max_speed=5.0,
        max_accel=5.0,
        max_thrust=5.0,
        mass=1.0,
        drag_coeff=0.0,
        dt=sim.config.dt,
        drag=0.0,
        grid_res=1.0,
        agent_radius=sim.config.agent_radius,
        wall_friction=sim.config.wall_friction,
    )


def test_rollout_scores_matches_scalar():
    sim = DummySim(
        pos=[[10.0, 10.0], [20.0, 20.0]],
        active=[True, True],
        threats=[DummyThreat([15.0, 10.0], 3.0, 0.8)],
    )
    policy = MPCLitePolicy(horizon=3, n_directions=8, fallback_astar=False)
    target = np.array([30.0, 10.0], dtype=np.float32)
    state = _make_state(sim, target)
    policy.set_context(state)

    act = np.array([1.0, 0.0], dtype=np.float32)
    scalar_score, scalar_safe = policy._rollout_metrics(0, act, 3, target)
    vec_scores, vec_safe = policy._rollout_scores(
        0,
        act[None, :],
        3,
        target,
    )
    assert np.isclose(vec_scores[0], scalar_score, rtol=1e-4, atol=1e-4)
    assert bool(vec_safe[0]) == bool(scalar_safe)


def test_cem_action_bounds():
    sim = DummySim(
        pos=[[5.0, 5.0], [10.0, 10.0]],
        active=[True, True],
        threats=[],
    )
    policy = MPCLitePolicy(
        horizon=2,
        fallback_astar=False,
        cem_enabled=True,
        cem_iters=1,
        cem_samples=8,
        cem_elite=2,
        cem_seed=42,
    )
    target = np.array([20.0, 5.0], dtype=np.float32)
    state = _make_state(sim, target)
    policy.set_context(state)

    obs = {"vector": np.zeros((13,), dtype=np.float32), "grid": None}
    info = {"agent_index": 0, "pos": sim.agents_pos[0], "target_pos": target}
    act = policy.get_action("drone_0", obs, state, info)
    assert act.shape == (2,)
    assert float(np.linalg.norm(act)) <= 1.0 + 1e-6


def test_step_batch_uses_native_batch_path(monkeypatch):
    sim = DummySim(
        pos=[[5.0, 5.0], [10.0, 10.0]],
        active=[True, True],
        threats=[],
    )
    policy = MPCLitePolicy(horizon=2, fallback_astar=False, use_numba=False)
    state = _make_state(sim, np.array([20.0, 5.0], dtype=np.float32))
    obs_map = {
        "drone_0": {"vector": np.zeros((13,), dtype=np.float32), "grid": None},
        "drone_1": {"vector": np.zeros((13,), dtype=np.float32), "grid": None},
    }
    infos = {
        "drone_0": {"agent_index": 0, "pos": sim.agents_pos[0], "target_pos": state.target_pos},
        "drone_1": {"agent_index": 1, "pos": sim.agents_pos[1], "target_pos": state.target_pos},
    }
    native = {
        "drone_0": np.array([0.1, 0.2], dtype=np.float32),
        "drone_1": np.array([0.3, 0.4], dtype=np.float32),
    }

    def fake_get_actions(obs_dict, state_obj, infos_dict=None):
        assert obs_dict is obs_map
        assert state_obj is state
        assert infos_dict is infos
        return native

    def fail_plan(*args, **kwargs):
        raise AssertionError("native batch path should bypass per-agent plan")

    monkeypatch.setattr(policy, "get_actions", fake_get_actions)
    monkeypatch.setattr(policy, "plan", fail_plan)

    out = policy.step_batch(obs_map, state, infos)

    assert out is native


def test_velocity_streaming_scores_match_hist_path():
    sim = DummySim(
        pos=[[10.0, 10.0], [15.0, 12.0]],
        active=[True, True],
        threats=[DummyThreat([18.0, 10.0], 2.5, 0.7)],
    )
    policy = MPCLitePolicy(horizon=3, n_directions=8, fallback_astar=False, use_numba=False)
    target = np.array([30.0, 10.0], dtype=np.float32)
    state = _make_state(sim, target)
    policy.set_context(state)
    ctx = policy._prepare_rollout_context(0, target, 3)
    assert ctx is not None
    assert not ctx.has_walls
    actions = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 0.0]], dtype=np.float32)

    hist_scores, hist_safe = policy._rollout_scores_velocity(0, actions, 3, target, ctx=ctx)
    stream_scores, stream_safe, pos_end, vel_end = policy._rollout_scores_velocity_streaming_core(actions, ctx, target)

    assert np.allclose(stream_scores, hist_scores, rtol=1e-4, atol=1e-4)
    assert np.array_equal(stream_safe, hist_safe)
    assert pos_end.shape == actions.shape
    assert vel_end.shape == actions.shape


def test_wall_numba_scores_match_python_path():
    if not mpc_lite._NUMBA_AVAILABLE:
        pytest.skip("numba is not available")
    sim = DummySim(
        pos=[[10.0, 10.0], [30.0, 30.0]],
        active=[True, True],
        threats=[DummyThreat([18.0, 10.0], 2.5, 0.7)],
        walls=[(14.0, 6.0, 15.0, 16.0)],
    )
    target = np.array([30.0, 10.0], dtype=np.float32)
    state = _make_state(sim, target)
    actions = np.array([[1.0, 0.0], [0.5, 0.0], [0.0, 0.0]], dtype=np.float32)
    py_policy = MPCLitePolicy(horizon=3, n_directions=8, fallback_astar=False, use_numba=False)
    nb_policy = MPCLitePolicy(horizon=3, n_directions=8, fallback_astar=False, use_numba=True)
    py_policy.set_context(state)
    nb_policy.set_context(state)
    py_ctx = py_policy._prepare_rollout_context(0, target, 3)
    nb_ctx = nb_policy._prepare_rollout_context(0, target, 3)

    py_scores, py_safe = py_policy._rollout_scores_with_context(0, py_ctx, actions, target)
    nb_scores, nb_safe = nb_policy._rollout_scores_with_context(0, nb_ctx, actions, target)

    assert mpc_lite._NUMBA_WARMED
    np.testing.assert_allclose(nb_scores, py_scores, rtol=1e-4, atol=1e-4)
    np.testing.assert_array_equal(nb_safe, py_safe)
