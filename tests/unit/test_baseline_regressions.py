from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np

from baselines.astar_grid import AStarGridPolicy
from baselines.flow_field import GlobalFlowFieldPolicy
from baselines.mpc_lite import MPCLitePolicy
from baselines.policies import PotentialFieldsPolicy
from baselines.potential_fields import PotentialFieldPolicy


def test_potential_fields_wrapper_reset_clears_inner_agent_state():
    policy = PotentialFieldsPolicy(n_agents=2)
    policy._pf._agent_state[0] = {"best_dist": 1.0, "since_improve": 5}

    policy.reset(seed=123)

    assert policy._pf._agent_state == {}


def test_potential_field_policy_uses_runtime_batch_size():
    policy = PotentialFieldPolicy(n_agents=4, stuck_steps=1)

    actions = policy.get_actions(
        agents_pos=np.zeros((2, 2), dtype=np.float32),
        active_mask=np.array([True, True]),
        target_pos=np.array([1.0, 1.0], dtype=np.float32),
        threats=[],
        agents_vel=np.zeros((2, 2), dtype=np.float32),
        max_speed=1.0,
        max_accel=1.0,
        dt=0.1,
        drag=0.0,
    )

    assert actions.shape == (2, 2)


def test_flow_field_zero_norm_separation_is_warning_free():
    policy = GlobalFlowFieldPolicy()
    state = SimpleNamespace(
        pos=np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32),
        alive=np.array([True, True]),
        agent_radius=0.5,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = policy._compute_separation_vectors(state)

    assert out.shape == (2, 2)
    assert not any(isinstance(item.message, RuntimeWarning) for item in caught)


def test_astar_soft_failure_counter_tracks_memory_fallback():
    policy = AStarGridPolicy(numba_warmup=False)
    state = SimpleNamespace(
        field_size=100.0,
        max_speed=5.0,
        dt=0.1,
        pos=np.array([[0.0, 0.0]], dtype=np.float32),
        vel=np.zeros((1, 2), dtype=np.float32),
        alive=np.array([True]),
        target_pos=np.array([5.0, 0.0], dtype=np.float32),
        target_vel=np.zeros(2, dtype=np.float32),
        oracle_dir=None,
        threats=[],
        static_walls=[],
        risk_p=np.array([0.0], dtype=np.float32),
        dists=np.array([5.0], dtype=np.float32),
        in_goal=np.array([False]),
        agent_radius=0.5,
        wall_friction=0.0,
        grid_res=1.0,
    )
    obs = {"vector": np.zeros((13,), dtype=np.float32), "grid": np.zeros((5, 5), dtype=np.float32)}
    obs["vector"][:2] = [5.0, 0.0]
    policy.set_context(state)

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    policy._memory_touch = boom
    out = policy.plan(
        "drone_0",
        obs,
        state,
        {
            "agent_index": 0,
            "pos": state.pos[0],
            "target_pos": state.target_pos,
            "target_vel": state.target_vel,
            "risk_p": 0.0,
            "in_goal": 0.0,
            "dt": 0.1,
            "drag": 0.0,
            "grid_res": 1.0,
        },
    )

    assert np.asarray(out[0]).shape == (2,)
    assert policy._soft_failures["memory_touch"] == 1


def test_mpc_soft_failure_counter_tracks_target_parse_fallback():
    policy = MPCLitePolicy(fallback_astar=False, use_numba=False)
    state = SimpleNamespace(
        field_size=100.0,
        max_speed=5.0,
        max_accel=2.0,
        dt=0.1,
        drag=0.0,
        pos=np.array([[0.0, 0.0]], dtype=np.float32),
        vel=np.zeros((1, 2), dtype=np.float32),
        alive=np.array([True]),
        target_pos=np.array([5.0, 0.0], dtype=np.float32),
        target_vel=np.zeros(2, dtype=np.float32),
        static_walls=[],
        threats=[],
        agent_radius=0.5,
        wall_friction=0.0,
        grid_res=1.0,
    )
    obs = {"vector": np.zeros((13,), dtype=np.float32), "grid": np.zeros((5, 5), dtype=np.float32)}
    obs["vector"][:2] = [5.0, 0.0]

    out = policy.plan(
        "drone_0",
        obs,
        state,
        {
            "agent_index": 0,
            "pos": state.pos[0],
            "target_pos": object(),
            "target_vel": state.target_vel,
            "dt": 0.1,
            "drag": 0.0,
            "grid_res": 1.0,
        },
    )

    assert np.asarray(out[0]).shape == (2,)
    assert policy._soft_failures["target_pos_info"] == 1


def test_astar_reuse_fast_path_skips_memory_wall_updates(monkeypatch):
    policy = AStarGridPolicy(
        plan_reuse=True,
        memory_walls_enabled=True,
        memory_threats_enabled=False,
        replan_interval=10,
        replan_target_shift=0.0,
        replan_risk_threshold=1.1,
        stuck_steps=0,
        global_plan_enabled=False,
        escape_steps=0,
        los_enabled=False,
    )
    state = SimpleNamespace(
        field_size=100.0,
        max_speed=5.0,
        dt=0.1,
        pos=np.array([[10.0, 10.0]], dtype=np.float32),
        vel=np.zeros((1, 2), dtype=np.float32),
        alive=np.array([True]),
        target_pos=np.array([15.0, 10.0], dtype=np.float32),
        target_vel=np.zeros(2, dtype=np.float32),
        oracle_dir=None,
        threats=[],
        static_walls=[],
        risk_p=np.array([0.0], dtype=np.float32),
        dists=np.array([5.0], dtype=np.float32),
        in_goal=np.array([False]),
        agent_radius=0.5,
        wall_friction=0.0,
        grid_res=1.0,
    )
    obs = {"vector": np.zeros((13,), dtype=np.float32), "grid": np.zeros((5, 5), dtype=np.float32)}
    obs["vector"][:2] = [5.0, 0.0]
    info = {
        "agent_index": 0,
        "pos": state.pos[0],
        "target_pos": state.target_pos,
        "target_vel": state.target_vel,
        "risk_p": 0.0,
        "in_goal": 0.0,
        "dt": 0.1,
        "drag": 0.0,
        "grid_res": 1.0,
    }
    policy.set_context(state)
    agent_state = policy._agent_state.setdefault(0, policy._default_agent_state())
    base_cell = policy._cell_from_pos(state.pos[0])
    agent_state["current_path"] = [base_cell, (base_cell[0] + 1, base_cell[1])]
    agent_state["path_idx"] = 0
    agent_state["plan_step"] = 0
    agent_state["plan_target_pos"] = np.asarray(state.target_pos, dtype=np.float32)

    def fail_memory_update(*args, **kwargs):
        raise AssertionError("reuse fast path should bypass wall memory update")

    monkeypatch.setattr(policy, "_memory_walls_update", fail_memory_update)
    out = policy.plan("drone_0", obs, state, info)
    desired = np.asarray(out[0] if isinstance(out, tuple) else out, dtype=np.float32)

    assert desired.shape == (2,)
    assert np.linalg.norm(desired) > 0.0


def test_astar_reuse_fast_path_skips_wall_mask_for_immediate_waypoint(monkeypatch):
    policy = AStarGridPolicy(
        plan_reuse=True,
        memory_walls_enabled=False,
        memory_threats_enabled=False,
        replan_interval=10,
        replan_target_shift=0.0,
        replan_risk_threshold=1.1,
        stuck_steps=0,
        global_plan_enabled=False,
        escape_steps=0,
        los_enabled=True,
        waypoint_lookahead=1,
    )
    state = SimpleNamespace(
        field_size=100.0,
        max_speed=5.0,
        dt=0.1,
        pos=np.array([[10.0, 10.0]], dtype=np.float32),
        vel=np.zeros((1, 2), dtype=np.float32),
        alive=np.array([True]),
        target_pos=np.array([15.0, 10.0], dtype=np.float32),
        target_vel=np.zeros(2, dtype=np.float32),
        oracle_dir=None,
        threats=[],
        static_walls=[],
        risk_p=np.array([0.0], dtype=np.float32),
        dists=np.array([5.0], dtype=np.float32),
        in_goal=np.array([False]),
        agent_radius=0.5,
        wall_friction=0.0,
        grid_res=1.0,
    )
    obs = {"vector": np.zeros((13,), dtype=np.float32), "grid": np.zeros((5, 5), dtype=np.float32)}
    obs["vector"][:2] = [5.0, 0.0]
    info = {
        "agent_index": 0,
        "pos": state.pos[0],
        "target_pos": state.target_pos,
        "target_vel": state.target_vel,
        "risk_p": 0.0,
        "in_goal": 0.0,
        "dt": 0.1,
        "drag": 0.0,
        "grid_res": 1.0,
    }
    policy.set_context(state)
    agent_state = policy._agent_state.setdefault(0, policy._default_agent_state())
    base_cell = policy._cell_from_pos(state.pos[0])
    agent_state["current_path"] = [base_cell, (base_cell[0] + 1, base_cell[1])]
    agent_state["path_idx"] = 0
    agent_state["plan_step"] = 0
    agent_state["plan_target_pos"] = np.asarray(state.target_pos, dtype=np.float32)

    def fail_wall_mask(*args, **kwargs):
        raise AssertionError("immediate reuse waypoint should not need wall mask")

    monkeypatch.setattr(policy, "_wall_mask", fail_wall_mask)
    out = policy.plan("drone_0", obs, state, info)
    desired = np.asarray(out[0] if isinstance(out, tuple) else out, dtype=np.float32)

    assert desired.shape == (2,)
    assert np.linalg.norm(desired) > 0.0


def test_mpc_low_score_fallback_uses_confirmation_window(monkeypatch):
    policy = MPCLitePolicy(
        fallback_astar=False,
        fallback_score=-0.2,
        fallback_cooldown=0,
        fallback_confirm_steps=2,
        stuck_steps=50,
        use_numba=False,
    )
    state = SimpleNamespace(
        field_size=100.0,
        max_speed=5.0,
        max_accel=2.0,
        dt=0.1,
        drag=0.0,
        pos=np.array([[0.0, 0.0]], dtype=np.float32),
        vel=np.zeros((1, 2), dtype=np.float32),
        alive=np.array([True]),
        target_pos=np.array([5.0, 0.0], dtype=np.float32),
        target_vel=np.zeros(2, dtype=np.float32),
        static_walls=[],
        threats=[],
        agent_radius=0.5,
        wall_friction=0.0,
        grid_res=1.0,
        oracle_dir=None,
    )
    obs = {"vector": np.zeros((13,), dtype=np.float32), "grid": np.zeros((5, 5), dtype=np.float32)}
    obs["vector"][:2] = [5.0, 0.0]
    info = {
        "agent_index": 0,
        "pos": state.pos[0],
        "target_pos": state.target_pos,
        "target_vel": state.target_vel,
        "dt": 0.1,
        "drag": 0.0,
        "grid_res": 1.0,
    }
    policy.set_context(state)
    fallback_calls: list[str] = []

    class DummyFallback:
        def plan(self, agent_id, obs_arg, state_arg, info_arg):
            fallback_calls.append(agent_id)
            return np.array([0.9, 0.0], dtype=np.float32)

        def reset(self, seed=None):
            return None

        def set_context(self, state_arg):
            return None

    monkeypatch.setattr(policy, "_astar_fallback", DummyFallback())
    monkeypatch.setattr(
        policy,
        "_choose_action_discrete",
        lambda *args, **kwargs: (np.array([0.1, 0.0], dtype=np.float32), -1.0),
    )

    out1 = policy.plan("drone_0", obs, state, dict(info))
    out2 = policy.plan("drone_0", obs, state, dict(info))
    arr1 = np.asarray(out1[0] if isinstance(out1, tuple) else out1, dtype=np.float32)
    arr2 = np.asarray(out2[0] if isinstance(out2, tuple) else out2, dtype=np.float32)

    assert np.allclose(arr1, np.array([0.1, 0.0], dtype=np.float32))
    assert np.allclose(arr2, np.array([0.9, 0.0], dtype=np.float32))
    assert fallback_calls == ["drone_0"]
    assert policy._fallback_stats["low_score_deferred"] == 1
    assert policy._fallback_stats["used_low_score"] == 1
