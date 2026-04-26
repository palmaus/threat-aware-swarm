from types import SimpleNamespace

import numpy as np

from baselines.flow_field import GlobalFlowFieldPolicy


def test_flow_field_separation_cache_returns_vectors():
    policy = GlobalFlowFieldPolicy()
    state = SimpleNamespace(
        pos=np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        alive=np.array([True, True]),
        agent_radius=0.5,
        decision_step=0,
        timestep=0,
        oracle_dir=None,
        oracle_risk_grad=None,
        field_size=100.0,
    )
    obs = {"vector": np.array([1.0, 0.0], dtype=np.float32), "grid": None}

    out0 = policy.plan("drone_0", obs, state, {"agent_index": 0})
    out1 = policy.plan("drone_1", obs, state, {"agent_index": 1})

    assert out0.shape == (2,)
    assert out1.shape == (2,)


def test_flow_field_cache_not_reused_after_step_counter_reset():
    policy = GlobalFlowFieldPolicy(plan_interval=5)
    obs = {"vector": np.array([1.0, 0.0], dtype=np.float32), "grid": None}
    state = SimpleNamespace(
        pos=np.array([[0.0, 0.0]], dtype=np.float32),
        alive=np.array([True]),
        agent_radius=0.5,
        decision_step=100,
        timestep=100,
        oracle_dir=None,
        oracle_risk_grad=None,
        field_size=100.0,
    )

    first = policy.plan("drone_0", obs, state, {"agent_index": 0})
    state.decision_step = 0
    state.timestep = 0
    obs["vector"] = np.array([-1.0, 0.0], dtype=np.float32)
    second = policy.plan("drone_0", obs, state, {"agent_index": 0})

    assert np.allclose(first, [1.0, 0.0])
    assert np.allclose(second, [-1.0, 0.0])
