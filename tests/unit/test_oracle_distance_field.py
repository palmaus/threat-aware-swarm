import numpy as np

from env.config import EnvConfig
from env.oracles.manager import OracleManager


class _DummySim:
    def __init__(self, pos: np.ndarray):
        self.agents_pos = np.asarray(pos, dtype=np.float32)
        self.threats: list = []


class _DummyEnv:
    def __init__(self, config: EnvConfig, pos: np.ndarray, target: np.ndarray):
        self.config = config
        self.oracle_enabled = True
        self.oracle_cell_size = 1.0
        self.target_pos = np.asarray(target, dtype=np.float32)
        self.sim = _DummySim(pos)


def test_oracle_direction_to_goal_uses_dist_grad():
    cfg = EnvConfig(field_size=20.0, n_agents=1, agent_radius=0.5)
    start = np.array([[2.0, 2.0]], dtype=np.float32)
    target = np.array([18.0, 18.0], dtype=np.float32)
    env = _DummyEnv(cfg, start, target)
    oracle = OracleManager(env)
    env.oracle = oracle
    oracle.set_walls([(8.0, 0.0, 12.0, 16.0)])
    oracle.compute_lengths(clear_state=True)

    assert oracle.dist_field is not None
    assert oracle.dist_grad is not None
    vec = oracle.direction_to_goal(np.array([2.0, 2.0], dtype=np.float32))
    assert np.all(np.isfinite(vec))
    assert float(np.linalg.norm(vec)) > 0.1


def test_oracle_path_is_built_lazily(monkeypatch):
    cfg = EnvConfig(field_size=20.0, n_agents=1, agent_radius=0.5)
    start = np.array([[2.0, 2.0]], dtype=np.float32)
    target = np.array([18.0, 18.0], dtype=np.float32)
    env = _DummyEnv(cfg, start, target)
    oracle = OracleManager(env)
    env.oracle = oracle
    oracle.set_walls([(8.0, 0.0, 12.0, 16.0)])
    calls = {"count": 0}

    def _fake_path(*args, **kwargs):
        calls["count"] += 1
        return [(2.0, 2.0), (18.0, 18.0)]

    monkeypatch.setattr("env.oracles.manager.path_from_distance_field", _fake_path)
    oracle.compute_lengths(clear_state=True)
    assert calls["count"] == 0
    first = oracle.path
    second = oracle.path
    assert first == second
    assert calls["count"] == 1


def test_oracle_optimal_len_batch_lookup_matches_agent_count():
    cfg = EnvConfig(field_size=20.0, n_agents=3, agent_radius=0.5)
    start = np.array([[2.0, 2.0], [3.0, 3.0], [4.0, 4.0]], dtype=np.float32)
    target = np.array([18.0, 18.0], dtype=np.float32)
    env = _DummyEnv(cfg, start, target)
    oracle = OracleManager(env)
    env.oracle = oracle
    oracle.compute_lengths(clear_state=True)
    assert oracle.optimal_len.shape == (3,)
    assert np.all(np.isfinite(oracle.optimal_len))
