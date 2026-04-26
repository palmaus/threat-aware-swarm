"""Smoke‑проверка headless‑роллаута."""

from types import SimpleNamespace

import numpy as np
from gymnasium import spaces

from baselines.policies import RandomPolicy
from env.config import EnvConfig
from env.pz_env import SwarmPZEnv
from scripts.debug.headless_rollout import _run_episode


def test_headless_rollout_smoke():
    cfg = EnvConfig(
        n_agents=1,
        wall_count=0,
        random_threat_mode="none",
        random_threat_count_min=0,
        random_threat_count_max=0,
    )
    cfg.profile = "lite"
    env = SwarmPZEnv(cfg, max_steps=3, goal_radius=3.0, oracle_enabled=False)
    policy = RandomPolicy(seed=0)
    scene = {
        "id": "unit",
        "max_steps": 3,
        "target_pos": [30.0, 30.0],
        "agents_pos": [[10.0, 10.0]],
        "threats": [],
        "walls": [],
    }
    out = _run_episode(env, policy, scene, seed=0, ppo_model=None, deterministic=True, record_every=1)
    assert "summary" in out
    assert out["steps"] >= 1


def test_headless_rollout_resets_stateful_policy_each_episode():
    class StatefulPolicy:
        def __init__(self):
            self.reset_calls = []

        def reset(self, seed):
            self.reset_calls.append(seed)

        def step(self, agent_id, obs, context, info):
            return np.zeros(2, dtype=np.float32)

    cfg = EnvConfig(
        n_agents=1,
        wall_count=0,
        random_threat_mode="none",
        random_threat_count_min=0,
        random_threat_count_max=0,
    )
    cfg.profile = "lite"
    env = SwarmPZEnv(cfg, max_steps=2, goal_radius=3.0, oracle_enabled=False)
    policy = StatefulPolicy()
    scene = {
        "id": "unit",
        "max_steps": 2,
        "target_pos": [30.0, 30.0],
        "agents_pos": [[10.0, 10.0]],
        "threats": [],
        "walls": [],
    }

    _run_episode(env, policy, scene, seed=1, ppo_model=None, deterministic=True, record_every=1)
    _run_episode(env, policy, scene, seed=2, ppo_model=None, deterministic=True, record_every=1)

    assert policy.reset_calls == [1, 2]


def test_headless_rollout_passes_recurrent_state_to_recurrent_ppo():
    class FakeEnv:
        possible_agents = ["drone_0"]
        max_steps = 1

        def reset(self, seed=None, options=None):
            return {
                "drone_0": {"vec": np.zeros(2, dtype=np.float32)},
            }, {"drone_0": {"alive": 1.0, "finished": 0.0}}

        def step(self, actions):
            return (
                {"drone_0": {"vec": np.zeros(2, dtype=np.float32)}},
                {"drone_0": 0.0},
                {"drone_0": True},
                {"drone_0": False},
                {"drone_0": {"risk_p": 0.0, "alive": 1.0, "finished": 0.0}},
            )

    class FakeRecurrentPPO:
        def __init__(self):
            self.policy = SimpleNamespace(lstm_actor=object())
            self.observation_space = spaces.Dict({"vec": spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)})
            self.calls = []

        def predict(self, obs, **kwargs):
            self.calls.append(kwargs)
            return np.zeros(2, dtype=np.float32), "next-state"

    model = FakeRecurrentPPO()
    _run_episode(
        FakeEnv(),
        policy=None,
        scene={"id": "unit", "max_steps": 1},
        seed=0,
        ppo_model=model,
        deterministic=True,
        record_every=1,
    )

    assert len(model.calls) == 1
    assert model.calls[0]["state"] is None
    np.testing.assert_array_equal(model.calls[0]["episode_start"], np.asarray([True], dtype=bool))
