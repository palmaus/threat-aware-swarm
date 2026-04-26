"""Проверка контрактов reset/step и форматов наблюдений."""

import numpy as np


def test_reset_returns_obs_and_info(env):
    obs, infos = env.reset(seed=0)
    assert set(obs.keys()) == set(env.possible_agents)
    assert set(infos.keys()) == set(env.possible_agents)
    for o in obs.values():
        assert isinstance(o, dict)
        assert o["vector"].dtype == np.float32
        assert o["grid"].dtype == np.float32
        assert o["vector"].shape == (13,)
        assert o["grid"].shape == (1, 41, 41)


def test_step_returns_dicts(env):
    _obs, _infos = env.reset(seed=0)
    actions = {agent: np.zeros((2,), dtype=np.float32) for agent in env.possible_agents}
    out = env.step(actions)
    assert isinstance(out, tuple)
    assert len(out) == 5
    obs2, rewards, terminations, truncations, infos2 = out
    for d in (obs2, rewards, terminations, truncations, infos2):
        assert set(d.keys()) == set(env.possible_agents)
    for o in obs2.values():
        assert isinstance(o, dict)
        assert o["vector"].shape == (13,)
        assert o["grid"].shape == (1, 41, 41)


def test_deterministic_reset(env):
    obs1, _ = env.reset(seed=0)
    obs2, _ = env.reset(seed=0)
    assert obs1["drone_0"]["vector"].shape == obs2["drone_0"]["vector"].shape == (13,)
    assert obs1["drone_0"]["grid"].shape == obs2["drone_0"]["grid"].shape == (1, 41, 41)
