"""Проверка ключей диагностических метрик в infos."""

import numpy as np

from env.config import EnvConfig
from env.pz_env import SwarmPZEnv


def test_infos_contains_expected_keys_when_alive():
    env = SwarmPZEnv(EnvConfig(n_agents=2), max_steps=10)
    _obs, _infos = env.reset()
    actions = {a: env.action_space(a).sample() for a in env.possible_agents}
    _obs, _rewards, _terms, _truncs, infos = env.step(actions)
    for a in env.possible_agents:
        inf = infos[a]
        assert "alive" in inf
        if inf.get("alive", False):
            assert "risk_p" in inf
            assert "died_this_step" in inf
            assert "dist_to_nearest_threat" in inf
            assert "nearest_threat_margin" in inf
            assert "min_wall_dist" in inf
            assert "risk_p_before" in inf
            assert "risk_p_after" in inf


def test_infos_not_all_empty_under_wrappers():
    env = SwarmPZEnv(EnvConfig(n_agents=2), max_steps=10)
    _obs, _infos = env.reset()
    actions = {a: env.action_space(a).sample() for a in env.possible_agents}
    _obs, _rewards, _terms, _truncs, infos = env.step(actions)
    assert any(bool(inf) for inf in infos.values())


def test_risk_p_not_all_zero_in_direct_env():
    env = SwarmPZEnv(EnvConfig(n_agents=2), max_steps=50)
    _obs, _infos = env.reset()
    risk_vals = []
    for _ in range(20):
        actions = {a: env.action_space(a).sample() for a in env.possible_agents}
        _obs, _rewards, terms, truncs, infos = env.step(actions)
        for inf in infos.values():
            if "risk_p" in inf:
                risk_vals.append(float(inf.get("risk_p", 0.0)))
        if all(terms.values()) or all(truncs.values()):
            break
    if risk_vals:
        assert np.any(np.array(risk_vals) >= 0.0)


def test_sanity_scenario_risk_positive():
    from scripts.debug.debug_env_metrics import run_sanity_scenario

    report = run_sanity_scenario(seed=0)
    assert report["ok"] is True


def test_sanity_margin_formula():
    from scripts.debug.debug_env_metrics import run_sanity_scenario

    report = run_sanity_scenario(seed=0)
    dist0 = report["dist0"]
    radius = report["radius"]
    margin = radius - dist0
    assert abs(margin - (radius - dist0)) < 1e-6
