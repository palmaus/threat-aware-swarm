from __future__ import annotations

import numpy as np

from env.config import EnvConfig
from env.pz_env import SwarmPZEnv


def test_train_fast_runtime_skips_heavy_oracle_payload():
    cfg = EnvConfig(n_agents=1)
    cfg.runtime_mode = "train_fast"
    env = SwarmPZEnv(cfg, max_steps=5)
    env.reset(seed=0)
    state = env.get_state()
    assert state.oracle_dist is None
    assert state.oracle_risk is None
    assert state.oracle_risk_grad is None


def test_train_fast_runtime_skips_reset_contract_validation(monkeypatch):
    cfg = EnvConfig(n_agents=1)
    cfg.runtime_mode = "train_fast"
    env = SwarmPZEnv(cfg, max_steps=5)

    def _fail(*args, **kwargs):
        raise AssertionError("reset contract validation should be skipped in train_fast mode")

    monkeypatch.setattr("env.pz_env.maybe_validate_reset", _fail)
    env.reset(seed=0)


def test_full_runtime_keeps_oracle_payload():
    cfg = EnvConfig(n_agents=1)
    cfg.runtime_mode = "full"
    env = SwarmPZEnv(cfg, max_steps=5)
    env.reset(seed=0)
    state = env.get_state()
    assert state.oracle_dir is None or isinstance(state.oracle_dir, np.ndarray)
    assert state.oracle_dist is not None


def test_oracle_dir_is_not_materialized_when_visibility_is_none():
    cfg = EnvConfig(n_agents=1)
    cfg.runtime_mode = "full"
    cfg.oracle_visibility = "none"
    cfg.oracle_visible_to_agents = False
    cfg.oracle_visible_to_baselines = False
    env = SwarmPZEnv(cfg, max_steps=5)
    env.reset(seed=0)
    assert env.get_state().oracle_dir is None


def test_oracle_dir_materializes_for_baseline_visibility():
    cfg = EnvConfig(n_agents=1)
    cfg.runtime_mode = "full"
    cfg.oracle_visibility = "baseline"
    env = SwarmPZEnv(cfg, max_steps=5)
    env.reset(seed=0)
    oracle_dir = env.get_state().oracle_dir
    assert oracle_dir is not None
    assert oracle_dir.shape == (1, 2)


def test_train_fast_runtime_emits_minimal_infos():
    cfg = EnvConfig(n_agents=1)
    cfg.runtime_mode = "train_fast"
    env = SwarmPZEnv(cfg, max_steps=5)
    _obs, infos = env.reset(seed=0)
    reset_info = infos[env.possible_agents[0]]
    assert "target_pos" not in reset_info
    assert "path_len" not in reset_info

    _obs, _rewards, _terms, _truncs, infos = env.step({env.possible_agents[0]: [0.0, 0.0]})
    step_info = infos[env.possible_agents[0]]
    assert "target_pos" not in step_info
    assert "path_len" not in step_info
    assert "rew_total" in step_info
