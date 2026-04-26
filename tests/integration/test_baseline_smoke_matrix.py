from __future__ import annotations

from types import SimpleNamespace

import pytest

from baselines.policies import registered_policy_names
from env.config import EnvConfig
from env.pz_env import SwarmPZEnv
from scripts.bench.benchmark_baselines import run_episode
from scripts.eval.eval_scenarios import _build_policy


@pytest.mark.integration
@pytest.mark.parametrize("policy_name", registered_policy_names())
def test_registered_baselines_short_episode_smoke(policy_name: str):
    cfg = EnvConfig()
    cfg.n_agents = 4
    env = SwarmPZEnv(cfg, max_steps=12, goal_radius=3.0, oracle_enabled=True)
    policy, _ = _build_policy(cfg, policy_name, SimpleNamespace(model=None))

    if hasattr(policy, "reset"):
        policy.reset(0)
    ep_len, ep_ret, metrics, _ = run_episode(
        env,
        policy,
        max_steps=8,
        seed=0,
        update_context=True,
        collect_infos=False,
        success_threshold=0.3,
    )

    assert ep_len >= 0
    assert ep_ret == ep_ret
    assert "finished_frac_end" in metrics
