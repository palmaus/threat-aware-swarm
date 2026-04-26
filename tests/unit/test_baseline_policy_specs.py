from __future__ import annotations

import subprocess
import sys
import textwrap

from baselines.factory import create_baseline_policy
from baselines.policies import canonical_policy_name, default_registry, get_policy_spec
from common.oracle_visibility import oracle_visible_for_policy
from env.config import EnvConfig


def test_policy_spec_alias_resolution_and_registry_metadata():
    reg = default_registry()
    policy = reg.create("astar")

    assert canonical_policy_name("astar") == "baseline:astar_grid"
    assert getattr(policy, "policy_name", None) == "baseline:astar_grid"
    assert getattr(policy, "policy_spec", None) is not None
    assert policy.policy_spec.info_regime == "fair"
    assert policy.policy_spec.oracle_capability == "optional"
    assert policy.policy_spec.map_aware is True


def test_oracle_visibility_respects_policy_capability():
    cfg = EnvConfig()
    cfg.oracle_visibility = "baseline"

    greedy_spec = get_policy_spec("greedy")
    flow_spec = get_policy_spec("flow")
    assert greedy_spec is not None
    assert flow_spec is not None

    assert not oracle_visible_for_policy(cfg, policy_name="greedy")
    assert oracle_visible_for_policy(cfg, policy_name="flow")


def test_oracle_visibility_falls_back_to_agent_for_unknown_policy():
    cfg = EnvConfig()
    cfg.oracle_visibility = "baseline"
    cfg.oracle_visible_to_agents = False

    assert not oracle_visible_for_policy(cfg, policy_name="ppo")


def test_oracle_visibility_agent_mode_stays_hidden_from_baselines():
    cfg = EnvConfig()
    cfg.oracle_visibility = "agent"

    assert not oracle_visible_for_policy(cfg, policy_name="baseline:astar_grid")


def test_oracle_visibility_policy_specs_are_available_before_baselines_import():
    code = textwrap.dedent(
        """
        import sys
        from common.oracle_visibility import oracle_visible_for_policy

        class Config:
            oracle_visibility = "baseline"
            oracle_visible_to_baselines = True
            oracle_visible_to_agents = False

        assert "baselines.policies" not in sys.modules
        assert oracle_visible_for_policy(Config(), policy_name="baseline:astar_grid")
        assert not oracle_visible_for_policy(Config(), policy_name="baseline:greedy")
        assert "baselines.policies" not in sys.modules
        """
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_baseline_factory_derives_runtime_defaults_from_env():
    class DummyEnv:
        config = EnvConfig(n_agents=7, grid_res=2.5)
        n_agents = 7

    astar = create_baseline_policy("astar", env=DummyEnv())
    assert astar.cell_size == 2.5
    assert astar.global_plan_cell_size == 5.0

    fields = create_baseline_policy("potential_fields", env=DummyEnv())
    assert fields._pf.n_agents == 7

    mpc = create_baseline_policy("baseline:mpc_lite", env=DummyEnv(), ui_safe=True)
    assert mpc.use_numba is False
