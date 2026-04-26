"""Policy construction helpers for baseline tuning."""

from __future__ import annotations

from baselines.factory import create_baseline_policy
from env.pz_env import SwarmPZEnv


def create_tuning_policy(policy_name: str, params: dict, env: SwarmPZEnv, seed: int):
    """Builds a policy with the historical tuning-time defaults preserved."""

    return create_baseline_policy(policy_name, env=env, seed=seed, params=params)
