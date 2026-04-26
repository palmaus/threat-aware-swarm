"""Проверка импорта ключевых модулей и базового reset."""

from env.config import EnvConfig
from env.pz_env import SwarmPZEnv


def test_imports_and_env_smoke():
    env = SwarmPZEnv(EnvConfig(), max_steps=10, goal_radius=3.0)
    obs, _infos = env.reset(seed=0)
    assert isinstance(obs, dict)
