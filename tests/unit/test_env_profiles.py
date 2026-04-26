"""Проверка профилей среды (lite/full)."""

import numpy as np

from env.config import EnvConfig
from env.engine import SwarmEngine


def test_env_profile_lite_disables_wind_and_energy():
    cfg = EnvConfig()
    cfg.profile = "lite"
    cfg.wind.enabled = True
    cfg.battery.capacity = 10.0
    engine = SwarmEngine(cfg)
    engine.reset(seed=0)
    assert cfg.wind.enabled is True
    assert cfg.battery.capacity == 10.0
    assert engine.config.wind.enabled is False
    assert engine.config.battery.capacity == 0.0
    assert getattr(engine, "_wind_field", None) is None
    assert getattr(engine.sim, "_wind_field", None) is None
    assert np.allclose(engine.sim.energy, 0.0)


def test_env_profile_full_restores_base():
    cfg = EnvConfig()
    cfg.profile = "full"
    cfg.wind.enabled = True
    cfg.wind.ou_sigma = 0.42
    cfg.battery.capacity = 12.0
    engine = SwarmEngine(cfg)
    cfg.wind.enabled = False
    cfg.wind.ou_sigma = 0.01
    cfg.battery.capacity = 0.0
    engine.reset(seed=0)
    assert cfg.wind.enabled is False
    assert cfg.wind.ou_sigma == 0.01
    assert cfg.battery.capacity == 0.0
    assert engine.config.wind.enabled is True
    assert engine.config.wind.ou_sigma == 0.42
    assert engine.config.battery.capacity == 12.0
