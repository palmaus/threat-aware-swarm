import numpy as np

from env.config import EnvConfig
from env.pz_env import SwarmPZEnv


def _simple_scene():
    return {
        "id": "noise_scene",
        "max_steps": 10,
        "target_pos": [90.0, 90.0],
        "agents_pos": [[50.0, 50.0]],
        "threats": [],
        "walls": [],
        "seed": 0,
    }


def test_obs_noise_grid_changes_values():
    scene = _simple_scene()

    cfg_clean = EnvConfig(n_agents=1, obs_noise_grid=0.0)
    env_clean = SwarmPZEnv(cfg_clean)
    env_clean._set_rng(123)
    obs_clean, _ = env_clean.reset(options={"scene": scene})
    obs0 = obs_clean["drone_0"]

    cfg_noisy = EnvConfig(n_agents=1, obs_noise_grid=0.5)
    env_noisy = SwarmPZEnv(cfg_noisy)
    env_noisy._set_rng(123)
    obs_noisy, _ = env_noisy.reset(options={"scene": scene})
    obs1 = obs_noisy["drone_0"]

    grid0 = obs0["grid"][0]
    grid1 = obs1["grid"][0]

    assert np.abs(grid1 - grid0).sum() > 0.0


def test_domain_randomization_ranges_and_restore():
    cfg = EnvConfig(
        n_agents=1,
        max_speed=3.0,
        domain_randomization=True,
        dr_max_speed_min=1.0,
        dr_max_speed_max=2.0,
        dr_drag_min=0.1,
        dr_drag_max=0.2,
    )
    cfg.physics.drag_coeff = 0.05
    env = SwarmPZEnv(cfg)
    env._set_rng(42)
    env.reset()
    assert 1.0 <= env.config.max_speed <= 2.0
    assert 0.1 <= env.config.physics.drag_coeff <= 0.2

    env.config.domain_randomization = False
    env.config.max_speed = 10.0
    env.config.physics.drag_coeff = 0.5
    env.reset()
    assert env.config.max_speed == 3.0
    assert env.config.physics.drag_coeff == 0.05
