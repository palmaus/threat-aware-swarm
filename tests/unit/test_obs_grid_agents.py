"""Проверка наблюдений: сетка не должна включать соседей."""

import numpy as np
import pytest

from env.config import EnvConfig
from env.pz_env import SwarmPZEnv


def test_obs_grid_excludes_neighbors():
    cfg = EnvConfig(n_agents=2)
    env = SwarmPZEnv(cfg)

    scene = {
        "id": "test_neighbors",
        "max_steps": 10,
        "target_pos": [90.0, 90.0],
        "agents_pos": [
            [50.0, 50.0],
            [58.0, 50.0],
        ],
        "threats": [],
        "seed": 0,
    }

    obs, _ = env.reset(options={"scene": scene})
    obs0 = obs["drone_0"]
    obs1 = obs["drone_1"]

    grid0 = obs0["grid"][0]
    grid1 = obs1["grid"][0]
    width = grid0.shape[0]

    center = width // 2

    assert float(grid0[center, center]) == pytest.approx(0.1, abs=1e-6)
    assert float(grid1[center, center]) == pytest.approx(0.1, abs=1e-6)

    assert np.max(grid0) == pytest.approx(0.1, abs=1e-6)
    assert np.max(grid1) == pytest.approx(0.1, abs=1e-6)
