from __future__ import annotations

import numpy as np
import pytest

from env.config import EnvConfig
from env.pz_env import SwarmPZEnv


def test_scene_transaction_restores_runtime_state_on_failure():
    env = SwarmPZEnv(EnvConfig(n_agents=2), max_steps=9)
    env.reset(seed=3)
    before = env.get_runtime_snapshot(include_oracle=False)

    with pytest.raises(RuntimeError, match="boom"):
        with env.scene.transaction():
            env.max_steps = 17
            env.target_pos = np.array([12.0, 18.0], dtype=np.float32)
            env.set_walls([{"x1": 10.0, "y1": 10.0, "x2": 20.0, "y2": 20.0}])
            env.set_static_circles([(np.array([30.0, 35.0], dtype=np.float32), 4.0)])
            env.clear_threats()
            env.add_static_threat(np.array([25.0, 25.0], dtype=np.float32), radius=5.0, intensity=0.3)
            env.set_agent_positions(np.array([[11.0, 11.0], [22.0, 22.0]], dtype=np.float32))
            raise RuntimeError("boom")

    after = env.get_runtime_snapshot(include_oracle=False)
    assert env.max_steps == 9
    np.testing.assert_allclose(after["target_pos"], before["target_pos"])
    np.testing.assert_allclose(after["agents_pos"], before["agents_pos"])
    assert after["walls"] == before["walls"]
    assert after["static_circles"] == before["static_circles"]
    assert len(after["threats"]) == len(before["threats"])
