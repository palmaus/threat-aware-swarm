import numpy as np

from env.config import EnvConfig
from env.pz_env import SwarmPZEnv


def _threat_snapshot(threats):
    out = []
    for t in threats:
        data = {
            "kind": str(getattr(t, "kind", "")),
            "pos": np.asarray(getattr(t, "position", np.zeros(2)), dtype=np.float32).tolist(),
            "radius": float(getattr(t, "radius", 0.0)),
            "intensity": float(getattr(t, "intensity", 0.0)),
            "oracle_block": bool(getattr(t, "oracle_block", False)),
        }
        if hasattr(t, "velocity"):
            data["velocity"] = np.asarray(t.velocity, dtype=np.float32).tolist()
        if hasattr(t, "vision_radius"):
            data["vision_radius"] = float(t.vision_radius)
        if hasattr(t, "speed"):
            data["speed"] = float(t.speed)
        out.append(data)
    return out


def _assert_threats_equal(a, b):
    assert len(a) == len(b)
    for ta, tb in zip(a, b, strict=True):
        assert ta["kind"] == tb["kind"]
        assert ta["oracle_block"] == tb["oracle_block"]
        assert np.allclose(ta["pos"], tb["pos"], atol=1e-6)
        assert np.isclose(ta["radius"], tb["radius"], atol=1e-6)
        assert np.isclose(ta["intensity"], tb["intensity"], atol=1e-6)
        if "velocity" in ta or "velocity" in tb:
            assert np.allclose(ta.get("velocity", [0.0, 0.0]), tb.get("velocity", [0.0, 0.0]), atol=1e-6)
        if "vision_radius" in ta or "vision_radius" in tb:
            assert np.isclose(ta.get("vision_radius", 0.0), tb.get("vision_radius", 0.0), atol=1e-6)
        if "speed" in ta or "speed" in tb:
            assert np.isclose(ta.get("speed", 0.0), tb.get("speed", 0.0), atol=1e-6)


def test_env_determinism_same_seed():
    cfg = EnvConfig()
    env_a = SwarmPZEnv(cfg, max_steps=50, goal_radius=3.0, oracle_enabled=False, oracle_async=False)
    env_b = SwarmPZEnv(cfg, max_steps=50, goal_radius=3.0, oracle_enabled=False, oracle_async=False)

    _obs_a, _info_a = env_a.reset(seed=123)
    _obs_b, _info_b = env_b.reset(seed=123)

    assert np.allclose(env_a.sim.agents_pos, env_b.sim.agents_pos, atol=1e-6)
    assert np.allclose(env_a.target_pos, env_b.target_pos, atol=1e-6)
    assert env_a.sim.walls == env_b.sim.walls
    _assert_threats_equal(_threat_snapshot(env_a.sim.threats), _threat_snapshot(env_b.sim.threats))

    zero = np.zeros((2,), dtype=np.float32)
    for _ in range(5):
        actions_a = dict.fromkeys(env_a.possible_agents, zero)
        actions_b = dict.fromkeys(env_b.possible_agents, zero)
        env_a.step(actions_a)
        env_b.step(actions_b)
        assert np.allclose(env_a.sim.agents_pos, env_b.sim.agents_pos, atol=1e-6)
        assert np.allclose(env_a.sim.agents_vel, env_b.sim.agents_vel, atol=1e-6)
        assert np.array_equal(env_a.sim.agents_active, env_b.sim.agents_active)
        _assert_threats_equal(_threat_snapshot(env_a.sim.threats), _threat_snapshot(env_b.sim.threats))
