"""Проверка движения и конфигурации динамических угроз."""

import numpy as np

from env.config import EnvConfig
from env.physics.core import AGENT_DEAD
from env.pz_env import SwarmPZEnv
from env.scenes.threats import threat_from_config


def _make_env(n_agents: int = 1) -> SwarmPZEnv:
    cfg = EnvConfig(n_agents=n_agents)
    return SwarmPZEnv(cfg, max_steps=10)


def test_dynamic_linear_threat_moves():
    env = _make_env()
    scene = {
        "seed": 0,
        "field_size": 100.0,
        "agents_pos": [[10.0, 10.0]],
        "target_pos": [90.0, 90.0],
        "dynamic_threats": [
            {"type": "linear", "pos": [50.0, 50.0], "radius": 5.0, "intensity": 0.1, "vel": [2.0, 0.0]},
        ],
        "max_steps": 10,
    }
    env.reset(options={"scene": scene})
    assert len(env.sim.dynamic_threats) == 1
    t = env.sim.dynamic_threats[0]
    p0 = t.position.copy()
    actions = {agent: np.zeros((2,), dtype=np.float32) for agent in env.possible_agents}
    env.step(actions)
    p1 = t.position.copy()
    assert np.linalg.norm(p1 - p0) > 0.0


def test_dynamic_threat_oracle_block_config_preserved():
    threat = threat_from_config(
        {
            "type": "linear",
            "pos": [50.0, 50.0],
            "radius": 5.0,
            "intensity": 0.1,
            "vel": [1.0, 0.0],
            "oracle_block": True,
        }
    )

    assert threat.oracle_block is True
    assert threat.copy().oracle_block is True


def test_apply_scene_does_not_mutate_agents_pos():
    env = _make_env(n_agents=3)
    scene = {
        "seed": 0,
        "field_size": 100.0,
        "agents_pos": [[10.0, 10.0]],
        "target_pos": [90.0, 90.0],
        "max_steps": 10,
    }

    env.reset(options={"scene": scene})

    assert scene["agents_pos"] == [[10.0, 10.0]]


def test_oracle_accounts_for_static_threat_block():
    env = _make_env()
    start = np.array([10.0, 10.0], dtype=np.float32)
    goal = np.array([90.0, 90.0], dtype=np.float32)
    scene = {
        "seed": 0,
        "field_size": 100.0,
        "agents_pos": [start.tolist()],
        "target_pos": goal.tolist(),
        "threats": [
            {
                "type": "static",
                "pos": [50.0, 50.0],
                "radius": 15.0,
                "intensity": 0.1,
                "oracle_block": True,
            },
        ],
        "max_steps": 10,
    }
    env.reset(options={"scene": scene})
    opt = float(env._optimal_len[0])
    direct = float(np.linalg.norm(goal - start))
    assert np.isfinite(opt)
    assert opt > direct * 1.02


def test_infos_include_threat_metrics():
    env = _make_env(n_agents=2)
    _obs, _infos = env.reset(seed=0)
    actions = {agent: np.zeros((2,), dtype=np.float32) for agent in env.possible_agents}
    _obs, _rewards, _terms, _truncs, infos = env.step(actions)
    for inf in infos.values():
        assert "threat_collided" in inf
        assert "threat_collisions" in inf
        assert "min_dist_to_threat" in inf
        assert "survival_time" in inf


def test_chaser_switches_target_on_death():
    env = _make_env(n_agents=2)
    env.config.profile = "lite"
    scene = {
        "seed": 0,
        "field_size": 100.0,
        "agents_pos": [[10.0, 10.0], [90.0, 90.0]],
        "target_pos": [50.0, 50.0],
        "dynamic_threats": [
            {
                "type": "chaser",
                "pos": [50.0, 50.0],
                "radius": 3.0,
                "intensity": 0.0,
                "speed": 2.0,
                "vision_radius": 200.0,
            },
        ],
        "max_steps": 5,
    }
    env.reset(options={"scene": scene})
    t = env.sim.dynamic_threats[0]
    actions = {agent: np.zeros((2,), dtype=np.float32) for agent in env.possible_agents}
    env.step(actions)
    v1 = t.velocity.copy()
    assert v1[0] < 0.0 and v1[1] < 0.0
    env.sim.agent_state[0] = AGENT_DEAD
    env.sim.agents_active[0] = False
    env.step(actions)
    v2 = t.velocity.copy()
    assert v2[0] > 0.0 and v2[1] > 0.0


def test_dynamic_threat_radius_changes():
    env = _make_env()
    scene = {
        "seed": 0,
        "field_size": 100.0,
        "agents_pos": [[10.0, 10.0]],
        "target_pos": [90.0, 90.0],
        "dynamic_threats": [
            {
                "type": "linear",
                "pos": [50.0, 50.0],
                "radius": 3.0,
                "intensity": 0.1,
                "vel": [0.0, 0.0],
                "radius_range": [2.0, 4.0],
                "radius_speed": 1.0,
                "radius_phase": 0.0,
                "radius_mode": "sine",
            },
        ],
        "max_steps": 5,
    }
    env.reset(options={"scene": scene})
    t = env.sim.dynamic_threats[0]
    r0 = float(t.radius)
    actions = {agent: np.zeros((2,), dtype=np.float32) for agent in env.possible_agents}
    env.step(actions)
    r1 = float(t.radius)
    assert 2.0 <= r1 <= 4.0
    assert abs(r1 - r0) > 1e-6


def test_moving_goal_updates_position():
    env = _make_env()
    scene = {
        "seed": 0,
        "field_size": 100.0,
        "agents_pos": [[10.0, 10.0]],
        "target_pos": [60.0, 60.0],
        "target_motion": {
            "type": "circle",
            "center": [60.0, 60.0],
            "radius": 10.0,
            "angular_speed": 1.0,
            "phase": 0.0,
        },
        "max_steps": 5,
    }
    env.reset(options={"scene": scene})
    p0 = env.target_pos.copy()
    actions = {agent: np.zeros((2,), dtype=np.float32) for agent in env.possible_agents}
    env.step(actions)
    p1 = env.target_pos.copy()
    assert np.linalg.norm(p1 - p0) > 0.0
