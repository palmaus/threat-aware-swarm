"""Смоук‑тест базового API среды и набора ключей в info."""

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from env.config import EnvConfig
from env.pz_env import SwarmPZEnv

REQUIRED_INFO_KEYS = {
    "alive",
    "in_goal",
    "finished",
    "finished_alive",
    "in_goal_steps",
    "newly_finished",
    "dist",
    "risk_p",
    "min_neighbor_dist",
}


def test_env_obs_and_infos_smoke():
    cfg = EnvConfig()
    env = SwarmPZEnv(cfg, max_steps=50, goal_radius=3.0)
    observations, infos = env.reset(seed=0)

    assert isinstance(observations, dict)
    assert observations, "reset should return observations for all agents"

    for obs in observations.values():
        assert isinstance(obs, dict)
        assert obs["vector"].shape == (13,)
        assert obs["grid"].shape == (1, 41, 41)

    for _ in range(10):
        actions = {}
        for agent_id in env.possible_agents:
            actions[agent_id] = np.random.uniform(-1.0, 1.0, size=(2,)).astype(np.float32)

        observations, _rewards, terminations, truncations, infos = env.step(actions)

        assert isinstance(infos, dict)
        for agent_id, inf in infos.items():
            assert isinstance(inf, dict)
            missing = REQUIRED_INFO_KEYS.difference(inf.keys())
            assert not missing, f"missing info keys for {agent_id}: {missing}"

        if all(terminations.values()) or all(truncations.values()):
            break


def test_finished_agent_freezes_until_swarm_done():
    cfg = EnvConfig(n_agents=2, random_threat_mode="none", random_threat_count_min=0, random_threat_count_max=0)
    cfg.wind.enabled = False
    cfg.battery.capacity = 0.0
    cfg.physics.drag_coeff = 0.0
    env = SwarmPZEnv(cfg, max_steps=20, goal_radius=3.0, goal_hold_steps=1, oracle_enabled=False)
    env.reset(seed=1)
    env.engine.target_pos[:] = np.array([10.0, 10.0], dtype=np.float32)
    env.engine.sim.agents_pos[:] = np.array([[10.0, 10.0], [0.0, 0.0]], dtype=np.float32)
    env.engine.sim.agents_vel[:] = np.array([[5.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    env.engine.prev_dists[:] = np.linalg.norm(env.engine.sim.agents_pos - env.engine.target_pos[None, :], axis=1)

    env.step({})
    frozen_pos = env.engine.sim.agents_pos[0].copy()
    frozen_path = float(env.engine._path_len[0])
    env.step({})

    np.testing.assert_allclose(env.engine.sim.agents_pos[0], frozen_pos)
    np.testing.assert_allclose(env.engine.sim.agents_vel[0], np.zeros((2,), dtype=np.float32))
    assert float(env.engine._path_len[0]) == frozen_path


def test_step_preserves_newly_finished_from_intermediate_physics_tick():
    cfg = EnvConfig(
        n_agents=2,
        physics_ticks_per_action=3,
        random_threat_mode="none",
        random_threat_count_min=0,
        random_threat_count_max=0,
    )
    cfg.wind.enabled = False
    cfg.battery.capacity = 0.0
    env = SwarmPZEnv(cfg, max_steps=20, goal_radius=3.0, goal_hold_steps=1, oracle_enabled=False)
    env.reset(seed=1)
    env.engine.target_pos[:] = np.array([10.0, 10.0], dtype=np.float32)
    env.engine.sim.agents_pos[:] = np.array([[10.0, 10.0], [0.0, 0.0]], dtype=np.float32)
    env.engine.sim.agents_vel[:] = 0.0
    env.engine.prev_dists[:] = np.linalg.norm(env.engine.sim.agents_pos - env.engine.target_pos[None, :], axis=1)

    _obs, _rewards, _terminations, _truncations, infos = env.step({})

    assert infos["drone_0"]["finished"] == 1.0
    assert infos["drone_0"]["newly_finished"] == 1.0


def test_scene_field_size_is_episode_local_and_observations_sync():
    cfg = EnvConfig(
        n_agents=1,
        random_threat_mode="none",
        random_threat_count_min=0,
        random_threat_count_max=0,
    )
    cfg.wind.enabled = False
    cfg.battery.capacity = 0.0
    env = SwarmPZEnv(cfg, max_steps=7, goal_radius=3.0, oracle_enabled=False)

    big_scene = {
        "id": "big",
        "field_size": 200.0,
        "max_steps": 1,
        "agents_pos": [[150.0, 150.0]],
        "target_pos": [180.0, 180.0],
        "threats": [],
    }
    env.reset(seed=0, options={"scene": big_scene})
    assert env.config.field_size == 200.0
    assert env.max_steps == 1
    np.testing.assert_allclose(env.sim.agents_pos[0], [150.0, 150.0])
    np.testing.assert_allclose(env.target_pos, [180.0, 180.0])

    small_scene = {
        "id": "small",
        "field_size": 50.0,
        "agents_pos": [[0.0, 0.0]],
        "target_pos": [50.0, 0.0],
        "threats": [],
    }
    observations, _infos = env.reset(seed=1, options={"scene": small_scene})
    assert env.config.field_size == 50.0
    assert env.max_steps == 7
    assert env.observer.obs_builder.field_size == 50.0
    np.testing.assert_allclose(observations["drone_0"]["vector"][0:2], [1.0, 0.0], atol=1e-6)

    env.reset(seed=2)
    assert env.config.field_size == 100.0
    assert env.max_steps == 7


def test_episode_summary_decision_steps_ignore_intermediate_physics_ticks():
    cfg = EnvConfig(
        n_agents=1,
        physics_ticks_per_action=4,
        random_threat_mode="none",
        random_threat_count_min=0,
        random_threat_count_max=0,
    )
    cfg.wind.enabled = False
    cfg.battery.capacity = 0.0
    env = SwarmPZEnv(cfg, max_steps=2, goal_radius=1.0, oracle_enabled=False)
    env.reset(seed=3)
    for _ in range(2):
        env.step({"drone_0": np.zeros(2, dtype=np.float32)})

    summary = env.get_episode_summary()
    assert summary is not None
    assert summary.steps == 2
    assert summary.decision_steps == 2
    assert summary.physics_steps == 8


def test_scene_start_center_avoids_scene_threats():
    cfg = EnvConfig(
        n_agents=3,
        random_threat_mode="none",
        random_threat_count_min=0,
        random_threat_count_max=0,
    )
    cfg.wind.enabled = False
    cfg.battery.capacity = 0.0
    env = SwarmPZEnv(cfg, max_steps=5, goal_radius=1.0, oracle_enabled=False)

    scene = {
        "id": "spawn_safe",
        "field_size": 100.0,
        "start_center": [50.0, 50.0],
        "start_sigma": 0.0,
        "target_pos": [90.0, 90.0],
        "threats": [{"pos": [50.0, 50.0], "radius": 20.0, "intensity": 0.0}],
        "max_steps": 5,
    }
    env.reset(seed=0, options={"scene": scene})

    threat = env.sim.threats[0]
    margin = float(env.config.agent_radius) * 1.5
    dists = np.linalg.norm(env.sim.agents_pos - threat.position[None, :], axis=1)
    assert np.all(dists > (float(threat.radius) + margin - 1e-6))


def test_scene_start_center_avoids_non_oracle_block_dynamic_threats():
    cfg = EnvConfig(
        n_agents=2,
        random_threat_mode="none",
        random_threat_count_min=0,
        random_threat_count_max=0,
    )
    cfg.wind.enabled = False
    cfg.battery.capacity = 0.0
    env = SwarmPZEnv(cfg, max_steps=5, goal_radius=1.0, oracle_enabled=False)

    scene = {
        "id": "spawn_dynamic_safe",
        "field_size": 100.0,
        "start_center": [50.0, 50.0],
        "start_sigma": 0.0,
        "target_pos": [90.0, 90.0],
        "dynamic_threats": [
            {
                "type": "linear",
                "pos": [50.0, 50.0],
                "radius": 10.0,
                "intensity": 0.1,
                "speed": 0.0,
                "oracle_block": False,
            }
        ],
        "max_steps": 5,
    }
    env.reset(seed=0, options={"scene": scene})

    threat = env.sim.dynamic_threats[0]
    margin = float(env.config.agent_radius) * 1.5
    dists = np.linalg.norm(env.sim.agents_pos - threat.position[None, :], axis=1)
    assert np.all(dists > (float(threat.radius) + margin - 1e-6))


def test_safe_cluster_keeps_agents_separated_even_with_zero_sigma():
    cfg = EnvConfig(
        n_agents=3,
        random_threat_mode="none",
        random_threat_count_min=0,
        random_threat_count_max=0,
    )
    cfg.wind.enabled = False
    cfg.battery.capacity = 0.0
    env = SwarmPZEnv(cfg, max_steps=5, goal_radius=1.0, oracle_enabled=False)

    scene = {
        "id": "spawn_non_overlap",
        "field_size": 100.0,
        "start_center": [50.0, 50.0],
        "start_sigma": 0.0,
        "target_pos": [90.0, 90.0],
        "threats": [],
        "max_steps": 5,
    }
    env.reset(seed=0, options={"scene": scene})

    positions = env.sim.agents_pos
    min_pair = min(
        float(np.linalg.norm(positions[i] - positions[j]))
        for i in range(len(positions))
        for j in range(i + 1, len(positions))
    )
    assert min_pair >= (2.0 * float(env.config.agent_radius) - 1e-6)


def test_scene_start_centers_keep_agents_separated_across_clusters():
    cfg = EnvConfig(
        n_agents=4,
        random_threat_mode="none",
        random_threat_count_min=0,
        random_threat_count_max=0,
    )
    cfg.wind.enabled = False
    cfg.battery.capacity = 0.0
    env = SwarmPZEnv(cfg, max_steps=5, goal_radius=1.0, oracle_enabled=False)

    scene = {
        "id": "multi_center_non_overlap",
        "field_size": 100.0,
        "start_centers": [[50.0, 50.0], [50.0, 50.0]],
        "start_sigma": 0.0,
        "target_pos": [90.0, 90.0],
        "threats": [],
    }
    env.reset(seed=7, options={"scene": scene})

    positions = env.sim.agents_pos
    min_pair = min(
        float(np.linalg.norm(positions[i] - positions[j]))
        for i in range(len(positions))
        for j in range(i + 1, len(positions))
    )
    assert min_pair >= (2.0 * float(env.config.agent_radius) - 1e-6)


def test_scene_agents_pos_short_list_cycles_original_positions():
    cfg = EnvConfig(
        n_agents=4,
        random_threat_mode="none",
        random_threat_count_min=0,
        random_threat_count_max=0,
    )
    cfg.wind.enabled = False
    cfg.battery.capacity = 0.0
    env = SwarmPZEnv(cfg, max_steps=5, goal_radius=1.0, oracle_enabled=False)

    scene = {
        "id": "agents_pos_cycle",
        "field_size": 100.0,
        "agents_pos": [[10.0, 10.0], [20.0, 20.0]],
        "target_pos": [90.0, 90.0],
        "threats": [],
    }
    env.reset(seed=3, options={"scene": scene})

    expected = np.asarray(
        [[10.0, 10.0], [20.0, 20.0], [10.0, 10.0], [20.0, 20.0]],
        dtype=np.float32,
    )
    np.testing.assert_allclose(env.sim.agents_pos, expected)


def test_random_threat_spawn_avoids_agents_and_target_goal():
    cfg = EnvConfig(
        n_agents=1,
        random_threat_mode="static",
        random_threat_count_min=1,
        random_threat_count_max=1,
        random_threat_radius_min=15.0,
        random_threat_radius_max=15.0,
        start_pos_min=(40.0, 40.0),
        start_pos_max=(60.0, 60.0),
        target_pos_min=(40.0, 40.0),
        target_pos_max=(60.0, 60.0),
    )
    cfg.wind.enabled = False
    cfg.battery.capacity = 0.0
    env = SwarmPZEnv(cfg, max_steps=5, goal_radius=3.0, oracle_enabled=False)

    env.reset(seed=0)

    threat = env.sim.threats[0]
    agent_dist = float(np.linalg.norm(env.sim.agents_pos[0] - threat.position))
    target_dist = float(np.linalg.norm(env.target_pos - threat.position))
    assert agent_dist >= float(threat.radius) + float(env.config.agent_radius) * 1.5 - 1e-6
    assert target_dist >= float(threat.radius) + float(env.goal_radius) - 1e-6


def test_safe_pos_revalidates_after_jitter_near_threat_edges():
    from env.scenes.threats import StaticThreat

    cfg = EnvConfig(
        field_size=20.0,
        n_agents=1,
        random_threat_mode="none",
        random_threat_count_min=0,
        random_threat_count_max=0,
    )
    cfg.wind.enabled = False
    cfg.battery.capacity = 0.0
    env = SwarmPZEnv(cfg, max_steps=2, goal_radius=1.0, oracle_enabled=False)
    env.sim.threats = [StaticThreat([5.5, 5.5], radius=0.6, intensity=0.1, oracle_block=True)]
    env.sim.static_threats = list(env.sim.threats)
    env.sim.dynamic_threats = []

    for _ in range(128):
        pos = env.spawn.sample_safe_pos((5.0, 6.0), (6.0, 7.0), margin=0.0)
        assert env.spawn.is_safe_pos(pos, 0.0, float(env.config.field_size))


def test_scene_field_size_does_not_mutate_source_config():
    cfg = EnvConfig(
        field_size=100.0,
        n_agents=1,
        random_threat_mode="none",
        random_threat_count_min=0,
        random_threat_count_max=0,
    )
    cfg.wind.enabled = False
    cfg.battery.capacity = 0.0

    env = SwarmPZEnv(cfg, max_steps=5, goal_radius=1.0, oracle_enabled=False)
    scene = {
        "id": "field_leak_guard",
        "field_size": 120.0,
        "agents_pos": [[10.0, 10.0]],
        "target_pos": [100.0, 100.0],
        "threats": [],
    }
    env.reset(seed=0, options={"scene": scene})

    assert cfg.field_size == 100.0
    assert env.config.field_size == 120.0

    env2 = SwarmPZEnv(cfg, max_steps=5, goal_radius=1.0, oracle_enabled=False)
    env2.reset(seed=1)
    assert env2.config.field_size == 100.0


def test_env_config_from_dict_ignores_runtime_only_keys():
    cfg = EnvConfig.from_dict({"field_size": 42.0, "max_steps": 17, "goal_radius": 2.5})

    assert cfg.field_size == 42.0
    assert not hasattr(cfg, "max_steps")


def test_domain_randomization_is_seeded_before_first_reset_sampling():
    def _make_env() -> SwarmPZEnv:
        cfg = EnvConfig(n_agents=1, random_threat_mode="none", domain_randomization=True)
        cfg.wind.enabled = False
        cfg.battery.capacity = 0.0
        cfg.dr_max_speed_min = 1.0
        cfg.dr_max_speed_max = 9.0
        cfg.dr_drag_min = 0.1
        cfg.dr_drag_max = 0.9
        cfg.dr_mass_min = 0.5
        cfg.dr_mass_max = 2.0
        cfg.dr_max_accel_min = 1.0
        cfg.dr_max_accel_max = 4.0
        return SwarmPZEnv(cfg, max_steps=1, oracle_enabled=False)

    def _snapshot(env: SwarmPZEnv) -> tuple[float, float, float, float]:
        return (
            float(env.config.max_speed),
            float(env.config.physics.drag_coeff),
            float(env.config.physics.mass),
            float(env.config.physics.max_thrust),
        )

    env_a = _make_env()
    env_b = _make_env()
    env_a.reset(seed=123)
    env_b.reset(seed=123)
    np.testing.assert_allclose(_snapshot(env_a), _snapshot(env_b))

    env_a.reset(seed=123)
    np.testing.assert_allclose(_snapshot(env_a), _snapshot(env_b))


def test_scene_wind_overrides_apply_and_restore_to_base_profile():
    cfg = EnvConfig(
        n_agents=1,
        random_threat_mode="none",
        random_threat_count_min=0,
        random_threat_count_max=0,
    )
    cfg.wind.enabled = False
    cfg.wind.seed = None
    cfg.battery.capacity = 0.0
    env = SwarmPZEnv(cfg, max_steps=5, goal_radius=1.0, oracle_enabled=False)

    scene = {
        "id": "wind_scene",
        "agents_pos": [[10.0, 10.0]],
        "target_pos": [20.0, 20.0],
        "threats": [],
        "wind": {
            "enabled": True,
            "ou_theta": 0.77,
            "ou_sigma": 0.88,
            "seed": 999,
        },
    }
    env.reset(seed=0, options={"scene": scene})
    assert env.config.wind.enabled is True
    assert env.config.wind.ou_theta == pytest.approx(0.77)
    assert env.config.wind.ou_sigma == pytest.approx(0.88)
    assert env.config.wind.seed == 999
    assert env.engine._wind_field is not None

    env.reset(seed=1)
    assert env.config.wind.enabled is False
    assert env.config.wind.ou_theta == pytest.approx(0.15)
    assert env.config.wind.ou_sigma == pytest.approx(0.3)
    assert env.config.wind.seed is None
    assert env.engine._wind_field is None


def test_threat_collisions_stop_accumulating_after_death():
    cfg = EnvConfig(
        n_agents=2,
        random_threat_mode="none",
        random_threat_count_min=0,
        random_threat_count_max=0,
    )
    cfg.wind.enabled = False
    cfg.battery.capacity = 0.0
    env = SwarmPZEnv(cfg, max_steps=5, goal_radius=1.0, oracle_enabled=False)
    scene = {
        "id": "dead_collision_guard",
        "agents_pos": [[50.0, 50.0], [10.0, 10.0]],
        "target_pos": [95.0, 95.0],
        "threats": [{"pos": [50.0, 50.0], "radius": 10.0, "intensity": 1.0}],
        "max_steps": 5,
    }
    env.reset(seed=0, options={"scene": scene})

    counts = []
    for _ in range(4):
        _obs, _rewards, terminations, truncations, infos = env.step(
            {
                "drone_0": np.zeros(2, dtype=np.float32),
                "drone_1": np.zeros(2, dtype=np.float32),
            }
        )
        counts.append(float(infos["drone_0"]["threat_collisions"]))
        if all(terminations.values()) or all(truncations.values()):
            break

    assert counts[0] == 1.0
    assert counts == [1.0 for _ in counts]
