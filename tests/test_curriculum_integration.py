import numpy as np

from env.config import EnvConfig
from env.pz_env import SwarmPZEnv


def test_curriculum_apply_and_spawn_min_dist():
    cfg = EnvConfig()
    env = SwarmPZEnv(cfg, max_steps=20, goal_radius=3.0, oracle_enabled=False, oracle_async=False)

    stage = {
        "min_start_target_dist": 50.0,
        "start_pos_min": (5.0, 5.0),
        "start_pos_max": (95.0, 95.0),
        "target_pos_min": (5.0, 5.0),
        "target_pos_max": (95.0, 95.0),
        "goal_radius": 6.0,
        "reward": {"w_progress": 5.0, "w_in_goal_step": 0.1},
    }
    env.apply_curriculum(stage)
    obs, _info = env.reset(seed=1)
    assert obs, "reset() должен возвращать наблюдения"

    # Проверяем, что параметры действительно применились.
    assert abs(env.goal_radius - 6.0) < 1e-6
    assert abs(env.reward.w_progress - 5.0) < 1e-6
    assert abs(env.reward.w_in_goal_step - 0.1) < 1e-6

    # Дистанция между средним стартом и целью должна быть достаточно большой.
    start_center = np.mean(env.sim.agents_pos, axis=0)
    dist = float(np.linalg.norm(start_center - env.target_pos))
    assert dist >= 35.0


def test_target_motion_prob():
    cfg = EnvConfig()
    env = SwarmPZEnv(cfg, max_steps=10, goal_radius=3.0, oracle_enabled=False, oracle_async=False)

    env.apply_curriculum(
        {
            "target_motion": {"type": "linear", "angle": "random", "speed": 1.0},
            "target_motion_prob": 0.0,
        }
    )
    env.reset(seed=2)
    assert float(np.linalg.norm(env.target_vel)) < 1e-4

    env.apply_curriculum(
        {
            "target_motion": {"type": "linear", "angle": "random", "speed": 1.0},
            "target_motion_prob": 1.0,
        }
    )
    env.reset(seed=3)
    assert float(np.linalg.norm(env.target_vel)) > 0.1


def test_random_threat_mode_none_disables_threats():
    cfg = EnvConfig()
    cfg.random_threat_mode = "none"
    cfg.random_threat_count_min = 3
    cfg.random_threat_count_max = 5
    env = SwarmPZEnv(cfg, max_steps=10, goal_radius=3.0, oracle_enabled=False, oracle_async=False)
    env.reset(seed=4)
    assert len(env.sim.threats) == 0
    assert len(env.sim.static_threats) == 0
    assert len(env.sim.dynamic_threats) == 0


def test_shared_curriculum_sync_on_reset():
    cfg = EnvConfig()
    shared = {"params": {"goal_radius": 7.0, "goal_hold_steps": 2}, "version": 1}
    env = SwarmPZEnv(
        cfg, max_steps=10, goal_radius=3.0, oracle_enabled=False, oracle_async=False, shared_curriculum=shared
    )
    env.reset(seed=5)
    assert abs(env.goal_radius - 7.0) < 1e-6
    assert env.goal_hold_steps == 2


def test_curriculum_can_update_max_steps():
    cfg = EnvConfig()
    env = SwarmPZEnv(cfg, max_steps=20, goal_radius=3.0, oracle_enabled=False, oracle_async=False)
    env.apply_curriculum({"max_steps": 800})
    assert env.max_steps == 800


def test_curriculum_blocks_structural_runtime_shape_changes():
    cfg = EnvConfig(n_agents=2, random_threat_mode="none", random_threat_count_min=0, random_threat_count_max=0)
    env = SwarmPZEnv(cfg, max_steps=20, goal_radius=3.0, oracle_enabled=False, oracle_async=False)

    env.apply_curriculum({"n_agents": 3, "grid_width": 21, "field_size": 50.0})
    assert env.config.n_agents == 2
    assert env.config.grid_width == 41
    assert env.config.field_size == 50.0

    obs, _infos = env.reset(seed=6)
    assert set(obs) == {"drone_0", "drone_1"}
    assert env.observer.obs_builder.field_size == 50.0


def test_random_threat_spawn_scales_to_small_fields():
    cfg = EnvConfig(
        field_size=30.0,
        n_agents=2,
        random_threat_mode="static",
        random_threat_count_min=1,
        random_threat_count_max=1,
    )
    env = SwarmPZEnv(cfg, max_steps=10, goal_radius=3.0, oracle_enabled=False, oracle_async=False)
    env.reset(seed=7)
    assert len(env.sim.threats) == 1
    threat = env.sim.threats[0]
    assert 0.0 <= float(threat.position[0]) <= cfg.field_size
    assert 0.0 <= float(threat.position[1]) <= cfg.field_size


def test_random_threat_spawn_handles_dense_default_counts():
    cfg = EnvConfig(field_size=64.0, n_agents=2, random_threat_mode="static")
    env = SwarmPZEnv(cfg, max_steps=10, goal_radius=3.0, oracle_enabled=False, oracle_async=False)
    env.reset(seed=8)
    assert len(env.sim.threats) >= cfg.random_threat_count_min


def test_curriculum_wind_overrides_persist_across_reset():
    cfg = EnvConfig()
    cfg.wind.enabled = False
    env = SwarmPZEnv(cfg, max_steps=10, goal_radius=3.0, oracle_enabled=False, oracle_async=False)

    env.apply_curriculum({"wind": {"enabled": True, "ou_theta": 0.9, "ou_sigma": 1.1, "seed": 123}})
    env.reset(seed=9)

    assert env.config.wind.enabled is True
    assert env.config.wind.ou_theta == 0.9
    assert env.config.wind.ou_sigma == 1.1
    assert env.config.wind.seed == 123


def test_curriculum_battery_overrides_persist_across_reset():
    cfg = EnvConfig()
    cfg.battery.capacity = 0.0
    env = SwarmPZEnv(cfg, max_steps=10, goal_radius=3.0, oracle_enabled=False, oracle_async=False)

    env.apply_curriculum({"battery": {"capacity": 10.0, "drain_hover": 1.5, "drain_thrust": 2.5}})
    env.reset(seed=10)

    assert env.config.battery.capacity == 10.0
    assert env.config.battery.drain_hover == 1.5
    assert env.config.battery.drain_thrust == 2.5
