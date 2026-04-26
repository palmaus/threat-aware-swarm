"""Проверка публичного состояния для политик."""

import math

import numpy as np

from env.config import EnvConfig
from env.pz_env import SwarmPZEnv
from env.scenes.threats import StaticThreat


def test_public_state_masks_oracle():
    env = SwarmPZEnv(EnvConfig(n_agents=1), max_steps=5)
    env.reset(seed=0)
    public_state = env.get_public_state(include_oracle=False)
    assert public_state.oracle_dir is None


def test_public_state_returns_detached_view_within_tick():
    env = SwarmPZEnv(EnvConfig(n_agents=1), max_steps=5)
    env.reset(seed=0)
    public_a = env.get_public_state(include_oracle=True)
    public_b = env.get_public_state(include_oracle=True)
    assert public_a is not public_b
    public_a.pos[0, 0] = -123.0
    assert public_b.pos[0, 0] != -123.0


def test_public_get_state_returns_detached_view_within_tick():
    env = SwarmPZEnv(EnvConfig(n_agents=1), max_steps=5)
    env.reset(seed=0)
    state_a = env.get_state()
    state_b = env.get_state()
    assert state_a is not state_b
    state_a.pos[0, 0] = -123.0
    assert state_b.pos[0, 0] != -123.0
    assert env.engine.get_state().pos[0, 0] != -123.0


def test_env_facade_runtime_writes_delegate_to_engine():
    env = SwarmPZEnv(EnvConfig(n_agents=1), max_steps=5, oracle_enabled=False, oracle_async=False)
    env.oracle_enabled = True
    env.oracle_async = True
    env._base_max_thrust = 12.0
    assert env.engine.oracle_enabled is True
    assert env.engine.oracle_async is True
    assert math.isclose(env.engine._base_max_thrust, 12.0, rel_tol=1e-6)
    assert "oracle_enabled" not in env.__dict__
    assert "_base_max_thrust" not in env.__dict__


def test_public_state_cache_invalidates_after_step():
    env = SwarmPZEnv(EnvConfig(n_agents=1), max_steps=5)
    env.reset(seed=0)
    before = env.get_public_state(include_oracle=True)
    env.step({env.possible_agents[0]: [0.0, 0.0]})
    after = env.get_public_state(include_oracle=True)
    assert after is not before
    assert after.timestep != before.timestep


def test_public_state_caches_oracle_and_non_oracle_variants_separately():
    env = SwarmPZEnv(EnvConfig(n_agents=1), max_steps=5)
    env.reset(seed=0)
    with_oracle = env.get_public_state(include_oracle=True)
    without_oracle = env.get_public_state(include_oracle=False)
    assert with_oracle is not without_oracle
    assert without_oracle.oracle_dir is None


def test_public_state_does_not_mutate_engine_arrays_or_threats():
    env = SwarmPZEnv(EnvConfig(n_agents=1), max_steps=5)
    env.reset(seed=0)
    env.sim.threats = [StaticThreat(np.array([25.0, 25.0], dtype=np.float32), radius=3.0, intensity=0.2)]
    env.sim.static_threats = list(env.sim.threats)
    env.sim.dynamic_threats = []
    env.sim._sync_threat_arrays()
    env.engine._invalidate_export_cache()

    public_state = env.get_public_state(include_oracle=True)
    public_state.pos[0, 0] = 123.0
    public_state.threats[0].position[0] = 77.0

    assert env.sim.agents_pos[0, 0] != 123.0
    assert env.sim.threats[0].position[0] != 77.0


def test_public_state_keeps_static_scene_payload():
    env = SwarmPZEnv(EnvConfig(n_agents=1), max_steps=5)
    scene = {
        "agents_pos": [[10.0, 10.0]],
        "target_pos": [90.0, 90.0],
        "walls": [{"x1": 20.0, "y1": 20.0, "x2": 30.0, "y2": 25.0}],
        "circle_obstacles": [{"pos": [40.0, 50.0], "radius": 3.0}],
    }
    env.reset(seed=0, options={"scene": scene})
    public_state = env.get_public_state(include_oracle=True)
    assert public_state.static_walls == [(20.0, 20.0, 30.0, 25.0)]
    assert public_state.static_circles == [(40.0, 50.0, 3.0)]


def test_runtime_scalars_refresh_after_curriculum_reset():
    env = SwarmPZEnv(EnvConfig(n_agents=1), max_steps=5)
    env.apply_curriculum({"physics": {"mass": 2.0, "max_thrust": 6.0, "drag_coeff": 0.4}})
    env.reset(seed=0)
    state = env.get_state()
    assert math.isclose(state.max_accel, 3.0, rel_tol=1e-6)
    assert math.isclose(state.drag, 0.02, rel_tol=1e-6)
