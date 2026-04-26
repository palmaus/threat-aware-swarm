import numpy as np

from baselines.policies import default_registry
from env.config import EnvConfig
from env.pz_env import SwarmPZEnv
from scripts.debug.finish_debug import run_episode


def test_finish_debug_resets_policy_and_uses_one_based_finish_steps():
    cfg = EnvConfig(
        n_agents=1,
        random_threat_mode="none",
        random_threat_count_min=0,
        random_threat_count_max=0,
    )
    cfg.wind.enabled = False
    cfg.battery.capacity = 0.0
    cfg.physics.drag_coeff = 0.0
    env = SwarmPZEnv(cfg, max_steps=5, goal_radius=3.0, goal_hold_steps=1, oracle_enabled=False)
    original_reset = env.reset

    def reset_at_goal(seed=None, options=None):
        obs, infos = original_reset(seed=seed, options=options)
        env.engine.target_pos[:] = np.array([10.0, 10.0], dtype=np.float32)
        env.engine.sim.agents_pos[:] = np.array([[10.0, 10.0]], dtype=np.float32)
        env.engine.sim.agents_vel[:] = 0.0
        env.engine.prev_dists[:] = 0.0
        return obs, infos

    env.reset = reset_at_goal
    policy = default_registry().create("baseline:zero")
    reset_calls = []
    original_policy_reset = policy.reset

    def counted_reset(seed=None):
        reset_calls.append(seed)
        original_policy_reset(seed)

    policy.reset = counted_reset

    summary, _trace = run_episode(env, policy, max_steps=5, seed=123, trace=True)

    assert reset_calls == [123]
    assert summary["first_in_goal_step"] == 1
    assert summary["first_finish_step"] == 1
