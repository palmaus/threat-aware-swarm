"""Проверка совместимости VecEnv со Stable Baselines3."""

import numpy as np

from scripts.eval.eval_models import _env_overrides_from_meta, make_vec_env as make_eval_vec_env
from scripts.train.trained_ppo import make_pz_env, make_vec_env


def test_make_pz_env_applies_structural_overrides_before_construction():
    env = make_pz_env(
        max_steps=10,
        goal_radius=3.0,
        seed=0,
        env_overrides={"n_agents": 3, "field_size": 42.0},
    )

    assert len(env.possible_agents) == 3
    assert env.sim.n == 3
    assert env.get_state().pos.shape == (3, 2)
    assert env.config.field_size == 42.0


def test_eval_model_env_overrides_include_initial_stage_and_structural_fields():
    payload = _env_overrides_from_meta(
        {
            "env_config": {"n_agents": 2, "max_steps": 999},
            "initial_stage_params": {"max_speed": 7.5, "goal_radius": 9.0},
        }
    )
    env = make_eval_vec_env(max_steps=10, goal_radius=3.0, seed=0, env_overrides=payload)
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]

    assert payload == {"n_agents": 2, "max_speed": 7.5}
    assert obs["vector"].shape[0] == 2

    env.close()


def test_vecenv_reset_step_returns_ndarray():
    import os

    os.environ["TA_SKIP_VECS_MONITOR"] = "1"
    env = make_vec_env(
        max_steps=10,
        goal_radius=3.0,
        seed=0,
        num_vec_envs=1,
        num_cpus=1,
    )

    obs = env.reset()
    assert isinstance(obs, dict)
    assert "vector" in obs and "grid" in obs
    assert obs["vector"].ndim == 2
    assert obs["vector"].shape[1] == 13
    assert obs["grid"].shape[1:] == (1, 41, 41)

    actions = np.zeros((obs["vector"].shape[0], 2), dtype=np.float32)
    out = env.step(actions)
    assert isinstance(out, tuple)
    assert len(out) == 4
    obs2, _rewards, _dones, _infos = out
    assert isinstance(obs2, dict)
    assert obs2["vector"].shape[1] == 13
    assert obs2["grid"].shape[1:] == (1, 41, 41)

    env.close()
