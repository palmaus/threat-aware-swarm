from __future__ import annotations

import pytest

from scripts.train.trained_ppo import make_vec_env


@pytest.mark.integration
def test_recurrent_ppo_smoke():
    try:
        from sb3_contrib import RecurrentPPO
    except Exception:
        pytest.skip("sb3_contrib not available")

    env = make_vec_env(
        max_steps=10,
        goal_radius=3.0,
        seed=0,
        num_vec_envs=1,
        num_cpus=1,
    )
    model = RecurrentPPO(
        "MultiInputLstmPolicy",
        env,
        n_steps=2,
        batch_size=2,
        n_epochs=1,
        learning_rate=1e-4,
        gamma=0.9,
        device="cpu",
    )
    model.learn(total_timesteps=2)
