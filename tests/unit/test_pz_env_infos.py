import numpy as np

REQUIRED_INFO_KEYS = {
    "alive",
    "finished",
    "in_goal",
    "dist",
    "risk_p",
    "min_neighbor_dist",
    "finished_alive",
    "newly_finished",
    "in_goal_steps",
}


def test_infos_have_required_keys(env):
    _obs, _infos = env.reset(seed=0)
    actions = {agent: np.zeros((2,), dtype=np.float32) for agent in env.possible_agents}
    _, _, _, _, infos = env.step(actions)
    for agent_id, inf in infos.items():
        assert isinstance(inf, dict)
        missing = REQUIRED_INFO_KEYS.difference(inf.keys())
        assert not missing, f"missing info keys for {agent_id}: {missing}"


def test_dead_agent_obs_zero_and_infos_present(env):
    _obs, _infos = env.reset(seed=0)

    # Принудительно делаем агента мёртвым, чтобы проверить нулевое наблюдение.
    env.sim.agent_state[0] = 2
    env.sim.agents_active[0] = False

    actions = {agent: np.zeros((2,), dtype=np.float32) for agent in env.possible_agents}
    obs2, _, _, _, infos2 = env.step(actions)

    dead_obs = obs2["drone_0"]
    assert np.allclose(dead_obs["vector"], 0.0)
    assert np.allclose(dead_obs["grid"], 0.0)
    missing = REQUIRED_INFO_KEYS.difference(infos2["drone_0"].keys())
    assert not missing
