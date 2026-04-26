from env.rewards.dto import RewardOutput


def test_reward_output_as_tuple():
    out = RewardOutput(
        rewards={"a": 1.0},
        infos={"a": {"rew_total": 1.0}},
        terminations={"a": False},
        truncations={"a": False},
    )
    tup = out.as_tuple()
    assert tup[0]["a"] == 1.0
    assert tup[1]["a"]["rew_total"] == 1.0
    assert tup[2]["a"] is False
    assert tup[3]["a"] is False
