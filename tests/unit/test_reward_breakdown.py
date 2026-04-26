"""Проверка DTO RewardBreakdown."""

from env.rewards.dto import RewardBreakdown


def test_reward_breakdown_from_float():
    out = RewardBreakdown.from_output(1.25)
    assert out.total == 1.25
    assert out.parts == {}


def test_reward_breakdown_from_tuple():
    out = RewardBreakdown.from_output((2.0, {"rew_progress": 1.0, "rew_risk": -0.5}))
    assert out.total == 2.0
    assert out.parts["rew_progress"] == 1.0
    assert out.parts["rew_risk"] == -0.5
