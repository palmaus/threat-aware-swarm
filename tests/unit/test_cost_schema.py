"""Проверка схемы cost-метрик."""

from env.rewards.cost_schema import COST_KEYS, costs_from_parts, ensure_costs


def test_ensure_costs_fills_keys():
    info = {"cost_progress": 1.0}
    out = ensure_costs(info)
    for key in COST_KEYS:
        assert key in out


def test_costs_from_parts_negates_rewards():
    parts = {"rew_progress": 2.0, "rew_risk": -1.5}
    out = costs_from_parts(parts, include_time=True)
    assert out["cost_progress"] == -2.0
    assert out["cost_risk"] == 1.5
    assert out["cost_time"] == 1.0
