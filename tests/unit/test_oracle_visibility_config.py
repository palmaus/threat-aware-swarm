"""Проверка режима видимости оракула."""

from __future__ import annotations

from common.oracle_visibility import oracle_visible
from env.config import EnvConfig


def test_oracle_visibility_modes():
    cfg = EnvConfig()
    cfg.oracle_visibility = "none"
    assert not oracle_visible(cfg, consumer="baseline")
    assert not oracle_visible(cfg, consumer="agent")

    cfg.oracle_visibility = "baseline"
    assert oracle_visible(cfg, consumer="baseline")
    assert not oracle_visible(cfg, consumer="agent")

    cfg.oracle_visibility = "agent"
    assert not oracle_visible(cfg, consumer="baseline")
    assert oracle_visible(cfg, consumer="agent")


def test_oracle_visibility_fallback():
    cfg = EnvConfig()
    cfg.oracle_visibility = ""
    cfg.oracle_visible_to_baselines = False
    cfg.oracle_visible_to_agents = True
    assert not oracle_visible(cfg, consumer="baseline")
    assert oracle_visible(cfg, consumer="agent")
