"""Проверка разделения провайдеров карты и угроз."""

from env.config import EnvConfig
from env.engine import SwarmEngine


def test_map_provider_does_not_add_threats():
    engine = SwarmEngine(EnvConfig(n_agents=1))
    engine.sim.reset()
    scene = {
        "target_pos": [10.0, 10.0],
        "threats": [{"type": "static", "pos": [50.0, 50.0], "radius": 5.0, "intensity": 0.2}],
    }
    engine.map_provider.apply(engine, scene)
    assert engine.sim.threats == []


def test_threat_provider_adds_threats():
    engine = SwarmEngine(EnvConfig(n_agents=1))
    engine.sim.reset()
    scene = {
        "threats": [{"type": "static", "pos": [50.0, 50.0], "radius": 5.0, "intensity": 0.2}],
    }
    engine.threat_provider.apply(engine, scene)
    assert len(engine.sim.threats) == 1
