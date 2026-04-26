from types import SimpleNamespace

from env.event_handlers import EpisodeStatsCollector
from env.events import DecisionStepEvent, EpisodeEndEvent, EpisodeStartEvent, EventBus


def test_episode_stats_collector_builds_summary():
    bus = EventBus()
    collector = EpisodeStatsCollector(bus)

    start_state = SimpleNamespace(timestep=5)
    bus.emit(EpisodeStartEvent(state=start_state, seed=123, scene={"id": "scene_a"}))
    bus.emit(DecisionStepEvent(step=None, decision_index=1, done=False, is_timeout=False))
    bus.emit(DecisionStepEvent(step=None, decision_index=2, done=False, is_timeout=False))
    bus.emit(DecisionStepEvent(step=None, decision_index=2, done=False, is_timeout=False))

    collector.record_costs(
        {
            "a": {"cost_progress": 1.0, "cost_risk": 2.0},
            "b": {"cost_progress": 3.0, "cost_risk": 4.0},
        }
    )
    end_state = SimpleNamespace(timestep=9, dt=0.5)
    bus.emit(EpisodeEndEvent(state=end_state, steps=2, done=True, is_timeout=False))

    summary = collector.last_summary
    assert summary is not None
    assert summary.seed == 123
    assert summary.scene_id == "scene_a"
    assert summary.steps == 2
    assert summary.decision_steps == 2
    assert summary.done is True
    assert summary.is_timeout is False
    assert summary.start_timestep == 5
    assert summary.end_timestep == 9
    assert summary.physics_steps == 4
    assert summary.duration_sim_secs == 2.0
    assert summary.cost_means["cost_progress"] == 2.0
    assert summary.cost_means["cost_risk"] == 3.0
