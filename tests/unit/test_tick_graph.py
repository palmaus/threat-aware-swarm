"""Проверка разделения decision/physics шагов."""

from env.decision_loop import DecisionLoop
from env.physics.loop import PhysicsLoop


class _DummyState:
    def __init__(self, timestep: int) -> None:
        self.timestep = timestep


class _DummyStep:
    def __init__(self, timestep: int, done: bool) -> None:
        self.state = _DummyState(timestep)
        self.done = done


def test_tick_graph_counts_steps():
    physics = PhysicsLoop()
    decision = DecisionLoop(ticks_per_action=3, physics_loop=physics)
    counter = {"t": 0}

    def step_fn(_actions):
        counter["t"] += 1
        return _DummyStep(counter["t"], done=False)

    steps = decision.run(step_fn, actions=None)
    assert decision.decision_step == 1
    assert len(steps) == 3
    assert physics.physics_step == 3


def test_tick_graph_stops_on_done():
    physics = PhysicsLoop()
    decision = DecisionLoop(ticks_per_action=5, physics_loop=physics)
    counter = {"t": 0}

    def step_fn(_actions):
        counter["t"] += 1
        return _DummyStep(counter["t"], done=counter["t"] >= 2)

    steps = decision.run(step_fn, actions=None)
    assert len(steps) == 2
    assert physics.physics_step == 2
