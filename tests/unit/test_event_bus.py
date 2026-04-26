"""Проверка базовой работы EventBus."""

from env.events import DecisionStepEvent, EventBus


def test_event_bus_emits_and_catches():
    bus = EventBus()
    received: list[str] = []

    def handler_ok(event):
        received.append("ok")

    def handler_fail(event):
        raise RuntimeError("boom")

    bus.subscribe(DecisionStepEvent, handler_ok)
    bus.subscribe(DecisionStepEvent, handler_fail)

    bus.emit(DecisionStepEvent(step=None, decision_index=1, done=False, is_timeout=False))
    assert received == ["ok"]
