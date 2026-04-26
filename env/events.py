"""Событийная шина для побочных действий и телеметрии."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EpisodeStartEvent:
    state: Any
    seed: int | None
    scene: dict | None


@dataclass(frozen=True)
class DecisionStepEvent:
    step: Any
    decision_index: int
    done: bool
    is_timeout: bool


@dataclass(frozen=True)
class EpisodeEndEvent:
    state: Any
    steps: int
    done: bool
    is_timeout: bool


EventHandler = Callable[[Any], None]


class EventBus:
    """Простая шина событий с подписками по типу."""

    def __init__(self) -> None:
        self._subs: dict[type, list[EventHandler]] = defaultdict(list)

    def subscribe(self, event_type: type, handler: EventHandler) -> None:
        self._subs[event_type].append(handler)

    def emit(self, event: Any) -> None:
        handlers = list(self._subs.get(type(event), []))
        if not handlers:
            return
        for handler in handlers:
            try:
                handler(event)
            except Exception:
                # Не ломаем симуляцию из-за подписчика.
                continue
