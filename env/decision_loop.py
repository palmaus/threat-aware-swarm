from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from env.physics.loop import PhysicsLoop
    from env.runtime.results import StepResult


@dataclass
class DecisionLoop:
    """Управляет разницей между decision‑шагами и симуляционными тиками."""

    ticks_per_action: int = 1
    decision_step: int = 0
    physics_loop: PhysicsLoop | None = None

    def sync(self, ticks_per_action: int) -> None:
        self.ticks_per_action = max(1, int(ticks_per_action))

    def reset(self) -> None:
        self.decision_step = 0

    def run(self, step_fn: Callable[[dict | None], StepResult], actions: dict | None) -> list[StepResult]:
        self.decision_step += 1
        if self.physics_loop is None:
            return [step_fn(actions)]
        return self.physics_loop.run(step_fn, actions, self.ticks_per_action)

    def is_timeout(self, max_steps: int) -> bool:
        return self.decision_step >= int(max_steps)
