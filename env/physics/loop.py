"""Цикл физических тиков (simulation time)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from env.runtime.results import StepResult


@dataclass
class PhysicsLoop:
    """Управляет физическими тиками независимо от decision‑шага."""

    physics_step: int = 0

    def reset(self) -> None:
        self.physics_step = 0

    def run(self, step_fn: Callable[[dict | None], StepResult], actions: dict | None, ticks: int) -> list[StepResult]:
        steps: list[StepResult] = []
        for _ in range(max(1, int(ticks))):
            step = step_fn(actions)
            steps.append(step)
            try:
                self.physics_step = int(step.state.timestep)
            except Exception:
                self.physics_step += 1
            if step.done:
                break
        return steps
