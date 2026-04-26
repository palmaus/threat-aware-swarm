"""Neutral step/result DTOs for engine, physics loop and reward code."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from env.state import SimState


@dataclass
class ThreatMeta:
    dist: np.ndarray
    margin: np.ndarray
    radius: np.ndarray
    intensity: np.ndarray
    idx: np.ndarray
    inside_any: np.ndarray
    inside_any_intensity_sum: np.ndarray


@dataclass
class StepResult:
    prev_state: SimState
    state: SimState
    done: bool
    is_timeout: bool
    alive_before: np.ndarray
    died_this_step: np.ndarray
    threat_meta: ThreatMeta
    prev_threat_meta: ThreatMeta | None


@dataclass
class DecisionStepResult:
    steps: list[StepResult]

    @property
    def last(self) -> StepResult | None:
        if not self.steps:
            return None
        return self.steps[-1]

    @property
    def done(self) -> bool:
        last = self.last
        return bool(last.done) if last is not None else False

    @property
    def is_timeout(self) -> bool:
        last = self.last
        return bool(last.is_timeout) if last is not None else False
