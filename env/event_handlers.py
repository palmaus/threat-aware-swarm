"""Штатные обработчики событий симуляции."""

from __future__ import annotations

from dataclasses import dataclass

from env.events import DecisionStepEvent, EpisodeEndEvent, EpisodeStartEvent, EventBus


@dataclass(frozen=True)
class EpisodeSummary:
    """Краткая сводка эпизода для логирования/диагностики."""

    seed: int | None
    scene_id: str | None
    steps: int
    decision_steps: int
    done: bool
    is_timeout: bool
    start_timestep: int
    end_timestep: int
    physics_steps: int
    duration_sim_secs: float
    cost_means: dict[str, float]


class EpisodeStatsCollector:
    """Собирает агрегированную статистику по событиям эпизода."""

    def __init__(self, bus: EventBus | None = None) -> None:
        self._current_seed: int | None = None
        self._current_scene_id: str | None = None
        self._start_timestep = 0
        self._decision_steps = 0
        self._cost_sums: dict[str, float] = {}
        self._cost_counts: dict[str, int] = {}
        self._last_decision_index: int | None = None
        self.last_summary: EpisodeSummary | None = None
        if bus is not None:
            self.attach(bus)

    def attach(self, bus: EventBus) -> None:
        bus.subscribe(EpisodeStartEvent, self._on_start)
        bus.subscribe(DecisionStepEvent, self._on_step)
        bus.subscribe(EpisodeEndEvent, self._on_end)

    def reset(self) -> None:
        self._current_seed = None
        self._current_scene_id = None
        self._start_timestep = 0
        self._decision_steps = 0
        self._last_decision_index = None
        self._cost_sums = {}
        self._cost_counts = {}

    def _on_start(self, event: EpisodeStartEvent) -> None:
        self._current_seed = event.seed
        scene = event.scene or {}
        self._current_scene_id = (
            str(scene.get("id")) if isinstance(scene, dict) and scene.get("id") is not None else None
        )
        try:
            self._start_timestep = int(event.state.timestep)
        except Exception:
            self._start_timestep = 0
        self._decision_steps = 0
        self._last_decision_index = None
        self._cost_sums = {}
        self._cost_counts = {}

    def _on_step(self, event: DecisionStepEvent) -> None:
        decision_index = int(event.decision_index)
        if self._last_decision_index != decision_index:
            self._decision_steps += 1
            self._last_decision_index = decision_index

    def _on_end(self, event: EpisodeEndEvent) -> None:
        end_timestep = 0
        try:
            end_timestep = int(event.state.timestep)
        except Exception:
            end_timestep = 0
        physics_steps = max(0, end_timestep - int(self._start_timestep))
        dt = 0.0
        try:
            dt = float(getattr(event.state, "dt", 0.0))
        except Exception:
            dt = 0.0
        duration_sim_secs = float(physics_steps) * float(dt)
        cost_means: dict[str, float] = {}
        for key, total in self._cost_sums.items():
            count = int(self._cost_counts.get(key, 0))
            cost_means[key] = float(total) / float(count) if count > 0 else float("nan")
        self.last_summary = EpisodeSummary(
            seed=self._current_seed,
            scene_id=self._current_scene_id,
            steps=int(event.steps),
            decision_steps=int(self._decision_steps),
            done=bool(event.done),
            is_timeout=bool(event.is_timeout),
            start_timestep=int(self._start_timestep),
            end_timestep=end_timestep,
            physics_steps=physics_steps,
            duration_sim_secs=duration_sim_secs,
            cost_means=cost_means,
        )

    def record_costs(self, infos: dict[str, dict]) -> None:
        from env.rewards.cost_schema import COST_KEYS

        for key in COST_KEYS:
            vals = []
            for inf in infos.values():
                if key not in inf:
                    continue
                try:
                    val = float(inf[key])
                except Exception:
                    continue
                if val == val:
                    vals.append(val)
            if not vals:
                continue
            mean_val = float(sum(vals)) / float(len(vals))
            self._cost_sums[key] = float(self._cost_sums.get(key, 0.0) + mean_val)
            self._cost_counts[key] = int(self._cost_counts.get(key, 0) + 1)
