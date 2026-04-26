from __future__ import annotations

from typing import Any, TypedDict


class AggregateMetrics(TypedDict, total=False):
    success_rate: float
    finished_frac_end: float
    alive_frac_end: float
    deaths_mean: float
    time_to_goal_mean: float
    risk_integral_all: float
    risk_integral_alive: float
    collision_like: float
    episode_len_mean: float
    energy_efficiency: float
    safety_score: float
    cost_progress: float
    cost_risk: float
    cost_wall: float
    cost_collision: float
    cost_energy: float
    cost_jerk: float
    cost_time: float
    finished_frac_end_std: float
    alive_frac_end_std: float
    risk_integral_all_std: float
    risk_integral_alive_std: float
    time_to_goal_std: float
    path_ratio_std: float


class SceneMetrics(TypedDict):
    success: bool
    finished_frac_end: float
    alive_frac_end: float
    deaths: float
    time_to_goal_mean: float
    risk_integral_all: float
    risk_integral_alive: float
    collision_like: float
    episode_len: float
    energy_efficiency: float
    safety_score: float
    cost_progress: float
    cost_risk: float
    cost_wall: float
    cost_collision: float
    cost_energy: float
    cost_jerk: float
    cost_time: float


class TrialRecord(TypedDict, total=False):
    params: dict[str, Any]
    aggregate: AggregateMetrics
    score: tuple[float, float, float, float, float, float, float]
    score_scalar: float
    value: float
    state: str
    reported_step: int
    number: int
    scenes: dict[str, AggregateMetrics]
    search_stages: list[dict[str, Any]]


class StageBRecord(TypedDict, total=False):
    policy: str
    rank: int
    params: dict[str, Any]
    aggregate: AggregateMetrics
    score: tuple[float, float, float, float, float, float, float]
    score_scalar: float
    stageA_value: float
    scenes: dict[str, AggregateMetrics]
