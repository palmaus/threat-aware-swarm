from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from env.rewards.cost_schema import COST_KEYS
from scripts.common.tune_types import AggregateMetrics, SceneMetrics

METRIC_KEYS = [
    "success_rate",
    "finished_frac_end",
    "alive_frac_end",
    "deaths_mean",
    "risk_integral_all",
    "risk_integral_alive",
    "time_to_goal_mean",
    "collision_like",
    "episode_len_mean",
    "path_ratio",
    "action_smoothness",
    "energy_efficiency",
    "safety_score",
    *COST_KEYS,
]

VARIABILITY_KEYS = [
    "finished_frac_end_std",
    "alive_frac_end_std",
    "risk_integral_all_std",
    "risk_integral_alive_std",
    "time_to_goal_std",
    "path_ratio_std",
]

TUNING_METRIC_KEYS = [*METRIC_KEYS, *VARIABILITY_KEYS]


@dataclass
class ScoreWeights:
    w_finished: float = 1_000_000.0
    w_alive: float = 100_000.0
    w_deaths: float = 10_000.0
    w_risk: float = 1_000.0
    w_time: float = 1.0
    penalty_finish: float = 1.0e9
    penalty_alive: float = 1.0e8


def resolve_score_weights(weights: Mapping[str, float] | ScoreWeights | None) -> ScoreWeights:
    if isinstance(weights, ScoreWeights):
        return weights
    base = ScoreWeights()
    if weights is None:
        return base
    merged = base.__dict__.copy()
    for key, val in weights.items():
        if key in merged:
            try:
                merged[key] = float(val)
            except Exception:
                continue
    return ScoreWeights(**merged)


def aggregate_episode_metrics(rows: list[SceneMetrics]) -> AggregateMetrics:
    def mean(key: str) -> float:
        vals = [r.get(key, float("nan")) for r in rows]
        vals = [v for v in vals if v == v]
        return float(np.mean(vals)) if vals else float("nan")

    def std(key: str) -> float:
        vals = [r.get(key, float("nan")) for r in rows]
        vals = [v for v in vals if v == v]
        return float(np.std(vals)) if vals else float("nan")

    def rate(key: str) -> float:
        vals = [1.0 if r[key] else 0.0 for r in rows]
        return float(np.mean(vals)) if vals else float("nan")

    out = {
        "success_rate": rate("success"),
        "finished_frac_end": mean("finished_frac_end"),
        "alive_frac_end": mean("alive_frac_end"),
        "deaths_mean": mean("deaths"),
        "time_to_goal_mean": mean("time_to_goal_mean"),
        "risk_integral_all": mean("risk_integral_all"),
        "risk_integral_alive": mean("risk_integral_alive"),
        "collision_like": mean("collision_like"),
        "episode_len_mean": mean("episode_len"),
        "path_ratio": mean("path_ratio"),
        "action_smoothness": mean("action_smoothness"),
        "energy_efficiency": mean("energy_efficiency"),
        "safety_score": mean("safety_score"),
        "finished_frac_end_std": std("finished_frac_end"),
        "alive_frac_end_std": std("alive_frac_end"),
        "risk_integral_all_std": std("risk_integral_all"),
        "risk_integral_alive_std": std("risk_integral_alive"),
        "time_to_goal_std": std("time_to_goal_mean"),
        "path_ratio_std": std("path_ratio"),
    }
    for key in COST_KEYS:
        out[key] = mean(key)
    return out


def aggregate_scene_metrics(per_scene: Mapping[str, Mapping[str, float]]) -> AggregateMetrics:
    def mean(key: str) -> float:
        vals = [v.get(key, float("nan")) for v in per_scene.values()]
        vals = [val for val in vals if val == val]
        return float(np.mean(vals)) if vals else float("nan")

    out = {
        "success_rate": mean("success_rate"),
        "finished_frac_end": mean("finished_frac_end"),
        "alive_frac_end": mean("alive_frac_end"),
        "deaths_mean": mean("deaths_mean"),
        "time_to_goal_mean": mean("time_to_goal_mean"),
        "risk_integral_all": mean("risk_integral_all"),
        "risk_integral_alive": mean("risk_integral_alive"),
        "collision_like": mean("collision_like"),
        "episode_len_mean": mean("episode_len_mean"),
        "path_ratio": mean("path_ratio"),
        "action_smoothness": mean("action_smoothness"),
        "energy_efficiency": mean("energy_efficiency"),
        "safety_score": mean("safety_score"),
        "finished_frac_end_std": mean("finished_frac_end_std"),
        "alive_frac_end_std": mean("alive_frac_end_std"),
        "risk_integral_all_std": mean("risk_integral_all_std"),
        "risk_integral_alive_std": mean("risk_integral_alive_std"),
        "time_to_goal_std": mean("time_to_goal_std"),
        "path_ratio_std": mean("path_ratio_std"),
    }
    for key in COST_KEYS:
        out[key] = mean(key)
    return out


def metric_float(agg: Mapping[str, float], key: str, default: float = 0.0) -> float:
    val = agg.get(key, default)
    try:
        out = float(val)
    except Exception:
        return default
    if out != out:
        return default
    return out


def score_tuple(
    agg: Mapping[str, float], finish_min: float, alive_min: float
) -> tuple[float, float, float, float, float, float, float]:
    finished = metric_float(agg, "finished_frac_end", 0.0)
    alive = metric_float(agg, "alive_frac_end", 0.0)
    deaths = metric_float(agg, "deaths_mean", 0.0)
    risk = metric_float(agg, "risk_integral_alive", float("nan"))
    if risk != risk:
        risk = metric_float(agg, "risk_integral_all", 0.0)
    t = agg.get("time_to_goal_mean", float("nan"))
    t_score = -float(t) if t == t else -1e6
    finished_ok = 1.0 if finished >= finish_min else 0.0
    alive_ok = 1.0 if alive >= alive_min else 0.0
    return (
        finished_ok,
        alive_ok,
        finished,
        alive,
        -deaths,
        -risk,
        t_score,
    )


def score_scalar(
    agg: Mapping[str, float],
    finish_min: float,
    alive_min: float,
    weights: Mapping[str, float] | ScoreWeights | None = None,
) -> float:
    finished = metric_float(agg, "finished_frac_end", 0.0)
    alive = metric_float(agg, "alive_frac_end", 0.0)
    deaths = metric_float(agg, "deaths_mean", 0.0)
    risk = metric_float(agg, "risk_integral_alive", float("nan"))
    if risk != risk:
        risk = metric_float(agg, "risk_integral_all", 0.0)
    t = agg.get("time_to_goal_mean", float("nan"))
    t_score = -float(t) if t == t else -1e6

    w = resolve_score_weights(weights)
    if finished < finish_min:
        return -float(w.penalty_finish) + finished
    if alive < alive_min:
        return -float(w.penalty_alive) + alive

    return (
        finished * float(w.w_finished)
        + alive * float(w.w_alive)
        - deaths * float(w.w_deaths)
        - risk * float(w.w_risk)
        + t_score * float(w.w_time)
    )


def score_scalar_soft(
    agg: Mapping[str, float],
    finish_min: float | None = None,
    alive_min: float | None = None,
    weights: Mapping[str, float] | ScoreWeights | None = None,
) -> float:
    finished = metric_float(agg, "finished_frac_end", 0.0)
    alive = metric_float(agg, "alive_frac_end", 0.0)
    deaths = metric_float(agg, "deaths_mean", 0.0)
    risk = metric_float(agg, "risk_integral_alive", float("nan"))
    if risk != risk:
        risk = metric_float(agg, "risk_integral_all", 0.0)
    t = agg.get("time_to_goal_mean", float("nan"))
    t_score = -float(t) if t == t else -1e6

    w = resolve_score_weights(weights)
    return (
        finished * float(w.w_finished)
        + alive * float(w.w_alive)
        - deaths * float(w.w_deaths)
        - risk * float(w.w_risk)
        + t_score * float(w.w_time)
    )
