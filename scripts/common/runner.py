"""Агрегация метрик прогонов и единая обвязка для оценочных серий."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class SuiteConfig:
    mode: str
    n_episodes: int
    max_steps: int
    goal_radius: float
    seed: int


def _nanmean(xs: list[float]) -> float:
    arr = np.array(xs, dtype=np.float32)
    return float(np.nanmean(arr)) if arr.size else float("nan")


def _nanstd(xs: list[float]) -> float:
    arr = np.array(xs, dtype=np.float32)
    return float(np.nanstd(arr)) if arr.size else float("nan")


def _percentile(xs: list[float], q: float) -> float:
    arr = np.array(xs, dtype=np.float32)
    return float(np.nanpercentile(arr, q)) if arr.size else float("nan")


def _ci95(xs: list[float]) -> tuple[float, float]:
    arr = np.array(xs, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    # 1.96 — коэффициент для нормального распределения при 95% доверительном интервале.
    half = 1.96 * std / np.sqrt(arr.size)
    return mean - half, mean + half


def aggregate_episode_metrics(episodes: list[dict[str, float]], keys: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k in keys:
        vals = [ep.get(k, float("nan")) for ep in episodes]
        out[f"{k}_mean"] = _nanmean(vals)
        out[f"{k}_std"] = _nanstd(vals)
        out[f"{k}_p50"] = _percentile(vals, 50)
        out[f"{k}_p90"] = _percentile(vals, 90)
        ci_lo, ci_hi = _ci95(vals)
        out[f"{k}_ci95_lo"] = ci_lo
        out[f"{k}_ci95_hi"] = ci_hi
    return out


def run_suite(
    runner_fn,
    suite: SuiteConfig,
    metric_keys: list[str],
) -> dict[str, Any]:
    episodes: list[dict[str, float]] = []
    # Разделяем фиксированный и случайный режимы, чтобы детерминизм был явным.
    rng = np.random.RandomState(suite.seed + 1000) if suite.mode == "random" else None

    for ep in range(suite.n_episodes):
        if suite.mode == "fixed":
            ep_seed = suite.seed + ep
        else:
            ep_seed = int(rng.randint(0, 2**31 - 1))
        metrics = runner_fn(ep_seed, suite.max_steps)
        episodes.append(metrics)

    aggregates = aggregate_episode_metrics(episodes, metric_keys)
    return {
        "suite": {
            "mode": suite.mode,
            "n_episodes": suite.n_episodes,
            "max_steps": suite.max_steps,
            "goal_radius": suite.goal_radius,
            "seed": suite.seed,
        },
        "episodes": episodes,
        "aggregates": aggregates,
    }
