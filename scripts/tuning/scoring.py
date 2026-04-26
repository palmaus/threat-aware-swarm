"""Scoring adapters for baseline tuning orchestration."""

from __future__ import annotations

from collections.abc import Callable

from scripts.common.metrics_utils import ScoreWeights, score_scalar, score_scalar_soft


def make_weighted_scorers(weights: ScoreWeights) -> tuple[Callable, Callable]:
    """Bind scoring weights without hiding them in tune_baselines.run closures."""

    def hard(agg, finish_min, alive_min):
        return score_scalar(
            agg,
            finish_min=finish_min,
            alive_min=alive_min,
            weights=weights,
        )

    def soft(agg, finish_min, alive_min):
        return score_scalar_soft(
            agg,
            weights=weights,
        )

    return hard, soft
