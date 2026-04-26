import numpy as np
import pytest

pytest.importorskip("gymnasium")

from scripts.bench.benchmark_baselines import _append_death_events, make_env


def test_make_env_respects_debug_metrics_flag():
    env = make_env(max_steps=1, goal_radius=1.0, seed=0, debug_metrics=False)

    assert env.config.debug_metrics is False


def test_append_death_events_reads_numpy_positions():
    events = []
    _append_death_events(
        events,
        "flow",
        [
            {
                "drone_0": {
                    "died_this_step": 1.0,
                    "pos": np.array([1.5, 2.5], dtype=np.float32),
                    "dist_to_nearest_threat": 0.1,
                    "nearest_threat_margin": -0.2,
                    "min_wall_dist": 3.0,
                    "dist": 4.0,
                }
            }
        ],
    )

    assert len(events) == 1
    assert events[0]["pos_x"] == pytest.approx(1.5)
    assert events[0]["pos_y"] == pytest.approx(2.5)
