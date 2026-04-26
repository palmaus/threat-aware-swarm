import pytest


def _skip_if_no_gymnasium():
    try:
        import gymnasium  # noqa: F401
    except Exception:
        pytest.skip("gymnasium not installed")


def test_run_bench_v2_returns_metrics():
    _skip_if_no_gymnasium()
    from scripts.bench.bench_obs_perf import run_bench

    result = run_bench(schema="obs@1694:v5", steps=5, n_agents=2, seed=0)
    assert result["steps"] == 5
    assert result["steps_per_sec"] > 0
    assert result["obs_dim"] == 1694
    assert result["grid_width"] == 41
