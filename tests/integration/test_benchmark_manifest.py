from __future__ import annotations

from pathlib import Path

from scripts.bench.benchmark_baselines import BenchmarkBaselinesConfig, run
from scripts.common.path_utils import find_repo_root


def test_benchmark_writes_metrics_manifest(tmp_path: Path) -> None:
    root = find_repo_root(Path(__file__).resolve())
    metrics_schema = root / "configs" / "metrics_schema.yaml"
    cfg = BenchmarkBaselinesConfig(
        policy="random",
        n_episodes=1,
        max_steps=5,
        seed=0,
        out_dir=str(tmp_path),
        out_root=str(tmp_path),
        metrics_schema=str(metrics_schema),
    )
    run(cfg)

    manifest = tmp_path / "metric_manifest.json"
    assert manifest.exists()
