from __future__ import annotations

from pathlib import Path

import yaml

from scripts.common.path_utils import find_repo_root
from scripts.eval.eval_scenarios import EvalScenariosConfig, run


def _write_scene(path: Path) -> None:
    scene = {
        "id": "S_smoke_manifest",
        "seed": 0,
        "field_size": 50.0,
        "agents_pos": [[5.0, 5.0]],
        "target_pos": [45.0, 45.0],
        "threats": [],
        "max_steps": 5,
    }
    path.write_text(yaml.safe_dump(scene), encoding="utf-8")


def test_eval_scenarios_writes_metrics_manifest(tmp_path: Path) -> None:
    root = find_repo_root(Path(__file__).resolve())
    metrics_schema = root / "configs" / "metrics_schema.yaml"
    scene_path = tmp_path / "scene.yaml"
    _write_scene(scene_path)

    cfg = EvalScenariosConfig(
        policy="baseline:potential_fields",
        episodes=1,
        max_steps=5,
        seed=0,
        scenes=[str(scene_path)],
        oracle_enabled=False,
        out_dir=str(tmp_path),
        out_root=str(tmp_path),
        metrics_schema=str(metrics_schema),
    )
    run(cfg)

    manifest = tmp_path / "metric_manifest.json"
    assert manifest.exists()
