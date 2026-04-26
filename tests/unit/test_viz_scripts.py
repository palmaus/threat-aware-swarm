"""Проверка загрузки сцен и политик, используемых в UI."""

import json
from pathlib import Path

from ui.policies import create_policy
from ui.scenes import list_models, load_scenes


def test_ui_load_scenes(tmp_path: Path):
    scene_json = {"id": "scene_json", "seed": 0}
    (tmp_path / "a.json").write_text(json.dumps(scene_json), encoding="utf-8")
    scenes = load_scenes(tmp_path)
    names = [name for name, _ in scenes]
    assert "scene_json" in names


def test_ui_create_policy():
    from env.config import EnvConfig
    from env.pz_env import SwarmPZEnv

    env = SwarmPZEnv(EnvConfig(), max_steps=5, goal_radius=3.0)
    name, policy = create_policy("baseline:astar_grid", env, None, deterministic=True)
    assert name == "baseline:astar_grid"
    assert hasattr(policy, "get_action")


def test_ui_list_models(tmp_path: Path):
    runs_dir = tmp_path / "runs"
    models_dir = runs_dir / "run_20260101_000000" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "best_by_finished.zip"
    model_path.write_text("stub", encoding="utf-8")
    models = list_models(runs_dir)
    assert any(str(model_path).endswith(m.replace("/", "\\")) or m.endswith("best_by_finished.zip") for m in models)
