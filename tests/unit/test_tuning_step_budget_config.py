from pathlib import Path

import yaml


def _load_yaml(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def test_tune_baselines_defaults_include_step_budget_group() -> None:
    cfg = _load_yaml("configs/hydra/tuning/tune_baselines.yaml")
    defaults = cfg.get("defaults", [])
    assert any(item == {"/tuning/step_budget": "default"} for item in defaults)


def test_step_budget_presets_define_expected_max_steps() -> None:
    budgets = {
        "default": 600,
        "static": 900,
        "dynamic": 1000,
        "long": 1600,
    }
    for name, expected in budgets.items():
        cfg = _load_yaml(f"configs/hydra/tuning/step_budget/{name}.yaml")
        assert cfg["env"]["max_steps"] == expected
