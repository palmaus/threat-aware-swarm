"""Проверка загрузки спеки эксперимента."""

from pathlib import Path

from scripts.experiments.experiment_spec import build_command, load_experiment


def test_load_experiment(tmp_path: Path):
    spec_path = tmp_path / "exp.yaml"
    spec_path.write_text(
        "name: demo\n"
        "steps:\n"
        "  - name: step1\n"
        "    entrypoint: scripts.eval.eval_scenarios\n"
        "    overrides: [policy=baseline:random]\n",
        encoding="utf-8",
    )
    spec = load_experiment(spec_path)
    assert spec.name == "demo"
    assert len(spec.steps) == 1
    assert spec.steps[0].entrypoint == "scripts.eval.eval_scenarios"
    cmd = build_command(spec.steps[0], python_bin="python3")
    assert cmd[:3] == ["python3", "-m", "scripts.eval.eval_scenarios"]
