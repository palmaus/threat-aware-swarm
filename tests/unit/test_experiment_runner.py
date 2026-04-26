import sys

from scripts.experiments import run_experiment


def test_run_experiment_dry_run(tmp_path, monkeypatch):
    spec = tmp_path / "spec.yaml"
    spec.write_text(
        """
name: demo
steps:
  - name: step0
    entrypoint: scripts.debug.finish_debug
    overrides: []
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "argv", ["run_experiment", "--spec", str(spec), "--dry-run"])
    assert run_experiment.main() == 0


def test_run_experiment_list(tmp_path, monkeypatch, capsys):
    spec = tmp_path / "spec.yaml"
    spec.write_text(
        """
name: demo
steps:
  - name: step0
    entrypoint: scripts.debug.finish_debug
    overrides: ["foo=bar"]
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "argv", ["run_experiment", "--spec", str(spec), "--list"])
    assert run_experiment.main() == 0
    out = capsys.readouterr().out
    assert "entrypoint" in out
    assert "scripts.debug.finish_debug" in out
