from __future__ import annotations

from pathlib import Path

import yaml

from scripts.analysis.ablation_runner import main as ablation_main


def test_ablation_runner_dry_run(tmp_path: Path, monkeypatch) -> None:
    cfg = {
        "name": "ablation_dry",
        "runs": [
            {
                "name": "dummy",
                "command": ["{python}", "-c", "print('ok')"],
                "run_name": "dummy",
                "out_dir": "runs/ablation",
            }
        ],
        "metrics": [{"key": "metrics.last_eval.success_rate", "label": "success_rate"}],
    }
    cfg_path = tmp_path / "ablation.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    monkeypatch.setenv("PYTHONPATH", str(Path.cwd()))
    monkeypatch.setattr("sys.argv", ["ablation_runner", "--config", str(cfg_path), "--dry-run"])
    ablation_main()
