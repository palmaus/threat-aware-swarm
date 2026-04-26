from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class ExperimentResult:
    name: str
    seed: int | None
    config: dict[str, Any]
    metrics: dict[str, Any]
    artifacts: dict[str, Any]
    created_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "seed": self.seed,
            "config": self.config,
            "metrics": self.metrics,
            "artifacts": self.artifacts,
            "created_at": self.created_at or datetime.now().isoformat(timespec="seconds"),
        }


def write_experiment_result(run_dir: Path, result: ExperimentResult) -> Path:
    path = run_dir / "meta" / "experiment_result.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return path
