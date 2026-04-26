"""Спецификация эксперимента и сбор команд запуска."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ExperimentStep:
    name: str
    entrypoint: str
    overrides: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None
    skip_if_env_missing: list[str] = field(default_factory=list)


@dataclass
class ExperimentSpec:
    name: str
    steps: list[ExperimentStep]


def load_experiment(path: str | Path) -> ExperimentSpec:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Spec эксперимента должен быть dict.")
    name = str(data.get("name", Path(path).stem))
    steps_raw = data.get("steps") or []
    steps: list[ExperimentStep] = []
    for idx, raw in enumerate(steps_raw):
        if not isinstance(raw, dict):
            raise ValueError(f"Step #{idx} должен быть dict.")
        entrypoint = str(raw.get("entrypoint", "")).strip()
        if not entrypoint:
            raise ValueError(f"Step #{idx} без entrypoint.")
        steps.append(
            ExperimentStep(
                name=str(raw.get("name", f"step_{idx}")),
                entrypoint=entrypoint,
                overrides=[str(x) for x in (raw.get("overrides") or [])],
                env={str(k): str(v) for k, v in (raw.get("env") or {}).items()},
                cwd=str(raw.get("cwd")) if raw.get("cwd") else None,
                skip_if_env_missing=[str(x) for x in (raw.get("skip_if_env_missing") or [])],
            )
        )
    return ExperimentSpec(name=name, steps=steps)


def build_command(step: ExperimentStep, *, python_bin: str) -> list[str]:
    return [python_bin, "-m", step.entrypoint, *step.overrides]


def format_step(step: ExperimentStep) -> dict[str, Any]:
    return {
        "name": step.name,
        "entrypoint": step.entrypoint,
        "overrides": list(step.overrides),
        "env": dict(step.env),
        "cwd": step.cwd,
        "skip_if_env_missing": list(step.skip_if_env_missing),
    }
