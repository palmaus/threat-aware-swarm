"""Repository path helpers for runtime code and scripts."""

from __future__ import annotations

import os
from pathlib import Path


def _resolve_env_path(value: str, root: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def find_repo_root(start: Path | None = None) -> Path:
    env_root = os.environ.get("TA_ROOT")
    if env_root:
        return Path(env_root).resolve()
    p = (start or Path.cwd()).resolve()
    if p.is_file():
        p = p.parent
    for _ in range(8):
        if (p / "pyproject.toml").exists() or (p / "requirements.txt").exists() or (p / ".git").exists():
            return p
        p = p.parent
    return Path.cwd().resolve()


def get_project_root(start: Path | None = None) -> Path:
    return find_repo_root(start)


def get_out_dir(root: Path | None = None) -> Path:
    root = root or get_project_root()
    env_out = os.environ.get("TA_OUT")
    if env_out:
        return _resolve_env_path(env_out, root)
    return get_runs_dir(root)


def get_runs_dir(root: Path | None = None) -> Path:
    root = root or get_project_root()
    env_runs = os.environ.get("TA_RUNS")
    if env_runs:
        return _resolve_env_path(env_runs, root)
    return (root / "runs").resolve()


def resolve_repo_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    root = find_repo_root(Path(__file__).resolve())
    return root / p
