"""Оркестратор экспериментов: читает spec и выполняет шаги."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

from scripts.common.path_utils import find_repo_root, resolve_repo_path
from scripts.experiments.experiment_spec import build_command, format_step, load_experiment


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment Orchestrator")
    parser.add_argument("--spec", required=True, help="Путь к YAML-спеке эксперимента.")
    parser.add_argument("--dry-run", action="store_true", help="Печатает команды без запуска.")
    parser.add_argument("--list", action="store_true", help="Печатает шаги эксперимента и выходит.")
    parser.add_argument("--python-bin", default=sys.executable, help="Интерпретатор Python.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    spec_path = resolve_repo_path(args.spec)
    spec = load_experiment(spec_path)
    repo_root = find_repo_root(spec_path)
    if args.list:
        for step in spec.steps:
            print(format_step(step))
        return 0
    for step in spec.steps:
        missing_env = [key for key in step.skip_if_env_missing if not os.environ.get(key)]
        if missing_env:
            print(f"[experiment:{spec.name}] step={step.name} пропущен (нет env: {', '.join(missing_env)})")
            continue
        cmd = build_command(step, python_bin=str(args.python_bin))
        print(f"[experiment:{spec.name}] step={step.name} cmd={' '.join(cmd)}")
        if args.dry_run:
            continue
        env = os.environ.copy()
        env.update(step.env)
        if step.cwd:
            cwd = resolve_repo_path(step.cwd)
        else:
            cwd = repo_root
        result = subprocess.run(cmd, env=env, cwd=str(cwd), check=False)
        if result.returncode != 0:
            return int(result.returncode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
