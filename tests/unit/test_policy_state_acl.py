"""Проверка, что политики не импортируют SimState напрямую."""

from __future__ import annotations

import ast
from pathlib import Path


def test_baselines_do_not_import_simstate() -> None:
    baselines_dir = Path(__file__).resolve().parents[2] / "baselines"
    offenders: list[Path] = []
    for path in baselines_dir.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "env.state":
                if any(name.name == "SimState" for name in node.names):
                    offenders.append(path)
                    break
    assert not offenders, f"Запрещён импорт SimState в политиках: {offenders}"
