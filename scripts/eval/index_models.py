#!/usr/bin/env python3
"""
index_models.py — индексация ВСЕХ моделей .zip в указанных директориях.
Запуск:
  python -m scripts.eval.index_models --scan runs --out runs/index/model_registry.csv --rewrite
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from scripts.common.path_utils import resolve_repo_path

BASE_FIELDS = [
    "path",
    "name",
    "group",
    "size_bytes",
    "mtime",
]


def scan_roots(roots: list[str]) -> list[Path]:
    out: list[Path] = []
    for r in roots:
        root = resolve_repo_path(r)
        if not root.exists():
            continue
        for p in root.rglob("*.zip"):
            out.append(p)
    return sorted(set(out))


def infer_group(p: Path) -> str:
    s = str(p).replace("\\", "/")
    # Группировка нужна для удобной фильтрации в интерфейсе и отчетах.
    for key in ["run_"]:
        if key in s:
            return key
    return p.parent.name


def read_registry(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.exists():
        return [], BASE_FIELDS[:]
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    return rows, fields


def write_registry(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


@dataclass
class IndexModelsConfig:
    scan: list[str] = field(default_factory=list)
    out: str = "runs/index/model_registry.csv"
    rewrite: bool = False


def run(cfg: IndexModelsConfig) -> None:
    args = cfg

    out_path = resolve_repo_path(args.out)
    rows, fields = ([], BASE_FIELDS[:]) if args.rewrite else read_registry(out_path)
    if not fields:
        fields = BASE_FIELDS[:]
    for c in BASE_FIELDS:
        if c not in fields:
            fields.append(c)

    existing = {r.get("path"): r for r in rows if r.get("path")}

    models = scan_roots(args.scan)
    for p in models:
        stat = p.stat()
        rec = existing.get(str(p), {})
        rec.update(
            {
                "path": str(p),
                "name": p.stem,
                "group": infer_group(p),
                "size_bytes": str(stat.st_size),
                "mtime": str(stat.st_mtime),
            }
        )
        existing[str(p)] = rec

    new_rows = list(existing.values())
    # Стабильная сортировка по группе и имени делает вывод детерминированным.
    new_rows.sort(key=lambda r: (r.get("group", ""), r.get("name", "")))

    write_registry(out_path, new_rows, fields)
    print(f"[ОК] Индексировано {len(models)} моделей -> {out_path}")


def main() -> None:
    import hydra
    from omegaconf import DictConfig, OmegaConf

    from scripts.common.hydra_utils import apply_schema

    @hydra.main(version_base=None, config_path="../../configs/hydra", config_name="eval/index_models")
    def _run(cfg: DictConfig) -> None:
        cfg = apply_schema(cfg, IndexModelsConfig)
        data = OmegaConf.to_container(cfg, resolve=True)
        run(IndexModelsConfig(**data))

    _run()


if __name__ == "__main__":
    main()
