#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
index_models.py — индексация ВСЕХ моделей .zip в указанных директориях.
Запуск:
  python scripts/index_models.py --scan train/models runs --out model_registry.csv --rewrite
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple


BASE_FIELDS = [
    "path",
    "name",
    "group",
    "size_bytes",
    "mtime",
]


def scan_roots(roots: List[str]) -> List[Path]:
    out: List[Path] = []
    for r in roots:
        root = Path(r)
        if not root.exists():
            continue
        for p in root.rglob("*.zip"):
            out.append(p)
    return sorted(set(out))


def infer_group(p: Path) -> str:
    s = str(p).replace("\\", "/")
    # Простая эвристика для определения группы
    for key in ["run_", "ppo_v1", "marl_v1", "ppo_colab_heavy"]:
        if key in s:
            return key
    return p.parent.name


def read_registry(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    if not path.exists():
        return [], BASE_FIELDS[:]
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    return rows, fields


def write_registry(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", nargs="+", required=True)
    ap.add_argument("--out", default="model_registry.csv")
    ap.add_argument("--rewrite", action="store_true")
    args = ap.parse_args()

    out_path = Path(args.out)
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
        rec.update({
            "path": str(p),
            "name": p.stem,
            "group": infer_group(p),
            "size_bytes": str(stat.st_size),
            "mtime": str(stat.st_mtime),
        })
        existing[str(p)] = rec

    new_rows = list(existing.values())
    # Стабильная сортировка по группе, затем по имени
    new_rows.sort(key=lambda r: (r.get("group",""), r.get("name","")))

    write_registry(out_path, new_rows, fields)
    print(f"[OK] Индексировано {len(models)} моделей -> {out_path}")


if __name__ == "__main__":
    main()
