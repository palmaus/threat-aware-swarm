"""Загрузка сцен и моделей для UI из файловой структуры проекта."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from ui.scene_spec import SceneValidationError, sanitize_scene_id, validate_and_normalize

try:
    import yaml
except Exception:
    yaml = None


def load_scene_file(path: Path) -> dict | None:
    try:
        if path.suffix.lower() in {".json"}:
            data = json.loads(path.read_text(encoding="utf-8"))
        if path.suffix.lower() in {".yaml", ".yml"} and yaml is not None:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    res = validate_and_normalize(data, allow_missing_id=True)
    if res.errors:
        return data
    return res.scene


def _scene_key(item: tuple[str, dict]) -> tuple:
    name = item[0]
    stem = str(item[1].get("id") or name)
    if stem.startswith("S"):
        tail = stem[1:]
        num = ""
        for ch in tail:
            if ch.isdigit():
                num += ch
            else:
                break
        if num:
            return (0, int(num), stem.lower())
    return (1, stem.lower())


def load_scenes(root: Path) -> list[tuple[str, dict]]:
    if not root.exists():
        return []
    scenes = []
    for p in sorted(root.iterdir()):
        if p.suffix.lower() not in {".json", ".yaml", ".yml"}:
            continue
        data = load_scene_file(p)
        if isinstance(data, dict):
            name = data.get("id") or p.stem
            scenes.append((str(name), data))
    return sorted(scenes, key=_scene_key)


def load_scenes_from_roots(roots: Iterable[Path]) -> list[tuple[str, dict]]:
    merged = {name: data for root in roots for name, data in load_scenes(root)}
    return sorted(merged.items(), key=_scene_key)


def save_scene(scene: dict, root: Path) -> Path:
    res = validate_and_normalize(scene, allow_missing_id=True)
    if res.errors:
        raise SceneValidationError("; ".join(res.errors))
    data = res.scene or {}
    scene_id = data.get("id") or "custom_scene"
    scene_id = sanitize_scene_id(str(scene_id))
    data["id"] = scene_id
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{scene_id}.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def delete_scene(scene_id: str, root: Path) -> bool:
    scene_id = sanitize_scene_id(str(scene_id))
    removed = False
    for suffix in (".json", ".yaml", ".yml"):
        path = root / f"{scene_id}{suffix}"
        if path.exists():
            path.unlink()
            removed = True
    return removed


def list_models(root: Path) -> list[str]:
    if not root.exists():
        return []
    models = []
    for p in root.rglob("*.zip"):
        try:
            rel = p.relative_to(root.parent)
        except Exception:
            rel = p
        models.append(str(rel).replace("\\", "/"))
    return sorted(models)
