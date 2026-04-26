from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from scripts.common.path_utils import get_runs_dir, resolve_repo_path


def _load_demo_scenes(spec_path: Path) -> list[Path]:
    data = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    steps = data.get("steps") if isinstance(data, dict) else []
    for step in steps:
        if not isinstance(step, dict):
            continue
        if step.get("entrypoint") != "scripts.eval.eval_scenarios":
            continue
        overrides = step.get("overrides") or []
        for item in overrides:
            if not isinstance(item, str) or not item.startswith("scenes="):
                continue
            raw = item.split("=", 1)[1].strip()
            if raw.startswith("[") and raw.endswith("]"):
                inner = raw[1:-1]
                parts = [p.strip() for p in inner.split(",") if p.strip()]
                return [resolve_repo_path(p) for p in parts]
    return []


def _scene_id(path: Path) -> str:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        payload = None
    if isinstance(payload, dict) and payload.get("id"):
        return str(payload["id"])
    return path.stem


def _find_latest_gif(scene_id: str) -> Path | None:
    root = get_runs_dir()
    pattern = f"**/gifs/{scene_id}_*.gif"
    matches = sorted(root.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def _copy_gif(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src.read_bytes())
    return dst


def _build_markdown(items: list[dict[str, Any]]) -> str:
    lines = ["# Demo Story Pack", ""]
    lines.append(f"_Updated: {datetime.now().isoformat(timespec='seconds')}_")
    lines.append("")
    for item in items:
        lines.append(f"## {item['scene_id']}")
        if item.get("gif"):
            lines.append(f"![{item['scene_id']}]({item['gif']})")
        else:
            lines.append("_GIF не найден._")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Генерация demo story pack.")
    parser.add_argument("--spec", default="configs/experiments/demo_pack.yaml", help="Путь к demo_pack.yaml.")
    parser.add_argument("--out", default="docs/demo_story.md", help="Выходной Markdown.")
    parser.add_argument("--copy", action="store_true", help="Копировать GIF в docs/images/demo_story/.")
    args = parser.parse_args()

    spec_path = resolve_repo_path(args.spec)
    scene_paths = _load_demo_scenes(spec_path)
    if not scene_paths:
        raise SystemExit("Не удалось найти сцены в demo_pack.yaml.")

    items: list[dict[str, Any]] = []
    for scene_path in scene_paths:
        sid = _scene_id(scene_path)
        gif_path = _find_latest_gif(sid)
        gif_ref = None
        if gif_path and gif_path.exists():
            if args.copy:
                dst = resolve_repo_path(f"docs/images/demo_story/{gif_path.name}")
                _copy_gif(gif_path, dst)
                gif_ref = str(dst).replace("\\", "/")
            else:
                gif_ref = str(gif_path).replace("\\", "/")
        items.append({"scene_id": sid, "gif": gif_ref})

    out_path = resolve_repo_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_build_markdown(items), encoding="utf-8")

    manifest_path = get_runs_dir() / "demo_story" / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[story_pack] out={out_path}")
    print(f"[story_pack] manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
