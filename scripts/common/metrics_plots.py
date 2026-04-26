from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _safe_values(per_scene: Mapping[str, Mapping[str, Any]], key: str) -> list[tuple[str, float]]:
    items: list[tuple[str, float]] = []
    for scene_id, metrics in per_scene.items():
        try:
            val = float(metrics.get(key, float("nan")))
        except Exception:
            continue
        if val != val:
            continue
        items.append((str(scene_id), val))
    return items


def write_summary_plots(
    out_dir: Path | str,
    per_scene: Mapping[str, Mapping[str, Any]],
    *,
    title: str | None = None,
    formats: tuple[str, ...] = ("png", "svg"),
) -> dict[str, Path]:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception:
        return {}

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, Path] = {}

    # 1) Success rate by scene
    success_items = _safe_values(per_scene, "success_rate")
    if success_items:
        labels = [k for k, _ in success_items]
        values = [v for _, v in success_items]
        fig, ax = plt.subplots(figsize=(max(6.5, 0.45 * len(labels)), 3.6))
        ax.bar(range(len(values)), values, color="#4c9aff")
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("success rate")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_title(f"{title} • success rate by scene" if title else "success rate by scene")
        fig.tight_layout()
        for ext in formats:
            path = out_path / f"success_rate_by_scene.{ext}"
            fig.savefig(path, dpi=140)
            artifacts[f"success_rate_by_scene_{ext}"] = path
        plt.close(fig)

    # 2) Path ratio distribution (per-scene means)
    path_items = _safe_values(per_scene, "path_ratio")
    if path_items:
        values = [v for _, v in path_items]
        fig, ax = plt.subplots(figsize=(6.6, 3.4))
        ax.hist(values, bins=min(12, max(4, len(values))), color="#8b5cf6", alpha=0.85)
        ax.set_xlabel("path ratio")
        ax.set_ylabel("count")
        ax.set_title(f"{title} • path ratio distribution" if title else "path ratio distribution")
        fig.tight_layout()
        for ext in formats:
            path = out_path / f"path_ratio_distribution.{ext}"
            fig.savefig(path, dpi=140)
            artifacts[f"path_ratio_distribution_{ext}"] = path
        plt.close(fig)

    # 3) Collisions by scene (collision_like)
    collision_items = _safe_values(per_scene, "collision_like")
    if not collision_items:
        collision_items = _safe_values(per_scene, "deaths_mean")
    if collision_items:
        labels = [k for k, _ in collision_items]
        values = [v for _, v in collision_items]
        fig, ax = plt.subplots(figsize=(max(6.5, 0.45 * len(labels)), 3.6))
        ax.bar(range(len(values)), values, color="#ef4444")
        ax.set_ylabel("collision like" if _safe_values(per_scene, "collision_like") else "deaths mean")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_title(f"{title} • collisions by scene" if title else "collisions by scene")
        fig.tight_layout()
        for ext in formats:
            path = out_path / f"collisions_by_scene.{ext}"
            fig.savefig(path, dpi=140)
            artifacts[f"collisions_by_scene_{ext}"] = path
        plt.close(fig)

    return artifacts
