"""Evaluation cache and lineage hashing helpers for baseline tuning."""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any

from scripts.common.path_utils import resolve_repo_path

DEFAULT_CACHE_LINEAGE_PATHS = [
    "baselines",
    "env",
    "scripts/tuning",
    "scripts/common/metrics_utils.py",
    "scripts/eval/eval_scenarios.py",
    "common",
    "scripts/common/episode_metrics.py",
    "configs/hydra/tuning",
]


def json_normalize(value: Any):
    if isinstance(value, dict):
        return {str(key): json_normalize(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_normalize(item) for item in value]
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except Exception:
        pass
    return value


def cache_lineage_files(lineage_paths: list[str] | None = None) -> list[Path]:
    files: list[Path] = []
    for entry in lineage_paths or DEFAULT_CACHE_LINEAGE_PATHS:
        path = resolve_repo_path(entry)
        if path.is_dir():
            files.extend(sorted(p for p in path.rglob("*.py") if p.is_file()))
        elif path.is_file():
            files.append(path)
    unique: dict[str, Path] = {}
    for path in files:
        unique[str(path)] = path
    return [unique[key] for key in sorted(unique)]


def compute_cache_code_hash(cache_version: str, lineage_paths: list[str] | None = None) -> tuple[str, list[str]]:
    digest = hashlib.blake2b(digest_size=12)
    digest.update(str(cache_version).encode("utf-8"))
    rel_paths: list[str] = []
    repo_root = resolve_repo_path(".")
    for path in cache_lineage_files(lineage_paths):
        rel = str(path.relative_to(repo_root))
        rel_paths.append(rel)
        digest.update(rel.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest(), rel_paths


def cleanup_eval_cache(base_root: Path, *, active_namespace: str, ttl_days: int) -> dict[str, int]:
    stats = {"removed_namespaces": 0, "removed_files": 0}
    if ttl_days <= 0 or not base_root.exists():
        return stats
    ttl_seconds = int(ttl_days) * 24 * 60 * 60
    now = time.time()
    for child in base_root.iterdir():
        if child.name == active_namespace:
            continue
        try:
            mtime = child.stat().st_mtime
        except FileNotFoundError:
            continue
        if now - mtime < ttl_seconds:
            continue
        if child.is_dir():
            removed = sum(1 for _ in child.rglob("*") if _.is_file())
            shutil.rmtree(child, ignore_errors=True)
            stats["removed_namespaces"] += 1
            stats["removed_files"] += removed
        else:
            child.unlink(missing_ok=True)
            stats["removed_files"] += 1
    return stats


def episode_cache_env_hash(env_payload: dict, *, regime: str, goal_radius: float, success_threshold: float) -> str:
    payload = {
        "env": json_normalize(env_payload),
        "regime": regime,
        "goal_radius": float(goal_radius),
        "success_threshold": float(success_threshold),
    }
    return hashlib.blake2b(json.dumps(payload, sort_keys=True).encode("utf-8"), digest_size=8).hexdigest()


def make_eval_cache(args, *, out_dir: Path, env_payload: dict, regime: str, goal_radius: float):
    enabled = bool(getattr(args, "eval_cache", True))
    cache_dir_raw = str(getattr(args, "cache_dir", "") or "").strip()
    if cache_dir_raw:
        cache_root = resolve_repo_path(cache_dir_raw)
    else:
        cache_root = resolve_repo_path(args.out_root) / "tune_eval_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_version = str(getattr(args, "cache_version", "v1") or "v1").strip()
    code_hash, lineage_files = compute_cache_code_hash(cache_version)
    namespace = f"{cache_version}_{code_hash}"
    cache_namespace_root = cache_root / namespace
    cache_namespace_root.mkdir(parents=True, exist_ok=True)
    cleanup_stats = {"removed_namespaces": 0, "removed_files": 0}
    if bool(getattr(args, "cache_cleanup", True)):
        cleanup_stats = cleanup_eval_cache(
            cache_root,
            active_namespace=namespace,
            ttl_days=int(getattr(args, "cache_ttl_days", 14)),
        )
    env_hash = episode_cache_env_hash(
        env_payload,
        regime=regime,
        goal_radius=goal_radius,
        success_threshold=float(getattr(args, "success_threshold", 0.5)),
    )
    return {
        "enabled": enabled,
        "base_root": cache_root,
        "root": cache_namespace_root,
        "stats": {"hits": 0, "misses": 0, "writes": 0},
        "cleanup": cleanup_stats,
        "namespace": namespace,
        "cache_version": cache_version,
        "code_hash": code_hash,
        "lineage_files": lineage_files,
        "env_hash": env_hash,
    }


def episode_cache_key(
    *,
    policy_name: str,
    params: dict,
    scene: dict,
    scene_seed: int,
    env_hash: str,
    episode_max_steps: int,
) -> str:
    payload = {
        "policy": str(policy_name),
        "params": json_normalize(params),
        "scene": json_normalize(scene),
        "seed": int(scene_seed),
        "env_hash": str(env_hash),
        "max_steps": int(episode_max_steps),
    }
    return hashlib.blake2b(json.dumps(payload, sort_keys=True).encode("utf-8"), digest_size=16).hexdigest()


def eval_cache_read(cache: dict | None, *, key: str):
    if not cache or not cache.get("enabled"):
        return None
    path = Path(cache["root"]) / key[:2] / f"{key}.json"
    if not path.exists():
        cache["stats"]["misses"] = int(cache["stats"].get("misses", 0)) + 1
        return None
    try:
        cache["stats"]["hits"] = int(cache["stats"].get("hits", 0)) + 1
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        cache["stats"]["misses"] = int(cache["stats"].get("misses", 0)) + 1
        return None


def eval_cache_write(cache: dict | None, *, key: str, metrics: dict) -> None:
    if not cache or not cache.get("enabled"):
        return
    path = Path(cache["root"]) / key[:2] / f"{key}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_normalize(metrics), ensure_ascii=False, sort_keys=True), encoding="utf-8")
    cache["stats"]["writes"] = int(cache["stats"].get("writes", 0)) + 1
