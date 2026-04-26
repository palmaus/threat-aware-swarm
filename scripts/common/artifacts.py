from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from scripts.common.path_utils import resolve_repo_path

_SAFE_RE = re.compile(r"[^a-zA-Z0-9_.-]+")


def sanitize_name(value: str | None) -> str:
    if not value:
        return ""
    cleaned = _SAFE_RE.sub("_", str(value)).strip("_")
    return cleaned[:64] if cleaned else ""


def make_run_id(prefix: str | None = None, *, ts: datetime | None = None) -> str:
    stamp = (ts or datetime.now()).strftime("%Y%m%d_%H%M%S")
    suffix = sanitize_name(prefix)
    return f"{stamp}_{suffix}" if suffix else stamp


def resolve_out_root(out_root: str | Path) -> Path:
    return resolve_repo_path(out_root)


def ensure_run_dir(
    *,
    category: str,
    out_root: str | Path,
    run_id: str | None,
    prefix: str | None = None,
    out_dir: str | Path | None = None,
) -> Path:
    if out_dir:
        path = resolve_repo_path(out_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path
    root = resolve_out_root(out_root)
    rid = run_id or make_run_id(prefix)
    path = root / category / rid
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_manifest(run_dir: Path, *, config: dict[str, Any], command: list[str] | None = None) -> Path:
    config_json = json.dumps(config, ensure_ascii=False, sort_keys=True, default=str)
    config_hash = hashlib.sha256(config_json.encode("utf-8")).hexdigest()
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": config,
        "command": command or [],
        "config_hash": config_hash,
        "config_hash_algo": "sha256",
    }
    path = run_dir / "manifest.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _hash_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _hash_json(data: Any) -> str:
    raw = json.dumps(data, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return _hash_bytes(raw)


def _normalize_scenes(scenes: list[Any] | None) -> tuple[list[Any], str | None]:
    if not scenes:
        return [], None
    items: list[Any] = []
    blob = bytearray()
    for sc in scenes:
        if isinstance(sc, dict):
            payload = json.dumps(sc, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
            items.append(sc)
        else:
            text = str(sc)
            payload = text.encode("utf-8")
            # If looks like a file path, include file contents in hash.
            if any(text.endswith(ext) for ext in (".yaml", ".yml", ".json")):
                try:
                    path = resolve_repo_path(text)
                    if path.exists():
                        payload = path.read_bytes()
                        items.append(str(path))
                    else:
                        items.append(text)
                except Exception:
                    items.append(text)
            else:
                items.append(text)
        blob.extend(payload)
        blob.extend(b"\n")
    return items, _hash_bytes(bytes(blob)) if blob else None


def _env_hash(env_cfg: Any | None) -> tuple[str | None, str | None]:
    if env_cfg is None:
        return None, None
    if OmegaConf.is_config(env_cfg):
        data = OmegaConf.to_container(env_cfg, resolve=True)
    elif is_dataclass(env_cfg):
        data = asdict(env_cfg)
    elif isinstance(env_cfg, dict):
        data = env_cfg
    else:
        data = {"value": str(env_cfg)}
    profile = None
    if isinstance(data, dict):
        profile = data.get("profile")
    return profile, _hash_json(data)


def _override_values(resolved: dict[str, Any], overrides: list[str]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for ov in overrides:
        if "=" not in ov:
            items.append({"override": ov, "resolved": None})
            continue
        key, _val = ov.split("=", 1)
        key = key.lstrip("+~")
        key_path = key.replace("/", ".")
        cur: Any = resolved
        found = True
        for part in key_path.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                found = False
                break
        items.append({"override": ov, "resolved": cur if found else None})
    return items


def write_config_audit(
    run_dir: Path,
    *,
    cfg: Any,
    scenes: list[Any] | None = None,
    seeds: list[int] | None = None,
    env_cfg: Any | None = None,
) -> Path:
    resolved = OmegaConf.to_container(cfg, resolve=True) if OmegaConf.is_config(cfg) else cfg
    resolved = resolved if isinstance(resolved, dict) else {"value": resolved}

    overrides: list[str] = []
    try:
        overrides = list(HydraConfig.get().overrides.task or [])
    except Exception:
        overrides = []

    scenes_norm, scenes_hash = _normalize_scenes(scenes)
    seed_list = list(seeds or [])
    seed_hash = _hash_json(seed_list) if seed_list else None
    env_profile, env_hash = _env_hash(env_cfg)

    payload = {
        "resolved": resolved,
        "overrides": overrides,
        "overrides_resolved": _override_values(resolved, overrides),
        "seeds": seed_list,
        "seeds_hash": seed_hash,
        "scenes": scenes_norm,
        "scenes_hash": scenes_hash,
        "env_profile": env_profile,
        "env_hash": env_hash,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    path = run_dir / "config_audit.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
