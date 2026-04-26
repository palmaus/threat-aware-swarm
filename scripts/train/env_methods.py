"""Helpers for calling methods through SB3/SuperSuit wrapper stacks."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def call_env_method(env: Any, method_name: str, *args: Any, **kwargs: Any) -> list[Any]:
    """Call a method on base envs even when the wrapper stack has no env_method."""

    visited = set()
    targets: list[Any] = []

    def _visit(obj: Any) -> None:
        if obj is None:
            return
        obj_id = id(obj)
        if obj_id in visited:
            return
        visited.add(obj_id)

        if hasattr(obj, "venv"):
            _visit(obj.venv)
            return
        if hasattr(obj, "env"):
            _visit(obj.env)
            return
        if hasattr(obj, "envs"):
            try:
                for sub in obj.envs:
                    _visit(sub)
                return
            except Exception as exc:
                logger.warning("call_env_method: failed to traverse envs: %s", exc)
        if hasattr(obj, "par_env"):
            _visit(obj.par_env)
            return
        targets.append(obj)

    _visit(env)

    hits = [target for target in targets if hasattr(target, method_name)]
    if not hits:
        raise AttributeError(f"env_method '{method_name}' not found on wrapped envs")
    return [getattr(target, method_name)(*args, **kwargs) for target in hits]
