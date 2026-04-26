"""Правила видимости оракула для baseline/RL."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from common.policy.specs import get_policy_spec

PolicySpecResolver = Callable[[str | None], Any]
_policy_spec_resolver: PolicySpecResolver | None = None


def set_policy_spec_resolver(resolver: PolicySpecResolver | None) -> None:
    """Registers an optional resolver without making common depend on baselines."""

    global _policy_spec_resolver
    _policy_spec_resolver = resolver


def oracle_visible(config, *, consumer: str) -> bool:
    """Возвращает, должен ли оракул быть доступен потребителю.

    consumer: "baseline" | "agent"
    """
    visibility = getattr(config, "oracle_visibility", None)
    if visibility:
        mode = str(visibility).strip().lower()
        if mode == "none":
            return False
        if mode == "baseline":
            return consumer == "baseline"
        if mode == "agent":
            return consumer == "agent"
    if consumer == "agent":
        return bool(getattr(config, "oracle_visible_to_agents", False))
    return bool(getattr(config, "oracle_visible_to_baselines", True))


def oracle_visible_for_policy(config, *, policy_name: str | None = None, policy: object | None = None) -> bool:
    """Учитывает и env-гейтинг, и capability конкретной baseline-политики.

    Для RL/прочих политик без baseline-spec поведение остаётся агентским.
    Для baseline с `oracle_capability="none"` доступ к oracle_dir блокируется,
    даже если окружение разрешает baseline visibility.
    """

    spec = getattr(policy, "policy_spec", None)
    resolved_name = policy_name
    if resolved_name is None:
        resolved_name = getattr(policy, "policy_name", None)
    if spec is None and resolved_name:
        spec = get_policy_spec(resolved_name)
        resolver = _policy_spec_resolver
        if spec is None and resolver is not None:
            spec = resolver(resolved_name)
    if spec is None:
        return oracle_visible(config, consumer="agent")
    visible = oracle_visible(config, consumer="baseline")
    if spec.oracle_capability == "none":
        return False
    return visible
