"""Совместимость со старым импортом `common.context`."""

from common.policy.context import PolicyContext, PolicyContextData, context_from_payload, context_from_state

__all__ = [
    "PolicyContext",
    "PolicyContextData",
    "context_from_payload",
    "context_from_state",
]
