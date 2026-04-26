"""Совместимость со старым импортом `common.oracle_visibility`."""

from common.policy.oracle_visibility import oracle_visible, oracle_visible_for_policy, set_policy_spec_resolver

__all__ = ["oracle_visible", "oracle_visible_for_policy", "set_policy_spec_resolver"]
