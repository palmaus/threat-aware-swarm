"""Общие контракты и runtime-утилиты проекта.

Предпочтительные новые импорты:
- `common.policy.context`
- `common.policy.oracle_visibility`
- `common.runtime.contracts`
- `common.physics.model`

Старые плоские модули (`common.context`, `common.contracts`,
`common.oracle_visibility`, `common.physics_model`) оставлены как compatibility
layer и продолжают работать.
"""

from common.policy.context import PolicyContext, PolicyContextData, context_from_payload, context_from_state
from common.policy.oracle_visibility import oracle_visible, oracle_visible_for_policy
from common.runtime.contracts import (
    ContractResult,
    maybe_validate_reset,
    validate_observations,
    validate_public_state,
    validate_sim_state,
)

__all__ = [
    "ContractResult",
    "PolicyContext",
    "PolicyContextData",
    "context_from_payload",
    "context_from_state",
    "maybe_validate_reset",
    "oracle_visible",
    "oracle_visible_for_policy",
    "validate_observations",
    "validate_public_state",
    "validate_sim_state",
]
