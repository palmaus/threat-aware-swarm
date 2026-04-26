"""Совместимость со старым импортом `common.contracts`."""

from common.runtime.contracts import (
    ContractResult,
    maybe_validate_reset,
    validate_observations,
    validate_public_state,
    validate_sim_state,
)

__all__ = [
    "ContractResult",
    "maybe_validate_reset",
    "validate_observations",
    "validate_public_state",
    "validate_sim_state",
]
