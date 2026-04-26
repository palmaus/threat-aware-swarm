from common.context import PolicyContextData as LegacyPolicyContextData
from common.contracts import ContractResult as LegacyContractResult
from common.oracle_visibility import oracle_visible as legacy_oracle_visible
from common.physics import apply_accel_dynamics
from common.physics.model import apply_accel_dynamics as nested_apply_accel_dynamics
from common.physics_model import apply_accel_dynamics as legacy_apply_accel_dynamics
from common.policy import PolicyContextData, context_from_state, oracle_visible
from common.policy.context import PolicyContextData as NestedPolicyContextData
from common.runtime import ContractResult, maybe_validate_reset
from common.runtime.contracts import ContractResult as NestedContractResult


def test_common_policy_exports_keep_compatibility() -> None:
    assert PolicyContextData is NestedPolicyContextData
    assert LegacyPolicyContextData is NestedPolicyContextData
    assert context_from_state is not None
    assert oracle_visible is legacy_oracle_visible


def test_common_runtime_exports_keep_compatibility() -> None:
    assert ContractResult is NestedContractResult
    assert LegacyContractResult is NestedContractResult
    assert maybe_validate_reset is not None


def test_common_physics_exports_keep_compatibility() -> None:
    assert apply_accel_dynamics is nested_apply_accel_dynamics
    assert legacy_apply_accel_dynamics is nested_apply_accel_dynamics
