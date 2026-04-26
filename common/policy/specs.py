"""Lightweight policy metadata used before concrete baseline modules are imported."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

OracleCapability = Literal["none", "optional", "required"]
InfoRegime = Literal["fair", "privileged", "oracle_only"]

_POLICY_NAME_ALIASES: dict[str, str] = {
    "random": "baseline:random",
    "zero": "baseline:zero",
    "greedy": "baseline:greedy",
    "greedy_safe": "baseline:greedy_safe",
    "wall": "baseline:wall",
    "brake": "baseline:brake",
    "astar": "baseline:astar_grid",
    "astar_grid": "baseline:astar_grid",
    "mpc": "baseline:mpc_lite",
    "mpc_lite": "baseline:mpc_lite",
    "sep": "baseline:separation_steering",
    "pf": "baseline:potential_fields",
    "potential_fields": "baseline:potential_fields",
    "flow": "baseline:flow_field",
    "flow_field": "baseline:flow_field",
}


@dataclass(frozen=True)
class BaselinePolicySpec:
    name: str
    info_regime: InfoRegime = "fair"
    oracle_capability: OracleCapability = "none"
    map_aware: bool = False
    privileged_reference: bool = False
    notes: str = ""


_POLICY_SPECS: dict[str, BaselinePolicySpec] = {
    "baseline:random": BaselinePolicySpec(name="baseline:random"),
    "baseline:zero": BaselinePolicySpec(name="baseline:zero"),
    "baseline:greedy": BaselinePolicySpec(name="baseline:greedy"),
    "baseline:greedy_safe": BaselinePolicySpec(name="baseline:greedy_safe"),
    "baseline:wall": BaselinePolicySpec(name="baseline:wall"),
    "baseline:brake": BaselinePolicySpec(name="baseline:brake"),
    "baseline:separation_steering": BaselinePolicySpec(name="baseline:separation_steering"),
    "baseline:astar_grid": BaselinePolicySpec(
        name="baseline:astar_grid",
        info_regime="fair",
        oracle_capability="optional",
        map_aware=True,
        notes="Local grid planner; can optionally consume oracle_dir when baseline visibility is enabled.",
    ),
    "baseline:mpc_lite": BaselinePolicySpec(
        name="baseline:mpc_lite",
        info_regime="fair",
        oracle_capability="optional",
        notes="Reactive MPC baseline; can optionally consume oracle_dir as a global direction prior.",
    ),
    "baseline:potential_fields": BaselinePolicySpec(
        name="baseline:potential_fields",
        info_regime="fair",
        oracle_capability="optional",
        notes="Reactive potential fields; can optionally blend oracle_dir when exposed to baselines.",
    ),
    "baseline:flow_field": BaselinePolicySpec(
        name="baseline:flow_field",
        info_regime="privileged",
        oracle_capability="required",
        map_aware=True,
        privileged_reference=True,
        notes="Map-aware reference planner built around oracle distance fields.",
    ),
}


def canonical_policy_name(name: str | None) -> str | None:
    if name is None:
        return None
    raw = str(name)
    return _POLICY_NAME_ALIASES.get(raw, raw)


def register_policy_spec(
    name: str,
    *,
    info_regime: InfoRegime = "fair",
    oracle_capability: OracleCapability = "none",
    map_aware: bool = False,
    privileged_reference: bool = False,
    notes: str = "",
) -> BaselinePolicySpec:
    spec = BaselinePolicySpec(
        name=name,
        info_regime=info_regime,
        oracle_capability=oracle_capability,
        map_aware=map_aware,
        privileged_reference=privileged_reference,
        notes=notes,
    )
    _POLICY_SPECS[name] = spec
    return spec


def get_policy_spec(name: str | None) -> BaselinePolicySpec | None:
    canonical = canonical_policy_name(name)
    if not canonical:
        return None
    return _POLICY_SPECS.get(canonical)
