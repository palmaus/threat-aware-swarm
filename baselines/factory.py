"""Shared construction helpers for registered baseline policies."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from baselines.policies import adapt_policy, canonical_policy_name, default_registry


def _env_value(env: Any, attr: str, default: Any) -> Any:
    try:
        return getattr(env, attr)
    except Exception:
        return default


def _env_config_value(env: Any, attr: str, default: Any) -> Any:
    try:
        return getattr(getattr(env, "config", None), attr, default)
    except Exception:
        return default


def baseline_policy_params(
    name: str,
    *,
    env: Any | None = None,
    seed: int | None = None,
    params: Mapping[str, Any] | None = None,
    n_agents: int | None = None,
    grid_res: float | None = None,
    ui_safe: bool = False,
) -> dict[str, Any]:
    """Return canonical constructor kwargs for a baseline across UI/eval/bench/tuning."""

    canonical = canonical_policy_name(name)
    merged: dict[str, Any] = dict(params or {})
    if env is not None:
        n_agents = int(_env_value(env, "n_agents", _env_config_value(env, "n_agents", n_agents or 1)))
        grid_res = float(_env_config_value(env, "grid_res", grid_res if grid_res is not None else 1.0))
    if grid_res is None:
        grid_res = 1.0
    if n_agents is None:
        n_agents = 1

    if canonical == "baseline:random" and seed is not None:
        merged.setdefault("seed", int(seed))
    if canonical == "baseline:potential_fields":
        merged.setdefault("n_agents", int(n_agents))
    if canonical == "baseline:astar_grid":
        cell_size = float(grid_res)
        merged.setdefault("cell_size", cell_size)
        merged.setdefault("global_plan_cell_size", max(cell_size * 2.0, cell_size))
    if canonical == "baseline:mpc_lite" and ui_safe:
        # Numba-JIT в web-процессе может приводить к нативным крашам.
        merged.setdefault("use_numba", False)
    return merged


def create_baseline_policy(
    name: str,
    *,
    env: Any | None = None,
    seed: int | None = None,
    params: Mapping[str, Any] | None = None,
    n_agents: int | None = None,
    grid_res: float | None = None,
    ui_safe: bool = False,
    adapt: bool = False,
):
    canonical = canonical_policy_name(name)
    if canonical is None:
        raise KeyError(f"Unknown policy: {name}")
    kwargs = baseline_policy_params(
        canonical,
        env=env,
        seed=seed,
        params=params,
        n_agents=n_agents,
        grid_res=grid_res,
        ui_safe=ui_safe,
    )
    policy = default_registry().create(canonical, **kwargs)
    return adapt_policy(policy) if adapt else policy
