from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from env.state import PublicState, SimState


def _strict_debug() -> bool:
    return os.environ.get("TA_STRICT_DEBUG", "").strip() == "1"


@dataclass(frozen=True)
class ContractResult:
    ok: bool
    message: str


def _shape_ok(arr: np.ndarray, shape: tuple[int, ...]) -> bool:
    if arr.ndim != len(shape):
        return False
    for dim, expected in zip(arr.shape, shape):
        if expected >= 0 and dim != expected:
            return False
    return True


def validate_sim_state(state: "SimState") -> ContractResult:
    try:
        pos = np.asarray(state.pos)
        vel = np.asarray(state.vel)
        alive = np.asarray(state.alive)
    except Exception as exc:
        return ContractResult(False, f"sim_state: invalid arrays ({exc})")
    if not _shape_ok(pos, (-1, 2)):
        return ContractResult(False, "sim_state: pos shape must be (n,2)")
    if not _shape_ok(vel, pos.shape):
        return ContractResult(False, "sim_state: vel shape mismatch")
    if alive.shape[0] != pos.shape[0]:
        return ContractResult(False, "sim_state: alive length mismatch")
    return ContractResult(True, "ok")


def validate_public_state(state: "PublicState") -> ContractResult:
    try:
        pos = np.asarray(state.pos)
        vel = np.asarray(state.vel)
        alive = np.asarray(state.alive)
    except Exception as exc:
        return ContractResult(False, f"public_state: invalid arrays ({exc})")
    if not _shape_ok(pos, (-1, 2)):
        return ContractResult(False, "public_state: pos shape must be (n,2)")
    if not _shape_ok(vel, pos.shape):
        return ContractResult(False, "public_state: vel shape mismatch")
    if alive.shape[0] != pos.shape[0]:
        return ContractResult(False, "public_state: alive length mismatch")
    return ContractResult(True, "ok")


def validate_observations(observations: dict[str, dict[str, Any]], *, grid_width: int) -> ContractResult:
    for agent_id, obs in observations.items():
        if not isinstance(obs, dict):
            return ContractResult(False, f"obs[{agent_id}] not dict")
        vec = np.asarray(obs.get("vector", []), dtype=np.float32).reshape(-1)
        if vec.size < 13:
            return ContractResult(False, f"obs[{agent_id}] vector too short")
        grid = obs.get("grid")
        if grid is None:
            continue
        grid_arr = np.asarray(grid, dtype=np.float32)
        if grid_arr.ndim == 3:
            if grid_arr.shape[1] != grid_width or grid_arr.shape[2] != grid_width:
                return ContractResult(False, f"obs[{agent_id}] grid shape mismatch")
        elif grid_arr.ndim == 2:
            if grid_arr.shape[0] != grid_width or grid_arr.shape[1] != grid_width:
                return ContractResult(False, f"obs[{agent_id}] grid shape mismatch")
        else:
            return ContractResult(False, f"obs[{agent_id}] grid dim invalid")
    return ContractResult(True, "ok")


def maybe_validate_reset(
    *,
    state: "SimState",
    public_state: "PublicState",
    observations: dict[str, dict[str, Any]],
    grid_width: int,
) -> None:
    if not _strict_debug():
        return
    sim_check = validate_sim_state(state)
    if not sim_check.ok:
        raise ValueError(sim_check.message)
    pub_check = validate_public_state(public_state)
    if not pub_check.ok:
        raise ValueError(pub_check.message)
    obs_check = validate_observations(observations, grid_width=grid_width)
    if not obs_check.ok:
        raise ValueError(obs_check.message)
