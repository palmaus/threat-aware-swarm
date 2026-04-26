from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class PolicyContext(Protocol):
    pos: np.ndarray
    vel: np.ndarray
    alive: np.ndarray
    agent_state: np.ndarray
    target_pos: np.ndarray
    target_vel: np.ndarray
    threats: list[Any]
    dists: np.ndarray
    in_goal: np.ndarray
    risk_p: np.ndarray
    oracle_dir: np.ndarray | None
    static_walls: list[tuple[float, float, float, float]]
    field_size: float
    max_speed: float
    max_accel: float
    max_thrust: float
    mass: float
    drag_coeff: float
    dt: float
    drag: float
    grid_res: float
    agent_radius: float
    wall_friction: float
    energy_level: np.ndarray
    measured_accel: np.ndarray
    timestep: int
    decision_step: int
    control_mode: str
    oracle_dist: np.ndarray | None
    oracle_risk: np.ndarray | None
    oracle_risk_grad: np.ndarray | None


@dataclass
class PolicyContextData:
    pos: np.ndarray
    vel: np.ndarray
    alive: np.ndarray
    agent_state: np.ndarray
    target_pos: np.ndarray
    target_vel: np.ndarray
    threats: list[Any]
    dists: np.ndarray
    in_goal: np.ndarray
    risk_p: np.ndarray
    oracle_dir: np.ndarray | None
    static_walls: list[tuple[float, float, float, float]]
    field_size: float
    max_speed: float
    max_accel: float
    max_thrust: float
    mass: float
    drag_coeff: float
    dt: float
    drag: float
    grid_res: float
    agent_radius: float
    wall_friction: float
    energy_level: np.ndarray
    measured_accel: np.ndarray
    timestep: int
    decision_step: int
    control_mode: str
    oracle_dist: np.ndarray | None = None
    oracle_risk: np.ndarray | None = None
    oracle_risk_grad: np.ndarray | None = None


def context_from_state(state: Any, *, include_oracle: bool | None = None) -> PolicyContextData:
    missing = [name for name in ("pos", "vel", "alive", "target_pos", "target_vel") if not hasattr(state, name)]
    if missing:
        raise TypeError(
            "context_from_state ожидает state-like объект из env.get_public_state(); "
            f"нет полей: {', '.join(missing)}."
        )
    if hasattr(state, "newly_finished") or hasattr(state, "last_action"):
        raise TypeError("context_from_state ожидает public state DTO, а не mutable SimState.")
    if include_oracle is None:
        include_oracle = True
    pos = np.asarray(state.pos, dtype=np.float32)
    vel = np.asarray(state.vel, dtype=np.float32)
    alive = np.asarray(state.alive, dtype=bool)
    target_pos = np.asarray(state.target_pos, dtype=np.float32)
    target_vel = np.asarray(state.target_vel, dtype=np.float32)
    dists = (
        np.asarray(state.dists, dtype=np.float32)
        if getattr(state, "dists", None) is not None
        else _dists(pos, target_pos)
    )
    in_goal = (
        np.asarray(state.in_goal, dtype=bool)
        if getattr(state, "in_goal", None) is not None
        else np.zeros((pos.shape[0],), dtype=bool)
    )
    risk_p = (
        np.asarray(state.risk_p, dtype=np.float32)
        if getattr(state, "risk_p", None) is not None
        else np.zeros((pos.shape[0],), dtype=np.float32)
    )
    oracle_dir = None
    oracle_dist = None
    oracle_risk = None
    oracle_risk_grad = None
    if include_oracle:
        if getattr(state, "oracle_dir", None) is not None:
            try:
                oracle_dir = np.asarray(state.oracle_dir, dtype=np.float32)
            except Exception:
                oracle_dir = None
        oracle_dist = getattr(state, "oracle_dist", None)
        oracle_risk = getattr(state, "oracle_risk", None)
        oracle_risk_grad = getattr(state, "oracle_risk_grad", None)
    static_walls = list(getattr(state, "static_walls", []) or [])
    agent_state = np.asarray(getattr(state, "agent_state", np.zeros((pos.shape[0],), dtype=np.int8)), dtype=np.int8)
    energy_level = np.asarray(
        getattr(state, "energy_level", np.zeros((pos.shape[0],), dtype=np.float32)), dtype=np.float32
    )
    measured_accel = np.asarray(
        getattr(state, "measured_accel", np.zeros_like(pos, dtype=np.float32)), dtype=np.float32
    )
    return PolicyContextData(
        pos=pos,
        vel=vel,
        alive=alive,
        agent_state=agent_state,
        target_pos=target_pos,
        target_vel=target_vel,
        threats=list(getattr(state, "threats", []) or []),
        dists=dists,
        in_goal=in_goal,
        risk_p=risk_p,
        oracle_dir=oracle_dir,
        static_walls=static_walls,
        field_size=float(getattr(state, "field_size", 0.0)),
        max_speed=float(getattr(state, "max_speed", 0.0)),
        max_accel=float(getattr(state, "max_accel", 0.0)),
        max_thrust=float(getattr(state, "max_thrust", 0.0)),
        mass=float(getattr(state, "mass", 0.0)),
        drag_coeff=float(getattr(state, "drag_coeff", 0.0)),
        dt=float(getattr(state, "dt", 0.0)),
        drag=float(getattr(state, "drag", 0.0)),
        grid_res=float(getattr(state, "grid_res", 1.0)),
        agent_radius=float(getattr(state, "agent_radius", 0.0)),
        wall_friction=float(getattr(state, "wall_friction", 0.0)),
        energy_level=energy_level,
        measured_accel=measured_accel,
        timestep=int(getattr(state, "timestep", 0)),
        decision_step=int(getattr(state, "decision_step", getattr(state, "timestep", 0))),
        control_mode=str(getattr(state, "control_mode", "waypoint")),
        oracle_dist=oracle_dist,
        oracle_risk=oracle_risk,
        oracle_risk_grad=oracle_risk_grad,
    )


def context_from_payload(payload: dict[str, Any], *, include_oracle: bool | None = None) -> PolicyContextData:
    if include_oracle is None:
        include_oracle = True
    pos = np.asarray(payload.get("agents_pos", []), dtype=np.float32)
    vel = np.asarray(payload.get("agents_vel", []), dtype=np.float32)
    alive = np.asarray(payload.get("agents_active", []), dtype=bool)
    agent_state = np.asarray(payload.get("agent_state", np.zeros_like(alive, dtype=np.int8)), dtype=np.int8)
    target_pos = np.asarray(payload.get("target_pos", np.zeros(2)), dtype=np.float32)
    target_vel = np.asarray(payload.get("target_vel", np.zeros(2)), dtype=np.float32)
    goal_radius = float(payload.get("goal_radius", 0.0))
    dists = _dists(pos, target_pos)
    in_goal = (dists <= goal_radius) & alive if dists.size else np.zeros_like(alive, dtype=bool)
    risk_p = np.zeros((pos.shape[0],), dtype=np.float32)
    oracle_dir = None
    if include_oracle and payload.get("oracle_dir") is not None:
        try:
            oracle_dir = np.asarray(payload.get("oracle_dir"), dtype=np.float32)
        except Exception:
            oracle_dir = None
    threats = []
    for t in payload.get("threats", []):
        try:
            t_pos, radius, intensity, vel_t = t
        except Exception:
            continue
        threats.append(
            SimpleNamespace(
                position=np.asarray(t_pos, dtype=np.float32),
                radius=float(radius),
                intensity=float(intensity),
                velocity=np.asarray(vel_t, dtype=np.float32),
            )
        )
    static_walls = [tuple(w) for w in (payload.get("walls") or [])]
    return PolicyContextData(
        pos=pos,
        vel=vel,
        alive=alive,
        agent_state=agent_state,
        target_pos=target_pos,
        target_vel=target_vel,
        threats=threats,
        dists=dists,
        in_goal=in_goal,
        risk_p=risk_p,
        oracle_dir=oracle_dir,
        static_walls=static_walls,
        field_size=float(payload.get("field_size", 0.0)),
        max_speed=float(payload.get("max_speed", 0.0)),
        max_accel=float(payload.get("max_accel", 0.0)),
        max_thrust=float(payload.get("max_thrust", 0.0)),
        mass=float(payload.get("mass", 0.0)),
        drag_coeff=float(payload.get("drag_coeff", 0.0)),
        dt=float(payload.get("dt", 0.0)),
        drag=float(payload.get("drag", 0.0)),
        grid_res=float(payload.get("grid_res", 1.0)),
        agent_radius=float(payload.get("agent_radius", 0.0)),
        wall_friction=float(payload.get("wall_friction", 0.0)),
        energy_level=np.asarray(
            payload.get("energy_level", np.zeros((pos.shape[0],), dtype=np.float32)), dtype=np.float32
        ),
        measured_accel=np.asarray(
            payload.get("measured_accel", np.zeros_like(pos, dtype=np.float32)), dtype=np.float32
        ),
        timestep=int(payload.get("timestep", 0)),
        decision_step=int(payload.get("decision_step", payload.get("timestep", 0))),
        control_mode=str(payload.get("control_mode", "waypoint")),
        oracle_dist=None,
        oracle_risk=None,
        oracle_risk_grad=None,
    )


def _dists(pos: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
    if pos.size == 0:
        return np.zeros((0,), dtype=np.float32)
    diff = pos - target_pos.reshape(1, 2)
    return np.linalg.norm(diff, axis=1).astype(np.float32)
