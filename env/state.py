from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class SimState:
    pos: np.ndarray  # Формат: (n, 2) float32.
    vel: np.ndarray  # Формат: (n, 2) float32.
    alive: np.ndarray  # Формат: (n,) bool.
    target_pos: np.ndarray  # Формат: (2,) float32.
    target_vel: np.ndarray  # Формат: (2,) float32.
    timestep: int
    threats: list[Any]
    dists: np.ndarray  # Формат: (n,) float32.
    in_goal: np.ndarray  # Формат: (n,) bool.
    in_goal_steps: np.ndarray  # Формат: (n,) int.
    finished: np.ndarray  # Формат: (n,) bool.
    newly_finished: np.ndarray  # Формат: (n,) bool.
    risk_p: np.ndarray  # Формат: (n,) float32.
    min_neighbor_dist: np.ndarray  # Формат: (n,) float32.
    last_action: np.ndarray  # Формат: (n, 2) float32.
    walls: np.ndarray  # Формат: (n, 4) float32.
    oracle_dir: np.ndarray | None  # Формат: (n, 2) float32 или None.
    static_walls: list[tuple[float, float, float, float]]
    static_circles: list[tuple[float, float, float]]
    collision_speed: np.ndarray  # Формат: (n,) float32.
    measured_accel: np.ndarray  # Формат: (n, 2) float32.
    energy: np.ndarray  # Формат: (n,) float32 (фактическая энергия).
    energy_level: np.ndarray  # Формат: (n,) float32 (0..1).
    agent_state: np.ndarray  # Формат: (n,) int8.
    control_mode: str
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
    decision_step: int = 0
    oracle_dist: np.ndarray | None = None  # Поле расстояний до цели (grid).
    oracle_risk: np.ndarray | None = None  # Поле риска (grid).
    oracle_risk_grad: np.ndarray | None = None  # Градиент риска (grid, 2 канала).


@dataclass
class PublicState:
    """Состояние, доступное политикам (без приватных полей ядра)."""

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
    static_circles: list[tuple[float, float, float]]
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


def _copy_array(value: Any, dtype: Any) -> np.ndarray:
    return np.array(value, dtype=dtype, copy=True)


def _copy_optional_array(value: Any, dtype: Any) -> np.ndarray | None:
    if value is None:
        return None
    return np.array(value, dtype=dtype, copy=True)


def _copy_threats(threats: list[Any]) -> list[Any]:
    copied = []
    for threat in threats:
        if hasattr(threat, "copy") and callable(threat.copy):
            copied.append(threat.copy())
            continue
        copied.append(copy.deepcopy(threat))
    return copied


def copy_public_state(state: PublicState) -> PublicState:
    return PublicState(
        pos=_copy_array(state.pos, np.float32),
        vel=_copy_array(state.vel, np.float32),
        alive=_copy_array(state.alive, bool),
        agent_state=_copy_array(state.agent_state, np.int8),
        target_pos=_copy_array(state.target_pos, np.float32),
        target_vel=_copy_array(state.target_vel, np.float32),
        threats=_copy_threats(list(state.threats)),
        dists=_copy_array(state.dists, np.float32),
        in_goal=_copy_array(state.in_goal, bool),
        risk_p=_copy_array(state.risk_p, np.float32),
        oracle_dir=_copy_optional_array(state.oracle_dir, np.float32),
        static_walls=list(state.static_walls),
        static_circles=list(state.static_circles),
        field_size=float(state.field_size),
        max_speed=float(state.max_speed),
        max_accel=float(state.max_accel),
        max_thrust=float(state.max_thrust),
        mass=float(state.mass),
        drag_coeff=float(state.drag_coeff),
        dt=float(state.dt),
        drag=float(state.drag),
        grid_res=float(state.grid_res),
        agent_radius=float(state.agent_radius),
        wall_friction=float(state.wall_friction),
        energy_level=_copy_array(state.energy_level, np.float32),
        measured_accel=_copy_array(state.measured_accel, np.float32),
        timestep=int(state.timestep),
        decision_step=int(state.decision_step),
        control_mode=str(state.control_mode),
        oracle_dist=_copy_optional_array(state.oracle_dist, np.float32),
        oracle_risk=_copy_optional_array(state.oracle_risk, np.float32),
        oracle_risk_grad=_copy_optional_array(state.oracle_risk_grad, np.float32),
    )


def copy_sim_state(state: SimState) -> SimState:
    """Возвращает detached-копию SimState для внешних read-only consumer'ов."""

    return SimState(
        pos=_copy_array(state.pos, np.float32),
        vel=_copy_array(state.vel, np.float32),
        alive=_copy_array(state.alive, bool),
        target_pos=_copy_array(state.target_pos, np.float32),
        target_vel=_copy_array(state.target_vel, np.float32),
        timestep=int(state.timestep),
        threats=_copy_threats(list(state.threats)),
        dists=_copy_array(state.dists, np.float32),
        in_goal=_copy_array(state.in_goal, bool),
        in_goal_steps=_copy_array(state.in_goal_steps, np.int32),
        finished=_copy_array(state.finished, bool),
        newly_finished=_copy_array(state.newly_finished, bool),
        risk_p=_copy_array(state.risk_p, np.float32),
        min_neighbor_dist=_copy_array(state.min_neighbor_dist, np.float32),
        last_action=_copy_array(state.last_action, np.float32),
        walls=_copy_array(state.walls, np.float32),
        oracle_dir=_copy_optional_array(state.oracle_dir, np.float32),
        static_walls=list(state.static_walls),
        static_circles=list(state.static_circles),
        collision_speed=_copy_array(state.collision_speed, np.float32),
        measured_accel=_copy_array(state.measured_accel, np.float32),
        energy=_copy_array(state.energy, np.float32),
        energy_level=_copy_array(state.energy_level, np.float32),
        agent_state=_copy_array(state.agent_state, np.int8),
        control_mode=str(state.control_mode),
        field_size=float(state.field_size),
        max_speed=float(state.max_speed),
        max_accel=float(state.max_accel),
        max_thrust=float(state.max_thrust),
        mass=float(state.mass),
        drag_coeff=float(state.drag_coeff),
        dt=float(state.dt),
        drag=float(state.drag),
        grid_res=float(state.grid_res),
        agent_radius=float(state.agent_radius),
        wall_friction=float(state.wall_friction),
        decision_step=int(state.decision_step),
        oracle_dist=_copy_optional_array(state.oracle_dist, np.float32),
        oracle_risk=_copy_optional_array(state.oracle_risk, np.float32),
        oracle_risk_grad=_copy_optional_array(state.oracle_risk_grad, np.float32),
    )


def public_state_from_state(state: SimState, *, include_oracle: bool = True) -> PublicState:
    oracle_dir = state.oracle_dir if include_oracle else None
    oracle_dist = state.oracle_dist if include_oracle else None
    oracle_risk = state.oracle_risk if include_oracle else None
    oracle_risk_grad = state.oracle_risk_grad if include_oracle else None
    return PublicState(
        pos=_copy_array(state.pos, np.float32),
        vel=_copy_array(state.vel, np.float32),
        alive=_copy_array(state.alive, bool),
        agent_state=_copy_array(state.agent_state, np.int8),
        target_pos=_copy_array(state.target_pos, np.float32),
        target_vel=_copy_array(state.target_vel, np.float32),
        threats=_copy_threats(list(state.threats)),
        dists=_copy_array(state.dists, np.float32),
        in_goal=_copy_array(state.in_goal, bool),
        risk_p=_copy_array(state.risk_p, np.float32),
        oracle_dir=_copy_optional_array(oracle_dir, np.float32),
        static_walls=list(state.static_walls),
        static_circles=list(state.static_circles),
        field_size=float(state.field_size),
        max_speed=float(state.max_speed),
        max_accel=float(state.max_accel),
        max_thrust=float(state.max_thrust),
        mass=float(state.mass),
        drag_coeff=float(state.drag_coeff),
        dt=float(state.dt),
        drag=float(state.drag),
        grid_res=float(state.grid_res),
        agent_radius=float(state.agent_radius),
        wall_friction=float(state.wall_friction),
        energy_level=_copy_array(state.energy_level, np.float32),
        measured_accel=_copy_array(state.measured_accel, np.float32),
        timestep=int(state.timestep),
        decision_step=int(state.decision_step),
        control_mode=str(state.control_mode),
        oracle_dist=_copy_optional_array(oracle_dist, np.float32),
        oracle_risk=_copy_optional_array(oracle_risk, np.float32),
        oracle_risk_grad=_copy_optional_array(oracle_risk_grad, np.float32),
    )


@dataclass
class RenderState:
    pos: np.ndarray
    vel: np.ndarray
    alive: np.ndarray
    target_pos: np.ndarray
    target_vel: np.ndarray
    timestep: int
    threats: list[Any]
    static_walls: list[tuple[float, float, float, float]]
    field_size: float
    goal_radius: float
    in_goal: np.ndarray | None = None
    finished: np.ndarray | None = None
