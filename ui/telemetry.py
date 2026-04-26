"""Формирование безопасной телеметрии для веб‑клиента."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from common.policy.obs_schema import obs_vector_to_fields
from ui.telemetry_dto import TelemetryAgent, TelemetryPayload, TelemetryStats, TelemetryThreat


def _json_float(val: Any) -> float | None:
    # JSON не принимает NaN/Inf, поэтому возвращаем None вместо некорректных чисел.
    try:
        out = float(val)
    except Exception:
        return None
    if not np.isfinite(out):
        return None
    return out


TELEMETRY_SCHEMA_VERSION = "telemetry@v2"


def build_stats(state: Any, last_infos: dict | None) -> TelemetryStats:
    # Значения по умолчанию нужны, чтобы фронт не падал при пустых info.
    stats = TelemetryStats(
        step=int(getattr(state, "timestep", 0)),
        alive=None,
        finished=None,
        in_goal=None,
        mean_dist=None,
        mean_risk=None,
        mean_path_ratio=None,
        mean_threat_collisions=None,
        mean_energy=None,
        mean_energy_level=None,
    )
    if last_infos:
        alive = [inf.get("alive", 0.0) for inf in last_infos.values()]
        finished = [inf.get("finished", 0.0) for inf in last_infos.values()]
        in_goal = [inf.get("in_goal", 0.0) for inf in last_infos.values()]
        dist = [inf.get("dist", float("nan")) for inf in last_infos.values()]
        risk = [inf.get("risk_p", float("nan")) for inf in last_infos.values()]
        path_ratio = [inf.get("path_ratio", float("nan")) for inf in last_infos.values()]
        threat_col = [inf.get("threat_collisions", float("nan")) for inf in last_infos.values()]
        energy = [inf.get("energy", float("nan")) for inf in last_infos.values()]
        energy_level = [inf.get("energy_level", float("nan")) for inf in last_infos.values()]
        stats.alive = int(np.nansum(alive))
        stats.finished = int(np.nansum(finished))
        stats.in_goal = int(np.nansum(in_goal))
        stats.mean_dist = _json_float(np.nanmean(np.asarray(dist, dtype=np.float32)))
        stats.mean_risk = _json_float(np.nanmean(np.asarray(risk, dtype=np.float32)))
        stats.mean_path_ratio = _json_float(np.nanmean(np.asarray(path_ratio, dtype=np.float32)))
        stats.mean_threat_collisions = _json_float(np.nanmean(np.asarray(threat_col, dtype=np.float32)))
        stats.mean_energy = _json_float(np.nanmean(np.asarray(energy, dtype=np.float32)))
        stats.mean_energy_level = _json_float(np.nanmean(np.asarray(energy_level, dtype=np.float32)))
    return stats


def build_agents(
    state: Any,
    agents: list[str],
    last_infos: dict | None,
    last_actions: dict | None,
) -> list[TelemetryAgent]:
    out: list[TelemetryAgent] = []
    pos = np.asarray(getattr(state, "pos", np.zeros((0, 2))), dtype=np.float32)
    vel = np.asarray(getattr(state, "vel", np.zeros((0, 2))), dtype=np.float32)
    alive = np.asarray(getattr(state, "alive", np.zeros((0,), dtype=bool)))
    target = getattr(state, "target_pos", None)
    for i, agent_id in enumerate(agents):
        inf = last_infos.get(agent_id, {}) if last_infos else {}
        if i < pos.shape[0]:
            pos_i = pos[i]
        else:
            pos_i = np.zeros((2,), dtype=np.float32)
        if i < vel.shape[0]:
            vel_i = vel[i]
        else:
            vel_i = np.zeros((2,), dtype=np.float32)
        alive_i = bool(alive[i]) if i < alive.shape[0] else False
        dist = _json_float(np.linalg.norm(pos_i - target)) if target is not None else None
        act = None
        if last_actions is not None and agent_id in last_actions:
            try:
                act_arr = np.asarray(last_actions[agent_id], dtype=np.float32).flatten()
                if act_arr.size >= 2:
                    act = [float(act_arr[0]), float(act_arr[1])]
            except Exception:
                act = None
        out.append(
            TelemetryAgent(
                id=agent_id,
                index=i,
                pos=[float(pos_i[0]), float(pos_i[1])],
                vel=[float(vel_i[0]), float(vel_i[1])],
                alive=bool(alive_i),
                finished=bool(inf.get("finished", False)),
                in_goal=bool(inf.get("in_goal", False)),
                dist=_json_float(inf.get("dist", dist)),
                risk_p=_json_float(inf.get("risk_p", None)),
                path_ratio=_json_float(inf.get("path_ratio", None)),
                collided=bool(inf.get("collided", False)),
                threat_collided=bool(inf.get("threat_collided", False)),
                min_dist_to_threat=_json_float(inf.get("min_dist_to_threat", None)),
                energy=_json_float(inf.get("energy", None)),
                energy_level=_json_float(inf.get("energy_level", None)),
                action=act,
            )
        )
    return out


def build_threats(state: Any) -> list[TelemetryThreat]:
    out: list[TelemetryThreat] = []
    for t in getattr(state, "threats", []) or []:
        pos = getattr(t, "position", [0.0, 0.0])
        out.append(
            TelemetryThreat(
                pos=[float(pos[0]), float(pos[1])],
                radius=float(getattr(t, "radius", 0.0)),
                intensity=float(getattr(t, "intensity", 0.0)),
                kind=str(getattr(t, "kind", "static")),
                dynamic=bool(getattr(t, "is_dynamic", False)),
                oracle_block=bool(getattr(t, "oracle_block", False)),
            )
        )
    return out


def build_payload(
    state: Any,
    last_infos: dict | None,
    *,
    screen_size: int,
    agents: list[str],
    agent_idx: int = 0,
    last_actions: dict | None = None,
    field_size: float = 0.0,
    goal_radius: float = 0.0,
    grid_res: float = 1.0,
    oracle_path: list[list[float]] | None = None,
    obs_provider: Callable[[int], dict[str, Any] | None] | None = None,
    wind: list[float] | None = None,
) -> TelemetryPayload:
    agent_grid = None
    agent_obs = None
    agent_id = None
    if agents:
        try:
            agent_idx = int(np.clip(agent_idx, 0, len(agents) - 1))
        except Exception:
            agent_idx = 0
        agent_id = agents[agent_idx]
        # Извлекаем локальную сетку только для выбранного агента, чтобы не раздувать трафик.
        try:
            obs = obs_provider(agent_idx) if obs_provider is not None else None
            if isinstance(obs, dict):
                grid = obs.get("grid", None)
                vec = obs.get("vector", None)
                if grid is not None:
                    grid_arr = np.asarray(grid, dtype=np.float32)
                    if grid_arr.ndim == 3:
                        grid_arr = grid_arr[0]
                    if grid_arr.ndim == 2:
                        agent_grid = grid_arr.tolist()
                if vec is not None:
                    agent_obs = obs_vector_to_fields(vec)
        except Exception:
            agent_grid = None
            agent_obs = None

    target_pos = getattr(state, "target_pos", None)
    target_val = None
    if target_pos is not None:
        try:
            target_val = [float(target_pos[0]), float(target_pos[1])]
        except Exception:
            target_val = None

    return TelemetryPayload(
        schema_version=TELEMETRY_SCHEMA_VERSION,
        stats=build_stats(state, last_infos),
        agents=build_agents(state, agents, last_infos, last_actions),
        threats=build_threats(state),
        walls=[list(w) for w in getattr(state, "static_walls", [])],
        oracle_path=[list(p) for p in (oracle_path or [])],
        field_size=float(field_size),
        goal_radius=float(goal_radius),
        target_pos=target_val,
        screen_size=int(screen_size),
        grid_res=float(grid_res),
        agent_index=agent_idx,
        agent_id=agent_id,
        agent_grid=agent_grid,
        agent_obs=agent_obs,
        wind=wind,
    )
