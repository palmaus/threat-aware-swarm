from __future__ import annotations

import numpy as np

from baselines.utils import obs_to_target, velocity_tracking_action_batch


def extract_plan_action(plan_out) -> np.ndarray:
    if isinstance(plan_out, tuple) and plan_out:
        plan_out = plan_out[0]
    return np.asarray(plan_out, dtype=np.float32)


def inside_margin(pos: np.ndarray, threats: list, margin: float) -> bool:
    if not threats:
        return False
    for t in threats:
        if float(np.linalg.norm(pos - t.position)) <= float(t.radius) + float(margin):
            return True
    return False


def inside_any(pos: np.ndarray, threats: list) -> bool:
    if not threats:
        return False
    for t in threats:
        if float(np.linalg.norm(pos - t.position)) <= float(t.radius):
            return True
    return False


def wall_penalty(pos: np.ndarray, field_size: float) -> float:
    size = float(field_size)
    d_left = pos[0] / size
    d_right = (size - pos[0]) / size
    d_down = pos[1] / size
    d_up = (size - pos[1]) / size
    min_wall = float(np.min([d_left, d_right, d_down, d_up]))
    if min_wall < 0.05:
        return float(0.05 - min_wall)
    return 0.0


def near_agents_penalty(idx: int, pos: np.ndarray, alive: np.ndarray) -> float:
    if pos.shape[0] <= 1 or not bool(alive[idx]):
        return 0.0
    others = []
    for j in range(pos.shape[0]):
        if j == idx or not bool(alive[j]):
            continue
        others.append(pos[j])
    if not others:
        return 0.0
    others_arr = np.stack(others, axis=0)
    dists = np.linalg.norm(others_arr - pos[idx], axis=1)
    min_dist = float(np.min(dists))
    if min_dist < 1.5:
        return float(1.5 - min_dist)
    return 0.0


def dist_from_info(info: dict, obs: dict | None) -> float | None:
    if "dist" in info:
        try:
            return float(info["dist"])
        except Exception:
            pass
    pos = info.get("pos")
    target = info.get("target_pos")
    if pos is not None and target is not None:
        try:
            return float(np.linalg.norm(np.asarray(target, dtype=np.float32) - np.asarray(pos, dtype=np.float32)))
        except Exception:
            pass
    if obs is not None:
        try:
            return float(np.linalg.norm(obs_to_target(obs)))
        except Exception:
            pass
    return None


def finalize_batch_waypoint_actions(
    *,
    agent_ids: list[str],
    desired: np.ndarray,
    to_target: np.ndarray,
    dist_m: np.ndarray,
    in_goal: np.ndarray,
    risk_p: np.ndarray,
    cur_vel: np.ndarray,
    obs_list: list[dict],
    info_list: list[dict],
    state,
    controller,
    stop_risk_threshold: float,
) -> dict[str, np.ndarray]:
    if controller is None:
        return {agent_id: np.asarray(desired[i], dtype=np.float32) for i, agent_id in enumerate(agent_ids)}
    desired_actions = controller.compute_batch(
        desired,
        to_target,
        dist_m,
        False,
        float(getattr(state, "field_size", 0.0)),
        in_goal=in_goal,
        risk_p=np.nan_to_num(risk_p, nan=0.0),
        stop_risk_threshold=float(stop_risk_threshold),
        vel=cur_vel,
    )
    accel_actions = velocity_tracking_action_batch(
        desired_actions,
        cur_vel,
        max_speed=float(getattr(state, "max_speed", 0.0)),
        max_accel=float(getattr(state, "max_accel", 0.0)),
        dt=float(getattr(state, "dt", 0.0)),
        drag=float(getattr(state, "drag", 0.0)),
        obs_list=obs_list,
        infos=info_list,
    )
    return {agent_id: accel_actions[i] for i, agent_id in enumerate(agent_ids)}
