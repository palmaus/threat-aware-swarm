"""Shared waypoint-to-control adapter used by env wrappers and baselines."""

from __future__ import annotations

import logging
import math
import os

import numpy as np

from common.physics.model import inverse_accel_for_velocity
from common.policy.obs_schema import normalize, obs_grid, obs_to_target, obs_vel, obs_walls

logger = logging.getLogger(__name__)


def _clip_action(vec: np.ndarray, max_mag: float = 1.0) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32)
    mag = float(np.linalg.norm(arr))
    if max_mag > 0.0 and mag > max_mag:
        arr = arr / mag * float(max_mag)
    return np.clip(arr, -1.0, 1.0).astype(np.float32)


def _wall_distance_along_dir(direction: np.ndarray, walls: np.ndarray) -> float | None:
    if walls is None or walls.size < 4:
        return None
    left, right, down, up = [float(x) for x in walls[:4]]
    ux, uy = float(direction[0]), float(direction[1])
    if abs(ux) <= 1e-6 and abs(uy) <= 1e-6:
        return None
    t_x = float("inf")
    t_y = float("inf")
    if abs(ux) > 1e-6:
        dist_x = right if ux > 0.0 else left
        t_x = dist_x / max(abs(ux), 1e-6)
    if abs(uy) > 1e-6:
        dist_y = up if uy > 0.0 else down
        t_y = dist_y / max(abs(uy), 1e-6)
    return float(min(t_x, t_y))


def _grid_wall_distance_along_dir(
    grid: np.ndarray,
    direction: np.ndarray,
    *,
    threshold: float,
    max_cells: int,
) -> float | None:
    if grid is None or grid.size == 0:
        return None
    w = grid.shape[0]
    if w <= 1:
        return None
    center = int(w // 2)
    dx = float(direction[0])
    dy = float(direction[1])
    if abs(dx) <= 1e-6 and abs(dy) <= 1e-6:
        return None
    for step in range(1, int(max_cells) + 1):
        x = round(center + dx * step)
        y = round(center + dy * step)
        if x < 0 or y < 0 or x >= w or y >= w:
            break
        if float(grid[y, x]) >= float(threshold):
            return float(step)
    return None


def _apply_brake_speed_cap(
    desired_vel: np.ndarray,
    obs: dict[str, np.ndarray],
    *,
    max_accel: float,
    brake_gain: float = 1.0,
) -> np.ndarray:
    speed = float(np.linalg.norm(desired_vel))
    if speed <= 1e-6 or max_accel <= 0.0:
        return desired_vel
    walls = obs_walls(obs)
    dist_dir = _wall_distance_along_dir(desired_vel / speed, walls)
    if dist_dir is None or not math.isfinite(dist_dir):
        return desired_vel
    brake_dist = (speed * speed) / (2.0 * max_accel)
    if brake_dist <= dist_dir:
        return desired_vel
    safe_speed = math.sqrt(max(0.0, 2.0 * max_accel * dist_dir)) * float(max(0.0, brake_gain))
    if safe_speed <= 1e-6:
        return np.zeros((2,), dtype=np.float32)
    scale = min(1.0, safe_speed / max(speed, 1e-6))
    return np.asarray(desired_vel, dtype=np.float32) * scale


def velocity_tracking_action(
    desired_action: np.ndarray,
    obs: dict[str, np.ndarray],
    info: dict,
    *,
    tau: float = 0.35,
    brake_gain: float = 1.0,
) -> np.ndarray:
    max_speed = float(info.get("max_speed", 0.0))
    max_accel = float(info.get("max_accel", 0.0))
    dt = float(info.get("dt", 0.0))
    drag = float(info.get("drag", 0.0))
    if max_speed <= 0.0 or max_accel <= 0.0 or dt <= 0.0:
        return np.asarray(desired_action, dtype=np.float32)
    tau = max(float(info.get("accel_tau", tau)), 1e-3)
    desired_vel = np.asarray(desired_action, dtype=np.float32) * max_speed
    desired_vel = _apply_brake_speed_cap(
        desired_vel,
        obs,
        max_accel=max_accel,
        brake_gain=float(info.get("brake_gain", brake_gain)),
    )
    desired_speed = float(np.linalg.norm(desired_vel))
    if desired_speed <= 1e-6:
        return np.zeros((2,), dtype=np.float32)

    cur_vel = info.get("cur_vel")
    cur_vel = obs_vel(obs) * max_speed if cur_vel is None else np.asarray(cur_vel, dtype=np.float32)
    speed = float(np.linalg.norm(cur_vel))
    grid = obs_grid(obs)
    if grid is not None and speed > 1e-6:
        grid_res = float(info.get("grid_res", 1.0))
        threshold = float(info.get("grid_wall_threshold", 0.85))
        lookahead = math.ceil((speed * speed) / (2.0 * max_accel * max(grid_res, 1e-6))) + 1
        lookahead = max(1, min(lookahead, int(grid.shape[0])))
        direction = desired_vel / max(desired_speed, 1e-6)
        wall_cells = _grid_wall_distance_along_dir(grid, direction, threshold=threshold, max_cells=lookahead)
        if wall_cells is not None:
            wall_dist = float(wall_cells) * max(grid_res, 1e-6)
            brake_dist = (speed * speed) / (2.0 * max_accel)
            if wall_dist < brake_dist:
                safe_speed = math.sqrt(max(0.0, 2.0 * max_accel * wall_dist))
                if safe_speed <= 1e-6:
                    desired_vel = np.zeros((2,), dtype=np.float32)
                else:
                    scale = min(1.0, safe_speed / max(speed, 1e-6))
                    desired_vel = desired_vel * scale

    accel = inverse_accel_for_velocity(desired_vel, cur_vel, dt=tau, drag=drag)
    accel = np.clip(accel, -max_accel, max_accel)
    return np.clip(accel / max_accel, -1.0, 1.0).astype(np.float32)


def velocity_tracking_action_batch(
    desired_actions: np.ndarray,
    cur_vel: np.ndarray,
    *,
    max_speed: float,
    max_accel: float,
    dt: float,
    drag: float = 0.0,
    tau: float = 0.35,
    obs_list: list[dict[str, np.ndarray]] | None = None,
    infos: list[dict] | None = None,
) -> np.ndarray:
    actions = np.asarray(desired_actions, dtype=np.float32)
    vel = np.asarray(cur_vel, dtype=np.float32)
    if max_speed <= 0.0 or max_accel <= 0.0 or dt <= 0.0:
        return actions.astype(np.float32)
    tau = max(float(tau), 1e-3)
    desired_vel = actions * float(max_speed)
    if obs_list is not None:
        info_list = infos or [None] * len(obs_list)
        for i, obs in enumerate(obs_list):
            if obs is None:
                continue
            info = info_list[i] or {}
            desired_vel[i] = _apply_brake_speed_cap(
                desired_vel[i],
                obs,
                max_accel=max_accel,
                brake_gain=float(info.get("brake_gain", 1.0)),
            )
            speed2 = float(desired_vel[i, 0] * desired_vel[i, 0] + desired_vel[i, 1] * desired_vel[i, 1])
            if speed2 <= 1e-12:
                continue
            speed = math.sqrt(speed2)
            grid = obs_grid(obs)
            if grid is None:
                continue
            grid_res = float(info.get("grid_res", 1.0))
            threshold = float(info.get("grid_wall_threshold", 0.85))
            lookahead = math.ceil((speed * speed) / (2.0 * max_accel * max(grid_res, 1e-6))) + 1
            lookahead = max(1, min(lookahead, int(grid.shape[0])))
            direction = desired_vel[i] / max(speed, 1e-6)
            wall_cells = _grid_wall_distance_along_dir(grid, direction, threshold=threshold, max_cells=lookahead)
            if wall_cells is None:
                continue
            wall_dist = float(wall_cells) * max(grid_res, 1e-6)
            brake_dist = (speed * speed) / (2.0 * max_accel)
            if wall_dist < brake_dist:
                safe_speed = math.sqrt(max(0.0, 2.0 * max_accel * wall_dist))
                if safe_speed <= 1e-6:
                    desired_vel[i] = np.zeros((2,), dtype=np.float32)
                else:
                    scale = min(1.0, safe_speed / max(speed, 1e-6))
                    desired_vel[i] = desired_vel[i] * scale
    accel = inverse_accel_for_velocity(desired_vel, vel, dt=tau, drag=drag)
    accel = np.clip(accel, -max_accel, max_accel)
    return np.clip(accel / max_accel, -1.0, 1.0).astype(np.float32)


def apply_control_mode(action: np.ndarray, obs: dict[str, np.ndarray], info: dict | None) -> np.ndarray:
    if not info:
        return np.asarray(action, dtype=np.float32)
    try:
        return velocity_tracking_action(action, obs, info)
    except Exception as exc:
        if os.getenv("TA_STRICT_CONTROL") == "1":
            raise
        logger.warning("Control adapter unavailable, returning waypoint action: %s", exc)
        return np.asarray(action, dtype=np.float32)


def maybe_accel_action(action: np.ndarray, obs: dict[str, np.ndarray], info: dict | None) -> np.ndarray:
    return apply_control_mode(action, obs, info)


class WaypointController:
    """Converts a desired waypoint vector into the current low-level control action."""

    def __init__(
        self,
        *,
        goal_radius_control: float,
        near_goal_speed_cap: float,
        near_goal_damping: float,
        near_goal_kp: float,
        risk_speed_scale: float,
        risk_speed_floor: float,
    ) -> None:
        self.goal_radius_control = float(goal_radius_control)
        self.near_goal_speed_cap = float(near_goal_speed_cap)
        self.near_goal_damping = float(near_goal_damping)
        self.near_goal_kp = float(near_goal_kp)
        self.risk_speed_scale = float(risk_speed_scale)
        self.risk_speed_floor = float(risk_speed_floor)

    def should_stop(self, in_goal: bool, risk_p: float, stop_risk_threshold: float) -> bool:
        return bool(in_goal and (risk_p <= stop_risk_threshold))

    def near_goal_active(self, dist_m: float | None, dist_normed: bool, field_size: float | None) -> bool:
        if dist_m is None:
            return False
        if dist_normed and field_size is None:
            return dist_m <= self.goal_radius_control
        return dist_m <= self.goal_radius_control

    def compute(
        self,
        desired_vec: np.ndarray,
        to_target: np.ndarray,
        obs: np.ndarray,
        dist_m: float | None,
        dist_normed: bool,
        field_size: float | None,
        *,
        in_goal: bool,
        risk_p: float,
        stop_risk_threshold: float,
    ) -> np.ndarray:
        if self.should_stop(in_goal, risk_p, stop_risk_threshold):
            return np.zeros((2,), dtype=np.float32)
        action = normalize(desired_vec)
        if self.near_goal_active(dist_m, dist_normed, field_size):
            action = self._near_goal_action(to_target, obs, dist_m)
        action = self._apply_risk_speed(action, risk_p)
        return _clip_action(action, 1.0)

    def compute_batch(
        self,
        desired_vecs: np.ndarray,
        to_target: np.ndarray,
        dist_m: np.ndarray | None,
        dist_normed: bool | np.ndarray,
        field_size: float | None,
        *,
        in_goal: np.ndarray,
        risk_p: np.ndarray,
        stop_risk_threshold: float,
        vel: np.ndarray | None = None,
    ) -> np.ndarray:
        desired = np.asarray(desired_vecs, dtype=np.float32)
        if desired.ndim != 2 or desired.shape[1] != 2:
            desired = np.reshape(desired, (-1, 2))
        n = desired.shape[0]
        actions = desired.copy()
        mags = np.linalg.norm(actions, axis=1)
        mask = mags > 1e-6
        if np.any(mask):
            actions[mask] = actions[mask] / mags[mask, None]
        actions = np.clip(actions, -1.0, 1.0)

        in_goal_arr = np.asarray(in_goal, dtype=bool).reshape(n)
        risk_arr = np.asarray(risk_p, dtype=np.float32).reshape(n)
        stop_mask = in_goal_arr & (risk_arr <= float(stop_risk_threshold))
        if np.any(stop_mask):
            actions[stop_mask] = 0.0

        dist_arr = None if dist_m is None else np.asarray(dist_m, dtype=np.float32).reshape(n)
        if dist_arr is not None:
            near_mask = dist_arr <= float(self.goal_radius_control)
            if np.any(near_mask):
                to_target_arr = np.asarray(to_target, dtype=np.float32).reshape(n, 2)
                t_norm = np.linalg.norm(to_target_arr, axis=1)
                safe_norm = np.where(t_norm > 1e-6, t_norm, 1.0)
                to_dir = to_target_arr / safe_norm[:, None]
                radius = max(float(self.goal_radius_control), 1e-3)
                speed_cap = float(np.clip(self.near_goal_speed_cap, 0.0, 1.0))
                speed = np.minimum(speed_cap, float(self.near_goal_kp) * (dist_arr / radius))
                desired_ng = to_dir * speed[:, None]
                if vel is not None:
                    vel_arr = np.asarray(vel, dtype=np.float32).reshape(n, 2)
                    desired_ng = desired_ng - (float(self.near_goal_damping) * vel_arr)
                mags_ng = np.linalg.norm(desired_ng, axis=1)
                over = mags_ng > speed_cap
                if np.any(over):
                    desired_ng[over] *= (speed_cap / (mags_ng[over] + 1e-6))[:, None]
                actions[near_mask] = np.clip(desired_ng[near_mask], -1.0, 1.0)

        if self.risk_speed_scale > 0.0:
            scale = 1.0 - (float(self.risk_speed_scale) * np.clip(risk_arr, 0.0, 1.0))
            actions = actions * np.maximum(scale, float(self.risk_speed_floor))[:, None]
        return np.clip(actions, -1.0, 1.0).astype(np.float32)

    def compute_action(
        self,
        desired_vec: np.ndarray,
        obs: np.ndarray,
        dist_m: float | None,
        dist_normed: bool,
        field_size: float | None,
        *,
        in_goal: bool,
        risk_p: float,
        stop_risk_threshold: float,
        info: dict | None,
        to_target: np.ndarray | None = None,
    ) -> np.ndarray:
        if to_target is None:
            to_target = obs_to_target(obs)
        action = self.compute(
            desired_vec,
            to_target,
            obs,
            dist_m,
            dist_normed,
            field_size,
            in_goal=in_goal,
            risk_p=risk_p,
            stop_risk_threshold=stop_risk_threshold,
        )
        return apply_control_mode(action, obs, info)

    def _near_goal_action(self, to_target: np.ndarray, obs: np.ndarray, dist_m: float | None) -> np.ndarray:
        if dist_m is None:
            return normalize(to_target)
        radius = max(self.goal_radius_control, 1e-3)
        speed_cap = float(np.clip(self.near_goal_speed_cap, 0.0, 1.0))
        speed = min(speed_cap, self.near_goal_kp * (dist_m / radius))
        desired = normalize(to_target) * speed
        action = desired - (self.near_goal_damping * obs_vel(obs))
        return _clip_action(action, speed_cap)

    def _apply_risk_speed(self, action: np.ndarray, risk_p: float) -> np.ndarray:
        if self.risk_speed_scale <= 0.0:
            return action
        scale = 1.0 - (self.risk_speed_scale * float(np.clip(risk_p, 0.0, 1.0)))
        scale = max(scale, float(self.risk_speed_floor))
        return np.asarray(action, dtype=np.float32) * scale
