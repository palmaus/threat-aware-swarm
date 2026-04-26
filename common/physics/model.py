from __future__ import annotations

import math

import numpy as np

try:  # Опциональная зависимость.
    from numba import njit
except Exception:  # pragma: no cover - опциональная
    njit = None


def compute_drag_factor(drag_coeff: float, mass: float, dt: float) -> float:
    """Переводит коэффициент сопротивления в dt-масштабированную форму."""
    return float(drag_coeff) / max(float(mass), 1e-6) * float(dt)


def apply_accel_dynamics(
    pos: np.ndarray,
    vel: np.ndarray,
    accel: np.ndarray,
    dt: float,
    *,
    drag: float = 0.0,
    max_speed: float = 0.0,
    wind: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    pos = np.asarray(pos, dtype=np.float32)
    vel = np.asarray(vel, dtype=np.float32)
    accel = np.asarray(accel, dtype=np.float32)
    if wind is None:
        wind = np.zeros_like(vel, dtype=np.float32)
    else:
        wind = np.asarray(wind, dtype=np.float32)
        if wind.shape != vel.shape:
            wind = np.zeros_like(vel, dtype=np.float32)

    drag = float(max(0.0, drag))
    # Сопротивление считается по относительной скорости к ветру.
    rel_vel = vel - wind
    next_vel = vel + (accel * float(dt)) - (drag * rel_vel)
    if max_speed > 0.0:
        speeds = np.linalg.norm(next_vel, axis=1, keepdims=True)
        scale = np.where(speeds > max_speed, max_speed / (speeds + 1e-8), 1.0)
        next_vel = next_vel * scale
    next_pos = pos + (next_vel * float(dt))
    return next_pos.astype(np.float32), next_vel.astype(np.float32)


def apply_accel_dynamics_vel(
    vel: np.ndarray,
    accel: np.ndarray,
    dt: float,
    *,
    drag: float = 0.0,
    max_speed: float = 0.0,
    wind: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    out: np.ndarray | None = None,
) -> np.ndarray:
    vel = np.asarray(vel, dtype=np.float32)
    accel = np.asarray(accel, dtype=np.float32)
    if wind is None:
        wind = None
    else:
        wind = np.asarray(wind, dtype=np.float32)
        if wind.shape != vel.shape:
            wind = None
    if mask is None:
        mask = None
    else:
        mask = np.asarray(mask, dtype=bool)

    if out is None:
        next_vel = vel.copy()
    else:
        next_vel = out
        np.copyto(next_vel, vel)
    drag = float(max(0.0, drag))
    dt = float(dt)

    if mask is None:
        if drag <= 0.0:
            next_vel = vel + (accel * dt)
        else:
            if wind is None:
                rel_vel = vel
            else:
                rel_vel = vel - wind
            next_vel = vel + (accel * dt) - (drag * rel_vel)
    else:
        if np.any(mask):
            if drag <= 0.0:
                next_vel[mask] = vel[mask] + (accel[mask] * dt)
            else:
                if wind is None:
                    rel_vel = vel[mask]
                else:
                    rel_vel = vel[mask] - wind[mask]
                next_vel[mask] = vel[mask] + (accel[mask] * dt) - (drag * rel_vel)

    if max_speed > 0.0:
        max_speed = float(max_speed)
        max_speed2 = max_speed * max_speed
        speed2 = np.sum(next_vel * next_vel, axis=1)
        over = speed2 > max_speed2
        if np.any(over):
            scale = (max_speed / (np.sqrt(speed2[over]) + 1e-8)).astype(np.float32)
            next_vel[over] = next_vel[over] * scale[:, None]
    return next_vel.astype(np.float32)


def inverse_accel_for_velocity(
    desired_vel: np.ndarray,
    cur_vel: np.ndarray,
    *,
    dt: float,
    drag: float = 0.0,
    wind: np.ndarray | None = None,
) -> np.ndarray:
    desired_vel = np.asarray(desired_vel, dtype=np.float32)
    cur_vel = np.asarray(cur_vel, dtype=np.float32)
    if wind is None:
        wind = np.zeros_like(cur_vel, dtype=np.float32)
    else:
        wind = np.asarray(wind, dtype=np.float32)
        if wind.shape != cur_vel.shape:
            wind = np.zeros_like(cur_vel, dtype=np.float32)
    denom = max(float(dt), 1e-6)
    accel = (desired_vel - cur_vel) / denom
    if drag > 0.0:
        rel_vel = cur_vel - wind
        accel = accel + (float(drag) / denom) * rel_vel
    return accel.astype(np.float32)


def _apply_accel_dynamics_step_py(
    vx: float,
    vy: float,
    ax: float,
    ay: float,
    dt: float,
    drag: float,
    max_speed: float,
    wind_x: float,
    wind_y: float,
) -> tuple[float, float]:
    drag = float(max(0.0, drag))
    rel_vx = vx - float(wind_x)
    rel_vy = vy - float(wind_y)
    vx = float(vx) + (float(ax) * float(dt)) - (drag * rel_vx)
    vy = float(vy) + (float(ay) * float(dt)) - (drag * rel_vy)
    if max_speed > 0.0:
        speed = math.sqrt(vx * vx + vy * vy)
        if speed > max_speed:
            scale = max_speed / (speed + 1e-8)
            vx *= scale
            vy *= scale
    return vx, vy


if njit is not None:
    apply_accel_dynamics_step = njit(cache=True, fastmath=True)(_apply_accel_dynamics_step_py)
else:  # pragma: no cover - опциональная зависимость
    apply_accel_dynamics_step = _apply_accel_dynamics_step_py
