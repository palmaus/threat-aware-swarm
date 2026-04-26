"""Runtime tunables adapter for the web UI."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _maybe_float(params: Mapping[str, Any], key: str) -> float | None:
    if key not in params:
        return None
    try:
        return float(params[key])
    except Exception:
        return None


def apply_tunables(env, params: Mapping[str, Any]) -> None:
    # UI keeps a stable schema, while runtime may derive equivalent physical params.
    wall_friction = _maybe_float(params, "wall_friction")
    if wall_friction is not None:
        env.config.wall_friction = wall_friction

    for reward_key in ("w_risk", "w_progress", "w_wall"):
        value = _maybe_float(params, reward_key)
        if value is None:
            continue
        try:
            setattr(env.reward, reward_key, value)
        except Exception:
            continue

    threat_speed = _maybe_float(params, "threat_speed")
    if threat_speed is not None:
        env.set_threat_speed_scale(threat_speed)

    if "threat_mode" in params:
        try:
            mode = str(params["threat_mode"]).lower()
        except Exception:
            mode = ""
        if mode in {"static", "dynamic", "mixed"}:
            env.config.random_threat_mode = mode

    max_accel = _maybe_float(params, "max_accel")
    if max_accel is not None:
        mass = float(getattr(env.config.physics, "mass", 1.0))
        env.set_runtime_defaults(max_thrust=max_accel * mass)

    max_thrust = _maybe_float(params, "max_thrust")
    if max_thrust is not None:
        env.set_runtime_defaults(max_thrust=max_thrust)

    mass = _maybe_float(params, "mass")
    if mass is not None:
        env.set_runtime_defaults(mass=mass)

    max_speed = _maybe_float(params, "max_speed")
    if max_speed is not None:
        env.set_runtime_defaults(max_speed=max_speed)

    drag = _maybe_float(params, "drag")
    if drag is not None:
        mass_value = float(getattr(env.config.physics, "mass", 1.0))
        dt = float(getattr(env.config, "dt", 0.0))
        env.set_runtime_defaults(drag_coeff=drag * mass_value / max(dt, 1.0e-6))

    drag_coeff = _maybe_float(params, "drag_coeff")
    if drag_coeff is not None:
        env.set_runtime_defaults(drag_coeff=drag_coeff)

    noise_keys = ("obs_noise_target", "obs_noise_vel", "obs_noise_grid")
    if any(key in params for key in noise_keys):
        for noise_key in noise_keys:
            value = _maybe_float(params, noise_key)
            if value is None:
                continue
            setattr(env.config, noise_key, value)
        env.obs_builder.set_noise(
            obs_noise_target=float(getattr(env.config, "obs_noise_target", 0.0)),
            obs_noise_vel=float(getattr(env.config, "obs_noise_vel", 0.0)),
            obs_noise_grid=float(getattr(env.config, "obs_noise_grid", 0.0)),
        )

    if "domain_randomization" in params:
        try:
            env.config.domain_randomization = bool(params["domain_randomization"])
        except Exception:
            pass

    for key in ("dr_max_speed_min", "dr_max_speed_max", "dr_drag_min", "dr_drag_max"):
        value = _maybe_float(params, key)
        if value is None:
            continue
        setattr(env.config, key, value)


def collect_tunables(env) -> dict[str, Any]:
    mass = float(getattr(env.config.physics, "mass", 1.0))
    max_thrust = float(getattr(env.config.physics, "max_thrust", 0.0))
    drag_coeff = float(getattr(env.config.physics, "drag_coeff", 0.0))
    max_accel = max_thrust / max(mass, 1.0e-6)
    drag = (drag_coeff / max(mass, 1.0e-6)) * float(getattr(env.config, "dt", 0.0))
    return {
        "wall_friction": float(env.config.wall_friction),
        "w_risk": float(env.reward.w_risk),
        "w_progress": float(env.reward.w_progress),
        "w_wall": float(env.reward.w_wall),
        "threat_speed": float(env.get_threat_speed_scale()),
        "threat_mode": str(getattr(env.config, "random_threat_mode", "static")),
        "max_accel": float(max_accel),
        "max_thrust": float(max_thrust),
        "mass": float(mass),
        "drag": float(drag),
        "drag_coeff": float(drag_coeff),
        "obs_noise_target": float(getattr(env.config, "obs_noise_target", 0.0)),
        "obs_noise_vel": float(getattr(env.config, "obs_noise_vel", 0.0)),
        "obs_noise_grid": float(getattr(env.config, "obs_noise_grid", 0.0)),
        "domain_randomization": bool(getattr(env.config, "domain_randomization", False)),
        "dr_max_speed_min": float(getattr(env.config, "dr_max_speed_min", -1.0)),
        "dr_max_speed_max": float(getattr(env.config, "dr_max_speed_max", -1.0)),
        "dr_drag_min": float(getattr(env.config, "dr_drag_min", -1.0)),
        "dr_drag_max": float(getattr(env.config, "dr_drag_max", -1.0)),
    }


__all__ = ["apply_tunables", "collect_tunables"]
