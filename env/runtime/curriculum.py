"""Runtime curriculum application helpers for SwarmEngine."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def apply_curriculum_params(engine: Any, stage_params: dict, *, unsafe_keys: Iterable[str]) -> None:
    """Apply episode/runtime curriculum overrides to an engine-like object."""

    if not stage_params:
        return
    unsafe = set(unsafe_keys)
    for key, value in stage_params.items():
        if key == "goal_hold_steps":
            engine.goal_hold_steps = int(value)
            continue
        if key == "max_steps":
            engine.max_steps = int(value)
            engine._base_max_steps = int(value)
            if hasattr(engine.config, "max_steps"):
                engine.config.max_steps = int(value)
            continue
        if key == "field_size":
            engine._set_field_size(float(value), update_base=True, clip_current=True)
            continue
        if key in unsafe:
            print(
                f"[Curriculum] Warning: параметр '{key}' требует пересоздания среды; "
                "runtime-изменение проигнорировано."
            )
            continue
        if key == "reward" and isinstance(value, dict):
            _set_section(engine.reward_cfg, value, "reward")
        elif key == "physics" and isinstance(value, dict):
            _set_section(engine.config.physics, value, "physics")
        elif key == "wind" and isinstance(value, dict):
            _set_section(engine.config.wind, value, "wind")
        elif key == "control_mode":
            mode = str(value).lower()
            if mode != "waypoint":
                raise ValueError(
                    f"Режим управления '{mode}' больше не поддерживается; используйте control_mode=waypoint."
                )
            engine.config.control_mode = mode
        elif key == "battery" and isinstance(value, dict):
            _set_section(engine.config.battery, value, "battery")
        elif key == "forest" and isinstance(value, dict):
            _set_section(engine.config.forest, value, "forest")
        elif key == "goal_radius":
            engine.goal_radius = float(value)
        elif key == "target_motion":
            engine.config.target_motion = value
        elif key == "target_motion_prob":
            engine.config.target_motion_prob = float(value)
        elif key in {"max_thrust", "mass", "drag_coeff"}:
            try:
                setattr(engine.config.physics, key, float(value))
            except Exception:
                print(f"[Curriculum] Warning: не удалось задать physics.{key}.")
        elif hasattr(engine.config, key):
            setattr(engine.config, key, value)
        elif hasattr(engine, key):
            setattr(engine, key, value)
        else:
            print(f"[Curriculum] Warning: неизвестный параметр '{key}' для EnvConfig/Env.")

    _refresh_runtime_base_snapshots(engine, stage_params)


def _set_section(section: Any, values: dict, name: str) -> None:
    if section is None:
        return
    for item_key, item_val in values.items():
        if hasattr(section, item_key):
            setattr(section, item_key, item_val)
        else:
            print(f"[Curriculum] Warning: неизвестный {name} '{item_key}'.")


def _refresh_runtime_base_snapshots(engine: Any, stage_params: dict) -> None:
    if "max_speed" in stage_params:
        try:
            engine._base_max_speed = float(engine.config.max_speed)
        except Exception:
            pass
    engine._sync_runtime_params()
    engine._invalidate_export_cache()
    engine._validate_control_mode()
    if "physics" in stage_params or any(k in stage_params for k in ("max_thrust", "mass", "drag_coeff")):
        try:
            engine._base_mass = float(getattr(engine.config.physics, "mass", 1.0))
            engine._base_drag_coeff = float(getattr(engine.config.physics, "drag_coeff", 0.0))
            engine._base_max_thrust = float(getattr(engine.config.physics, "max_thrust", 0.0))
        except Exception:
            pass
    if "wind" in stage_params:
        try:
            engine._base_wind_enabled = bool(getattr(engine.config.wind, "enabled", False))
            engine._base_wind_theta = float(getattr(engine.config.wind, "ou_theta", 0.15))
            engine._base_wind_sigma = float(getattr(engine.config.wind, "ou_sigma", 0.3))
            engine._base_wind_seed = getattr(engine.config.wind, "seed", None)
        except Exception:
            pass
    if "battery" in stage_params:
        try:
            engine._base_battery_capacity = float(getattr(engine.config.battery, "capacity", 0.0))
            engine._base_battery_drain_hover = float(getattr(engine.config.battery, "drain_hover", 0.0))
            engine._base_battery_drain_thrust = float(getattr(engine.config.battery, "drain_thrust", 0.0))
        except Exception:
            pass
