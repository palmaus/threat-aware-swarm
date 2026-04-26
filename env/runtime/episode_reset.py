"""Episode reset helpers for SwarmEngine."""

from __future__ import annotations

import numpy as np

from env.physics.wind import GlobalOUWind


def apply_episode_domain_randomization(engine) -> None:
    """Apply per-episode randomized runtime scalars or restore base values."""

    if bool(getattr(engine.config, "domain_randomization", False)):
        min_speed = float(getattr(engine.config, "dr_max_speed_min", -1.0))
        max_speed = float(getattr(engine.config, "dr_max_speed_max", -1.0))
        if min_speed <= 0.0 or max_speed <= 0.0:
            min_speed = engine._base_max_speed
            max_speed = engine._base_max_speed
        if max_speed < min_speed:
            max_speed, min_speed = min_speed, max_speed

        min_drag = float(getattr(engine.config, "dr_drag_min", -1.0))
        max_drag = float(getattr(engine.config, "dr_drag_max", -1.0))
        if min_drag < 0.0 or max_drag < 0.0:
            min_drag = engine._base_drag_coeff
            max_drag = engine._base_drag_coeff
        if max_drag < min_drag:
            max_drag, min_drag = min_drag, max_drag

        min_accel = float(getattr(engine.config, "dr_max_accel_min", -1.0))
        max_accel = float(getattr(engine.config, "dr_max_accel_max", -1.0))
        base_max_accel = engine._base_max_thrust / max(engine._base_mass, 1e-6)
        if min_accel <= 0.0 or max_accel <= 0.0:
            min_accel = base_max_accel
            max_accel = base_max_accel
        if max_accel < min_accel:
            max_accel, min_accel = min_accel, max_accel

        min_mass = float(getattr(engine.config, "dr_mass_min", -1.0))
        max_mass = float(getattr(engine.config, "dr_mass_max", -1.0))
        if min_mass <= 0.0 or max_mass <= 0.0:
            min_mass = engine._base_mass
            max_mass = engine._base_mass
        if max_mass < min_mass:
            max_mass, min_mass = min_mass, max_mass

        engine.config.max_speed = (
            float(engine.rng.uniform(min_speed, max_speed))
            if max_speed >= min_speed
            else float(engine._base_max_speed)
        )
        engine.config.physics.drag_coeff = (
            float(engine.rng.uniform(min_drag, max_drag))
            if max_drag >= min_drag
            else float(engine._base_drag_coeff)
        )
        mass = float(engine.rng.uniform(min_mass, max_mass)) if max_mass >= min_mass else float(engine._base_mass)
        engine.config.physics.mass = mass
        accel = (
            float(engine.rng.uniform(min_accel, max_accel)) if max_accel >= min_accel else float(base_max_accel)
        )
        engine.config.physics.max_thrust = float(accel * mass)
        return

    engine.config.max_speed = float(engine._base_max_speed)
    engine.config.physics.drag_coeff = float(engine._base_drag_coeff)
    engine.config.physics.mass = float(engine._base_mass)
    engine.config.physics.max_thrust = float(engine._base_max_thrust)


def apply_scene_wind_override(config, scene: dict | None) -> None:
    if not scene or not isinstance(scene.get("wind"), dict):
        return
    for w_key, w_val in scene.get("wind", {}).items():
        if hasattr(config.wind, w_key):
            setattr(config.wind, w_key, w_val)


def configure_episode_wind(engine) -> None:
    if bool(getattr(engine.config.wind, "enabled", False)):
        wind_seed = getattr(engine.config.wind, "seed", None)
        if wind_seed is not None:
            wind_rng = np.random.default_rng(int(wind_seed))
        else:
            wind_rng = engine.rng_registry.get("wind")
        engine._wind_field = GlobalOUWind(
            theta=float(getattr(engine.config.wind, "ou_theta", 0.15)),
            sigma=float(getattr(engine.config.wind, "ou_sigma", 0.3)),
            rng=wind_rng,
        )
        engine._wind_field.reset()
        engine.sim.set_wind_field(engine._wind_field)
    else:
        engine._wind_field = None
        engine.sim.set_wind_field(None)
