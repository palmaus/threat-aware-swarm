"""Совместимость со старым импортом `common.physics_model`."""

from common.physics.model import (
    apply_accel_dynamics,
    apply_accel_dynamics_step,
    apply_accel_dynamics_vel,
    compute_drag_factor,
    inverse_accel_for_velocity,
)

__all__ = [
    "apply_accel_dynamics",
    "apply_accel_dynamics_step",
    "apply_accel_dynamics_vel",
    "compute_drag_factor",
    "inverse_accel_for_velocity",
]
