import numpy as np

from common.physics_model import apply_accel_dynamics
from env.config import EnvConfig
from env.physics.core import PhysicsCore


def test_drag_relaxes_velocity_to_wind():
    # Без тяги скорость должна стремиться к скорости ветра.
    pos = np.zeros((1, 2), dtype=np.float32)
    vel = np.array([[10.0, 0.0]], dtype=np.float32)
    accel = np.zeros((1, 2), dtype=np.float32)
    wind = np.array([[2.0, 0.0]], dtype=np.float32)

    for _ in range(200):
        pos, vel = apply_accel_dynamics(pos, vel, accel, 0.1, drag=0.2, max_speed=0.0, wind=wind)

    assert np.linalg.norm(vel - wind) < 0.05


def test_speed_limit_is_respected():
    # Ограничение max_speed не должно превышаться.
    pos = np.zeros((1, 2), dtype=np.float32)
    vel = np.zeros((1, 2), dtype=np.float32)
    accel = np.array([[100.0, 0.0]], dtype=np.float32)

    _, next_vel = apply_accel_dynamics(pos, vel, accel, 0.1, drag=0.0, max_speed=5.0, wind=None)
    assert float(np.linalg.norm(next_vel)) <= 5.01


def test_energy_monotonic_with_thrust():
    # Энергия должна убывать при ненулевой тяге и не становиться отрицательной.
    cfg = EnvConfig(n_agents=1)
    cfg.dt = 0.1
    cfg.battery.capacity = 5.0
    cfg.battery.drain_hover = 0.0
    cfg.battery.drain_thrust = 0.5
    cfg.physics.max_thrust = 4.0

    core = PhysicsCore(cfg, rng=np.random.default_rng(0))
    core.reset()
    energy0 = float(core.energy[0])

    thrust = np.array([[cfg.physics.max_thrust, 0.0]], dtype=np.float32)
    for _ in range(10):
        core.step(thrust)

    assert float(core.energy[0]) < energy0
    assert float(core.energy[0]) >= 0.0
