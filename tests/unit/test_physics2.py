import numpy as np

from env.config import EnvConfig
from env.physics.core import AGENT_ACTIVE, AGENT_DEAD, _collect_circle_obstacles
from env.physics.wind import WindField
from env.pz_env import SwarmPZEnv
from env.scenes.procedural import ForestConfig, generate_forest


class _ConstantWind(WindField):
    def __init__(self, value: tuple[float, float]) -> None:
        self._value = np.asarray(value, dtype=np.float32)

    def reset(self, seed: int | None = None) -> None:  # pragma: no cover - детерминированно
        return

    def step(self, dt: float) -> None:  # pragma: no cover - детерминированно
        return

    def sample(self, positions: np.ndarray) -> np.ndarray:
        pos = np.asarray(positions, dtype=np.float32)
        if pos.ndim == 1:
            return self._value.copy()
        return np.broadcast_to(self._value, pos.shape).astype(np.float32)


def test_wind_moves_passive_agent():
    cfg = EnvConfig(n_agents=1)
    cfg.physics.drag_coeff = 1.0
    cfg.physics.max_thrust = 1.0
    cfg.max_speed = 50.0
    cfg.dt = 1.0

    env = SwarmPZEnv(cfg)
    env.reset(seed=0)
    env.engine.sim.set_wind_field(_ConstantWind((1.0, 0.0)))

    v0 = env.engine.sim.agents_vel.copy()
    env.step({"drone_0": np.zeros((2,), dtype=np.float32)})
    v1 = env.engine.sim.agents_vel.copy()

    assert v1[0, 0] > v0[0, 0]


def test_energy_drain_scales_with_thrust():
    cfg = EnvConfig(n_agents=1)
    cfg.physics.max_thrust = 2.0
    cfg.battery.capacity = 10.0
    cfg.battery.drain_hover = 0.0
    cfg.battery.drain_thrust = 1.0
    cfg.dt = 1.0

    env = SwarmPZEnv(cfg)
    env.reset(seed=0)
    env.engine.sim.energy[:] = 10.0

    env.step({"drone_0": np.array([1.0, 0.0], dtype=np.float32)})
    energy = float(env.engine.sim.energy[0])

    assert np.isclose(energy, 8.0, atol=1e-4)


def test_physics_step_reuses_agent_arrays():
    cfg = EnvConfig(n_agents=1)
    env = SwarmPZEnv(cfg)
    env.reset(seed=0)

    pos_id = id(env.engine.sim.agents_pos)
    vel_id = id(env.engine.sim.agents_vel)
    env.step({"drone_0": np.array([0.2, 0.0], dtype=np.float32)})
    env.step({"drone_0": np.array([0.0, 0.0], dtype=np.float32)})

    assert id(env.engine.sim.agents_pos) == pos_id
    assert id(env.engine.sim.agents_vel) == vel_id


def test_dead_agents_become_circle_obstacles():
    static_pos = np.zeros((0, 2), dtype=np.float32)
    static_rad = np.zeros((0,), dtype=np.float32)
    agents_pos = np.array([[10.0, 10.0], [20.0, 20.0]], dtype=np.float32)
    agent_state = np.array([AGENT_DEAD, AGENT_ACTIVE], dtype=np.int8)

    pos, rad = _collect_circle_obstacles(static_pos, static_rad, agents_pos, agent_state, agent_radius=0.5)

    assert pos.shape[0] == 1
    assert np.allclose(pos[0], [10.0, 10.0])
    assert np.isclose(rad[0], 0.5, atol=1e-6)


def test_poisson_forest_has_min_center_distance():
    rng = np.random.default_rng(123)
    cfg = ForestConfig(
        enabled=True,
        count=20,
        radius_min=1.0,
        radius_max=1.0,
        min_dist=3.0,
        region_min=(0.0, 0.0),
        region_max=(50.0, 50.0),
    )
    obstacles = generate_forest(rng, config=cfg)
    centers = np.asarray([pos for pos, _ in obstacles], dtype=np.float32)
    if centers.shape[0] < 2:
        return
    for i in range(centers.shape[0]):
        for j in range(i + 1, centers.shape[0]):
            diff = centers[i] - centers[j]
            dist = float(np.sqrt(diff[0] * diff[0] + diff[1] * diff[1]))
            assert dist >= (cfg.min_dist - 1e-3)
