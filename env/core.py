import numpy as np
from dataclasses import dataclass
from env.config import EnvConfig

@dataclass
class Threat:
    position: np.ndarray
    radius: float
    intensity: float

class SwarmSimulator:
    def __init__(self, config: EnvConfig):
        self.config = config
        self.n = config.n_agents

        # State
        self.agents_pos = np.zeros((self.n, 2), dtype=np.float32)
        self.agents_vel = np.zeros((self.n, 2), dtype=np.float32) # Храним скорость для Observation
        self.agents_active = np.ones(self.n, dtype=bool)

        self.threats = []
        self.time_step = 0

    def add_threat(self, position, radius, intensity):
        self.threats.append(Threat(
            position=np.array(position, dtype=np.float32),
            radius=radius,
            intensity=intensity
        ))

    def reset(self):
        # Сброс состояния симуляции
        self.time_step = 0
        self.agents_active[:] = True
        self.threats = []
        # Случайный старт в углу (логика расстановки будет в Env)
        self.agents_pos = np.random.uniform(0, 20, (self.n, 2))
        self.agents_vel = np.zeros((self.n, 2), dtype=np.float32)
        return self.get_state()

    def step(self, velocity_actions: np.ndarray):
        """
        velocity_actions: (N, 2) - желаемые векторы скорости
        """
        if self.n == 0: return self.get_state()

        # 1. Ограничение скорости (Clipping)
        # v1: Мгновенное изменение скорости (без инерции/ускорения)
        speeds = np.linalg.norm(velocity_actions, axis=1, keepdims=True)
        scale = np.where(speeds > self.config.max_speed,
                         self.config.max_speed / (speeds + 1e-8), 1.0)
        clamped_vel = velocity_actions * scale

        # Сохраняем скорость (важно для obs)
        self.agents_vel = clamped_vel

        # 2. Обновление позиций
        active_indices = np.where(self.agents_active)[0]
        # v1: pos = pos + vel * dt
        self.agents_pos[active_indices] += clamped_vel[active_indices] * self.config.dt

        # 3. Границы и Угрозы
        self._apply_boundaries()
        self._update_survival()

        self.time_step += 1
        return self.get_state()

    def _apply_boundaries(self):
        # Жесткое ограничение поля. Агент не может вылететь за [0, field_size].
        # Награда будет наказывать за приближение к этим стенкам.
        self.agents_pos = np.clip(self.agents_pos, 0, self.config.field_size)

    def _update_survival(self):
        # v1: Вероятностная модель смерти
        active_indices = np.where(self.agents_active)[0]
        if len(active_indices) == 0: return

        positions = self.agents_pos[active_indices]
        survival_probs = np.ones(len(active_indices))

        for threat in self.threats:
            dists = np.linalg.norm(positions - threat.position, axis=1)
            in_range_mask = dists <= threat.radius
            survival_probs[in_range_mask] *= (1.0 - threat.intensity)

        random_vals = np.random.random(len(active_indices))
        survived = random_vals < survival_probs

        killed_indices = active_indices[~survived]
        self.agents_active[killed_indices] = False

    def get_wall_distances(self, agent_idx):
        """
        Возвращает нормализованные расстояния до 4-х стен [Left, Right, Down, Up].
        Диапазон [0, 1].
        """
        pos = self.agents_pos[agent_idx]
        size = self.config.field_size

        d_left = pos[0] / size
        d_right = (size - pos[0]) / size
        d_down = pos[1] / size
        d_up = (size - pos[1]) / size

        return np.array([d_left, d_right, d_down, d_up], dtype=np.float32)

    def get_state(self):
        return self.agents_pos.copy(), self.agents_active.copy()