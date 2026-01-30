import numpy as np

class PotentialFieldPolicy:
    def __init__(self, n_agents, k_att=1.0, k_rep=50.0, safety_margin=5.0):
        """
        :param n_agents: количество агентов
        :param k_att: коэффициент притяжения к цели
        :param k_rep: коэффициент отталкивания от угроз
        :param safety_margin: запас расстояния (насколько метров облетать угрозу)
        """
        self.n_agents = n_agents
        self.k_att = k_att
        self.k_rep = k_rep
        self.safety_margin = safety_margin

    def get_actions(self, agents_pos: np.ndarray, active_mask: np.ndarray,
                    target_pos: np.ndarray, threats: list):
        """
        Вычисляет действия для агентов на основе потенциалов
         1. Притяжение к цели
         2. Отталкивание от угроз
        """
        total_force = np.zeros_like(agents_pos) # (n_agents, 2)
        to_target = target_pos - agents_pos  # вектор от агента к цели

        # Притяжение к цели
        dists_target = np.linalg.norm(to_target, axis=1, keepdims=True)
        to_target_norm = to_target / (dists_target + 1e-8) # нормализованный вектор

        total_force += to_target_norm * self.k_att

        # Отталкивание от угрозы
        for threat in threats:
            # Вектор от угрозы к агенту
            from_threat = agents_pos - threat.position
            dists_threat = np.linalg.norm(from_threat, axis=1, keepdims=True)

            push_dir = from_threat / (dists_threat + 1e-8) # нормализованный вектор

            # Определяем зону влияния
            # Мы хотим отворачивать !ЗАРАНЕЕ!, до того как влетим в радиус поражения
            # Influence radius = радиус угрозы + запас безопасности
            influence_radius = threat.radius + self.safety_margin

            # Считаем силу оттакливания
            # если мы далеко, сила 0
            # если мы близко, сила растет линейно или квадратично

            repulsion_mag = np.maximum(0, influence_radius - dists_threat)

            total_force += push_dir * repulsion_mag * self.k_rep

        # Обнуляем мертвых агентов
        total_force[~active_mask] = 0.0

        return total_force


