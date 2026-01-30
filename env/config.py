from dataclasses import dataclass

@dataclass
class EnvConfig:
    field_size: float = 100.0 # Размер игрового поля (100 на 100 метров)
    n_agents: int = 20 # Количество агентов в среде
    dt: float = 0.1 # Шаг времени симуляции в секундах
    max_speed: float = 5.0 # Максимальная скорость (м/c) v1
    comm_radius: float = 20.0 # Радиус связи между агентами (метры)
