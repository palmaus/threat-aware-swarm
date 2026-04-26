from dataclasses import dataclass, field, fields
from typing import Any

from env.scenes.procedural import ForestConfig


@dataclass
class PhysicsConfig:
    mass: float = 1.0  # Масса агента (кг).
    max_thrust: float = 8.0  # Максимальная тяга (Н).
    drag_coeff: float = 0.5  # Коэффициент сопротивления (Н / (м/с)).
    collision_elasticity: float = 0.0  # Эластичность столкновений (0..1).


@dataclass
class WindConfig:
    enabled: bool = True  # Включить ветер.
    ou_theta: float = 0.15  # Скорость возврата OU.
    ou_sigma: float = 0.3  # Интенсивность шума OU.
    seed: int | None = None  # Отдельный seed для ветра.


@dataclass
class BatteryConfig:
    capacity: float = 100.0  # Ёмкость батареи (условные единицы).
    drain_hover: float = 0.1  # Базовый расход в секунду (ховер/hover).
    drain_thrust: float = 0.05  # Расход на модуль тяги (на Н) в секунду.


@dataclass
class EnvConfig:
    field_size: float = 100.0  # Размер стороны поля в метрах (квадрат 100×100).
    n_agents: int = 20  # Число агентов в эпизоде.
    dt: float = 0.1  # Шаг симуляции в секундах.
    max_speed: float = 5.0  # Верхняя граница скорости, м/с.
    comm_radius: float = 20.0  # Радиус связи агентов в метрах.
    agent_radius: float = 0.5  # Геометрический радиус агента в метрах.
    physics_ticks_per_action: int = 1  # Число сим‑тиков на одно решение (шаг решения).
    grid_width: int = 41  # Размер локальной риск-сетки (квадрат).
    grid_res: float = 1.0  # Разрешение риск-сетки (метров на клетку).
    obs_schema_version: str = "obs@1694:v5"  # Версия схемы наблюдений (единственная поддерживаемая).
    walls: list[dict[str, Any]] = field(default_factory=list)  # Статические стены (прямоугольники).
    wall_count: int = 0  # Если > 0, генерировать случайные стены при сбросе.
    wall_size_min: float = 5.0  # Минимальная сторона стены в метрах.
    wall_size_max: float = 15.0  # Максимальная сторона стены в метрах.
    wall_friction: float = 0.0  # Трение при скольжении (0..1).
    target_pos_min: tuple[float, float] = (80.0, 80.0)  # Минимум для случайной цели (x, y).
    target_pos_max: tuple[float, float] = (95.0, 95.0)  # Максимум для случайной цели (x, y).
    start_pos_min: tuple[float, float] = (5.0, 5.0)  # Минимум для старта (x, y).
    start_pos_max: tuple[float, float] = (20.0, 20.0)  # Максимум для старта (x, y).
    min_start_target_dist: float = 0.0  # Минимальная дистанция старт-цель (метры).
    target_motion: dict[str, Any] | None = None  # Движение цели (None или dict с параметрами).
    target_motion_prob: float = 1.0  # Вероятность включения движения цели (0..1).
    random_threat_mode: str = "static"  # Режим угроз для случайной карты: "none" | "static" | "dynamic" | "mixed".
    random_threat_dynamic_frac: float = 0.5  # Доля динамических угроз в режиме "mixed" (0..1).
    random_threat_count_min: int = 3  # Минимум угроз на случайной карте.
    random_threat_count_max: int = 5  # Максимум угроз на случайной карте.
    random_threat_radius_min: float = 10.0  # Минимальный радиус угроз.
    random_threat_radius_max: float = 15.0  # Максимальный радиус угроз.
    random_threat_intensity: float = 0.1  # Интенсивность угроз (0..1).
    random_threat_speed_min: float = 1.0  # Минимальная скорость для динамических угроз.
    random_threat_speed_max: float = 3.5  # Максимальная скорость для динамических угроз.
    random_threat_noise_scale_min: float = 0.2  # Минимальная шумность для броуновских угроз.
    random_threat_noise_scale_max: float = 0.8  # Максимальная шумность для броуновских угроз.
    random_threat_chaser_vision_min: float = 20.0  # Минимальная дальность зрения чейзера.
    random_threat_chaser_vision_max: float = 40.0  # Максимальная дальность зрения чейзера.
    debug_metrics: bool = True  # Включать расширенные info и дополнительные метрики.
    debug_metrics_mode: str = "full"  # full|lite (lite отключает часть тяжёлых debug-полей).
    infos_mode: str = "full"  # full|compact (compact сокращает infos до ключевых метрик).
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    wind: WindConfig = field(default_factory=WindConfig)
    battery: BatteryConfig = field(default_factory=BatteryConfig)
    forest: ForestConfig = field(default_factory=ForestConfig)
    obs_noise_target: float = 0.0  # Шум вектора к цели (стандартное отклонение/std).
    obs_noise_vel: float = 0.0  # Шум скорости (стандартное отклонение/std).
    obs_noise_grid: float = 0.0  # Шум риска в локальной сетке (стандартное отклонение/std).
    control_mode: str = "waypoint"  # Режим управления (единственный поддерживаемый: "waypoint").
    profile: str = "full"  # Профиль среды: "lite" (без ветра/энергии) | "full".
    runtime_mode: str = "full"  # Режим runtime: "full" | "train_fast".
    domain_randomization: bool = False  # Включить доменную рандомизацию параметров среды.
    dr_max_speed_min: float = -1.0  # Минимум max_speed для доменной рандомизации (<=0 = базовое значение).
    dr_max_speed_max: float = -1.0  # Максимум max_speed для доменной рандомизации (<=0 = базовое значение).
    dr_drag_min: float = -1.0  # Минимум drag_coeff для доменной рандомизации (<=0 = базовое значение).
    dr_drag_max: float = -1.0  # Максимум drag_coeff для доменной рандомизации (<=0 = базовое значение).
    dr_max_accel_min: float = -1.0  # Минимум max_thrust/mass для доменной рандомизации.
    dr_max_accel_max: float = -1.0  # Максимум max_thrust/mass для доменной рандомизации.
    dr_mass_min: float = -1.0  # Минимум mass для доменной рандомизации (<=0 = базовое значение).
    dr_mass_max: float = -1.0  # Максимум mass для доменной рандомизации (<=0 = базовое значение).
    oracle_risk_weight: float = 2.0  # Вес риска для оракульного пути (0 = игнорировать угрозы).
    oracle_astar_weight: float = 1.0  # Вес эвристики в A* (1.0 = обычный A*).
    oracle_inflation_buffer: float = 0.2  # Доп. буфер безопасности для C-space (метры).
    oracle_update_interval: int = 1  # Как часто обновлять оракульный путь (в шагах симуляции).
    oracle_visibility: str = "baseline"  # none|baseline|agent (глобальное правило видимости оракула).
    oracle_visible_to_baselines: bool = True  # Разрешать бейзлайнам доступ к oracle_dir.
    oracle_visible_to_agents: bool = False  # Разрешать агентам (RL) доступ к oracle_dir.

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "EnvConfig":
        payload = dict(data or {})
        valid_keys = {item.name for item in fields(cls)}
        payload = {key: value for key, value in payload.items() if key in valid_keys}
        payload["physics"] = _coerce_dataclass(payload.get("physics"), PhysicsConfig)
        payload["wind"] = _coerce_dataclass(payload.get("wind"), WindConfig)
        payload["battery"] = _coerce_dataclass(payload.get("battery"), BatteryConfig)
        payload["forest"] = _coerce_dataclass(payload.get("forest"), ForestConfig)
        return cls(**payload)


def _coerce_dataclass(value: Any, cls: type) -> Any:
    if isinstance(value, cls):
        return value
    if isinstance(value, dict):
        valid_keys = {item.name for item in fields(cls)}
        payload = {key: val for key, val in value.items() if key in valid_keys}
        try:
            return cls(**payload)
        except Exception:
            return cls()
    return cls()
