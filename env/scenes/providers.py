"""Провайдеры генерации карты и угроз."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from env.scenes.procedural import ForestConfig, generate_forest
from env.scenes.threats import threat_from_config


class MapProvider(Protocol):
    """Интерфейс генератора карты и стартовых позиций."""

    def apply(self, env: Any, scene: dict | None) -> None:
        """Применяет карту и позиции (без угроз)."""


class ThreatProvider(Protocol):
    """Интерфейс генератора угроз."""

    def apply(self, env: Any, scene: dict | None) -> None:
        """Применяет угрозы (без стен/карты)."""


@dataclass
class DefaultMapProvider:
    """Дефолтная генерация карты на основе конфигов и SceneSpec."""

    def apply(self, env: Any, scene: dict | None) -> None:
        if scene:
            env.scene.apply_scene_layout(scene)
            return

        if env.config.walls:
            env.scene.set_walls(env.config.walls)
        elif env.config.wall_count > 0:
            env.scene.set_walls(env.scene.generate_random_walls())
        else:
            env.scene.set_walls([])

        forest_cfg = getattr(env.config, "forest", ForestConfig())
        if isinstance(forest_cfg, dict):
            try:
                forest_cfg = ForestConfig(**forest_cfg)
            except Exception:
                forest_cfg = ForestConfig()
        if getattr(forest_cfg, "enabled", False) and int(getattr(forest_cfg, "count", 0)) > 0:
            if getattr(forest_cfg, "region_max", None) == (100.0, 100.0):
                forest_cfg = ForestConfig(
                    enabled=True,
                    count=int(forest_cfg.count),
                    radius_min=float(forest_cfg.radius_min),
                    radius_max=float(forest_cfg.radius_max),
                    min_dist=float(forest_cfg.min_dist),
                    region_min=tuple(getattr(forest_cfg, "region_min", (0.0, 0.0))),
                    region_max=(float(env.config.field_size), float(env.config.field_size)),
                )
            env.set_static_circles(generate_forest(env.rng_registry.get("procedural"), config=forest_cfg))
        else:
            env.set_static_circles([])

        # Безопасная инициализация: исключаем старт внутри стен и угроз.
        tmin = np.asarray(getattr(env.config, "target_pos_min", (80.0, 80.0)), dtype=np.float32)
        tmax = np.asarray(getattr(env.config, "target_pos_max", (95.0, 95.0)), dtype=np.float32)
        smin = np.asarray(getattr(env.config, "start_pos_min", (5.0, 5.0)), dtype=np.float32)
        smax = np.asarray(getattr(env.config, "start_pos_max", (20.0, 20.0)), dtype=np.float32)

        if tmax.shape[0] >= 2 and tmin.shape[0] >= 2:
            tx = (float(tmin[0]), float(tmax[0]))
            ty = (float(tmin[1]), float(tmax[1]))
        else:
            tx = (80.0, 95.0)
            ty = (80.0, 95.0)
        if smax.shape[0] >= 2 and smin.shape[0] >= 2:
            sx = (float(smin[0]), float(smax[0]))
            sy = (float(smin[1]), float(smax[1]))
        else:
            sx = (5.0, 20.0)
            sy = (5.0, 20.0)

        env.target_pos = env.spawn.sample_safe_pos(
            tx,
            ty,
            margin=env.config.agent_radius * 1.5,
        ).astype(np.float32)
        min_dist = float(getattr(env.config, "min_start_target_dist", 0.0))
        start_center = env.spawn.sample_safe_pos_with_min_dist(
            sx,
            sy,
            margin=env.config.agent_radius * 1.5,
            target_pos=env.target_pos,
            min_dist=min_dist,
        ).astype(np.float32)
        env.set_agent_positions(env.spawn.sample_safe_cluster(start_center))

        motion_cfg = getattr(env.config, "target_motion", None)
        motion_prob = float(getattr(env.config, "target_motion_prob", 1.0))
        motion_prob = float(np.clip(motion_prob, 0.0, 1.0))
        if motion_cfg and float(env.rng_registry.get("engine").random()) < motion_prob:
            env.configure_target_motion(motion_cfg)


@dataclass
class DefaultThreatProvider:
    """Дефолтная генерация угроз на основе конфигов и SceneSpec."""

    def apply(self, env: Any, scene: dict | None) -> None:
        if scene:
            env.scene.apply_scene_threats(scene)
            env.scene.apply_scene_agents(scene)
            return

        rng_threats = env.rng_registry.get("threats")
        threat_mode = str(getattr(env.config, "random_threat_mode", "static")).lower()
        if threat_mode not in {"none", "static", "dynamic", "mixed"}:
            threat_mode = "static"
        if threat_mode != "none":
            dynamic_frac = float(getattr(env.config, "random_threat_dynamic_frac", 0.5))
            dynamic_frac = float(np.clip(dynamic_frac, 0.0, 1.0))
            dynamic_types = ("linear", "brownian", "chaser")
            count_min = int(getattr(env.config, "random_threat_count_min", 3))
            count_max = int(getattr(env.config, "random_threat_count_max", 5))
            if count_max < count_min:
                count_max = count_min
            if count_min < 0:
                count_min = 0
            if count_max < 0:
                count_max = 0
            n_threats = 0
            if count_max > 0:
                n_threats = int(rng_threats.integers(count_min, count_max + 1))

            t_radius_min = float(getattr(env.config, "random_threat_radius_min", 10.0))
            t_radius_max = float(getattr(env.config, "random_threat_radius_max", 15.0))
            if t_radius_max < t_radius_min:
                t_radius_min, t_radius_max = t_radius_max, t_radius_min
            field_size = float(getattr(env.config, "field_size", 100.0))
            grid_res = float(max(getattr(env.config, "grid_res", 1.0), 1e-6))
            max_fit_radius = max(0.0, (field_size - grid_res) * 0.5)
            t_radius_max = min(t_radius_max, max_fit_radius)
            t_radius_min = min(t_radius_min, t_radius_max)
            intensity = float(getattr(env.config, "random_threat_intensity", 0.1))
            speed_min = float(getattr(env.config, "random_threat_speed_min", 1.0))
            speed_max = float(getattr(env.config, "random_threat_speed_max", 3.5))
            noise_min = float(getattr(env.config, "random_threat_noise_scale_min", 0.2))
            noise_max = float(getattr(env.config, "random_threat_noise_scale_max", 0.8))
            vision_min = float(getattr(env.config, "random_threat_chaser_vision_min", 20.0))
            vision_max = float(getattr(env.config, "random_threat_chaser_vision_max", 40.0))

            for _ in range(n_threats):
                t_radius = float(rng_threats.uniform(t_radius_min, t_radius_max))
                x_range = _central_spawn_range(field_size, t_radius)
                y_range = _central_spawn_range(field_size, t_radius)
                t_pos = env.spawn.sample_safe_threat_pos(
                    x_range,
                    y_range,
                    margin=t_radius,
                    min_agent_dist=t_radius + float(env.config.agent_radius) * 1.5,
                    target_pos=env.target_pos,
                    min_target_dist=t_radius + float(getattr(env, "goal_radius", 0.0)),
                ).astype(np.float32)
                use_dynamic = threat_mode == "dynamic" or (
                    threat_mode == "mixed" and float(rng_threats.random()) < dynamic_frac
                )
                if use_dynamic:
                    t_type = str(rng_threats.choice(dynamic_types))
                    cfg = {
                        "type": t_type,
                        "pos": t_pos,
                        "radius": t_radius,
                        "intensity": intensity,
                    }
                    if t_type == "linear":
                        cfg["speed"] = float(rng_threats.uniform(speed_min, speed_max))
                        cfg["angle"] = float(rng_threats.uniform(0.0, 360.0))
                    elif t_type == "brownian":
                        cfg["speed"] = float(rng_threats.uniform(speed_min, speed_max))
                        cfg["noise_scale"] = float(rng_threats.uniform(noise_min, noise_max))
                    elif t_type == "chaser":
                        cfg["speed"] = float(rng_threats.uniform(speed_min, speed_max))
                        cfg["vision_radius"] = float(rng_threats.uniform(vision_min, vision_max))
                    threat = threat_from_config(cfg, rng=rng_threats)
                    env.add_threat_object(threat)
                else:
                    env.add_static_threat(t_pos, t_radius, intensity, oracle_block=True)
        else:
            # Защита от "призрачных" угроз: режим none гарантирует пустые списки.
            env.clear_threats()


def _central_spawn_range(field_size: float, radius: float) -> tuple[float, float]:
    field = float(max(field_size, 0.0))
    radius = float(max(radius, 0.0))
    low = max(radius, field * 0.3)
    high = min(max(field - radius, 0.0), field * 0.7)
    if high < low:
        low = min(radius, field)
        high = max(low, min(max(field - radius, 0.0), field))
    return (float(low), float(high))
