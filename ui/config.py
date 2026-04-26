"""Конфигурация интерфейса, отделённая от конфигурации среды обучения."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class UIConfig:
    screen_size: int = 900
    max_steps: int = 600
    goal_radius: float = 3.0
    seed: int = 0
    fps: int = 20
    policy_workers: int = 4
    oracle_enabled: bool = False
    oracle_async: bool = True
    oracle_update_interval: int = 10
    attention_stride: int = 4


@dataclass
class WebUIConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    screen_size: int = 900
    max_steps: int = 600
    goal_radius: float = 3.0
    seed: int = 0
    fps: int = 20
    policy_workers: int = 4
    oracle_enabled: bool = False
    oracle_async: bool = True
    oracle_update_interval: int = 10
    attention_stride: int = 4
    compare: bool = False
    scene_root: str = "scenarios"
    user_scene_root: str = "scenarios/user"
    runs_root: str = "runs"
    hydra: dict = field(default_factory=dict)
