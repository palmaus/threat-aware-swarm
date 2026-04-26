from __future__ import annotations

import logging

from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


def validate_config_guardrails(cfg) -> None:
    """Жёсткие проверки конфигов, чтобы ловить ошибки до старта симуляции."""
    if OmegaConf.is_config(cfg):
        try:
            OmegaConf.set_struct(cfg, True)
        except Exception as exc:  # pragma: no cover - защита от экзотики Hydra
            logger.warning("Не удалось включить struct‑режим для OmegaConf: %s", exc)

    try:
        mode = str(cfg.env.control_mode)
    except Exception:
        mode = None
    if mode is not None and mode != "waypoint":
        raise ValueError(f"Неподдерживаемый control_mode: {mode}. Допустим только 'waypoint'.")

    try:
        ticks = int(cfg.env.physics_ticks_per_action)
    except Exception:
        ticks = 1
    if ticks < 1:
        raise ValueError("physics_ticks_per_action должен быть >= 1.")
