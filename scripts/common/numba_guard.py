from __future__ import annotations

import logging


def _is_numba_compiled(fn) -> bool:
    return bool(getattr(fn, "signatures", None))


def _has_numba_dispatcher(fn) -> bool:
    return hasattr(fn, "py_func")


def log_numba_status(logger: logging.Logger | None = None) -> None:
    """Логирует статус Numba для критических функций."""
    if logger is None:
        logger = logging.getLogger(__name__)

    try:
        from common import physics_model

        if _is_numba_compiled(physics_model.apply_accel_dynamics_step):
            pass
        elif _has_numba_dispatcher(physics_model.apply_accel_dynamics_step):
            logger.info("Numba доступна для apply_accel_dynamics_step, но ядро ещё не прогрето; первый вызов заплатит JIT-cost.")
        else:
            logger.warning("Numba недоступна для apply_accel_dynamics_step; будет Python-fallback.")
    except Exception as exc:  # pragma: no cover - диагностика
        logger.warning("Проверка Numba для physics_model недоступна: %s", exc)

    try:
        from env.oracles import grid_oracle as oracle_mod

        if not getattr(oracle_mod, "_NUMBA_AVAILABLE", False):
            logger.warning("Numba недоступна для oracle; возможна деградация производительности.")
    except Exception as exc:  # pragma: no cover - диагностика
        logger.warning("Проверка Numba для oracle недоступна: %s", exc)

    try:
        from baselines import astar_grid as astar_mod

        if not getattr(astar_mod, "_NUMBA_AVAILABLE", False):
            logger.warning("Numba недоступна для astar_grid; возможна деградация производительности.")
    except Exception as exc:  # pragma: no cover - диагностика
        logger.warning("Проверка Numba для astar_grid недоступна: %s", exc)
