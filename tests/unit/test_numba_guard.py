import logging
from types import SimpleNamespace

import baselines.astar_grid as astar_mod
import common.physics_model as physics_model
import env.oracles.grid_oracle as oracle_mod
from scripts.common.numba_guard import log_numba_status


def test_log_numba_status_uses_new_oracle_import_and_non_warning_for_unwarmed_kernel(monkeypatch, caplog) -> None:
    fake_dispatcher = SimpleNamespace(signatures=[], py_func=object())

    monkeypatch.setattr(physics_model, "apply_accel_dynamics_step", fake_dispatcher)
    monkeypatch.setattr(oracle_mod, "_NUMBA_AVAILABLE", True)
    monkeypatch.setattr(astar_mod, "_NUMBA_AVAILABLE", True)

    logger = logging.getLogger("test_numba_guard")
    with caplog.at_level(logging.INFO, logger=logger.name):
        log_numba_status(logger)

    text = caplog.text
    assert "ядро ещё не прогрето" in text
    assert "Проверка Numba для oracle недоступна" not in text
    assert "Python-fallback" not in text
