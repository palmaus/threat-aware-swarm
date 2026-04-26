"""Проверка ADR/ALP обновлений в curriculum."""

from __future__ import annotations

from scripts.train.curriculum_manager import CurriculumManager


def test_curriculum_adr_updates_ranges():
    adr_cfg = {
        "enabled": True,
        "metric": "swarm/finished_frac",
        "threshold": 0.8,
        "interval_steps": 1,
        "step": 0.5,
        "params": {
            "drag_coeff": {"base": 0.5, "min": 0.1, "max": 1.0},
            "obs_noise_vel": {"base": 0.0, "min": 0.0, "max": 0.2},
        },
    }
    manager = CurriculumManager(stages=[{}], adr_config=adr_cfg)
    manager.initialize(0)
    update = manager.step(10, 0.9)
    assert update.apply_params.get("domain_randomization") is True
    assert "dr_drag_min" in update.apply_params
    assert "dr_drag_max" in update.apply_params
    assert "obs_noise_vel" in update.apply_params


def test_curriculum_alp_selects_bucket():
    alp_cfg = {
        "enabled": True,
        "metric": "swarm/finished_frac",
        "interval_steps": 1,
        "epsilon": 0.1,
        "seed": 0,
        "buckets": [
            {"params": {"random_threat_count_min": 1, "random_threat_count_max": 2}},
            {"params": {"random_threat_count_min": 3, "random_threat_count_max": 4}},
        ],
    }
    manager = CurriculumManager(stages=[{}], alp_config=alp_cfg)
    init = manager.initialize(0)
    assert any(key in init.apply_params for key in ("random_threat_count_min", "random_threat_count_max"))
    update = manager.step(10, 0.9)
    assert "curriculum/alp_bucket" in update.log_values
