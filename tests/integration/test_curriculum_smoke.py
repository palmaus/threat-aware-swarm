from __future__ import annotations

from scripts.train.curriculum_manager import CurriculumManager


def test_curriculum_stage_switch_smoke():
    stages = [
        {"name": "s0", "params": {"max_steps": 5}, "threshold": 0.0, "min_steps": 0},
        {"name": "s1", "params": {"max_steps": 6}, "threshold": 0.0, "min_steps": 0},
    ]
    manager = CurriculumManager(stages=stages, metric_key="swarm/finished_frac", threshold=0.0, min_steps_per_stage=0)
    update = manager.initialize(0)
    assert update.apply_params.get("max_steps") == 5
    update2 = manager.step(1, 1.0)
    assert update2.apply_params.get("max_steps") == 6
