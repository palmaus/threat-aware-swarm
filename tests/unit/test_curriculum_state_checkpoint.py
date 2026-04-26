from __future__ import annotations

import pickle
from pathlib import Path

from scripts.train.curriculum_manager import CurriculumManager


def test_curriculum_state_persists_adr_alp(tmp_path: Path) -> None:
    state_json = tmp_path / "curriculum_state.json"
    state_pkl = tmp_path / "alp_state.pkl"
    adr_cfg = {
        "enabled": True,
        "metric": "swarm/finished_frac",
        "threshold": 0.5,
        "interval_steps": 1,
        "step": 0.5,
        "params": {"drag_coeff": {"base": 0.5, "min": 0.1, "max": 1.0}},
    }
    alp_cfg = {
        "enabled": True,
        "metric": "swarm/finished_frac",
        "interval_steps": 1,
        "epsilon": 0.1,
        "seed": 123,
        "buckets": [
            {"params": {"random_threat_count_min": 1, "random_threat_count_max": 2}},
            {"params": {"random_threat_count_min": 3, "random_threat_count_max": 4}},
        ],
    }
    manager = CurriculumManager(
        stages=[{}],
        adr_config=adr_cfg,
        alp_config=alp_cfg,
        state_path=state_json,
        state_pickle_path=state_pkl,
    )
    manager.initialize(0)
    manager.step(10, 0.9)
    assert state_json.exists()
    assert state_pkl.exists()

    raw = pickle.loads(state_pkl.read_bytes())
    assert "alp" in raw
    assert "adr" in raw

    manager2 = CurriculumManager(
        stages=[{}],
        adr_config=adr_cfg,
        alp_config=alp_cfg,
        state_path=state_json,
        state_pickle_path=state_pkl,
    )
    manager2.initialize(0)
    assert manager2._adr_scale >= 0.0
