from pathlib import Path

import numpy as np
import yaml

from env.config import EnvConfig
from env.pz_env import SwarmPZEnv


def test_scenarios_reset_smoke():
    cfg = EnvConfig()
    env = SwarmPZEnv(cfg, max_steps=20, goal_radius=3.0, oracle_enabled=False, oracle_async=False)
    scene_dir = Path("scenarios")
    scenes = sorted(scene_dir.glob("S*.yaml"))
    assert scenes, "Не найдено ни одной сцены в scenarios/"

    for scene_path in scenes:
        scene = yaml.safe_load(scene_path.read_text())
        obs, _info = env.reset(seed=0, options={"scene": scene})
        assert obs, f"reset() не вернул наблюдение для {scene_path.name}"
        if "target_pos" in scene:
            target = np.asarray(scene["target_pos"], dtype=np.float32)
            motion = scene.get("target_motion") or {}
            motion_type = str(motion.get("type", "")).lower()
            if motion_type in {"circle", "circular"}:
                center = np.asarray(motion.get("center", target), dtype=np.float32)
                radius = float(motion.get("radius", 0.0))
                phase = float(motion.get("phase", 0.0))
                expected = center + radius * np.array([np.cos(phase), np.sin(phase)], dtype=np.float32)
                assert np.allclose(env.target_pos, expected)
            else:
                assert np.allclose(env.target_pos, target)
        if not scene.get("threats") and not scene.get("static_threats") and not scene.get("dynamic_threats"):
            assert len(env.sim.threats) == 0
