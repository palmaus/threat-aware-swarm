import math
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import yaml
from gymnasium import spaces

import scripts.eval.eval_scenarios as es


def _scene(scene_id: str) -> dict:
    return {
        "id": scene_id,
        "seed": 0,
        "field_size": 100.0,
        "start_center": [10.0, 10.0],
        "start_sigma": 0.1,
        "target_pos": [12.0, 12.0],
        "threats": [],
        "max_steps": 50,
    }


def _args(workers: int) -> SimpleNamespace:
    return SimpleNamespace(
        policy="baseline:greedy_safe",
        model="",
        deterministic=True,
        episodes=1,
        max_steps=50,
        seed=0,
        success_threshold=0.5,
        oracle_enabled=False,
        workers=workers,
    )


def test_evaluate_scenes_parallel_matches_serial():
    scenes = [_scene("T1"), _scene("T2")]

    serial = es.evaluate_scenes(scenes, _args(0))
    parallel = es.evaluate_scenes(scenes, _args(2))

    assert set(serial.keys()) == set(parallel.keys())
    for sid, metrics in serial.items():
        other = parallel[sid]
        for key, val in metrics.items():
            pval = other.get(key)
            if isinstance(val, float) and math.isnan(val):
                assert isinstance(pval, float) and math.isnan(pval)
            else:
                assert pval == pytest.approx(val, rel=1e-6, abs=1e-6)


def test_load_scenes_uses_path_stem_for_missing_id(tmp_path: Path):
    scene_path = tmp_path / "custom_scene.yaml"
    payload = _scene("")
    payload.pop("id")
    scene_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    scenes = es.load_scenes([scene_path])

    assert scenes[0]["id"] == "custom_scene"


def test_run_episode_gif_passes_recurrent_state(monkeypatch, tmp_path: Path):
    class FakeEnv:
        config = SimpleNamespace()
        possible_agents = ["drone_0"]
        max_steps = 1

        def reset(self, seed=None, options=None):
            return (
                {"drone_0": {"vec": np.zeros(2, dtype=np.float32)}},
                {"drone_0": {"alive": 1.0, "finished": 0.0}},
            )

        def step(self, actions):
            return (
                {"drone_0": {"vec": np.zeros(2, dtype=np.float32)}},
                {"drone_0": 0.0},
                {"drone_0": True},
                {"drone_0": False},
                {"drone_0": {"alive": 1.0, "finished": 0.0}},
            )

    class FakeRecurrentPPO:
        def __init__(self):
            self.policy = SimpleNamespace(lstm_actor=object())
            self.observation_space = spaces.Dict({"vec": spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)})
            self.calls = []

        def predict(self, obs, **kwargs):
            self.calls.append(kwargs)
            return np.zeros(2, dtype=np.float32), "next-state"

    class FakeWriter:
        def __init__(self):
            self.frames = []
            self.closed = False

        def append_data(self, frame):
            self.frames.append(frame)

        def close(self):
            self.closed = True

    class FakeRenderer:
        def __init__(self, config, screen_size):
            self.config = config
            self.screen_size = screen_size

        def render_array(self, env, overlay=None, trails=None, agent_idx=0):
            return np.zeros((2, 2, 3), dtype=np.uint8)

    class FakeOverlayFlags:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    writer = FakeWriter()
    monkeypatch.setattr(es, "imageio", SimpleNamespace(get_writer=lambda *args, **kwargs: writer))
    fake_renderer_module = ModuleType("ui.renderer")
    fake_renderer_module.OverlayFlags = FakeOverlayFlags
    fake_renderer_module.PygameRenderer = FakeRenderer
    monkeypatch.setitem(sys.modules, "ui.renderer", fake_renderer_module)

    model = FakeRecurrentPPO()
    es.run_episode_gif(
        FakeEnv(),
        policy=None,
        scene={"id": "unit", "max_steps": 1},
        seed=0,
        ppo_model=model,
        deterministic=True,
        gif_path=tmp_path / "out.gif",
    )

    assert writer.closed is True
    assert len(writer.frames) == 2
    assert len(model.calls) == 1
    assert model.calls[0]["state"] is None
    np.testing.assert_array_equal(model.calls[0]["episode_start"], np.asarray([True], dtype=bool))
