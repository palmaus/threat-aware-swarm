from __future__ import annotations

import pytest

import common.runtime.env_factory as env_factory


def test_make_pz_env_fails_closed_on_runtime_curriculum_error(monkeypatch):
    closed = {"value": False}

    class DummyEnv:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def apply_curriculum(self, payload):
            raise ValueError(f"boom:{payload}")

        def close(self):
            closed["value"] = True

    monkeypatch.setattr(env_factory, "SwarmPZEnv", DummyEnv)

    with pytest.raises(RuntimeError, match="apply failed"):
        env_factory.make_pz_env(
            max_steps=5,
            goal_radius=3.0,
            reset=False,
            initial_stage_params={"goal_radius": 9.0},
        )

    assert closed["value"] is True
