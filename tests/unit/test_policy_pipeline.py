from types import SimpleNamespace

import numpy as np

from baselines.policies import (
    ControllerStage,
    PerceptionAdapter,
    PerceptionStage,
    PlannerAdapter,
    PlannerPolicy,
    PlannerStage,
    PlanResult,
    PolicyPipeline,
    WaypointControllerStage,
)


def test_policy_pipeline_calls_stages_in_order():
    calls = []

    class SpyPerception(PerceptionStage):
        def prepare(self, agent_id, obs, state, info=None):
            calls.append("perception")
            vec = np.asarray(obs["vector"], dtype=np.float32) * 2.0
            return {"vector": vec, "grid": obs.get("grid")}

    class SpyPlanner(PlannerStage):
        def plan(self, agent_id, obs, state, info=None):
            calls.append("planner")
            assert np.allclose(obs["vector"], np.array([2.0, 4.0], dtype=np.float32))
            return PlanResult(desired=np.array([1.0, -1.0], dtype=np.float32), extra={"dist_m": 1.0})

    class SpyController(ControllerStage):
        def control(self, agent_id, desired_vec, obs, state, info=None, *, extra=None):
            calls.append("controller")
            extra = extra or {}
            assert extra.get("dist_m") == 1.0
            return np.asarray(desired_vec, dtype=np.float32) + 1.0

    pipeline = PolicyPipeline(
        perception=SpyPerception(),
        planner=SpyPlanner(),
        controller=SpyController(),
    )
    state = SimpleNamespace()
    obs = {"vector": np.array([1.0, 2.0], dtype=np.float32), "grid": None}
    action = pipeline.step("drone_0", obs, state, {})

    assert calls == ["perception", "planner", "controller"]
    assert np.allclose(action, np.array([2.0, 0.0], dtype=np.float32))


def test_planner_adapter_uses_plan():
    calls = []

    class DummyPlanner:
        def plan(self, agent_id, obs, state, info=None):
            calls.append("plan")
            return np.array([0.2, 0.0], dtype=np.float32)

    adapter = PlannerAdapter(DummyPlanner())
    obs = {"vector": np.zeros((2,), dtype=np.float32), "grid": None}
    out = adapter.plan("drone_0", obs, SimpleNamespace(), {})

    assert calls == ["plan"]
    assert np.allclose(out.desired, np.array([0.2, 0.0], dtype=np.float32))


def test_planner_policy_as_pipeline():
    class DummyPolicy(PlannerPolicy):
        def plan(self, agent_id, obs, state, info=None):
            return np.array([0.0, 0.0], dtype=np.float32)

    policy = DummyPolicy()
    pipeline = policy.as_pipeline()

    assert isinstance(pipeline, PolicyPipeline)
    assert isinstance(pipeline.planner, PlannerAdapter)
    assert isinstance(pipeline.perception, PerceptionAdapter)
    assert isinstance(pipeline.controller, WaypointControllerStage)
