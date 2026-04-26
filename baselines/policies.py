from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from baselines.controllers import WaypointController
from baselines.params import get_best_params
from baselines.potential_fields import PotentialFieldPolicy
from baselines.utils import (
    agent_index_from_id,
    grid_avoid_from_cost,
    maybe_accel_action,
    normalize,
    obs_to_target,
    obs_vel,
    obs_walls,
    split_obs,
    velocity_tracking_action_batch,
    wall_avoid_from_distances,
)
from common.policy.context import PolicyContext
from common.policy.specs import (
    BaselinePolicySpec,
    InfoRegime,
    OracleCapability,
    canonical_policy_name,
    get_policy_spec,
    register_policy_spec,
)

PolicyFactory = Callable[..., "BasePolicy"]
_POLICY_FACTORIES: dict[str, PolicyFactory] = {}
ObsDict = dict[str, np.ndarray]


@dataclass(frozen=True)
class PlanResult:
    """Результат планирования: desired‑вектор + опциональные метаданные."""

    desired: np.ndarray
    extra: dict[str, Any] | None = None


def plan_result_from(value: Any) -> PlanResult:
    """Нормализует формат plan(): ndarray | (ndarray, dict) | PlanResult."""
    if isinstance(value, PlanResult):
        return value
    extra: dict[str, Any] | None = None
    desired = value
    if isinstance(value, tuple) and value:
        desired = value[0]
        if len(value) > 1 and isinstance(value[1], dict):
            extra = value[1]
    desired_arr = np.asarray(desired, dtype=np.float32)
    return PlanResult(desired=desired_arr, extra=extra)


def _build_default_controller() -> WaypointController:
    return WaypointController(
        goal_radius_control=4.0,
        near_goal_speed_cap=0.6,
        near_goal_damping=0.7,
        near_goal_kp=0.8,
        risk_speed_scale=0.65,
        risk_speed_floor=0.3,
    )


def register_policy(
    name: str,
    *,
    info_regime: InfoRegime | None = None,
    oracle_capability: OracleCapability | None = None,
    map_aware: bool | None = None,
    privileged_reference: bool | None = None,
    notes: str | None = None,
):
    def _decorator(factory: PolicyFactory):
        _POLICY_FACTORIES[name] = factory
        base_spec = get_policy_spec(name)
        register_policy_spec(
            name=name,
            info_regime=info_regime if info_regime is not None else getattr(base_spec, "info_regime", "fair"),
            oracle_capability=oracle_capability
            if oracle_capability is not None
            else getattr(base_spec, "oracle_capability", "none"),
            map_aware=map_aware if map_aware is not None else bool(getattr(base_spec, "map_aware", False)),
            privileged_reference=privileged_reference
            if privileged_reference is not None
            else bool(getattr(base_spec, "privileged_reference", False)),
            notes=notes if notes is not None else str(getattr(base_spec, "notes", "")),
        )
        return factory

    return _decorator


def registered_policy_names() -> tuple[str, ...]:
    return tuple(sorted(_POLICY_FACTORIES.keys()))


class BasePolicy:
    """Контракт политики: reset/set_context/prepare_obs/step/cleanup."""

    def reset(self, seed: int | None = None) -> None:
        return None

    def set_context(self, state: PolicyContext | None = None) -> None:
        """Опциональный контекст среды (для планировщиков)."""
        return None

    def _ensure_info(self, info: dict | None, state: PolicyContext | None) -> dict:
        out = {} if info is None else info
        if state is None:
            return out
        if "control_mode" not in out:
            out["control_mode"] = getattr(state, "control_mode", "waypoint")
        if "max_speed" not in out:
            out["max_speed"] = getattr(state, "max_speed", 0.0)
        if "max_accel" not in out:
            out["max_accel"] = getattr(state, "max_accel", 0.0)
        if "dt" not in out:
            out["dt"] = getattr(state, "dt", 0.0)
        if "drag" not in out:
            out["drag"] = getattr(state, "drag", 0.0)
        if "grid_res" not in out:
            out["grid_res"] = getattr(state, "grid_res", 1.0)
        return out

    def _apply_controller(
        self,
        desired_vec: np.ndarray,
        obs: ObsDict,
        state: PolicyContext | None,
        info: dict | None,
        *,
        to_target: np.ndarray | None = None,
        dist_m: float | None = None,
        in_goal: bool | None = None,
        risk_p: float | None = None,
        dist_normed: bool = False,
    ) -> np.ndarray:
        controller = getattr(self, "_controller", None)
        stop_risk_threshold = float(getattr(self, "stop_risk_threshold", 0.4))
        return _apply_controller_core(
            controller,
            stop_risk_threshold,
            desired_vec,
            obs,
            state,
            info,
            to_target=to_target,
            dist_m=dist_m,
            in_goal=in_goal,
            risk_p=risk_p,
            dist_normed=dist_normed,
        )

    def prepare_obs(
        self,
        agent_id: str,
        obs: ObsDict,
        state: PolicyContext,
        info: dict | None = None,
    ) -> ObsDict:
        return obs

    def step(
        self,
        agent_id: str,
        obs: ObsDict,
        state: PolicyContext,
        info: dict | None = None,
    ) -> np.ndarray:
        info = self._ensure_info(info, state)
        obs = self.prepare_obs(agent_id, obs, state, info)
        return self.get_action(agent_id, obs, state, info)

    def step_batch(
        self,
        obs_map: dict[str, ObsDict],
        state: PolicyContext,
        infos: dict | None = None,
    ) -> dict[str, np.ndarray]:
        actions: dict[str, np.ndarray] = {}
        infos = infos or {}
        for agent_id, obs in obs_map.items():
            info = self._ensure_info(infos.get(agent_id), state)
            actions[agent_id] = self.step(agent_id, obs, state, info)
        return actions

    def cleanup(self) -> None:
        return None

    def get_action(
        self,
        agent_id: str,
        obs: ObsDict,
        state: PolicyContext,
        info: dict | None = None,
    ) -> np.ndarray:
        raise NotImplementedError

    def get_actions(
        self,
        obs_map: dict[str, ObsDict],
        state: PolicyContext,
        infos: dict | None = None,
    ) -> dict[str, np.ndarray]:
        return self.step_batch(obs_map, state, infos)


class PerceptionStage:
    """Стадия Perception: нормализует/объединяет наблюдения перед планированием."""

    def reset(self, seed: int | None = None) -> None:
        return None

    def set_context(self, state: PolicyContext | None = None) -> None:
        return None

    def prepare(
        self,
        agent_id: str,
        obs: ObsDict,
        state: PolicyContext,
        info: dict | None = None,
    ) -> ObsDict:
        return obs


class IdentityPerception(PerceptionStage):
    """Perception по умолчанию: возвращает obs как есть."""


class PerceptionAdapter(PerceptionStage):
    """Адаптер для старых политик с prepare_obs или пользовательской функции."""

    def __init__(self, policy: object | Callable[..., ObsDict]):
        self._policy = policy

    def reset(self, seed: int | None = None) -> None:
        if hasattr(self._policy, "reset"):
            try:
                self._policy.reset(seed)
            except Exception:
                self._policy.reset()

    def set_context(self, state: PolicyContext | None = None) -> None:
        if hasattr(self._policy, "set_context"):
            self._policy.set_context(state)

    def prepare(
        self,
        agent_id: str,
        obs: ObsDict,
        state: PolicyContext,
        info: dict | None = None,
    ) -> ObsDict:
        if callable(self._policy) and not hasattr(self._policy, "prepare_obs"):
            return self._policy(agent_id, obs, state, info)
        if hasattr(self._policy, "prepare_obs"):
            return self._policy.prepare_obs(agent_id, obs, state, info)
        return obs


class PlannerStage:
    """Стадия Planner: выдаёт desired‑вектор (или desired + метаданные)."""

    def reset(self, seed: int | None = None) -> None:
        return None

    def set_context(self, state: PolicyContext | None = None) -> None:
        return None

    def plan(
        self,
        agent_id: str,
        obs: ObsDict,
        state: PolicyContext,
        info: dict | None = None,
    ) -> PlanResult:
        raise NotImplementedError


class PlannerAdapter(PlannerStage):
    """Адаптер планировщика для политик с plan/get_action/predict."""

    def __init__(self, policy: object):
        self._policy = policy

    def reset(self, seed: int | None = None) -> None:
        if hasattr(self._policy, "reset"):
            try:
                self._policy.reset(seed)
            except Exception:
                self._policy.reset()

    def set_context(self, state: PolicyContext | None = None) -> None:
        if hasattr(self._policy, "set_context"):
            self._policy.set_context(state)

    def plan(
        self,
        agent_id: str,
        obs: ObsDict,
        state: PolicyContext,
        info: dict | None = None,
    ) -> PlanResult:
        if hasattr(self._policy, "plan"):
            return plan_result_from(self._policy.plan(agent_id, obs, state, info))
        if hasattr(self._policy, "get_action"):
            return plan_result_from(self._policy.get_action(agent_id, obs, state, info))
        if hasattr(self._policy, "predict"):
            act, _ = self._policy.predict(obs)
            return plan_result_from(np.asarray(act, dtype=np.float32))
        raise NotImplementedError("PlannerAdapter: нет plan/get_action/predict")


class ControllerStage:
    """Стадия Controller: превращает desired‑вектор в действие среды."""

    def reset(self, seed: int | None = None) -> None:
        return None

    def set_context(self, state: PolicyContext | None = None) -> None:
        return None

    def control(
        self,
        agent_id: str,
        desired_vec: np.ndarray,
        obs: ObsDict,
        state: PolicyContext | None,
        info: dict | None = None,
        *,
        extra: dict | None = None,
    ) -> np.ndarray:
        raise NotImplementedError


class DirectControllerStage(ControllerStage):
    """Прямой контроллер: нормализация desired и проброс в режим управления."""

    def control(
        self,
        agent_id: str,
        desired_vec: np.ndarray,
        obs: ObsDict,
        state: PolicyContext | None,
        info: dict | None = None,
        *,
        extra: dict | None = None,
    ) -> np.ndarray:
        return maybe_accel_action(normalize(desired_vec), obs, info)


class WaypointControllerStage(ControllerStage):
    """Контроллер waypoint с торможением у цели и риск‑масштабированием."""

    def __init__(
        self,
        *,
        controller: WaypointController | None = None,
        stop_risk_threshold: float = 0.4,
    ) -> None:
        self._controller = controller or _build_default_controller()
        self.stop_risk_threshold = float(stop_risk_threshold)

    def control(
        self,
        agent_id: str,
        desired_vec: np.ndarray,
        obs: ObsDict,
        state: PolicyContext | None,
        info: dict | None = None,
        *,
        extra: dict | None = None,
    ) -> np.ndarray:
        extra = extra or {}
        return _apply_controller_core(
            self._controller,
            self.stop_risk_threshold,
            desired_vec,
            obs,
            state,
            info,
            to_target=extra.get("to_target"),
            dist_m=extra.get("dist_m"),
            in_goal=extra.get("in_goal"),
            risk_p=extra.get("risk_p"),
            dist_normed=bool(extra.get("dist_normed", False)),
        )


class PolicyPipeline(BasePolicy):
    """Композиция Perception → Planner → Controller."""

    def __init__(
        self,
        *,
        planner: PlannerStage,
        perception: PerceptionStage | None = None,
        controller: ControllerStage | None = None,
    ) -> None:
        self.perception = perception or IdentityPerception()
        self.planner = planner
        self.controller = controller or DirectControllerStage()

    def reset(self, seed: int | None = None) -> None:
        self.perception.reset(seed)
        self.planner.reset(seed)
        self.controller.reset(seed)

    def set_context(self, state: PolicyContext | None = None) -> None:
        self.perception.set_context(state)
        self.planner.set_context(state)
        self.controller.set_context(state)

    def prepare_obs(
        self,
        agent_id: str,
        obs: ObsDict,
        state: PolicyContext,
        info: dict | None = None,
    ) -> ObsDict:
        return self.perception.prepare(agent_id, obs, state, info)

    def step(
        self,
        agent_id: str,
        obs: ObsDict,
        state: PolicyContext,
        info: dict | None = None,
    ) -> np.ndarray:
        info = self._ensure_info(info, state)
        prepared = self.perception.prepare(agent_id, obs, state, info)
        plan_res = plan_result_from(self.planner.plan(agent_id, prepared, state, info))
        extra = plan_res.extra or {}
        return self.controller.control(
            agent_id,
            plan_res.desired,
            prepared,
            state,
            info,
            extra=extra,
        )

    def step_batch(
        self,
        obs_map: dict[str, ObsDict],
        state: PolicyContext,
        infos: dict | None = None,
    ) -> dict[str, np.ndarray]:
        actions: dict[str, np.ndarray] = {}
        infos = infos or {}
        for agent_id, obs in obs_map.items():
            info = self._ensure_info(infos.get(agent_id), state)
            actions[agent_id] = self.step(agent_id, obs, state, info)
        return actions


class PlannerPolicy(BasePolicy):
    """Политика‑планировщик: возвращает только desired‑вектор, контроллер рулит действием."""

    def plan(
        self,
        agent_id: str,
        obs: ObsDict,
        state: PolicyContext,
        info: dict | None = None,
    ) -> PlanResult:
        raise NotImplementedError

    def get_action(
        self,
        agent_id: str,
        obs: ObsDict,
        state: PolicyContext,
        info: dict | None = None,
    ) -> np.ndarray:
        info = self._ensure_info(info, state)
        obs = self.prepare_obs(agent_id, obs, state, info)
        plan_res = plan_result_from(self.plan(agent_id, obs, state, info))
        extra = plan_res.extra or {}
        allowed = {"to_target", "dist_m", "in_goal", "risk_p", "dist_normed"}
        extra = {key: value for key, value in extra.items() if key in allowed}
        return self._apply_controller(plan_res.desired, obs, state, info, **extra)

    def step_batch(
        self,
        obs_map: dict[str, ObsDict],
        state: PolicyContext,
        infos: dict | None = None,
    ) -> dict[str, np.ndarray]:
        if state is None or getattr(self, "_controller", None) is None:
            return super().step_batch(obs_map, state, infos)
        infos = infos or {}
        agent_ids = list(obs_map.keys())
        n = len(agent_ids)
        desired = np.zeros((n, 2), dtype=np.float32)
        to_target = np.zeros((n, 2), dtype=np.float32)
        dist_m = np.full((n,), np.nan, dtype=np.float32)
        in_goal = np.zeros((n,), dtype=bool)
        risk_p = np.full((n,), np.nan, dtype=np.float32)
        cur_vel = np.zeros((n, 2), dtype=np.float32)
        obs_list: list[ObsDict] = []
        info_list: list[dict] = []

        for i, agent_id in enumerate(agent_ids):
            obs = obs_map[agent_id]
            info = self._ensure_info(infos.get(agent_id), state)
            info_list.append(info)
            prepared = self.prepare_obs(agent_id, obs, state, info)
            obs_list.append(prepared)
            plan_res = plan_result_from(self.plan(agent_id, prepared, state, info))
            desired[i] = plan_res.desired
            extra = plan_res.extra or {}

            idx = agent_index_from_id(agent_id, info)
            if idx is None:
                idx = 0
            if state.dists is not None and 0 <= idx < state.dists.shape[0]:
                dist_m[i] = float(state.dists[idx])
            if state.in_goal is not None and 0 <= idx < state.in_goal.shape[0]:
                in_goal[i] = bool(state.in_goal[idx])
            if state.risk_p is not None and 0 <= idx < state.risk_p.shape[0]:
                risk_p[i] = float(state.risk_p[idx])
            if getattr(state, "vel", None) is not None and 0 <= idx < state.vel.shape[0]:
                cur_vel[i] = state.vel[idx]
            else:
                cur_vel[i] = obs_vel(prepared) * float(getattr(state, "max_speed", 0.0))

            if extra:
                if extra.get("to_target") is not None:
                    to_target[i] = np.asarray(extra["to_target"], dtype=np.float32)
                if extra.get("dist_m") is not None:
                    dist_m[i] = float(extra["dist_m"])
                if extra.get("in_goal") is not None:
                    in_goal[i] = bool(extra["in_goal"])
                if extra.get("risk_p") is not None:
                    risk_p[i] = float(extra["risk_p"])
            if not np.isfinite(dist_m[i]):
                dist_m[i] = float("nan")
            if not np.isfinite(risk_p[i]):
                risk_p[i] = 0.0
            if not np.any(to_target[i]):
                to_target[i] = obs_to_target(prepared)

        controller: WaypointController = getattr(self, "_controller", None)
        desired_actions = controller.compute_batch(
            desired,
            to_target,
            dist_m,
            False,
            float(getattr(state, "field_size", 0.0)),
            in_goal=in_goal,
            risk_p=risk_p,
            stop_risk_threshold=float(getattr(self, "stop_risk_threshold", 0.4)),
            vel=cur_vel,
        )
        accel_actions = velocity_tracking_action_batch(
            desired_actions,
            cur_vel,
            max_speed=float(getattr(state, "max_speed", 0.0)),
            max_accel=float(getattr(state, "max_accel", 0.0)),
            dt=float(getattr(state, "dt", 0.0)),
            drag=float(getattr(state, "drag", 0.0)),
            obs_list=obs_list,
            infos=info_list,
        )
        return {agent_id: accel_actions[i] for i, agent_id in enumerate(agent_ids)}

    def as_pipeline(
        self,
        *,
        perception: PerceptionStage | None = None,
        controller: ControllerStage | None = None,
    ) -> PolicyPipeline:
        """Строит пайплайн Perception → Planner → Controller вокруг планировщика."""
        if perception is None:
            perception = PerceptionAdapter(self)
        if controller is None:
            controller = WaypointControllerStage(
                controller=getattr(self, "_controller", None),
                stop_risk_threshold=float(getattr(self, "stop_risk_threshold", 0.4)),
            )
        planner = PlannerAdapter(self)
        return PolicyPipeline(perception=perception, planner=planner, controller=controller)


class PolicyAdapter(BasePolicy):
    """Адаптер для внешних политик без полного контракта BasePolicy."""

    def __init__(self, policy: object):
        self._policy = policy

    def reset(self, seed: int | None = None) -> None:
        if hasattr(self._policy, "reset"):
            try:
                self._policy.reset(seed)
            except Exception:
                self._policy.reset()

    def set_context(self, state: PolicyContext | None = None) -> None:
        if hasattr(self._policy, "set_context"):
            self._policy.set_context(state)

    def prepare_obs(
        self,
        agent_id: str,
        obs: ObsDict,
        state: PolicyContext,
        info: dict | None = None,
    ) -> ObsDict:
        if hasattr(self._policy, "prepare_obs"):
            return self._policy.prepare_obs(agent_id, obs, state, info)
        return obs

    def step(
        self,
        agent_id: str,
        obs: ObsDict,
        state: PolicyContext,
        info: dict | None = None,
    ) -> np.ndarray:
        obs = self.prepare_obs(agent_id, obs, state, info)
        if hasattr(self._policy, "step"):
            return self._policy.step(agent_id, obs, state, info)
        if hasattr(self._policy, "get_action"):
            return self._policy.get_action(agent_id, obs, state, info)
        if hasattr(self._policy, "predict"):
            act, _ = self._policy.predict(obs)
            return np.asarray(act, dtype=np.float32)
        raise NotImplementedError("PolicyAdapter: no compatible step/get_action/predict method")

    def step_batch(
        self,
        obs_map: dict[str, ObsDict],
        state: PolicyContext,
        infos: dict | None = None,
    ) -> dict[str, np.ndarray]:
        if hasattr(self._policy, "step_batch"):
            return self._policy.step_batch(obs_map, state, infos)
        if hasattr(self._policy, "get_actions"):
            return self._policy.get_actions(obs_map, state, infos)
        return super().step_batch(obs_map, state, infos)

    def cleanup(self) -> None:
        if hasattr(self._policy, "cleanup"):
            self._policy.cleanup()


def _apply_controller_core(
    controller: WaypointController | None,
    stop_risk_threshold: float,
    desired_vec: np.ndarray,
    obs: ObsDict,
    state: PolicyContext | None,
    info: dict | None,
    *,
    to_target: np.ndarray | None = None,
    dist_m: float | None = None,
    in_goal: bool | None = None,
    risk_p: float | None = None,
    dist_normed: bool = False,
) -> np.ndarray:
    if info is None:
        info = {}
    if controller is None or state is None:
        return maybe_accel_action(normalize(desired_vec), obs, info)
    idx = agent_index_from_id(info.get("agent_id", ""), info)
    if idx is None and info.get("agent_index") is None and state is not None:
        idx = agent_index_from_id("", info)
    if idx is None:
        idx = 0
    if dist_m is None and state is not None and state.dists is not None and idx < state.dists.shape[0]:
        dist_m = float(state.dists[idx])
    if in_goal is None and state is not None and state.in_goal is not None and idx < state.in_goal.shape[0]:
        in_goal = bool(state.in_goal[idx])
    if risk_p is None and state is not None and state.risk_p is not None and idx < state.risk_p.shape[0]:
        risk_p = float(state.risk_p[idx])
    if "cur_vel" not in info and state is not None and getattr(state, "vel", None) is not None:
        try:
            if 0 <= idx < state.vel.shape[0]:
                info["cur_vel"] = state.vel[idx]
        except Exception:
            pass
    return controller.compute_action(
        desired_vec,
        obs,
        dist_m,
        dist_normed,
        float(getattr(state, "field_size", 0.0)),
        in_goal=bool(in_goal) if in_goal is not None else False,
        risk_p=float(risk_p) if risk_p is not None else 0.0,
        stop_risk_threshold=float(stop_risk_threshold),
        info=info,
        to_target=to_target,
    )


@register_policy("baseline:random")
class RandomPolicy(PlannerPolicy):
    def __init__(self, seed: int = 0):
        self.rng = np.random.RandomState(seed)
        self.stop_risk_threshold = 0.4
        self._controller = _build_default_controller()

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.rng = np.random.RandomState(int(seed))

    def plan(self, agent_id: str, obs: ObsDict, state: PolicyContext, info: dict | None = None) -> np.ndarray:
        return self.rng.uniform(-1.0, 1.0, size=(2,)).astype(np.float32)


@register_policy("baseline:zero")
class ZeroPolicy(PlannerPolicy):
    def __init__(self):
        self.stop_risk_threshold = 0.4
        self._controller = _build_default_controller()

    def plan(self, agent_id: str, obs: ObsDict, state: PolicyContext, info: dict | None = None) -> np.ndarray:
        return np.zeros((2,), dtype=np.float32)


@register_policy("baseline:greedy")
class GreedyToTargetPolicy(PlannerPolicy):
    def __init__(self):
        self.stop_risk_threshold = 0.4
        self._controller = _build_default_controller()

    def plan(self, agent_id: str, obs: ObsDict, state: PolicyContext, info: dict | None = None) -> np.ndarray:
        info = info or {}
        target = info.get("target_pos")
        if target is None and state is not None:
            target = state.target_pos
        pos = info.get("pos")
        if pos is None and state is not None:
            idx = agent_index_from_id(agent_id, info)
            if idx is not None and 0 <= idx < state.pos.shape[0]:
                pos = state.pos[idx]
        if target is not None and pos is not None:
            return np.asarray(target, dtype=np.float32) - np.asarray(pos, dtype=np.float32)
        return obs_to_target(obs)


@register_policy("baseline:greedy_safe")
class GreedySafePolicy(PlannerPolicy):
    def __init__(
        self,
        w_goal: float = 1.0,
        w_avoid: float = 0.7,
        w_wall: float = 0.4,
        grid_radius: int = 2,
        **_,
    ):
        self.w_goal = float(w_goal)
        self.w_avoid = float(w_avoid)
        self.w_wall = float(w_wall)
        self.grid_radius = int(max(1, grid_radius))
        self.stop_risk_threshold = 0.4
        self._controller = _build_default_controller()

    def plan(self, agent_id: str, obs: ObsDict, state: PolicyContext, info: dict | None = None) -> np.ndarray:
        to_target, _, walls, grid = split_obs(obs)
        goal_vec = normalize(to_target)
        avoid_vec = (
            grid_avoid_from_cost(grid, radius=self.grid_radius) if grid is not None else np.zeros(2, dtype=np.float32)
        )
        wall_vec = wall_avoid_from_distances(walls)

        return (self.w_goal * goal_vec) + (self.w_avoid * avoid_vec) + (self.w_wall * wall_vec)


class PotentialFieldsPolicy(PlannerPolicy):
    def __init__(
        self,
        n_agents: int,
        k_att: float = 1.0,
        k_rep: float = 50.0,
        safety_margin: float = 5.0,
        stop_risk_threshold: float = 0.4,
        goal_radius_control: float = 4.0,
        near_goal_speed_cap: float = 0.6,
        near_goal_damping: float = 0.7,
        near_goal_kp: float = 0.8,
        risk_speed_scale: float = 0.65,
        risk_speed_floor: float = 0.3,
    ):
        self._pf = PotentialFieldPolicy(n_agents=n_agents, k_att=k_att, k_rep=k_rep, safety_margin=safety_margin)
        self.stop_risk_threshold = float(stop_risk_threshold)
        self._controller = WaypointController(
            goal_radius_control=goal_radius_control,
            near_goal_speed_cap=near_goal_speed_cap,
            near_goal_damping=near_goal_damping,
            near_goal_kp=near_goal_kp,
            risk_speed_scale=risk_speed_scale,
            risk_speed_floor=risk_speed_floor,
        )
        self._cache_step: int | None = None
        self._cache_actions: np.ndarray | None = None

    def reset(self, seed: int | None = None) -> None:
        self._pf.reset(seed)
        self._cache_step = None
        self._cache_actions = None

    def _get_cached_actions(self, state: PolicyContext) -> np.ndarray:
        step = int(getattr(state, "decision_step", getattr(state, "timestep", 0)))
        if (
            self._cache_actions is None
            or self._cache_step != step
            or self._cache_actions.shape[0] != state.pos.shape[0]
        ):
            self._cache_actions = self._pf.get_actions(
                state.pos,
                state.alive,
                state.target_pos,
                state.threats,
                walls=state.static_walls,
                target_vel=state.target_vel,
                max_speed=state.max_speed,
                agents_vel=state.vel,
                max_accel=state.max_accel,
                dt=state.dt,
                drag=state.drag,
                oracle_dir=state.oracle_dir,
                output_mode="desired",
            )
            self._cache_step = step
        return self._cache_actions

    def plan(self, agent_id: str, obs: ObsDict, state: PolicyContext, info: dict | None = None) -> np.ndarray:
        idx = agent_index_from_id(agent_id, info)
        if idx is None or idx < 0 or idx >= state.pos.shape[0]:
            return np.zeros((2,), dtype=np.float32)
        acts = self._get_cached_actions(state)
        if idx >= acts.shape[0]:
            return np.zeros((2,), dtype=np.float32)
        return np.asarray(acts[idx], dtype=np.float32)

    def get_actions(
        self,
        obs_map: dict[str, ObsDict],
        state: PolicyContext,
        infos: dict | None = None,
    ) -> dict[str, np.ndarray]:
        infos = infos or {}
        acts = self._get_cached_actions(state)
        actions: dict[str, np.ndarray] = {}
        for agent_id, obs in obs_map.items():
            info = self._ensure_info(infos.get(agent_id), state)
            idx = agent_index_from_id(agent_id, info)
            if idx is None or idx < 0 or idx >= acts.shape[0]:
                actions[agent_id] = np.zeros((2,), dtype=np.float32)
                continue
            to_target = obs_to_target(obs)
            actions[agent_id] = self._apply_controller(
                acts[idx],
                obs,
                state,
                info,
                to_target=to_target,
            ).astype(np.float32)
        return actions

    def act_batch(
        self,
        agents_pos: np.ndarray,
        active_mask: np.ndarray,
        target_pos: np.ndarray,
        threats: list,
        *,
        walls: list | None = None,
        target_vel: np.ndarray | list | None = None,
        max_speed: float | None = None,
        agents_vel: np.ndarray | None = None,
        max_accel: float | None = None,
        dt: float | None = None,
        drag: float | None = None,
    ) -> np.ndarray:
        # Пакетный вызов используется в векторизованных оценках.
        return self._pf.get_actions(
            agents_pos,
            active_mask,
            target_pos,
            threats,
            walls=walls,
            target_vel=target_vel,
            max_speed=max_speed,
            agents_vel=agents_vel,
            max_accel=max_accel,
            dt=dt,
            drag=drag,
        )


if __name__ == "__main__":
    pass


@register_policy("baseline:wall")
class WallAvoidPolicy(PlannerPolicy):
    def __init__(self):
        self.stop_risk_threshold = 0.4
        self._controller = _build_default_controller()

    def plan(self, agent_id: str, obs: ObsDict, state: PolicyContext, info: dict | None = None) -> np.ndarray:
        walls = obs_walls(obs)
        left, right, down, up = walls
        return np.array([right - left, up - down], dtype=np.float32)


@register_policy("baseline:brake")
class BrakeNearGoalPolicy(PlannerPolicy):
    def __init__(self):
        self.stop_risk_threshold = 0.4
        self._controller = _build_default_controller()

    def plan(self, agent_id: str, obs: ObsDict, state: PolicyContext, info: dict | None = None) -> np.ndarray:
        to_target = obs_to_target(obs)
        dist = float(np.linalg.norm(to_target))
        if dist < 0.05:
            return np.zeros((2,), dtype=np.float32)
        return to_target


class PolicyRegistry:
    def __init__(self):
        self._factories = {}

    def register(self, name: str, factory):
        self._factories[name] = factory

    def create(self, name: str, **kwargs):
        canonical = canonical_policy_name(name)
        if canonical not in self._factories:
            raise KeyError(f"Неизвестная политика: {name}")
        # Лучшие параметры подмешиваются по умолчанию, но их можно переопределить.
        params = {}
        params.update(get_best_params(canonical))
        params.update(kwargs)
        policy = self._factories[canonical](**params)
        spec = get_policy_spec(canonical)
        try:
            policy.policy_name = canonical
            policy.policy_spec = spec
        except Exception:
            pass
        return policy


@register_policy("baseline:astar_grid")
def _make_astar_grid(**kwargs):
    from baselines.astar_grid import AStarGridPolicy

    return AStarGridPolicy(**kwargs)


@register_policy("baseline:mpc_lite")
def _make_mpc_lite(**kwargs):
    from baselines.mpc_lite import MPCLitePolicy

    return MPCLitePolicy(**kwargs)


@register_policy("baseline:separation_steering")
def _make_separation_steering(**kwargs):
    from baselines.separation_steering import SeparationSteeringPolicy

    return SeparationSteeringPolicy(**kwargs)


@register_policy("baseline:potential_fields")
def _make_potential_fields(n_agents=1, **kwargs):
    return PotentialFieldsPolicy(n_agents=n_agents, **kwargs)


@register_policy("baseline:flow_field")
def _make_flow_field(**kwargs):
    from baselines.flow_field import FlowFieldPolicy

    return FlowFieldPolicy(**kwargs)


def default_registry() -> PolicyRegistry:
    # Единая точка регистрации нужна для командного интерфейса, веб‑интерфейса и бенчмарков.
    reg = PolicyRegistry()
    for name, factory in _POLICY_FACTORIES.items():
        reg.register(name, factory)
    return reg


def adapt_policy(policy: object) -> BasePolicy:
    if isinstance(policy, BasePolicy):
        return policy
    return PolicyAdapter(policy)
