"""Process-backed policy helpers for the web UI controller."""

from __future__ import annotations

import inspect
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np

from baselines.factory import create_baseline_policy
from baselines.policies import canonical_policy_name
from common.policy.context import PolicyContextData, context_from_payload
from common.runtime.episode_runner import build_info_batch, build_policy_context
from env.pz_env import SwarmPZEnv

logger = logging.getLogger(__name__)

_ASTAR_WORKER_POLICY = None
_MPC_WORKER_POLICY = None
_WORKER_ERROR_KEY = "__worker_error__"


def _astar_worker_init(policy_name: str, policy_params: dict) -> None:
    global _ASTAR_WORKER_POLICY

    _ASTAR_WORKER_POLICY = create_baseline_policy(policy_name, params=policy_params)


def _astar_worker_action(payload: tuple[str, np.ndarray, dict, dict]) -> np.ndarray:
    if _ASTAR_WORKER_POLICY is None:
        return np.zeros((2,), dtype=np.float32)
    agent_id, obs, info, sim_payload = payload
    context = _payload_to_context(sim_payload)
    _ASTAR_WORKER_POLICY.set_context(context)
    if hasattr(_ASTAR_WORKER_POLICY, "step"):
        return _ASTAR_WORKER_POLICY.step(agent_id, obs, context, info)
    return _ASTAR_WORKER_POLICY.get_action(agent_id, obs, context, info)


def _astar_worker_reset(seed: int | None) -> None:
    if _ASTAR_WORKER_POLICY is None:
        return
    if hasattr(_ASTAR_WORKER_POLICY, "reset"):
        _ASTAR_WORKER_POLICY.reset(seed)


def _mpc_worker_init(policy_name: str, policy_params: dict, enable_jit: bool) -> None:
    global _MPC_WORKER_POLICY
    if enable_jit:
        os.environ["NUMBA_DISABLE_JIT"] = "0"
        os.environ.setdefault("NUMBA_NUM_THREADS", "1")
        os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")

    _MPC_WORKER_POLICY = create_baseline_policy(policy_name, params=policy_params)


def _payload_to_context(sim_payload: dict) -> PolicyContextData:
    include_oracle = bool(sim_payload.get("oracle_visible", True))
    return context_from_payload(sim_payload, include_oracle=include_oracle)


def _worker_error(exc: BaseException | str) -> dict:
    return {_WORKER_ERROR_KEY: str(exc)}


def _mpc_worker_action_batch(payload: tuple[dict, dict, dict]) -> dict:
    if _MPC_WORKER_POLICY is None:
        return _worker_error("MPC worker policy is not initialized")
    obs_map, info_map, sim_payload = payload
    try:
        context = _payload_to_context(sim_payload)
        _MPC_WORKER_POLICY.set_context(context)
        if hasattr(_MPC_WORKER_POLICY, "step_batch"):
            return dict(_MPC_WORKER_POLICY.step_batch(obs_map, context, info_map))
        if hasattr(_MPC_WORKER_POLICY, "get_actions"):
            return dict(_MPC_WORKER_POLICY.get_actions(obs_map, context, info_map))
        actions = {}
        for agent_id, obs in obs_map.items():
            info = info_map.get(agent_id, {})
            actions[agent_id] = _MPC_WORKER_POLICY.get_action(agent_id, obs, context, info)
        return actions
    except Exception as exc:  # pragma: no cover - child-process safety net
        return _worker_error(exc)


def _mpc_worker_reset(seed: int | None) -> None:
    if _MPC_WORKER_POLICY is None:
        return
    if hasattr(_MPC_WORKER_POLICY, "reset"):
        _MPC_WORKER_POLICY.reset(seed)


def policy_params_from_instance(policy) -> dict:
    seen: set[int] = set()
    while hasattr(policy, "_policy") and id(policy) not in seen:
        seen.add(id(policy))
        policy = getattr(policy, "_policy")

    params: dict = {}
    sig = inspect.signature(policy.__init__)
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            continue
        if hasattr(policy, name):
            params[name] = getattr(policy, name)
    return params


def build_policy_pool(policy, policy_name: str, env: SwarmPZEnv, ui_cfg, *, log: logging.Logger | None = None):
    """Build an optional process-backed adapter without leaking concrete classes into ui.controller."""

    workers = int(getattr(ui_cfg, "policy_workers", 0) or 0)
    log = log or logger
    canonical = canonical_policy_name(policy_name)
    if canonical == "baseline:mpc_lite":
        if workers <= 1:
            pool_workers = min(4, int(env.n_agents))
            log.info("MPC workers=auto -> %s процессов.", pool_workers)
        else:
            pool_workers = int(max(1, workers))
        params = policy_params_from_instance(policy)
        params["use_numba"] = True
        try:
            pool = MpcProcessPool(
                params,
                workers=pool_workers,
                enable_jit=True,
                policy_name=policy_name,
            )
            log.info("Запущен пул MPC на %s процессов.", pool_workers)
            return pool
        except (OSError, PermissionError, RuntimeError) as exc:
            log.warning("Не удалось запустить пул MPC: %s. Продолжаем без параллелизма.", exc)
            return None
    if workers <= 1:
        return None
    if canonical != "baseline:astar_grid":
        log.info("Параллельные workers=%s активны только для baseline:astar_grid.", workers)
        return None
    params = policy_params_from_instance(policy)
    try:
        pool = AStarProcessPool(
            params,
            workers=workers,
            policy_name=policy_name,
        )
        log.info("Запущен пул A* на %s процессов.", workers)
        return pool
    except (OSError, PermissionError, RuntimeError) as exc:
        log.warning("Не удалось запустить пул A*: %s. Продолжаем без параллелизма.", exc)
        return None


def _build_sim_payload(
    env: SwarmPZEnv,
    state: PolicyContextData | None = None,
    policy: object | None = None,
    policy_name: str | None = None,
) -> dict:
    payload = env.get_runtime_snapshot(include_oracle=True, policy=policy, policy_name=policy_name)
    if bool(payload.get("oracle_visible")) and state is not None and getattr(state, "oracle_dir", None) is not None:
        try:
            payload["oracle_dir"] = np.asarray(state.oracle_dir, dtype=np.float32).copy()
        except Exception:
            payload["oracle_dir"] = None
    return payload


def _valid_actions_or_log(result: object, *, policy_name: str) -> dict:
    if isinstance(result, dict) and _WORKER_ERROR_KEY in result:
        logger.warning("%s worker failed: %s", policy_name, result.get(_WORKER_ERROR_KEY))
        return {}
    if isinstance(result, dict):
        return result
    logger.warning("%s worker returned invalid result type: %s", policy_name, type(result).__name__)
    return {}


class AStarProcessPool:
    def __init__(
        self,
        policy_params: dict,
        workers: int,
        *,
        policy_name: str = "baseline:astar_grid",
    ):
        self.workers = int(max(1, workers))
        self.executors = []
        self.policy_name = str(policy_name)
        for _ in range(self.workers):
            executor = ProcessPoolExecutor(
                max_workers=1,
                initializer=_astar_worker_init,
                initargs=(policy_name, policy_params),
            )
            self.executors.append(executor)

    def shutdown(self) -> None:
        for ex in self.executors:
            ex.shutdown(wait=False, cancel_futures=True)
        self.executors = []

    def reset(self, seed: int | None) -> None:
        for ex in self.executors:
            ex.submit(_astar_worker_reset, seed)

    def compute(self, env: SwarmPZEnv, obs_map: dict | None, infos_map: dict | None) -> dict:
        actions = {}
        _include_oracle, _state, context = build_policy_context(env, policy_name=self.policy_name)
        info_batch = build_info_batch(env, infos_map, context)
        sim_payload = _build_sim_payload(env, context, None, self.policy_name)
        futures = []
        for i, agent_id in enumerate(env.possible_agents):
            info = info_batch.get(agent_id, {})
            obs = obs_map.get(agent_id) if obs_map is not None else None
            if obs is None:
                obs = env.get_agent_observation(i)
            payload = (agent_id, obs, info, sim_payload)
            ex = self.executors[i % self.workers]
            futures.append((agent_id, ex.submit(_astar_worker_action, payload)))
        for agent_id, fut in futures:
            actions[agent_id] = fut.result()
        return actions


class MpcProcessPool:
    def __init__(
        self,
        policy_params: dict,
        *,
        workers: int,
        enable_jit: bool = True,
        policy_name: str = "baseline:mpc_lite",
    ):
        self.workers = int(max(1, workers))
        self.executors = []
        self.policy_name = str(policy_name)
        mp_ctx = get_context("spawn")
        for _ in range(self.workers):
            executor = ProcessPoolExecutor(
                max_workers=1,
                mp_context=mp_ctx,
                initializer=_mpc_worker_init,
                initargs=(policy_name, policy_params, bool(enable_jit)),
            )
            self.executors.append(executor)

    def shutdown(self) -> None:
        for ex in self.executors:
            ex.shutdown(wait=False, cancel_futures=True)
        self.executors = []

    def reset(self, seed: int | None) -> None:
        for ex in self.executors:
            ex.submit(_mpc_worker_reset, seed)

    def compute(self, env: SwarmPZEnv, obs_map: dict | None, infos_map: dict | None) -> dict:
        actions = {}
        _include_oracle, _state, context = build_policy_context(env, policy_name=self.policy_name)
        info_batch = build_info_batch(env, infos_map, context)
        sim_payload = _build_sim_payload(env, context, None, self.policy_name)
        agent_ids = list(env.possible_agents)
        if not agent_ids:
            return actions

        obs_payload: dict = {}
        info_payload: dict = {}
        for i, agent_id in enumerate(agent_ids):
            info = info_batch.get(agent_id, {})
            obs = obs_map.get(agent_id) if obs_map is not None else None
            if obs is None:
                obs = env.get_agent_observation(i)
            obs_payload[agent_id] = obs
            info_payload[agent_id] = info

        if self.workers <= 1:
            ex = self.executors[0]
            try:
                fut = ex.submit(_mpc_worker_action_batch, (obs_payload, info_payload, sim_payload))
                actions.update(_valid_actions_or_log(fut.result(), policy_name=self.policy_name))
            except Exception as exc:
                logger.warning("%s worker future failed: %s", self.policy_name, exc)
            for agent_id in agent_ids:
                actions.setdefault(agent_id, np.zeros((2,), dtype=np.float32))
            return actions

        chunk_size = max(1, (len(agent_ids) + self.workers - 1) // self.workers)
        futures: list[tuple[list[str], object]] = []
        for worker_idx, ex in enumerate(self.executors):
            start = worker_idx * chunk_size
            if start >= len(agent_ids):
                break
            chunk_ids = agent_ids[start : start + chunk_size]
            chunk_obs = {agent_id: obs_payload[agent_id] for agent_id in chunk_ids}
            chunk_info = {agent_id: info_payload[agent_id] for agent_id in chunk_ids}
            try:
                fut = ex.submit(_mpc_worker_action_batch, (chunk_obs, chunk_info, sim_payload))
            except Exception as exc:
                logger.warning("%s worker submit failed: %s", self.policy_name, exc)
                for agent_id in chunk_ids:
                    actions[agent_id] = np.zeros((2,), dtype=np.float32)
                continue
            futures.append((chunk_ids, fut))

        for chunk_ids, fut in futures:
            try:
                result = _valid_actions_or_log(fut.result(), policy_name=self.policy_name)
            except Exception as exc:
                logger.warning("%s worker future failed: %s", self.policy_name, exc)
                result = {}
            for agent_id in chunk_ids:
                if agent_id in result:
                    actions[agent_id] = result[agent_id]
                else:
                    actions[agent_id] = np.zeros((2,), dtype=np.float32)
        return actions
