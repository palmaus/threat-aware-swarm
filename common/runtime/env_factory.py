"""Shared Swarm env construction helpers for UI/train/eval/bench/debug entrypoints."""

from __future__ import annotations

import copy
import logging
import os
from dataclasses import asdict
from functools import partial
from importlib import import_module
from typing import Any

from common.runtime.env_overrides import split_env_constructor_overrides
from env.config import EnvConfig
from env.pz_env import SwarmPZEnv

logger = logging.getLogger(__name__)


def _import_supersuit():
    return import_module("supersuit")


def apply_lite_metrics_cfg(cfg: EnvConfig, enabled: bool) -> EnvConfig:
    """Apply the shared compact-info profile used by fast eval/perf paths."""

    if bool(enabled):
        cfg.debug_metrics_mode = "lite"
        cfg.infos_mode = "compact"
    return cfg


def _merge_config(config: EnvConfig | None, cfg_payload: dict[str, Any]) -> EnvConfig:
    if config is None:
        return EnvConfig.from_dict(cfg_payload)
    payload = asdict(copy.deepcopy(config))
    payload.update(cfg_payload)
    return EnvConfig.from_dict(payload)


def make_pz_env(
    *,
    max_steps: int,
    goal_radius: float,
    seed: int | None = None,
    config: EnvConfig | None = None,
    env_overrides: dict[str, Any] | None = None,
    initial_stage_params: dict[str, Any] | None = None,
    shared_curriculum: dict[str, Any] | None = None,
    oracle_enabled: bool = True,
    oracle_cell_size: float = 1.0,
    oracle_async: bool = False,
    lite_metrics: bool = False,
    debug_metrics: bool | None = None,
    reset: bool = True,
) -> SwarmPZEnv:
    """Create a PettingZoo env with consistent config/runtime override semantics."""

    cfg_payload, runtime_payloads = split_env_constructor_overrides(
        EnvConfig,
        dict(env_overrides or {}),
        dict(initial_stage_params or {}),
    )
    runtime_env_overrides, runtime_stage_params = runtime_payloads
    cfg = _merge_config(config, cfg_payload)
    apply_lite_metrics_cfg(cfg, lite_metrics)
    if debug_metrics is not None:
        cfg.debug_metrics = bool(debug_metrics)

    env = SwarmPZEnv(
        cfg,
        max_steps=int(max_steps),
        goal_radius=float(goal_radius),
        oracle_enabled=bool(oracle_enabled),
        oracle_cell_size=float(oracle_cell_size),
        oracle_async=bool(oracle_async),
        shared_curriculum=shared_curriculum,
    )
    for payload, label in (
        (runtime_env_overrides, "Env override"),
        (runtime_stage_params, "Initial stage"),
    ):
        if not payload:
            continue
        try:
            env.apply_curriculum(payload)
        except Exception as exc:
            try:
                env.close()
            except Exception:
                pass
            raise RuntimeError(f"{label} apply failed for payload={payload!r}") from exc

    if reset:
        env.reset(seed=seed)
    return env


def pettingzoo_to_vec_env(pz_env):
    return _import_supersuit().pettingzoo_env_to_vec_env_v1(pz_env)


def concat_vec_envs_safe(
    env_fns,
    obs_space,
    act_space,
    num_vec_envs: int,
    num_cpus: int,
    base_class: str = "stable_baselines3",
):
    if num_vec_envs <= 1:
        return env_fns[0]()
    from supersuit import vector as sv

    constructor = sv.MakeCPUAsyncConstructor(min(int(num_cpus), int(num_vec_envs)))
    vec = constructor(env_fns, obs_space, act_space)
    if base_class == "gymnasium":
        return vec
    if base_class == "stable_baselines":
        from supersuit.vector.sb_vector_wrapper import SBVecEnvWrapper

        return SBVecEnvWrapper(vec)
    if base_class == "stable_baselines3":
        from supersuit.vector.sb3_vector_wrapper import SB3VecEnvWrapper

        return SB3VecEnvWrapper(vec)
    raise ValueError("Unsupported base_class for concat_vec_envs_safe")


def make_vec_env(
    *,
    max_steps: int,
    goal_radius: float,
    seed: int,
    num_vec_envs: int = 1,
    num_cpus: int = 1,
    config: EnvConfig | None = None,
    env_overrides: dict[str, Any] | None = None,
    initial_stage_params: dict[str, Any] | None = None,
    shared_curriculum: dict[str, Any] | None = None,
    oracle_enabled: bool = True,
    lite_metrics: bool = False,
    train_wrappers: bool = False,
    vec_monitor: bool = False,
    smoke_check: bool = False,
    base_class: str = "stable_baselines3",
) -> Any:
    """Create a vectorized Swarm env while preserving entrypoint-specific wrappers."""

    def _make_single(seed_offset: int):
        pz_env = make_pz_env(
            max_steps=max_steps,
            goal_radius=goal_radius,
            seed=int(seed) + int(seed_offset),
            config=config,
            env_overrides=env_overrides,
            initial_stage_params=initial_stage_params,
            shared_curriculum=shared_curriculum,
            oracle_enabled=oracle_enabled,
            lite_metrics=lite_metrics,
            reset=False,
        )
        from env.wrappers import WaypointActionWrapper

        pz_env = WaypointActionWrapper(pz_env)
        return pettingzoo_to_vec_env(pz_env)

    base_vec = _make_single(0)
    obs_space = getattr(base_vec, "observation_space", None)
    act_space = getattr(base_vec, "action_space", None)

    if int(num_vec_envs) > 1:
        env_fns = [partial(_make_single, i) for i in range(int(num_vec_envs))]
        vec_env = concat_vec_envs_safe(
            env_fns,
            obs_space,
            act_space,
            num_vec_envs=int(num_vec_envs),
            num_cpus=int(num_cpus),
            base_class=base_class,
        )
        try:
            base_vec.close()
        except Exception as exc:
            logger.warning("Base vec close failed: %s", exc)
    else:
        vec_env = base_vec

    if train_wrappers:
        from env.wrappers import ResetStepAdapter, ResetStepVecWrapper, SafeEnvIsWrapped

        vec_env = ResetStepAdapter(vec_env)
        vec_env = ResetStepVecWrapper(vec_env)
        vec_env = SafeEnvIsWrapped(vec_env)
        if smoke_check:
            obs = vec_env.reset()
            assert not isinstance(obs, tuple), "reset() returned tuple; adapter should unwrap obs"

    if vec_monitor and os.environ.get("TA_SKIP_VECS_MONITOR") != "1":
        try:
            from stable_baselines3.common.vec_env import VecMonitor

            vec_env = VecMonitor(vec_env)
        except TypeError as exc:
            print(f"[ПРЕД] VecMonitor отключён: {exc}")
    return vec_env
