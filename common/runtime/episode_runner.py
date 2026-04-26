"""Shared helpers for policy-driven episode loops."""

from __future__ import annotations

from typing import Any

import numpy as np

from common.policy.context import PolicyContextData, context_from_state
from common.policy.oracle_visibility import oracle_visible_for_policy


def build_policy_context(
    env: Any,
    *,
    policy: object | None = None,
    policy_name: str | None = None,
) -> tuple[bool, Any, PolicyContextData]:
    include_oracle = oracle_visible_for_policy(env.config, policy=policy, policy_name=policy_name)
    public_state = env.get_public_state(include_oracle=include_oracle)
    context = context_from_state(public_state, include_oracle=include_oracle)
    return bool(include_oracle), public_state, context


def build_policy_info(
    env: Any,
    infos: dict | None,
    agent_id: str,
    idx: int,
    state: PolicyContextData,
) -> dict:
    info = {}
    if infos is not None:
        try:
            raw = infos.get(agent_id, {})
            info.update(raw if isinstance(raw, dict) else {})
        except Exception:
            info = {}
    info.setdefault("target_pos", state.target_pos)
    info.setdefault("target_vel", state.target_vel)
    info.setdefault("pos", state.pos[idx])
    info.setdefault("max_speed", float(state.max_speed))
    info.setdefault("max_accel", float(state.max_accel))
    info.setdefault("max_thrust", float(getattr(state, "max_thrust", 0.0)))
    info.setdefault("mass", float(getattr(state, "mass", 0.0)))
    info.setdefault("drag_coeff", float(getattr(state, "drag_coeff", 0.0)))
    info.setdefault("dt", float(state.dt))
    info.setdefault("drag", float(state.drag))
    info.setdefault("grid_res", float(state.grid_res))
    info["agent_index"] = int(idx)
    return info


def build_info_batch(env: Any, infos: dict | None, context: PolicyContextData) -> dict[str, dict]:
    return {
        agent_id: build_policy_info(env, infos, agent_id, idx, context)
        for idx, agent_id in enumerate(env.possible_agents)
    }


def reset_policy_context(
    env: Any,
    policy: object | None,
    seed: int | None = None,
    *,
    policy_name: str | None = None,
    set_context: bool = True,
) -> None:
    if policy is None:
        return
    if hasattr(policy, "reset"):
        policy.reset(seed)
    if set_context and hasattr(policy, "set_context"):
        _include_oracle, _public_state, context = build_policy_context(
            env,
            policy=policy,
            policy_name=policy_name,
        )
        policy.set_context(context)


def policy_actions(
    env: Any,
    policy: object,
    obs_map: dict | None,
    infos_map: dict | None,
    *,
    policy_name: str | None = None,
) -> dict[str, np.ndarray]:
    _include_oracle, _public_state, context = build_policy_context(
        env,
        policy=policy,
        policy_name=policy_name,
    )
    info_batch = build_info_batch(env, infos_map, context)

    def get_obs(idx: int, agent_id: str):
        obs = obs_map.get(agent_id) if obs_map is not None else None
        if obs is not None:
            return obs
        return env.get_agent_observation(idx)

    obs_payload = {agent_id: get_obs(idx, agent_id) for idx, agent_id in enumerate(env.possible_agents)}
    if hasattr(policy, "step_batch"):
        return dict(policy.step_batch(obs_payload, context, info_batch))
    if hasattr(policy, "get_actions"):
        return dict(policy.get_actions(obs_payload, context, info_batch))

    actions: dict[str, np.ndarray] = {}
    for idx, agent_id in enumerate(env.possible_agents):
        if hasattr(policy, "step"):
            actions[agent_id] = policy.step(agent_id, obs_payload[agent_id], context, info_batch[agent_id])
        else:
            actions[agent_id] = policy.get_action(agent_id, obs_payload[agent_id], context, info_batch[agent_id])
    return actions
