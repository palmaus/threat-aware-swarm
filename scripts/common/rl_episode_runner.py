"""Shared PPO/RecurrentPPO and policy episode runners."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from gymnasium import spaces

from scripts.common.episode_metrics import EpisodeMetricsAccumulator
from scripts.common.episode_runner import policy_actions, reset_policy_context


@dataclass
class PettingZooModelState:
    lstm_states: dict[int, object] = field(default_factory=dict)
    episode_starts: dict[int, bool] = field(default_factory=dict)

    def mark_dones(self, dones: dict[str, bool], agent_ids: list[str]) -> None:
        for idx, agent_id in enumerate(agent_ids):
            self.episode_starts[idx] = bool(dones.get(agent_id, False))


@dataclass
class VectorModelState:
    lstm_state: object | None = None
    episode_start: np.ndarray | None = None

    def mark_dones(self, dones: Any) -> None:
        self.episode_start = np.asarray(dones, dtype=bool)


@dataclass
class EpisodeRunResult:
    summary: dict[str, float]
    steps: int
    trace: list[dict[str, Any]]
    last_infos: dict[str, dict[str, Any]] | None = None


def is_recurrent_model(model: Any) -> bool:
    return bool(getattr(getattr(model, "policy", None), "lstm_actor", None))


def adapt_obs_for_model(obs: Any, model: Any) -> Any:
    try:
        space = model.observation_space
    except Exception as exc:  # pragma: no cover - unstable model objects
        raise RuntimeError("PPO model lacks observation_space") from exc
    if not isinstance(space, spaces.Dict):
        raise RuntimeError("Legacy obs spaces are not supported; use Dict obs models.")
    return obs


def predict_pettingzoo_model_actions(
    model: Any,
    env: Any,
    obs_map: dict[str, Any],
    state: PettingZooModelState,
    *,
    deterministic: bool,
) -> dict[str, np.ndarray]:
    actions: dict[str, np.ndarray] = {}
    recurrent = is_recurrent_model(model)
    for idx, agent_id in enumerate(getattr(env, "possible_agents", []) or []):
        model_obs = adapt_obs_for_model(obs_map[agent_id], model)
        if recurrent:
            lstm_state = state.lstm_states.get(idx, None)
            start = np.asarray([state.episode_starts.get(idx, True)], dtype=bool)
            action, next_state = model.predict(
                model_obs,
                state=lstm_state,
                episode_start=start,
                deterministic=deterministic,
            )
            state.lstm_states[idx] = next_state
            state.episode_starts[idx] = False
        else:
            action, _ = model.predict(model_obs, deterministic=deterministic)
        actions[agent_id] = action
    return actions


def predict_vector_model_actions(
    model: Any,
    obs: Any,
    state: VectorModelState,
    *,
    deterministic: bool,
) -> Any:
    if is_recurrent_model(model):
        if state.episode_start is None:
            n_envs = int(obs["vector"].shape[0])
            state.episode_start = np.ones((n_envs,), dtype=bool)
        actions, next_state = model.predict(
            obs,
            state=state.lstm_state,
            episode_start=state.episode_start,
            deterministic=deterministic,
        )
        state.lstm_state = next_state
        return actions
    actions, _ = model.predict(obs, deterministic=deterministic)
    return actions


def _scene_with_seed(scene: dict | None, seed: int | None) -> dict | None:
    if scene is None or seed is None:
        return scene
    try:
        payload = copy.deepcopy(scene)
        payload["seed"] = int(seed)
        return payload
    except Exception:
        return scene


def run_pettingzoo_episode(
    env: Any,
    policy: Any,
    scene: dict | None,
    seed: int,
    *,
    success_threshold: float = 0.5,
    ppo_model: Any | None = None,
    deterministic: bool = False,
    policy_name: str | None = None,
    max_steps: int | None = None,
    trace_every: int = 0,
    trace_builder: Callable[[int, Any, dict[str, dict[str, Any]], Any], dict[str, Any]] | None = None,
) -> EpisodeRunResult:
    obs, infos = env.reset(seed=seed, options={"scene": _scene_with_seed(scene, seed)})
    if policy is not None:
        reset_policy_context(env, policy, seed, policy_name=policy_name)

    model_state = PettingZooModelState()
    metrics = EpisodeMetricsAccumulator.from_env(env, success_threshold=success_threshold)
    trace: list[dict[str, Any]] = []
    steps = 0
    limit = int(max_steps if max_steps is not None else (scene or {}).get("max_steps", getattr(env, "max_steps", 0)))

    for t in range(limit):
        if ppo_model is not None:
            actions = predict_pettingzoo_model_actions(
                ppo_model,
                env,
                obs,
                model_state,
                deterministic=deterministic,
            )
        else:
            actions = policy_actions(env, policy, obs, infos, policy_name=policy_name)

        obs, _rewards, terminations, truncations, infos = env.step(actions)
        steps += 1
        try:
            decision_step = int(getattr(env.get_state(), "decision_step", steps))
        except Exception:
            decision_step = steps
        metrics.update(env, infos, decision_step=decision_step)

        if trace_every > 0 and (t % int(trace_every) == 0):
            if trace_builder is None:
                trace.append({"t": t})
            else:
                trace.append(trace_builder(t, env, infos, obs))

        if ppo_model is not None and is_recurrent_model(ppo_model):
            dones = {
                agent_id: bool(terminations.get(agent_id, False) or truncations.get(agent_id, False))
                for agent_id in getattr(env, "possible_agents", []) or []
            }
            model_state.mark_dones(dones, list(getattr(env, "possible_agents", []) or []))

        if all(terminations.values()) or all(truncations.values()):
            break

    return EpisodeRunResult(
        summary=metrics.summary(env, steps=steps),
        steps=steps,
        trace=trace,
        last_infos=metrics.last_infos,
    )
