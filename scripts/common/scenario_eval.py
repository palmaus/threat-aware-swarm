from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import yaml
from stable_baselines3 import PPO

try:
    from sb3_contrib import RecurrentPPO
except Exception:  # pragma: no cover - optional dependency
    RecurrentPPO = None

try:
    import imageio.v2 as imageio
except Exception:  # pragma: no cover - optional dependency
    imageio = None

from baselines.factory import create_baseline_policy
from common.runtime.env_factory import make_pz_env as _common_make_pz_env
from env.config import EnvConfig
from env.pz_env import SwarmPZEnv
from scripts.common.episode_runner import policy_actions, reset_policy_context
from scripts.common.path_utils import resolve_repo_path
from scripts.common.rl_episode_runner import (
    PettingZooModelState,
    predict_pettingzoo_model_actions,
    run_pettingzoo_episode,
)

SUCCESS_THRESHOLD_DEFAULT = 0.5

_SCENE_PRESETS: dict[str, list[str]] = {
    "preset:sanity": [
        "scenarios/S0_sanity_no_threats.yaml",
        "scenarios/S1_single_threat_blocking.yaml",
        "scenarios/S2_corridor_two_threats.yaml",
        "scenarios/S3_high_intensity_near_goal.yaml",
    ],
    "preset:static": [
        "scenarios/S0_sanity_no_threats.yaml",
        "scenarios/S1_single_threat_blocking.yaml",
        "scenarios/S2_corridor_two_threats.yaml",
        "scenarios/S3_high_intensity_near_goal.yaml",
        "scenarios/S4_goal_between_threats.yaml",
        "scenarios/S8_u_trap_escape.yaml",
        "scenarios/S9_bottleneck_passage.yaml",
        "scenarios/S10_risk_vs_efficiency.yaml",
        "scenarios/S12_cross_traffic.yaml",
        "scenarios/S13_dense_forest_static.yaml",
    ],
    "preset:dynamic": [
        "scenarios/S5_dynamic_linear.yaml",
        "scenarios/S6_dynamic_brownian.yaml",
        "scenarios/S7_dynamic_chaser.yaml",
        "scenarios/S11_dynamic_gate.yaml",
        "scenarios/S14_moving_goal.yaml",
        "scenarios/S15_interceptor_threat.yaml",
        "scenarios/S16_dynamic_breathing.yaml",
    ],
    "preset:ood": [
        "scenarios/ood/OOD1_wind_cross.yaml",
        "scenarios/ood/OOD2_dynamic_swarm.yaml",
        "scenarios/ood/OOD3_maze_long.yaml",
    ],
}


def scene_id(scene: dict, fallback: str = "scene") -> str:
    sid = scene.get("id") if isinstance(scene, dict) else None
    return str(sid) if sid not in (None, "") else str(fallback)


def load_scenes(paths) -> list[dict]:
    scenes: list[dict] = []
    for raw_path in paths:
        path = Path(raw_path)
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            data = {}
        if not data.get("id"):
            data = dict(data)
            data["id"] = path.stem
        scenes.append(data)
    return scenes


def resolve_scene_paths(scene_args: list[str] | None) -> list[Path]:
    if not scene_args:
        return sorted(resolve_repo_path("scenarios").glob("S*.yaml"))
    resolved: list[Path] = []
    for raw in scene_args:
        if raw in _SCENE_PRESETS:
            for candidate in _SCENE_PRESETS[raw]:
                path = resolve_repo_path(candidate)
                if path.exists():
                    resolved.append(path)
            continue
        path = resolve_repo_path(raw)
        if path.is_dir():
            resolved.extend(sorted(path.glob("*.yaml")))
        else:
            resolved.append(path)
    return resolved


def make_env(
    cfg: EnvConfig,
    max_steps: int,
    goal_radius: float,
    *,
    oracle_enabled: bool,
    wrap_waypoint_actions: bool = False,
) -> SwarmPZEnv:
    env = _common_make_pz_env(
        max_steps=max_steps,
        goal_radius=goal_radius,
        config=cfg,
        oracle_enabled=oracle_enabled,
        reset=False,
    )
    if wrap_waypoint_actions:
        from env.wrappers import WaypointActionWrapper

        env = WaypointActionWrapper(env)
    return env


def build_policy(cfg: EnvConfig, policy_name: str, *, model_path: str | None = None):
    if policy_name == "ppo":
        if not model_path:
            raise SystemExit("--model is required for PPO policy")
        custom_objects = {"learning_rate": 0.0, "lr_schedule": (lambda _: 0.0), "clip_range": 0.2}
        try:
            ppo_model = PPO.load(model_path, device="cpu", custom_objects=custom_objects)
        except Exception:
            if RecurrentPPO is None:
                raise
            ppo_model = RecurrentPPO.load(model_path, device="cpu", custom_objects=custom_objects)
        return None, ppo_model
    return (
        create_baseline_policy(
            policy_name,
            n_agents=int(getattr(cfg, "n_agents", 1)),
            grid_res=float(getattr(cfg, "grid_res", 1.0)),
        ),
        None,
    )


def run_episode(
    env,
    policy,
    scene,
    seed,
    success_threshold,
    *,
    ppo_model=None,
    deterministic: bool = False,
) -> dict:
    result = run_pettingzoo_episode(
        env,
        policy,
        scene,
        seed,
        success_threshold=success_threshold,
        ppo_model=ppo_model,
        deterministic=deterministic,
    )
    summary = result.summary
    return {
        "success": bool(summary["success"]),
        "finished_frac_end": summary["finished_frac_end"],
        "alive_frac_end": summary["alive_frac_end"],
        "deaths": summary["deaths"],
        "time_to_goal_mean": summary["time_to_goal_mean"],
        "risk_integral_all": summary["risk_integral_all"],
        "risk_integral_alive": summary["risk_integral_alive"],
        "collision_like": summary["collision_like"],
        "episode_len": result.steps,
        "path_ratio": summary["path_ratio"],
        "action_smoothness": summary["action_smoothness"],
        "energy_efficiency": summary["energy_efficiency"],
        "safety_score": summary["safety_score"],
        **{key: value for key, value in summary.items() if key.startswith("cost_")},
    }


def run_episode_gif(
    env,
    policy,
    scene,
    seed,
    *,
    ppo_model=None,
    deterministic: bool = False,
    gif_path: Path,
    gif_fps: int = 20,
    screen_size: int = 900,
) -> None:
    if imageio is None:
        raise RuntimeError("imageio not installed")
    try:
        from ui.renderer import OverlayFlags, PygameRenderer
    except Exception as exc:  # pragma: no cover - optional UI dependency
        raise RuntimeError("pygame renderer not available") from exc

    scene_payload = scene
    if scene is not None and seed is not None:
        try:
            scene_payload = copy.deepcopy(scene)
            scene_payload["seed"] = int(seed)
        except Exception:
            scene_payload = scene

    obs, infos = env.reset(seed=seed, options={"scene": scene_payload})
    if policy is not None:
        reset_policy_context(env, policy, seed)
    model_state = PettingZooModelState()
    renderer = PygameRenderer(env.config, screen_size=int(screen_size))
    overlay = OverlayFlags(show_grid=False, show_trails=False, show_threats=True)
    writer = imageio.get_writer(str(gif_path), mode="I", fps=int(gif_fps))
    try:
        writer.append_data(renderer.render_array(env, overlay=overlay, trails=None, agent_idx=0))
        max_steps = int((scene or {}).get("max_steps", env.max_steps))
        for _ in range(max_steps):
            actions = {}
            if ppo_model is not None:
                actions = predict_pettingzoo_model_actions(
                    ppo_model,
                    env,
                    obs,
                    model_state,
                    deterministic=deterministic,
                )
            else:
                actions.update(policy_actions(env, policy, obs, infos))
            obs, _rewards, terminations, truncations, infos = env.step(actions)
            if ppo_model is not None:
                dones = {
                    agent_id: bool(terminations.get(agent_id, False) or truncations.get(agent_id, False))
                    for agent_id in env.possible_agents
                }
                model_state.mark_dones(dones, list(env.possible_agents))
            writer.append_data(renderer.render_array(env, overlay=overlay, trails=None, agent_idx=0))
            if all(terminations.values()) or all(truncations.values()):
                break
    finally:
        writer.close()


__all__ = [
    "SUCCESS_THRESHOLD_DEFAULT",
    "build_policy",
    "load_scenes",
    "make_env",
    "resolve_scene_paths",
    "run_episode",
    "run_episode_gif",
    "scene_id",
]
