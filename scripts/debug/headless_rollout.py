from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import hydra
import numpy as np
from omegaconf import OmegaConf

from env.config import EnvConfig
from scripts.common.hydra_utils import apply_schema
from scripts.common.model_registry import resolve_model_path
from scripts.common.path_utils import get_runs_dir
from scripts.common.rl_episode_runner import run_pettingzoo_episode
from scripts.common.scenario_eval import build_policy, load_scenes, make_env, resolve_scene_paths


@dataclass
class HeadlessRolloutConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    policy: str = "baseline:flow_field"
    model: str | None = None
    scenes: list[str] = field(default_factory=lambda: ["preset:sanity"])
    seed: int = 0
    max_steps: int = 600
    goal_radius: float = 3.0
    oracle_enabled: bool = True
    deterministic: bool = True
    record_every: int = 1
    out_dir: str = ""
    out_name: str = ""


def _resolve_out_dir(cfg: HeadlessRolloutConfig) -> Path:
    if cfg.out_dir:
        return Path(cfg.out_dir).expanduser().resolve()
    return get_runs_dir() / "debug"


def _run_episode(env, policy, scene, seed, *, ppo_model=None, deterministic: bool, record_every: int) -> dict:
    def _trace_builder(t, env_obj, infos, _obs):
        agent_id = env_obj.possible_agents[0] if env_obj.possible_agents else None
        if agent_id is None:
            return {}
        info = infos.get(agent_id, {})
        return {
            "t": t,
            "dist": info.get("dist", None),
            "risk_p": info.get("risk_p", None),
            "energy": info.get("energy", None),
            "energy_level": info.get("energy_level", None),
            "alive": info.get("alive", None),
            "finished": info.get("finished", None),
        }

    result = run_pettingzoo_episode(
        env,
        policy,
        scene,
        seed,
        ppo_model=ppo_model,
        deterministic=deterministic,
        trace_every=record_every,
        trace_builder=_trace_builder,
    )
    summary_src = result.summary
    summary = {
        "alive_end": int(summary_src.get("alive_count_end", 0.0)),
        "finished_end": int(summary_src.get("finished_count_end", 0.0)),
        "alive_frac_end": summary_src.get("alive_frac_end", float("nan")),
        "finished_frac_end": summary_src.get("finished_frac_end", float("nan")),
        "risk_integral_all": summary_src.get("risk_integral_all", float("nan")),
        "risk_integral_alive": summary_src.get("risk_integral_alive", float("nan")),
    }
    return {
        "summary": summary,
        "steps": result.steps,
        "trace": result.trace,
    }


@hydra.main(config_path="../../configs/hydra", config_name="debug/headless_rollout", version_base=None)
def main(cfg: HeadlessRolloutConfig) -> int:
    cfg = apply_schema(cfg, HeadlessRolloutConfig)
    data = OmegaConf.to_container(cfg, resolve=True)
    data = data if isinstance(data, dict) else {}
    env_cfg = EnvConfig.from_dict(data.get("env") or {})
    cfg_obj = HeadlessRolloutConfig(**{**data, "env": env_cfg})

    scene_paths = resolve_scene_paths(cfg_obj.scenes)
    scenes = load_scenes(scene_paths)
    out_dir = _resolve_out_dir(cfg_obj)
    out_dir.mkdir(parents=True, exist_ok=True)

    policy_name = cfg_obj.policy
    model_path = cfg_obj.model
    if policy_name == "ppo":
        if not model_path:
            raise SystemExit("Для PPO требуется model")
        model_path = str(resolve_model_path(model_path))
    policy, ppo_model = build_policy(env_cfg, policy_name, model_path=model_path)

    results = []
    for scene, path in zip(scenes, scene_paths):
        scene_payload = scene
        scene_payload["id"] = scene_payload.get("id") or Path(path).stem
        env = make_env(
            env_cfg,
            cfg_obj.max_steps,
            cfg_obj.goal_radius,
            oracle_enabled=cfg_obj.oracle_enabled,
            wrap_waypoint_actions=ppo_model is not None,
        )
        results.append(
            {
                "scene": scene_payload.get("id"),
                "run": _run_episode(
                    env,
                    policy,
                    scene_payload,
                    cfg_obj.seed,
                    ppo_model=ppo_model,
                    deterministic=cfg_obj.deterministic,
                    record_every=cfg_obj.record_every,
                ),
            }
        )

    out_name = cfg_obj.out_name or f"headless_{cfg_obj.seed}"
    out_path = out_dir / f"{out_name}.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[headless] JSON: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
