from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import hydra
from omegaconf import OmegaConf

from env.config import EnvConfig
from scripts.common.hydra_utils import apply_schema
from scripts.common.metrics_utils import aggregate_episode_metrics, aggregate_scene_metrics
from scripts.common.model_registry import resolve_model_path
from scripts.common.path_utils import get_runs_dir
from scripts.common.scenario_eval import build_policy, load_scenes, make_env, resolve_scene_paths, run_episode


@dataclass
class BenchmarkSuiteConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    policies: list[str] = field(default_factory=lambda: ["baseline:flow_field", "baseline:astar_grid"])
    scenes: list[str] = field(default_factory=lambda: ["preset:sanity"])
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    success_threshold: float = 0.5
    max_steps: int = 600
    goal_radius: float = 3.0
    oracle_enabled: bool = True
    deterministic: bool = True
    out_dir: str = ""
    out_name: str = ""
    model: str | None = None


def _resolve_out_dir(cfg: BenchmarkSuiteConfig) -> Path:
    if cfg.out_dir:
        return Path(cfg.out_dir).expanduser().resolve()
    return get_runs_dir() / "benchmarks"


def _parse_policy(entry: str, default_model: str | None) -> tuple[str, str | None, str]:
    if entry.startswith("ppo:"):
        return "ppo", entry[len("ppo:") :], "ppo"
    if entry == "ppo":
        return "ppo", default_model, "ppo"
    return entry, None, entry


def _format_float(val: float | None) -> str:
    try:
        if val is None or val != val:
            return "-"
        return f"{float(val):.3f}"
    except Exception:
        return "-"


def _write_markdown(out_path: Path, results: dict) -> None:
    lines = [
        "# Benchmark suite",
        "",
        "| Policy | Success | Alive | Risk | Energy | Safety |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for policy, payload in results.items():
        overall = payload.get("overall", {})
        lines.append(
            "| "
            + " | ".join(
                [
                    policy,
                    _format_float(overall.get("success_rate")),
                    _format_float(overall.get("alive_frac_end")),
                    _format_float(overall.get("risk_integral_alive")),
                    _format_float(overall.get("energy_efficiency")),
                    _format_float(overall.get("safety_score")),
                ]
            )
            + " |"
        )
    out_path.write_text("\n".join(lines), encoding="utf-8")


@hydra.main(config_path="../../configs/hydra", config_name="analysis/benchmark_suite", version_base=None)
def main(cfg: BenchmarkSuiteConfig) -> int:
    cfg = apply_schema(cfg, BenchmarkSuiteConfig)
    data = OmegaConf.to_container(cfg, resolve=True)
    data = data if isinstance(data, dict) else {}
    env_cfg = EnvConfig.from_dict(data.get("env") or {})
    suite = BenchmarkSuiteConfig(**{**data, "env": env_cfg})

    scene_paths = resolve_scene_paths(suite.scenes)
    scenes = load_scenes(scene_paths)
    seeds = suite.seeds if suite.seeds else [None]

    out_dir = _resolve_out_dir(suite)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    base_name = suite.out_name or f"bench_{stamp}"

    results: dict[str, dict] = {}

    for entry in suite.policies:
        policy_name, model_path, label = _parse_policy(entry, suite.model)
        if policy_name == "ppo":
            if not model_path:
                raise SystemExit("PPO политика требует model или запись ppo:<path>")
            model_path = str(resolve_model_path(model_path))
        policy, ppo_model = build_policy(env_cfg, policy_name, model_path=model_path)
        env = make_env(
            env_cfg,
            suite.max_steps,
            suite.goal_radius,
            oracle_enabled=suite.oracle_enabled,
            wrap_waypoint_actions=ppo_model is not None,
        )
        per_scene: dict[str, dict] = {}
        for path, scene in zip(scene_paths, scenes):
            scene_id = str(scene.get("id") or Path(path).stem)
            rows = []
            for seed in seeds:
                row = run_episode(
                    env,
                    policy,
                    scene,
                    seed,
                    suite.success_threshold,
                    ppo_model=ppo_model,
                    deterministic=suite.deterministic,
                )
                rows.append(row)
            per_scene[scene_id] = aggregate_episode_metrics(rows)
        results[label] = {
            "per_scene": per_scene,
            "overall": aggregate_scene_metrics(per_scene),
        }

    json_path = out_dir / f"{base_name}.json"
    md_path = out_dir / f"{base_name}.md"
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_markdown(md_path, results)
    print(f"[benchmark] JSON: {json_path}")
    print(f"[benchmark] MD:   {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
