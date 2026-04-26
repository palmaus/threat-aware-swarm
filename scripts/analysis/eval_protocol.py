from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from math import sqrt
from pathlib import Path
from types import SimpleNamespace

import hydra
import numpy as np
from omegaconf import OmegaConf

from env.config import EnvConfig
from scripts.common.artifacts import write_config_audit
from scripts.common.experiment_result import ExperimentResult, write_experiment_result
from scripts.common.hydra_utils import apply_schema
from scripts.common.logging_utils import init_mlflow, log_artifacts_mlflow, log_metrics, log_params
from scripts.common.metrics_plots import write_summary_plots
from scripts.common.metrics_schema import load_metrics_schema, validate_metrics, write_metrics_manifest
from scripts.common.metrics_utils import METRIC_KEYS, aggregate_episode_metrics, aggregate_scene_metrics
from scripts.common.model_registry import resolve_model_path
from scripts.common.path_utils import get_runs_dir, resolve_repo_path
from scripts.common.scenario_eval import build_policy, load_scenes, make_env, resolve_scene_paths, run_episode


@dataclass
class EvalProtocolConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    policy: str = "baseline:potential_fields"
    model: str | None = None
    scenes: list[str] = field(default_factory=lambda: ["preset:sanity"])
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    episodes_per_seed: int = 1
    success_threshold: float = 0.5
    max_steps: int = 600
    goal_radius: float = 3.0
    oracle_enabled: bool = False
    deterministic: bool = True
    out_dir: str = ""
    out_name: str = ""
    metrics_schema: str = "configs/metrics_schema.yaml"
    strict_schema: bool = True
    logging: dict = field(default_factory=dict)


def _resolve_out_dir(cfg: EvalProtocolConfig) -> Path:
    if cfg.out_dir:
        return resolve_repo_path(cfg.out_dir)
    return get_runs_dir() / "eval_protocol"


def _parse_policy(entry: str, default_model: str | None) -> tuple[str, str | None, str]:
    if entry.startswith("ppo:"):
        return "ppo", entry[len("ppo:") :], "ppo"
    if entry == "ppo":
        return "ppo", default_model, "ppo"
    return entry, None, entry


def _mean_std_ci(values: list[float]) -> dict[str, float]:
    vals = [v for v in values if v == v]
    if not vals:
        return {"mean": float("nan"), "std": float("nan"), "ci95": float("nan"), "n": 0}
    arr = np.asarray(vals, dtype=np.float64)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    ci95 = float(1.96 * std / sqrt(arr.size)) if arr.size > 1 else 0.0
    return {"mean": mean, "std": std, "ci95": ci95, "n": int(arr.size)}


def _format_stat(stat: dict[str, float]) -> str:
    mean = stat.get("mean", float("nan"))
    std = stat.get("std", float("nan"))
    ci = stat.get("ci95", float("nan"))
    if mean != mean:
        return "-"
    return f"{mean:.4f} ± {std:.4f} (±{ci:.4f})"


def _write_report(
    out_path: Path,
    cfg: EvalProtocolConfig,
    aggregate_stats: dict[str, dict[str, float]],
    per_scene_stats: dict[str, dict[str, dict[str, float]]],
) -> None:
    lines: list[str] = []
    lines.append("# Eval protocol report")
    lines.append("")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append("## Config")
    lines.append(f"- Policy: `{cfg.policy}`")
    if cfg.model:
        lines.append(f"- Model: `{cfg.model}`")
    lines.append(f"- Scenes: {', '.join(cfg.scenes) if cfg.scenes else 'default'}")
    lines.append(f"- Seeds: {', '.join(str(s) for s in cfg.seeds)}")
    lines.append(f"- Episodes per seed: {cfg.episodes_per_seed}")
    try:
        visibility = getattr(cfg.env, "oracle_visibility", "") or ""
        vis_base = getattr(cfg.env, "oracle_visible_to_baselines", None)
        vis_agent = getattr(cfg.env, "oracle_visible_to_agents", None)
        if visibility or vis_base is not None or vis_agent is not None:
            lines.append(f"- Oracle visibility: `{visibility or 'default'}`")
            if vis_base is not None:
                lines.append(f"- Oracle visible to baselines: `{bool(vis_base)}`")
            if vis_agent is not None:
                lines.append(f"- Oracle visible to agents: `{bool(vis_agent)}`")
    except Exception:
        pass
    lines.append("")

    lines.append("## Aggregate metrics (mean ± std, 95% CI)")
    lines.append("| Metric | Value |")
    lines.append("| --- | --- |")
    for key in METRIC_KEYS:
        stat = aggregate_stats.get(key, {})
        lines.append(f"| {key} | {_format_stat(stat)} |")

    if per_scene_stats:
        lines.append("")
        lines.append("## Per‑scene (success_rate / alive / risk / time)")
        lines.append("| Scene | Success | Alive | Risk | Time |")
        lines.append("| --- | --- | --- | --- | --- |")
        for scene_id, stats in per_scene_stats.items():
            lines.append(
                "| "
                + " | ".join(
                    [
                        scene_id,
                        _format_stat(stats.get("success_rate", {})),
                        _format_stat(stats.get("alive_frac_end", {})),
                        _format_stat(stats.get("risk_integral_alive", {})),
                        _format_stat(stats.get("time_to_goal_mean", {})),
                    ]
                )
                + " |"
            )

    out_path.write_text("\n".join(lines), encoding="utf-8")


@hydra.main(config_path="../../configs/hydra", config_name="analysis/eval_protocol", version_base=None)
def main(cfg: EvalProtocolConfig) -> int:
    cfg = apply_schema(cfg, EvalProtocolConfig)
    cfg_raw = cfg
    data = OmegaConf.to_container(cfg, resolve=True)
    data = data if isinstance(data, dict) else {}
    env_cfg = EnvConfig.from_dict(data.get("env") or {})
    cfg = EvalProtocolConfig(**{**data, "env": env_cfg})

    policy_name, model_path, label = _parse_policy(cfg.policy, cfg.model)
    if policy_name == "ppo":
        if not model_path:
            raise SystemExit("PPO политика требует model или запись ppo:<path>")
        model_path = str(resolve_model_path(model_path))

    scene_paths = resolve_scene_paths(cfg.scenes)
    scenes = load_scenes(scene_paths)
    seeds = cfg.seeds if cfg.seeds else [0]

    policy, ppo_model = build_policy(env_cfg, policy_name, model_path=model_path)
    env = make_env(
        env_cfg,
        cfg.max_steps,
        cfg.goal_radius,
        oracle_enabled=cfg.oracle_enabled,
        wrap_waypoint_actions=ppo_model is not None,
    )

    out_dir = _resolve_out_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_config_audit(
        out_dir,
        cfg=cfg_raw,
        scenes=list(cfg.scenes),
        seeds=list(seeds),
        env_cfg=cfg.env,
    )
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    base_name = cfg.out_name or f"eval_{stamp}"

    per_seed: dict[int, dict] = {}
    for seed in seeds:
        per_scene: dict[str, dict] = {}
        for path, scene in zip(scene_paths, scenes):
            scene_id = str(scene.get("id") or Path(path).stem)
            rows = []
            for ep in range(max(1, int(cfg.episodes_per_seed))):
                ep_seed = int(seed) * 1000 + ep
                row = run_episode(
                    env,
                    policy,
                    scene,
                    ep_seed,
                    cfg.success_threshold,
                    ppo_model=ppo_model,
                    deterministic=cfg.deterministic,
                )
                rows.append(row)
            per_scene[scene_id] = aggregate_episode_metrics(rows)
        per_seed[int(seed)] = {
            "per_scene": per_scene,
            "aggregate": aggregate_scene_metrics(per_scene),
        }

    aggregate_stats: dict[str, dict[str, float]] = {}
    for key in METRIC_KEYS:
        vals = [payload["aggregate"].get(key, float("nan")) for payload in per_seed.values()]
        aggregate_stats[key] = _mean_std_ci(vals)

    per_scene_stats: dict[str, dict[str, dict[str, float]]] = {}
    for path, scene in zip(scene_paths, scenes):
        sid = str(scene.get("id") or Path(path).stem)
        per_scene_stats[sid] = {}
        for key in METRIC_KEYS:
            vals = [payload["per_scene"].get(sid, {}).get(key, float("nan")) for payload in per_seed.values()]
            per_scene_stats[sid][key] = _mean_std_ci(vals)

    json_path = out_dir / f"{base_name}.json"
    report_path = out_dir / f"{base_name}.md"
    json_path.write_text(
        json.dumps(
            {
                "config": OmegaConf.to_container(OmegaConf.create(data), resolve=True),
                "policy": label,
                "seeds": seeds,
                "per_seed": per_seed,
                "aggregate": aggregate_stats,
                "per_scene": per_scene_stats,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_report(report_path, cfg, aggregate_stats, per_scene_stats)

    schema_path = Path(cfg.metrics_schema) if cfg.metrics_schema else None
    schema = load_metrics_schema(schema_path)
    missing, extra = validate_metrics(aggregate_stats.keys(), schema.keys)
    write_metrics_manifest(
        out_dir,
        schema_path=schema_path,
        schema=schema,
        actual_keys=aggregate_stats.keys(),
        missing=missing,
        extra=extra,
    )
    if (missing or extra) and cfg.strict_schema:
        raise SystemExit(f"Metric schema mismatch. missing={missing} extra={extra}")

    plot_dir = out_dir / "plots"
    plot_payload: dict[str, dict[str, float]] = {}
    for scene_id, stats in per_scene_stats.items():
        plot_payload[scene_id] = {
            "success_rate": float(stats.get("success_rate", {}).get("mean", float("nan"))),
            "path_ratio": float(stats.get("path_ratio", {}).get("mean", float("nan"))),
            "collision_like": float(stats.get("collision_like", {}).get("mean", float("nan"))),
            "deaths_mean": float(stats.get("deaths_mean", {}).get("mean", float("nan"))),
        }
    plot_paths = write_summary_plots(plot_dir, plot_payload, title=label)

    write_experiment_result(
        out_dir,
        ExperimentResult(
            name="eval_protocol",
            seed=int(seeds[0]) if seeds else 0,
            config=OmegaConf.to_container(OmegaConf.create(data), resolve=True),
            metrics={"aggregate": aggregate_stats, "per_scene": per_scene_stats},
            artifacts={
                "report": str(report_path),
                "json": str(json_path),
                "plots": str(plot_dir) if plot_paths else "",
            },
        ),
    )

    tracking_cfg = {}
    if isinstance(cfg.logging, dict):
        tracking_cfg = cfg.logging.get("tracking", {}) or {}
    mlflow_run = init_mlflow(tracking_cfg.get("mlflow", {}), run_name=base_name)
    if mlflow_run is not None:
        try:
            log_params(mlflow_run, None, {"eval_protocol": OmegaConf.to_container(OmegaConf.create(data), resolve=True)})
            metrics_flat: dict[str, float] = {"eval/n_seeds": float(len(seeds))}
            for key, stat in aggregate_stats.items():
                metrics_flat[f"eval/mean/{key}"] = float(stat.get("mean", float("nan")))
                metrics_flat[f"eval/std/{key}"] = float(stat.get("std", float("nan")))
                metrics_flat[f"eval/ci95/{key}"] = float(stat.get("ci95", float("nan")))
            log_metrics(mlflow_run, None, metrics_flat, step=0)
            log_artifacts_mlflow(mlflow_run, [json_path, report_path], artifact_path="eval_protocol")
        except Exception:
            pass

    print(f"[eval_protocol] JSON: {json_path}")
    print(f"[eval_protocol] MD:   {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
