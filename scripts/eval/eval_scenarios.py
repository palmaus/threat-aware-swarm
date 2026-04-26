import csv
import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace

from env.config import EnvConfig
from scripts.common.artifacts import ensure_run_dir, write_config_audit, write_manifest
from scripts.common.config_guardrails import validate_config_guardrails
from common.runtime.env_factory import apply_lite_metrics_cfg
from scripts.common.experiment_result import ExperimentResult, write_experiment_result
from scripts.common.metrics_plots import write_summary_plots
from scripts.common.metrics_schema import load_metrics_schema, validate_metrics, write_metrics_manifest
from scripts.common.metrics_utils import aggregate_episode_metrics, aggregate_scene_metrics
from scripts.common.model_registry import resolve_model_path
from scripts.common.numba_guard import log_numba_status
from scripts.common.rng_manager import SeedManager
from scripts.common import scenario_eval as scenario_eval_helpers
from scripts.common.scenario_eval import SUCCESS_THRESHOLD_DEFAULT
from scripts.common.scenario_eval import build_policy as _shared_build_policy
from scripts.common.scenario_eval import load_scenes, make_env, resolve_scene_paths
from scripts.common.scenario_eval import run_episode, scene_id as _scene_id

logger = logging.getLogger(__name__)
imageio = scenario_eval_helpers.imageio


def _apply_lite_metrics_cfg(cfg: EnvConfig, args: object) -> None:
    apply_lite_metrics_cfg(cfg, bool(getattr(args, "lite_metrics", False)))


def _env_audit_payload(args: object, *, goal_radius: float = 3.0) -> dict:
    cfg = EnvConfig()
    _apply_lite_metrics_cfg(cfg, args)
    payload = asdict(cfg)
    payload["max_steps"] = int(getattr(args, "max_steps", 0) or 0)
    payload["goal_radius"] = float(goal_radius)
    payload["oracle_enabled"] = bool(getattr(args, "oracle_enabled", False))
    return payload


def _build_policy(cfg: EnvConfig, policy_name: str, args):
    return _shared_build_policy(cfg, policy_name, model_path=getattr(args, "model", None))


def run_episode_gif(*args, **kwargs):
    previous_imageio = scenario_eval_helpers.imageio
    scenario_eval_helpers.imageio = imageio
    try:
        return scenario_eval_helpers.run_episode_gif(*args, **kwargs)
    finally:
        scenario_eval_helpers.imageio = previous_imageio


def _evaluate_scene(scene: dict, args_dict: dict) -> tuple[str, dict]:
    args = SimpleNamespace(**args_dict)
    cfg = EnvConfig()
    cfg.max_steps = int(args.max_steps)
    _apply_lite_metrics_cfg(cfg, args)
    policy, ppo_model = _build_policy(cfg, args.policy, args)
    env = make_env(
        cfg,
        max_steps=int(args.max_steps),
        goal_radius=3.0,
        oracle_enabled=bool(args.oracle_enabled),
        wrap_waypoint_actions=ppo_model is not None,
    )
    rows = []
    for ep in range(args.episodes):
        rows.append(
            run_episode(
                env,
                policy,
                scene,
                args.seed + ep,
                args.success_threshold,
                ppo_model=ppo_model,
                deterministic=args.deterministic,
            ),
        )
    return _scene_id(scene), aggregate_episode_metrics(rows)


def evaluate_scenes(scenes: list[dict], args) -> dict[str, dict]:
    workers = int(getattr(args, "workers", 0) or 0)
    if workers > 1 and len(scenes) > 1:
        results = {}
        args_dict = vars(args)
        try:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futures = [ex.submit(_evaluate_scene, scene, args_dict) for scene in scenes]
                for fut in as_completed(futures):
                    scene_id, agg = fut.result()
                    results[scene_id] = agg
            return results
        except (PermissionError, OSError):
            pass

    cfg = EnvConfig()
    cfg.max_steps = int(args.max_steps)
    _apply_lite_metrics_cfg(cfg, args)
    policy, ppo_model = _build_policy(cfg, args.policy, args)
    env = make_env(
        cfg,
        max_steps=int(args.max_steps),
        goal_radius=3.0,
        oracle_enabled=bool(args.oracle_enabled),
        wrap_waypoint_actions=ppo_model is not None,
    )
    per_scene = {}
    for idx, scene in enumerate(scenes):
        rows = []
        for ep in range(args.episodes):
            rows.append(
                run_episode(
                    env,
                    policy,
                    scene,
                    args.seed + ep,
                    args.success_threshold,
                    ppo_model=ppo_model,
                    deterministic=args.deterministic,
                ),
            )
        per_scene[_scene_id(scene, f"scene_{idx}")] = aggregate_episode_metrics(rows)
    return per_scene


def compare_golden(results, golden, thresholds):
    failures = []
    for scene_id, agg in results.items():
        g = golden.get(scene_id)
        if not g:
            continue
        if agg.get("success_rate") == agg.get("success_rate"):
            if scene_id != "S3_high_intensity_near_goal":
                if agg["success_rate"] < g["success_rate"] * (1.0 - thresholds["success_rate_rel"]):
                    failures.append((scene_id, "success_rate"))
        if agg.get("time_to_goal_mean") == agg.get("time_to_goal_mean") and g.get("time_to_goal_mean") == g.get(
            "time_to_goal_mean"
        ):
            if agg["time_to_goal_mean"] > g["time_to_goal_mean"] * (1.0 + thresholds["time_to_goal_rel"]):
                failures.append((scene_id, "time_to_goal_mean"))
        if agg.get("deaths_mean") == agg.get("deaths_mean"):
            if agg["deaths_mean"] > g["deaths_mean"] + thresholds["death_rate_abs"]:
                failures.append((scene_id, "deaths_mean"))
        if agg.get("risk_integral_all") == agg.get("risk_integral_all"):
            if agg["risk_integral_all"] > g["risk_integral_all"] * (1.0 + thresholds["risk_integral_rel"]):
                failures.append((scene_id, "risk_integral_all"))
    return failures


@dataclass
class EvalScenariosConfig:
    policy: str = "baseline:potential_fields"
    model: str = ""
    mlflow_run_id: str = ""
    mlflow_artifact_path: str = "models/final.zip"
    mlflow_tracking_uri: str = ""
    deterministic: bool = False
    episodes: int = 20
    max_steps: int = 600
    seed: int = 0
    success_threshold: float = SUCCESS_THRESHOLD_DEFAULT
    scenes: list[str] = field(default_factory=list)
    oracle_enabled: bool = False
    out_dir: str = ""
    out_root: str = "runs"
    run_id: str = ""
    update_golden: bool = False
    check_golden: bool = False
    workers: int = 0
    export_gif: bool = False
    gif_fps: int = 20
    gif_screen_size: int = 900
    gif_dir: str = ""
    metrics_schema: str = "configs/metrics_schema.yaml"
    lite_metrics: bool = False


def run(cfg: EvalScenariosConfig, *, cfg_raw: object | None = None) -> None:
    args = SimpleNamespace(**cfg.__dict__)
    validate_config_guardrails(cfg)
    SeedManager(int(args.seed)).seed_all()
    log_numba_status(logger)

    if args.policy == "ppo" and not args.model and args.mlflow_run_id:
        model_ref = f"mlflow:{args.mlflow_run_id}"
        model_path = resolve_model_path(
            model_ref,
            artifact_path=args.mlflow_artifact_path,
            download_dir=Path(args.out_root) / "mlflow_models" / args.mlflow_run_id,
            tracking_uri=args.mlflow_tracking_uri or None,
        )
        args.model = str(model_path)
    elif args.model and (str(args.model).startswith("mlflow:") or str(args.model).startswith("clearml:")):
        model_path = resolve_model_path(
            str(args.model),
            artifact_path=args.mlflow_artifact_path,
            download_dir=Path(args.out_root) / "models",
            tracking_uri=args.mlflow_tracking_uri or None,
        )
        args.model = str(model_path)

    scene_paths = resolve_scene_paths(args.scenes)
    scenes = load_scenes(scene_paths)

    policy_name = args.policy
    per_scene = evaluate_scenes(scenes, args)

    out_dir = ensure_run_dir(
        category="scenarios",
        out_root=args.out_root,
        run_id=args.run_id or None,
        prefix=args.policy.replace(":", "_"),
        out_dir=args.out_dir or None,
    )
    json_path = out_dir / "results.json"
    csv_path = out_dir / "results.csv"

    json_path.write_text(
        json.dumps(
            {
                "config": vars(args),
                "policy_name": policy_name,
                "scenes": per_scene,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["scene_id", "policy_name", *next(iter(per_scene.values())).keys()],
        )
        writer.writeheader()
        for sid, agg in per_scene.items():
            writer.writerow({"scene_id": sid, "policy_name": policy_name, **agg})

    if args.export_gif:
        if imageio is None:
            raise SystemExit("imageio not installed; set export_gif=false or install imageio")
        gif_root = Path(args.gif_dir) if args.gif_dir else (out_dir / "gifs")
        gif_root.mkdir(parents=True, exist_ok=True)
        cfg = EnvConfig()
        cfg.max_steps = int(args.max_steps)
        _apply_lite_metrics_cfg(cfg, args)
        policy, ppo_model = _build_policy(cfg, args.policy, args)
        env = make_env(
            cfg,
            max_steps=int(args.max_steps),
            goal_radius=3.0,
            oracle_enabled=bool(args.oracle_enabled),
            wrap_waypoint_actions=ppo_model is not None,
        )
        slug_policy = args.policy.replace(":", "_")
        for idx, scene in enumerate(scenes):
            scene_id = _scene_id(scene, f"scene_{idx}")
            gif_path = gif_root / f"{scene_id}_{slug_policy}.gif"
            run_episode_gif(
                env,
                policy,
                scene,
                args.seed,
                ppo_model=ppo_model,
                deterministic=args.deterministic,
                gif_path=gif_path,
                gif_fps=int(args.gif_fps),
                screen_size=int(args.gif_screen_size),
            )

    golden_path = Path("scenarios/golden/baseline_potential_fields.json")
    if args.update_golden:
        golden_path.write_text(json.dumps(per_scene, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.check_golden and golden_path.exists():
        golden = json.loads(golden_path.read_text() or "{}")
        if not golden:
            print("[ПРЕД] Golden‑файл пустой, проверка пропущена")
            golden = None
        thresholds = {
            "success_rate_rel": 0.10,
            "time_to_goal_rel": 0.15,
            "death_rate_abs": 0.10,
            "risk_integral_rel": 0.20,
        }
        if golden:
            failures = compare_golden(per_scene, golden, thresholds)
            if failures:
                raise SystemExit(f"Regression detected: {failures}")

    aggregate = aggregate_scene_metrics(per_scene)
    schema_path = Path(args.metrics_schema) if args.metrics_schema else None
    schema = load_metrics_schema(schema_path)
    missing, extra = validate_metrics(aggregate.keys(), schema.keys)
    write_metrics_manifest(
        out_dir,
        schema_path=schema_path,
        schema=schema,
        actual_keys=aggregate.keys(),
        missing=missing,
        extra=extra,
    )
    if missing or extra:
        strict = bool(args.check_golden or args.update_golden)
        msg = f"Metric schema mismatch. missing={missing} extra={extra}"
        if strict:
            raise SystemExit(msg)
        print(f"[WARN] {msg}")
    plot_dir = out_dir / "plots"
    plot_paths = write_summary_plots(plot_dir, per_scene, title=policy_name)
    write_experiment_result(
        out_dir,
        ExperimentResult(
            name="eval_scenarios",
            seed=int(args.seed),
            config=vars(args),
            metrics={"aggregate": aggregate, "per_scene": per_scene},
            artifacts={
                "results_json": str(json_path),
                "results_csv": str(csv_path),
                "plots": str(plot_dir) if plot_paths else "",
            },
        ),
    )
    write_manifest(out_dir, config=vars(args), command=["eval_scenarios"])
    audit_cfg = cfg_raw if cfg_raw is not None else vars(args)
    eval_seeds = [int(args.seed) + i for i in range(int(args.episodes))] if int(args.episodes) > 0 else []
    write_config_audit(
        out_dir,
        cfg=audit_cfg,
        scenes=[str(p) for p in scene_paths],
        seeds=eval_seeds,
        env_cfg=_env_audit_payload(args),
    )
    print(f"[ОК] {json_path}")
    print(f"[ОК] {csv_path}")


def main() -> None:
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base=None, config_path="../../configs/hydra", config_name="eval/scenarios")
    def _run(cfg: DictConfig) -> None:
        from omegaconf import OmegaConf

        from scripts.common.hydra_utils import apply_schema

        cfg = apply_schema(cfg, EvalScenariosConfig)
        data = OmegaConf.to_container(cfg, resolve=True)
        run(EvalScenariosConfig(**data), cfg_raw=cfg)

    _run()


if __name__ == "__main__":
    main()
