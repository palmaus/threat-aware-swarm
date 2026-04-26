"""Запуск бенчмарка базовых политик с сохранением метрик и отчёта."""

import csv
import json
import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from baselines.factory import create_baseline_policy
from env.config import EnvConfig
from env.pz_env import SwarmPZEnv
from scripts.common.artifacts import ensure_run_dir, write_config_audit, write_manifest
from scripts.common.config_guardrails import validate_config_guardrails
from common.runtime.env_factory import make_pz_env as _common_make_pz_env
from scripts.common.episode_metrics import EpisodeMetricsAccumulator
from scripts.common.episode_metrics import aggregate_infos
from scripts.common.episode_runner import policy_actions, reset_policy_context
from scripts.common.experiment_result import ExperimentResult, write_experiment_result
from scripts.common.metrics_schema import load_metrics_schema, validate_metrics, write_metrics_manifest
from scripts.common.metrics_utils import aggregate_episode_metrics
from scripts.common.numba_guard import log_numba_status
from scripts.common.rng_manager import SeedManager

logger = logging.getLogger(__name__)


def make_env(max_steps: int, goal_radius: float, seed: int, *, debug_metrics: bool = True) -> SwarmPZEnv:
    cfg = EnvConfig()
    return _common_make_pz_env(
        max_steps=max_steps,
        goal_radius=goal_radius,
        seed=seed,
        config=cfg,
        debug_metrics=bool(debug_metrics),
        reset=False,
    )


def _resolve_mp_context(mode: str | None) -> mp.context.BaseContext:
    if mode and mode != "auto":
        return mp.get_context(mode)
    for candidate in ("fork", "spawn", "forkserver"):
        try:
            return mp.get_context(candidate)
        except Exception:
            continue
    return mp.get_context()


def _build_policy(name: str, env: SwarmPZEnv, seed: int):
    return create_baseline_policy(name, env=env, seed=seed)


def _run_episode_worker(payload: tuple[str, dict, int]) -> tuple[int, float, dict]:
    name, args_dict, ep_idx = payload
    args = SimpleNamespace(**args_dict)
    env = make_env(
        max_steps=args.max_steps,
        goal_radius=3.0,
        seed=args.seed + ep_idx,
        debug_metrics=bool(getattr(args, "debug_metrics", False)),
    )
    policy = _build_policy(name, env, args.seed)
    policy.reset(args.seed + ep_idx)
    ep_len, ep_ret, metrics, _ = run_episode(
        env,
        policy,
        args.max_steps,
        args.seed + ep_idx,
        update_context=True,
        collect_infos=False,
        success_threshold=float(getattr(args, "success_threshold", 0.5)),
    )
    return ep_len, ep_ret, metrics


def _benchmark_policy(name: str, args_dict: dict, env: SwarmPZEnv | None = None) -> dict:
    args = SimpleNamespace(**args_dict)
    episode_workers = max(0, int(getattr(args, "episode_workers", 0)))
    if args.workers and args.workers > 1 and episode_workers > 1:
        print("[WARN] nested parallelism disabled; set --workers=1 to enable episode parallelism.")
        episode_workers = 0
    local_env = env or make_env(
        max_steps=args.max_steps,
        goal_radius=3.0,
        seed=args.seed,
        debug_metrics=bool(args.debug_metrics),
    )
    policy = _build_policy(name, local_env, args.seed)
    reset_policy_context(local_env, policy, args.seed, policy_name=name)

    debug_examples = []
    debug_counts = {}
    risk_timeline = None
    debug_metrics = bool(args.debug_metrics)
    if episode_workers > 1 and debug_metrics:
        print("[WARN] debug-metrics disabled for episode parallelism.")
        debug_metrics = False
    if debug_metrics:
        debug_counts = {
            "steps": 0,
            "infos_empty": 0,
            "risk_missing": 0,
            "risk_nan": 0,
            "risk_zero": 0,
            "agents_seen": 0,
            "keys": {},
        }

        def debug_hook(step_idx, infos, *, _counts=debug_counts, _examples=debug_examples):
            _counts["steps"] += 1
            if not infos:
                _counts["infos_empty"] += 1
                return
            for inf in infos.values():
                _counts["agents_seen"] += 1
                if not inf:
                    _counts["infos_empty"] += 1
                    continue
                for k in inf.keys():
                    _counts["keys"][k] = _counts["keys"].get(k, 0) + 1
                if "risk_p" not in inf:
                    _counts["risk_missing"] += 1
                else:
                    try:
                        val = float(inf.get("risk_p"))
                        if val != val:
                            _counts["risk_nan"] += 1
                        elif val == 0.0:
                            _counts["risk_zero"] += 1
                    except Exception:
                        _counts["risk_nan"] += 1
            if step_idx < 5 and len(_examples) < 5:
                _examples.append(_jsonify_infos(step_idx, infos))

        policy.debug_hook = debug_hook

    ep_metrics = []
    ep_returns = []
    ep_lens = []
    death_events = []
    if episode_workers > 1:
        ctx = _resolve_mp_context(getattr(args, "mp_context", "auto"))
        try:
            with ProcessPoolExecutor(max_workers=episode_workers, mp_context=ctx) as executor:
                futures = [executor.submit(_run_episode_worker, (name, args_dict, ep)) for ep in range(args.n_episodes)]
                for fut in as_completed(futures):
                    try:
                        ep_len, ep_ret, metrics = fut.result()
                    except Exception as exc:
                        print(f"[WARN] episode failed for {name}: {exc}")
                        continue
                    ep_lens.append(ep_len)
                    ep_returns.append(ep_ret)
                    ep_metrics.append(metrics)
        except PermissionError as exc:
            print(f"[WARN] episode parallelism failed ({exc}); falling back to sequential.")
            episode_workers = 0

    if episode_workers <= 1:
        collect_infos = True
        for ep in range(args.n_episodes):
            policy.reset(args.seed + ep)
            ep_len, ep_ret, metrics, per_step_infos = run_episode(
                local_env,
                policy,
                args.max_steps,
                args.seed + ep,
                update_context=True,
                collect_infos=collect_infos,
                success_threshold=float(getattr(args, "success_threshold", 0.5)),
            )
            ep_lens.append(ep_len)
            ep_returns.append(ep_ret)
            ep_metrics.append(metrics)
            _append_death_events(death_events, name, per_step_infos)
            if debug_metrics and risk_timeline is None:
                if _episode_has_death(per_step_infos):
                    risk_timeline = _build_risk_timeline(per_step_infos)

    def mean_key(k, _ep_metrics=ep_metrics):
        arr = np.array([m.get(k, float("nan")) for m in _ep_metrics], dtype=np.float32)
        return float(np.nanmean(arr)) if arr.size else float("nan")

    def mean_std_key(k, _ep_metrics=ep_metrics):
        arr = np.array([m.get(k, float("nan")) for m in _ep_metrics], dtype=np.float32)
        if not arr.size:
            return float("nan"), float("nan")
        return float(np.nanmean(arr)), float(np.nanstd(arr))

    alive_mean, alive_std = mean_std_key("alive_frac")
    fga_mean, fga_std = mean_std_key("finished_given_alive")
    dist_mean, dist_std = mean_std_key("mean_dist")

    agg_metrics = aggregate_episode_metrics(ep_metrics) if ep_metrics else {}

    row = {
        "policy": name,
        "n_episodes": args.n_episodes,
        "seed": args.seed,
        "episode_return_mean": float(np.mean(ep_returns)) if ep_returns else float("nan"),
        "episode_len_mean": float(np.mean(ep_lens)) if ep_lens else float("nan"),
        "alive_frac": alive_mean,
        "alive_frac_std": alive_std,
        "finished_frac": mean_key("finished_frac"),
        "finished_given_alive": fga_mean,
        "finished_given_alive_std": fga_std,
        "in_goal_frac": mean_key("in_goal_frac"),
        "mean_dist": dist_mean,
        "mean_dist_std": dist_std,
        "mean_risk_p": mean_key("mean_risk_p"),
        "success_all_finished": mean_key("success_all_finished"),
        "success_any_finished": mean_key("success_any_finished"),
        "success_frac_ge_0_5": mean_key("success_frac_ge_0_5"),
        "first_finish_step_mean": mean_key("first_finish_step"),
        "finished_count_end_mean": mean_key("finished_count_end"),
        "alive_count_end_mean": mean_key("alive_count_end"),
        "finished_given_alive_end_mean": mean_key("finished_given_alive_end"),
        "eval_dt": datetime.now().isoformat(timespec="seconds"),
    }
    row.update(agg_metrics)

    if debug_metrics and hasattr(policy, "debug_hook"):
        delattr(policy, "debug_hook")

    return {
        "policy": name,
        "row": row,
        "death_events": death_events,
        "risk_timeline": risk_timeline,
        "debug_examples": debug_examples,
        "debug_counts": debug_counts,
    }


def _episode_summary(per_step_infos, n_agents: int):
    finished_end = 0
    alive_end = 0
    first_finish_step = None
    for t, infos in enumerate(per_step_infos):
        for inf in infos.values():
            if not isinstance(inf, dict):
                continue
            if first_finish_step is None and inf.get("newly_finished", 0.0):
                first_finish_step = t + 1
    if per_step_infos:
        last = per_step_infos[-1]
        for inf in last.values():
            if not isinstance(inf, dict):
                continue
            if inf.get("finished", False):
                finished_end += 1
            if inf.get("alive", False):
                alive_end += 1
    finished_frac_end = finished_end / max(n_agents, 1)
    finished_given_alive_end = finished_end / max(alive_end, 1) if alive_end > 0 else float("nan")
    return {
        "finished_count_end": finished_end,
        "alive_count_end": alive_end,
        "first_finish_step": first_finish_step if first_finish_step is not None else -1,
        "finished_given_alive_end": finished_given_alive_end,
        "success_all_finished": float(finished_end == n_agents),
        "success_any_finished": float(finished_end > 0),
        "success_frac_ge_0_5": float(finished_frac_end >= 0.5),
    }


def _append_death_events(events, policy_name, per_step_infos):
    if not per_step_infos:
        return

    def _coord(value, index: int) -> float:
        try:
            arr = np.asarray(value, dtype=np.float32).reshape(-1)
        except Exception:
            return float("nan")
        if arr.size <= index:
            return float("nan")
        return float(arr[index])

    for t, infos in enumerate(per_step_infos):
        for agent_id, inf in infos.items():
            if not isinstance(inf, dict):
                continue
            if float(inf.get("died_this_step", 0.0)) > 0.0:
                events.append(
                    {
                        "policy": policy_name,
                        "step_idx": t,
                        "agent_id": agent_id,
                        "pos_x": _coord(inf.get("pos"), 0),
                        "pos_y": _coord(inf.get("pos"), 1),
                        "died_this_step": float(inf.get("died_this_step", 0.0)),
                        "dist_to_nearest_threat": float(inf.get("dist_to_nearest_threat", float("nan"))),
                        "nearest_threat_margin": float(inf.get("nearest_threat_margin", float("nan"))),
                        "min_wall_dist": float(inf.get("min_wall_dist", float("nan"))),
                        "dist_to_target": float(inf.get("dist", float("nan"))),
                    }
                )


def _episode_has_death(per_step_infos):
    for infos in per_step_infos:
        for inf in infos.values():
            if isinstance(inf, dict) and float(inf.get("died_this_step", 0.0)) > 0.0:
                return True
    return False


def _build_risk_timeline(per_step_infos):
    rows = []
    for t, infos in enumerate(per_step_infos):
        for agent_id, inf in infos.items():
            if not isinstance(inf, dict):
                continue
            rows.append(
                {
                    "step_idx": t,
                    "agent_id": agent_id,
                    "alive_before": inf.get("alive_before", float("nan")),
                    "alive_after": inf.get("alive_after", float("nan")),
                    "died_this_step": inf.get("died_this_step", float("nan")),
                    "pos_before": inf.get("pos_before", None),
                    "pos_after": inf.get("pos_after", None),
                    "dist_to_nearest_threat_before": inf.get("dist_to_nearest_threat_before", float("nan")),
                    "dist_to_nearest_threat_after": inf.get("dist_to_nearest_threat", float("nan")),
                    "nearest_threat_radius": inf.get("nearest_threat_radius", float("nan")),
                    "nearest_threat_intensity_before": inf.get("nearest_threat_intensity_before", float("nan")),
                    "nearest_threat_intensity_after": inf.get("nearest_threat_intensity", float("nan")),
                    "nearest_threat_id_before": inf.get("nearest_threat_id_before", float("nan")),
                    "nearest_threat_id_after": inf.get("nearest_threat_id", float("nan")),
                    "inside_nearest_before": inf.get("inside_nearest_before", float("nan")),
                    "inside_any_before": inf.get("inside_any_before", float("nan")),
                    "any_inside_intensity_sum_before": inf.get("any_inside_intensity_sum_before", float("nan")),
                    "margin_before": inf.get("nearest_threat_margin_before", float("nan")),
                    "margin_after": inf.get("nearest_threat_margin", float("nan")),
                    "risk_p_before": inf.get("risk_p_before", float("nan")),
                    "risk_p_after": inf.get("risk_p_after", float("nan")),
                    "risk_p_logged": inf.get("risk_p", float("nan")),
                }
            )
    return rows


def _jsonify_infos(step_idx, infos):
    def _norm(v):
        if isinstance(v, (int, float, str, bool)) or v is None:
            return v
        if isinstance(v, np.ndarray):
            return v.tolist()
        try:
            return float(v)
        except Exception:
            return str(v)

    out = {"step": int(step_idx), "infos": {}}
    for k, inf in infos.items():
        if not isinstance(inf, dict):
            out["infos"][k] = str(inf)
            continue
        out["infos"][k] = {ik: _norm(iv) for ik, iv in inf.items()}
    return out


def _render_infos_schema(counts):
    lines = ["# Схема infos", ""]
    for policy, c in counts.items():
        lines.append(f"## {policy}")
        lines.append("")
        lines.append(f"- шаги: {c.get('steps', 0)}")
        lines.append(f"- агентов просмотрено: {c.get('agents_seen', 0)}")
        lines.append(f"- пустых infos: {c.get('infos_empty', 0)}")
        lines.append(f"- отсутствие risk_p: {c.get('risk_missing', 0)}")
        lines.append(f"- risk_p = NaN: {c.get('risk_nan', 0)}")
        lines.append(f"- risk_p = 0: {c.get('risk_zero', 0)}")
        lines.append("")
        lines.append("| ключ | частота |")
        lines.append("| --- | --- |")
        for k, v in sorted(c.get("keys", {}).items(), key=lambda kv: kv[0]):
            lines.append(f"| {k} | {v} |")
        lines.append("")
    return "\n".join(lines)


def run_episode(
    env: SwarmPZEnv,
    policy,
    max_steps: int,
    seed: int,
    *,
    update_context: bool = False,
    collect_infos: bool = True,
    success_threshold: float = 0.5,
):
    try:
        obs, infos = env.reset(seed=seed)
    except Exception:
        obs, infos = env.reset()

    reset_policy_context(env, policy, seed, set_context=bool(update_context))

    ep_return = 0.0
    last_infos = None
    per_step_infos = [] if collect_infos else None
    metrics = EpisodeMetricsAccumulator.from_env(env, success_threshold=success_threshold)
    for t in range(max_steps):
        actions = policy_actions(env, policy, obs, infos)

        obs, rewards, terminations, truncations, infos = env.step(actions)
        if per_step_infos is not None:
            per_step_infos.append(infos)
        if hasattr(policy, "debug_hook") and callable(policy.debug_hook):
            policy.debug_hook(t, infos)
        decision_step = t + 1
        metrics.update(env, infos, decision_step=decision_step)
        last_infos = infos
        try:
            ep_return += float(np.mean(list(rewards.values())))
        except Exception:
            pass
        if all(terminations.values()) or all(truncations.values()):
            break

    steps = t + 1 if last_infos is not None else 0
    summary = aggregate_infos(last_infos or {})
    summary.update(metrics.summary(env, steps=steps))
    return steps, ep_return, summary, per_step_infos


@dataclass
class BenchmarkBaselinesConfig:
    policy: str = "all"
    n_episodes: int = 20
    max_steps: int = 600
    seed: int = 0
    success_threshold: float = 0.5
    out: str = ""
    out_dir: str = ""
    out_root: str = "runs"
    run_id: str = ""
    runs_dir: str = ""
    debug_metrics: bool = False
    workers: int = 0
    episode_workers: int = 0
    mp_context: str = "auto"
    metrics_schema: str = "configs/metrics_schema.yaml"


def _env_audit_payload(args: BenchmarkBaselinesConfig, *, goal_radius: float = 3.0) -> dict:
    payload = asdict(EnvConfig())
    payload["max_steps"] = int(args.max_steps)
    payload["goal_radius"] = float(goal_radius)
    return payload


def run(cfg: BenchmarkBaselinesConfig, *, cfg_raw: object | None = None) -> None:
    args = cfg
    validate_config_guardrails(cfg)
    SeedManager(int(args.seed)).seed_all()
    log_numba_status(logger)

    if args.debug_metrics and (
        (args.workers and args.workers > 1) or (args.episode_workers and args.episode_workers > 1)
    ):
        print("[WARN] debug-metrics disabled in parallel mode; set workers=1 to enable diagnostics.")
        args.debug_metrics = False

    out_dir_arg = args.out_dir or args.out
    run_dir = ensure_run_dir(
        category="bench",
        out_root=args.out_root,
        run_id=args.run_id or None,
        prefix=args.policy,
        out_dir=out_dir_arg or None,
    )

    all_policies = [
        "random",
        "zero",
        "greedy",
        "greedy_safe",
        "pf",
        "flow",
        "wall",
        "brake",
        "astar",
        "sep",
    ]
    selected = all_policies if args.policy == "all" else [args.policy]

    rows_by_policy = {}
    death_events = []
    risk_timeline = {}
    debug_examples = {}
    debug_counts = {}

    if args.workers and args.workers > 1 and len(selected) > 1:
        ctx = _resolve_mp_context(args.mp_context)
        args_dict = vars(args)
        try:
            with ProcessPoolExecutor(max_workers=int(args.workers), mp_context=ctx) as executor:
                futures = {executor.submit(_benchmark_policy, name, args_dict): name for name in selected}
                for fut in as_completed(futures):
                    name = futures[fut]
                    try:
                        result = fut.result()
                    except Exception as exc:
                        print(f"[WARN] policy benchmark failed for {name}: {exc}")
                        continue
                    rows_by_policy[name] = result["row"]
                    death_events.extend(result["death_events"])
                    if args.debug_metrics:
                        debug_examples[name] = result.get("debug_examples", [])
                        debug_counts[name] = result.get("debug_counts", {})
                        if result.get("risk_timeline"):
                            risk_timeline[name] = result.get("risk_timeline")
        except PermissionError as exc:
            print(f"[WARN] policy parallelism failed ({exc}); falling back to sequential.")
            args.workers = 0

    if not rows_by_policy:
        env = make_env(
            max_steps=args.max_steps,
            goal_radius=3.0,
            seed=args.seed,
            debug_metrics=bool(args.debug_metrics),
        )
        args_dict = vars(args)
        for name in selected:
            result = _benchmark_policy(name, args_dict, env=env)
            rows_by_policy[name] = result["row"]
            death_events.extend(result["death_events"])
            if args.debug_metrics:
                debug_examples[name] = result.get("debug_examples", [])
                debug_counts[name] = result.get("debug_counts", {})
                if result.get("risk_timeline"):
                    risk_timeline[name] = result.get("risk_timeline")

    rows = [rows_by_policy[name] for name in selected if name in rows_by_policy]
    if not rows:
        raise SystemExit("No benchmark results produced.")

    schema_path = Path(args.metrics_schema) if args.metrics_schema else None
    schema = load_metrics_schema(schema_path)
    missing, extra = validate_metrics(rows[0].keys(), schema.keys)
    write_metrics_manifest(
        run_dir,
        schema_path=schema_path,
        schema=schema,
        actual_keys=rows[0].keys(),
        missing=missing,
        extra=extra,
    )
    if missing:
        raise SystemExit(f"Metric schema mismatch (missing={missing}).")
    if extra:
        print(f"[WARN] Extra metrics not in schema: {extra}")

    csv_path = run_dir / "results.csv"
    json_path = run_dir / "results.json"

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    write_experiment_result(
        run_dir,
        ExperimentResult(
            name="benchmark_baselines",
            seed=int(args.seed),
            config=vars(args),
            metrics={"rows": rows},
            artifacts={"results_json": str(json_path), "results_csv": str(csv_path)},
        ),
    )

    json_path.write_text(
        json.dumps(
            {
                "config": vars(args),
                "rows": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    eval_results = []
    for r in rows:
        eval_results.append(
            {
                "model_name": r["policy"],
                "eval_fixed/alive_frac_mean": r["alive_frac"],
                "eval_fixed/finished_given_alive_mean": r["finished_given_alive"],
                "eval_fixed/mean_dist_mean": r["mean_dist"],
                "eval_fixed/alive_frac_std": r.get("alive_frac_std", float("nan")),
                "eval_fixed/finished_given_alive_std": r.get("finished_given_alive_std", float("nan")),
                "eval_fixed/mean_dist_std": r.get("mean_dist_std", float("nan")),
                "eval_dt": r["eval_dt"],
            }
        )

    eval_path = run_dir / "eval_results.json"
    eval_path.write_text(json.dumps(eval_results, ensure_ascii=False, indent=2), encoding="utf-8")

    report_csv = run_dir / "report.csv"
    report_md = run_dir / "report.md"
    death_csv = run_dir / "death_events.csv"
    with report_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    lines = [
        "# Отчёт по бейзлайнам",
        "",
        "| policy | finished_frac | alive_frac | finished_given_alive | mean_dist | mean_risk_p | success_all | success_any | success_ge_0_5 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append(
            f"| {r['policy']} | {r.get('finished_frac', '')} | {r.get('alive_frac', '')} | "
            f"{r.get('finished_given_alive', '')} | {r.get('mean_dist', '')} | {r.get('mean_risk_p', '')} | "
            f"{r.get('success_all_finished', '')} | {r.get('success_any_finished', '')} | {r.get('success_frac_ge_0_5', '')} |"
        )
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if death_events:
        for e in death_events:
            if "dist_to_target" not in e:
                e["dist_to_target"] = float("nan")
        conds = {
            "within_threat": 0,
            "near_wall": 0,
            "other": 0,
        }
        for e in death_events:
            margin = e.get("nearest_threat_margin", float("nan"))
            min_wall = e.get("min_wall_dist", float("nan"))
            if margin == margin and margin >= 0.0:
                conds["within_threat"] += 1
            elif min_wall == min_wall and min_wall < 0.05:
                conds["near_wall"] += 1
            else:
                conds["other"] += 1
        print("[ИНФО] Наиболее частые условия смерти:", sorted(conds.items(), key=lambda kv: kv[1], reverse=True)[:3])

    if death_events:
        with death_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(death_events[0].keys()))
            writer.writeheader()
            for r in death_events:
                writer.writerow(r)

    write_manifest(run_dir, config=vars(args), command=["benchmark_baselines"])
    audit_cfg = cfg_raw if cfg_raw is not None else vars(args)
    eval_seeds = [int(args.seed) + i for i in range(int(args.n_episodes))] if int(args.n_episodes) > 0 else []
    write_config_audit(
        run_dir,
        cfg=audit_cfg,
        seeds=eval_seeds,
        env_cfg=_env_audit_payload(args),
    )
    print(f"[ОК] {csv_path}")


def main() -> None:
    import hydra
    from omegaconf import DictConfig, OmegaConf

    from scripts.common.hydra_utils import apply_schema

    @hydra.main(version_base=None, config_path="../../configs/hydra", config_name="bench/benchmark_baselines")
    def _run(cfg: DictConfig) -> None:
        cfg = apply_schema(cfg, BenchmarkBaselinesConfig)
        data = OmegaConf.to_container(cfg, resolve=True)
        run(BenchmarkBaselinesConfig(**data), cfg_raw=cfg)

    _run()


if __name__ == "__main__":
    main()
