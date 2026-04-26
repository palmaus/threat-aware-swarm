from __future__ import annotations

import argparse
import cProfile
import json
import pstats
import time
from pathlib import Path

import yaml

from baselines.factory import create_baseline_policy
from env.config import EnvConfig
from env.pz_env import SwarmPZEnv
from scripts.common.artifacts import ensure_run_dir
from common.runtime.env_factory import apply_lite_metrics_cfg
from common.runtime.env_factory import make_pz_env as _common_make_pz_env
from scripts.common.episode_runner import policy_actions
from scripts.common.path_utils import resolve_repo_path

DEFAULT_POLICIES = ("baseline:astar_grid", "baseline:mpc_lite")
DEFAULT_SCENARIO = "scenarios/S7_dynamic_chaser.yaml"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fixed-step profiler for baseline policies.")
    parser.add_argument("--scenario", default=DEFAULT_SCENARIO)
    parser.add_argument("--policies", nargs="*", default=list(DEFAULT_POLICIES))
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--goal-radius", type=float, default=3.0)
    parser.add_argument("--lite-metrics", action="store_true")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--compare", default="", help="Эталонный JSON для сравнения.")
    parser.add_argument("--metric", choices=("wall", "profile"), default="wall")
    parser.add_argument("--max-regression-pct", type=float, default=-1.0)
    return parser.parse_args()


def normalize_policy_name(name: str) -> str:
    return name if name.startswith("baseline:") else f"baseline:{name}"


def metric_from_row(row: dict, *, preferred: str) -> float | None:
    if preferred == "wall":
        for key in ("wall_time_s_per_step", "step_only_s_per_step", "sec_per_step"):
            val = row.get(key)
            if val is not None:
                return float(val)
        return None
    for key in ("step_only_s_per_step", "wall_time_s_per_step", "sec_per_step"):
        val = row.get(key)
        if val is not None:
            return float(val)
    return None


def compare_profiles(current: dict, reference: dict, *, preferred: str) -> dict:
    summary: dict[str, dict] = {}
    for policy, row in current.items():
        cur = metric_from_row(row, preferred=preferred)
        ref_row = reference.get(policy)
        ref = metric_from_row(ref_row, preferred=preferred) if isinstance(ref_row, dict) else None
        delta_pct = None
        if cur is not None and ref not in (None, 0.0):
            delta_pct = ((cur - ref) / ref) * 100.0
        summary[policy] = {
            "current": cur,
            "reference": ref,
            "delta_pct": delta_pct,
            "preferred_metric": preferred,
        }
    return summary


def gate_regressions(comparison: dict, *, max_regression_pct: float) -> list[str]:
    if max_regression_pct < 0.0:
        return []
    failures: list[str] = []
    for policy, row in comparison.items():
        delta = row.get("delta_pct")
        if delta is not None and float(delta) > float(max_regression_pct):
            failures.append(policy)
    return failures


def _build_env(*, steps: int, goal_radius: float, lite_metrics: bool) -> SwarmPZEnv:
    cfg = EnvConfig()
    apply_lite_metrics_cfg(cfg, lite_metrics)
    return _common_make_pz_env(max_steps=int(steps), goal_radius=float(goal_radius), config=cfg, reset=False)


def _make_policy(name: str, env: SwarmPZEnv):
    return create_baseline_policy(name, env=env)


def _top_profile_rows(stats: pstats.Stats, *, limit: int = 15) -> list[dict]:
    rows: list[dict] = []
    for func, stat in sorted(stats.stats.items(), key=lambda kv: kv[1][3], reverse=True)[:limit]:
        _cc, nc, tt, ct, _callers = stat
        filename, line, fn = func
        rows.append(
            {
                "file": filename,
                "line": int(line),
                "func": fn,
                "cumtime": float(ct),
                "tottime": float(tt),
                "calls": int(nc),
            }
        )
    return rows


def profile_policy(
    policy_name: str,
    *,
    scene_path: str,
    steps: int,
    seed: int,
    goal_radius: float,
    lite_metrics: bool,
) -> dict:
    scene = yaml.safe_load(resolve_repo_path(scene_path).read_text(encoding="utf-8"))

    def _run_with_optional_profile(use_profile: bool) -> tuple[float, pstats.Stats | None]:
        env = _build_env(steps=steps, goal_radius=goal_radius, lite_metrics=lite_metrics)
        policy = _make_policy(policy_name, env)
        policy.reset(seed)
        obs, infos = env.reset(seed=seed, options={"scene": scene})
        profiler = cProfile.Profile() if use_profile else None
        if profiler is not None:
            profiler.enable()
        start = time.perf_counter()
        for _ in range(int(steps)):
            acts = policy_actions(env, policy, obs, infos, policy_name=policy_name)
            obs, _rew, _term, _trunc, infos = env.step(acts)
        elapsed = time.perf_counter() - start
        if profiler is None:
            return elapsed, None
        profiler.disable()
        return elapsed, pstats.Stats(profiler)

    wall_total, _ = _run_with_optional_profile(False)
    _elapsed, stats = _run_with_optional_profile(True)
    assert stats is not None
    return {
        "scene": scene_path,
        "max_steps": int(steps),
        "wall_time_total_sec": float(wall_total),
        "wall_time_s_per_step": float(wall_total / max(int(steps), 1)),
        "step_only_total_time_sec": float(stats.total_tt),
        "step_only_s_per_step": float(stats.total_tt / max(int(steps), 1)),
        "top": _top_profile_rows(stats),
    }


def write_summary_markdown(
    out_path: Path,
    *,
    scenario: str,
    steps: int,
    results: dict,
    comparison: dict | None = None,
) -> None:
    lines = [
        "# Профиль производительности baseline'ов",
        "",
        f"- Сцена: `{scenario}`",
        f"- Шаги: `{steps}` fixed steps",
        "",
        "| Policy | wall s/step | cProfile s/step |",
        "| --- | ---: | ---: |",
    ]
    for policy, row in results.items():
        lines.append(
            f"| `{policy}` | {row['wall_time_s_per_step']:.5f} | {row['step_only_s_per_step']:.5f} |"
        )
    if comparison:
        lines.extend(
            [
                "",
                "## Сравнение",
                "",
                "| Policy | Эталон | Текущее | Delta % | Metric |",
                "| --- | ---: | ---: | ---: | --- |",
            ]
        )
        for policy, row in comparison.items():
            ref = row.get("reference")
            cur = row.get("current")
            delta = row.get("delta_pct")
            ref_s = "-" if ref is None else f"{ref:.5f}"
            cur_s = "-" if cur is None else f"{cur:.5f}"
            delta_s = "-" if delta is None else f"{delta:+.1f}%"
            lines.append(f"| `{policy}` | {ref_s} | {cur_s} | {delta_s} | {row['preferred_metric']} |")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    policies = [normalize_policy_name(name) for name in args.policies]
    results = {
        policy: profile_policy(
            policy,
            scene_path=args.scenario,
            steps=args.steps,
            seed=args.seed,
            goal_radius=args.goal_radius,
            lite_metrics=bool(args.lite_metrics),
        )
        for policy in policies
    }
    run_dir = ensure_run_dir(
        category="perf",
        out_root="runs",
        run_id=None,
        out_dir=args.out_dir or None,
        prefix="baseline_profile",
    )
    profile_path = run_dir / "profile.json"
    profile_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    comparison = None
    failures: list[str] = []
    if args.compare:
        reference = json.loads(resolve_repo_path(args.compare).read_text(encoding="utf-8"))
        comparison = compare_profiles(results, reference, preferred=args.metric)
        (run_dir / "comparison.json").write_text(
            json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        failures = gate_regressions(comparison, max_regression_pct=float(args.max_regression_pct))

    write_summary_markdown(
        run_dir / "summary.md",
        scenario=args.scenario,
        steps=int(args.steps),
        results=results,
        comparison=comparison,
    )
    print(f"[profile_baselines] out={run_dir}")
    if failures:
        print(f"[profile_baselines] regression gate failed: {', '.join(failures)}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
