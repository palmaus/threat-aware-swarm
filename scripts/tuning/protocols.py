"""Shared tuning protocol helpers kept outside the main Optuna orchestrator."""

from __future__ import annotations

import hashlib
import math
import random
from pathlib import Path

from scripts.common.metrics_utils import aggregate_scene_metrics, metric_float
from scripts.common.scenario_eval import load_scenes, resolve_scene_paths
from scripts.common.tune_types import AggregateMetrics


def _stable_hash(text: str) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "little")


def _scene_id(scene: dict, fallback: str) -> str:
    return str(scene.get("id") or fallback)


def infer_scene_family(scene: dict) -> str:
    scene_id = str(scene.get("id", "")).lower()
    has_dynamic = bool(scene.get("dynamic_threats")) or bool(scene.get("target_motion"))
    has_walls = bool(scene.get("walls"))
    threats = len(scene.get("threats") or [])
    if "maze" in scene_id or "trap" in scene_id or "gap" in scene_id or "bottleneck" in scene_id or "corridor" in scene_id:
        topology = "maze"
    elif "forest" in scene_id or threats >= 8:
        topology = "dense"
    elif has_walls:
        topology = "walls"
    else:
        topology = "open"

    if "moving_goal" in scene_id:
        theme = "moving_goal"
    elif "chaser" in scene_id or "interceptor" in scene_id:
        theme = "pursuit"
    elif "risk" in scene_id:
        theme = "tradeoff"
    elif threats == 0 and not has_dynamic:
        theme = "no_threat"
    elif has_dynamic:
        theme = "dynamic"
    else:
        theme = "navigation"
    motion = "dynamic" if has_dynamic else "static"
    return f"{motion}_{topology}_{theme}"


def _dedupe_scene_entries(entries: list[tuple[Path, dict]]) -> list[tuple[Path, dict]]:
    seen: set[str] = set()
    out: list[tuple[Path, dict]] = []
    for path, scene in entries:
        scene_id = _scene_id(scene, path.stem)
        if scene_id in seen:
            continue
        seen.add(scene_id)
        out.append((path, scene))
    return out


def _split_scene_entries(entries: list[tuple[Path, dict]]) -> dict[str, list[tuple[Path, dict]]]:
    groups: dict[str, list[tuple[Path, dict]]] = {}
    for path, scene in entries:
        family = infer_scene_family(scene)
        groups.setdefault(family, []).append((path, scene))
    search: list[tuple[Path, dict]] = []
    holdout: list[tuple[Path, dict]] = []
    benchmark: list[tuple[Path, dict]] = []
    for family_entries in groups.values():
        ordered = sorted(family_entries, key=lambda item: _scene_id(item[1], item[0].stem))
        count = len(ordered)
        if count == 1:
            search.extend(ordered)
        elif count == 2:
            search.append(ordered[0])
            holdout.append(ordered[1])
        else:
            search.extend(ordered[:-2])
            holdout.append(ordered[-2])
            benchmark.append(ordered[-1])
    if not benchmark and len(holdout) >= 2:
        benchmark.append(holdout.pop())
    elif not benchmark and len(search) >= 3:
        benchmark.append(search.pop())
    if not holdout and benchmark:
        holdout = list(benchmark)
    return {
        "search": _dedupe_scene_entries(search),
        "holdout": _dedupe_scene_entries(holdout),
        "benchmark": _dedupe_scene_entries(benchmark),
    }


def _scene_list_hash(scenes: list[dict]) -> int:
    payload = ",".join(sorted(_scene_id(scene, f"scene_{idx}") for idx, scene in enumerate(scenes)))
    return _stable_hash(payload)


def _seed_pack_hash(seeds: list[int]) -> int:
    payload = ",".join(str(int(seed)) for seed in seeds)
    return _stable_hash(payload)


def _resolve_seed_pack(explicit: list[int], *, base_seed: int, count: int, offset: int) -> list[int]:
    if explicit:
        return [int(seed) for seed in explicit]
    return [int(base_seed + offset + idx) for idx in range(max(0, count))]


def _scene_family_map(scenes: list[dict]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for idx, scene in enumerate(scenes):
        mapping[_scene_id(scene, f"scene_{idx}")] = infer_scene_family(scene)
    return mapping


def _family_balanced_scene_subset(
    scenes: list[dict],
    scene_families: dict[str, str],
    target_count: int,
) -> list[dict]:
    if target_count >= len(scenes):
        return list(scenes)
    grouped: dict[str, list[dict]] = {}
    for idx, scene in enumerate(scenes):
        scene_id = _scene_id(scene, f"scene_{idx}")
        family = scene_families.get(scene_id, "ungrouped")
        grouped.setdefault(family, []).append(scene)
    ordered_families = sorted(grouped)
    selected: list[dict] = []
    while len(selected) < target_count:
        progressed = False
        for family in ordered_families:
            bucket = grouped.get(family) or []
            if not bucket:
                continue
            selected.append(bucket.pop(0))
            progressed = True
            if len(selected) >= target_count:
                break
        if not progressed:
            break
    return selected


def _stage_count(total: int, *, fraction: float | None, limit: int | None) -> int:
    if total <= 0:
        return 0
    if limit is not None and int(limit) > 0:
        return min(total, int(limit))
    frac = 1.0 if fraction is None else float(fraction)
    frac = min(1.0, max(0.0, frac))
    return max(1, min(total, math.ceil(total * frac)))


def _resolve_search_stages(
    args,
    *,
    search_scenes: list[dict],
    eval_seeds: list[int],
    max_steps: int,
    scene_families: dict[str, str],
    default_tuning_profiles: dict[str, list[dict]],
    policy_tuning_profile_overrides: dict[str, dict[str, list[dict]]],
    policy_name: str | None = None,
) -> list[dict]:
    raw_stages = list(getattr(args, "search_fidelity", []) or [])
    if not raw_stages:
        profile_name = str(getattr(args, "tuning_profile", "balanced") or "balanced").strip().lower()
        policy_profiles = getattr(args, "policy_search_fidelity", {}) or {}
        if policy_name and policy_name in policy_profiles and profile_name in (policy_profiles.get(policy_name) or {}):
            raw_stages = policy_profiles[policy_name][profile_name]
        elif policy_name and profile_name in policy_tuning_profile_overrides.get(policy_name, {}):
            raw_stages = policy_tuning_profile_overrides[policy_name][profile_name]
        else:
            raw_stages = default_tuning_profiles.get(profile_name, default_tuning_profiles["balanced"])
    stages: list[dict] = []
    seen_signatures: set[tuple[int, int, int]] = set()
    for idx, raw_stage in enumerate(raw_stages):
        scene_count = _stage_count(
            len(search_scenes),
            fraction=raw_stage.get("scene_frac"),
            limit=raw_stage.get("scene_limit"),
        )
        seed_count = _stage_count(
            len(eval_seeds),
            fraction=raw_stage.get("seed_frac"),
            limit=raw_stage.get("seed_limit"),
        )
        max_steps_frac = float(raw_stage.get("max_steps_frac", 1.0))
        stage_max_steps = max(1, round(max_steps * max_steps_frac))
        signature = (scene_count, seed_count, stage_max_steps)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        stage_scenes = _family_balanced_scene_subset(search_scenes, scene_families, scene_count)
        stage_seeds = list(eval_seeds[:seed_count])
        stage_ids = [_scene_id(scene, f"scene_{scene_idx}") for scene_idx, scene in enumerate(stage_scenes)]
        stages.append(
            {
                "label": str(raw_stage.get("label", f"stage_{idx + 1}")),
                "scenes": stage_scenes,
                "scene_ids": stage_ids,
                "scene_families": {scene_id: scene_families.get(scene_id, "ungrouped") for scene_id in stage_ids},
                "seeds": stage_seeds,
                "max_steps": stage_max_steps,
            }
        )
    if not stages:
        stages.append(
            {
                "label": "full",
                "scenes": list(search_scenes),
                "scene_ids": [_scene_id(scene, f"scene_{idx}") for idx, scene in enumerate(search_scenes)],
                "scene_families": dict(scene_families),
                "seeds": list(eval_seeds),
                "max_steps": max_steps,
            }
        )
    return stages


def _aggregate_by_family(per_scene: dict[str, AggregateMetrics], scene_families: dict[str, str]) -> AggregateMetrics:
    if not per_scene:
        return aggregate_scene_metrics({})
    families: dict[str, dict[str, AggregateMetrics]] = {}
    for scene_id, metrics in per_scene.items():
        family = scene_families.get(scene_id, "ungrouped")
        families.setdefault(family, {})[scene_id] = metrics
    family_metrics = {family: aggregate_scene_metrics(metrics) for family, metrics in families.items()}
    return aggregate_scene_metrics(family_metrics)


def _family_breakdown(per_scene: dict[str, AggregateMetrics], scene_families: dict[str, str]) -> dict[str, AggregateMetrics]:
    families: dict[str, dict[str, AggregateMetrics]] = {}
    for scene_id, metrics in per_scene.items():
        family = scene_families.get(scene_id, "ungrouped")
        families.setdefault(family, {})[scene_id] = metrics
    return {family: aggregate_scene_metrics(metrics) for family, metrics in families.items()}


def _aggregate_with_protocol(
    per_scene: dict[str, AggregateMetrics],
    *,
    scene_families: dict[str, str] | None,
    aggregate_by_family: bool,
) -> AggregateMetrics:
    if aggregate_by_family and scene_families:
        return _aggregate_by_family(per_scene, scene_families)
    return aggregate_scene_metrics(per_scene)


def _risk_metric(agg: dict | AggregateMetrics) -> float:
    risk = metric_float(agg, "risk_integral_alive", float("nan"))
    if risk != risk:
        risk = metric_float(agg, "risk_integral_all", 0.0)
    return risk


def _gate_threshold(value: float, fallback: float) -> float:
    if value < 0.0:
        return float(fallback)
    return float(value)


def _passes_gates(agg: dict | AggregateMetrics, args) -> bool:
    finish_gate = _gate_threshold(float(args.gate_finish_min), float(args.finish_min))
    alive_gate = _gate_threshold(float(args.gate_alive_min), float(args.alive_min))
    risk_gate = float(getattr(args, "gate_max_risk", -1.0))
    if metric_float(agg, "finished_frac_end", 0.0) < finish_gate:
        return False
    if metric_float(agg, "alive_frac_end", 0.0) < alive_gate:
        return False
    if risk_gate >= 0.0 and _risk_metric(agg) > risk_gate:
        return False
    return True


def _uncertainty_metric(agg: dict | AggregateMetrics) -> float:
    risk_std = metric_float(agg, "risk_integral_alive_std", float("nan"))
    if risk_std != risk_std:
        risk_std = metric_float(agg, "risk_integral_all_std", 0.0)
    return (
        metric_float(agg, "finished_frac_end_std", 0.0) * 100.0
        + metric_float(agg, "alive_frac_end_std", 0.0) * 50.0
        + risk_std * 10.0
        + metric_float(agg, "path_ratio_std", 0.0) * 5.0
    )


def _safe_sort_value(value: float, *, default: float) -> float:
    return value if value == value else default


def _dominates(left: dict, right: dict) -> bool:
    left_agg = left.get("aggregate") or {}
    right_agg = right.get("aggregate") or {}
    left_metrics = (
        metric_float(left_agg, "finished_frac_end", 0.0),
        metric_float(left_agg, "alive_frac_end", 0.0),
        -_risk_metric(left_agg),
        -metric_float(left_agg, "time_to_goal_mean", 1.0e9),
    )
    right_metrics = (
        metric_float(right_agg, "finished_frac_end", 0.0),
        metric_float(right_agg, "alive_frac_end", 0.0),
        -_risk_metric(right_agg),
        -metric_float(right_agg, "time_to_goal_mean", 1.0e9),
    )
    ge_all = all(left_val >= right_val for left_val, right_val in zip(left_metrics, right_metrics))
    gt_any = any(left_val > right_val for left_val, right_val in zip(left_metrics, right_metrics))
    return ge_all and gt_any


def _pareto_frontier(rows: list[dict], *, args, trial_value_fn) -> list[dict]:
    candidates = [row for row in rows if _passes_gates(row.get("aggregate") or {}, args)]
    if not candidates:
        candidates = list(rows)
    frontier = []
    for row in candidates:
        if any(_dominates(other, row) for other in candidates if other is not row):
            continue
        frontier.append(row)
    return sorted(frontier, key=trial_value_fn, reverse=True)


def _select_champions(rows: list[dict], *, args, trial_value_fn) -> dict[str, dict]:
    if not rows:
        return {}
    candidates = [row for row in rows if _passes_gates(row.get("aggregate") or {}, args)]
    if not candidates:
        candidates = list(rows)

    def balanced_key(row: dict):
        agg = row.get("aggregate") or {}
        return (trial_value_fn(row), -_uncertainty_metric(agg))

    def safe_key(row: dict):
        agg = row.get("aggregate") or {}
        return (
            -_risk_metric(agg),
            metric_float(agg, "alive_frac_end", 0.0),
            metric_float(agg, "finished_frac_end", 0.0),
            -metric_float(agg, "time_to_goal_mean", 1.0e9),
        )

    def fast_key(row: dict):
        agg = row.get("aggregate") or {}
        return (
            -metric_float(agg, "time_to_goal_mean", 1.0e9),
            metric_float(agg, "finished_frac_end", 0.0),
            metric_float(agg, "alive_frac_end", 0.0),
            -_risk_metric(agg),
        )

    def stable_key(row: dict):
        agg = row.get("aggregate") or {}
        return (
            -_uncertainty_metric(agg),
            metric_float(agg, "finished_frac_end", 0.0),
            metric_float(agg, "alive_frac_end", 0.0),
            -_risk_metric(agg),
            -metric_float(agg, "time_to_goal_mean", 1.0e9),
        )

    return {
        "balanced": max(candidates, key=balanced_key),
        "safe": max(candidates, key=safe_key),
        "fast": max(candidates, key=fast_key),
        "stable": max(candidates, key=stable_key),
    }


def _champion_reason(name: str, agg: dict) -> str:
    if name == "safe":
        return f"lowest risk focus (risk={_risk_metric(agg):.4f}, alive={metric_float(agg, 'alive_frac_end', 0.0):.3f})"
    if name == "fast":
        return f"speed-first winner (time={metric_float(agg, 'time_to_goal_mean', float('nan')):.2f})"
    if name == "stable":
        return f"lowest uncertainty winner (uncertainty={_uncertainty_metric(agg):.3f})"
    return (
        f"best balanced objective (finish={metric_float(agg, 'finished_frac_end', 0.0):.3f}, "
        f"alive={metric_float(agg, 'alive_frac_end', 0.0):.3f}, risk={_risk_metric(agg):.4f}, "
        f"uncertainty={_uncertainty_metric(agg):.3f})"
    )


def _build_tuning_protocol(
    args,
    explicit_scene_paths: list[Path],
    explicit_scenes: list[dict],
    *,
    resolve_scene_paths_fn=resolve_scene_paths,
    load_scenes_fn=load_scenes,
) -> dict:
    explicit_pairs = list(zip(explicit_scene_paths, explicit_scenes))
    if args.search_scenes or args.holdout_scenes or args.benchmark_scenes:
        search_paths = resolve_scene_paths_fn(args.search_scenes) if args.search_scenes else explicit_scene_paths
        holdout_paths = resolve_scene_paths_fn(args.holdout_scenes) if args.holdout_scenes else []
        benchmark_paths = resolve_scene_paths_fn(args.benchmark_scenes) if args.benchmark_scenes else []
        search_pairs = list(zip(search_paths, load_scenes_fn(search_paths)))
        holdout_pairs = list(zip(holdout_paths, load_scenes_fn(holdout_paths)))
        benchmark_pairs = list(zip(benchmark_paths, load_scenes_fn(benchmark_paths)))
    else:
        split = _split_scene_entries(explicit_pairs)
        search_pairs = split["search"]
        holdout_pairs = split["holdout"]
        benchmark_pairs = split["benchmark"]

    ood_paths = resolve_scene_paths_fn(args.ood_scenes) if args.ood_scenes else resolve_scene_paths_fn(["preset:ood"])
    ood_pairs = list(zip(ood_paths, load_scenes_fn(ood_paths)))

    search_scenes = [scene for _, scene in search_pairs]
    holdout_scenes = [scene for _, scene in holdout_pairs]
    benchmark_scenes = [scene for _, scene in benchmark_pairs]
    ood_scenes = [scene for _, scene in ood_pairs]
    search_paths = [path for path, _ in search_pairs]
    holdout_paths = [path for path, _ in holdout_pairs]
    benchmark_paths = [path for path, _ in benchmark_pairs]
    ood_paths = [path for path, _ in ood_pairs]

    search_seed_count = int(args.episodes)
    holdout_seed_count = int(args.episodes_eval if args.stage_b else args.episodes)
    report_seed_count = int(args.episodes_eval)
    search_seeds = _resolve_seed_pack(args.search_seeds, base_seed=int(args.seed), count=search_seed_count, offset=0)
    holdout_seeds = _resolve_seed_pack(
        args.holdout_seeds,
        base_seed=int(args.seed),
        count=holdout_seed_count,
        offset=10_000,
    )
    report_seeds = _resolve_seed_pack(args.report_seeds, base_seed=int(args.seed), count=report_seed_count, offset=20_000)

    return {
        "search_paths": search_paths,
        "holdout_paths": holdout_paths,
        "benchmark_paths": benchmark_paths,
        "ood_paths": ood_paths,
        "search_scenes": search_scenes,
        "holdout_scenes": holdout_scenes,
        "benchmark_scenes": benchmark_scenes,
        "ood_scenes": ood_scenes,
        "search_seeds": search_seeds,
        "holdout_seeds": holdout_seeds,
        "report_seeds": report_seeds,
        "summary": {
            "search_scene_ids": [_scene_id(scene, path.stem) for path, scene in search_pairs],
            "holdout_scene_ids": [_scene_id(scene, path.stem) for path, scene in holdout_pairs],
            "benchmark_scene_ids": [_scene_id(scene, path.stem) for path, scene in benchmark_pairs],
            "ood_scene_ids": [_scene_id(scene, path.stem) for path, scene in ood_pairs],
            "search_scene_hash": _scene_list_hash(search_scenes),
            "holdout_scene_hash": _scene_list_hash(holdout_scenes),
            "benchmark_scene_hash": _scene_list_hash(benchmark_scenes),
            "ood_scene_hash": _scene_list_hash(ood_scenes),
            "search_seed_hash": _seed_pack_hash(search_seeds),
            "holdout_seed_hash": _seed_pack_hash(holdout_seeds),
            "report_seed_hash": _seed_pack_hash(report_seeds),
        },
    }


def _sample_from_spec(spec: dict, rng: random.Random):
    if "values" in spec:
        return rng.choice(list(spec["values"]))
    lo = spec.get("min")
    hi = spec.get("max")
    if lo is None or hi is None:
        raise ValueError(f"Bad range spec: {spec}")
    spec_type = spec.get("type", "float")
    if spec.get("log", False):
        lo = math.log(float(lo))
        hi = math.log(float(hi))
        value = math.exp(rng.uniform(lo, hi))
    else:
        value = rng.uniform(float(lo), float(hi))
    if spec_type == "int":
        return round(value)
    return float(value)


__all__ = [
    "_aggregate_with_protocol",
    "_build_tuning_protocol",
    "_champion_reason",
    "_family_breakdown",
    "_pareto_frontier",
    "_passes_gates",
    "_resolve_search_stages",
    "_resolve_seed_pack",
    "_risk_metric",
    "_sample_from_spec",
    "_scene_family_map",
    "_scene_id",
    "_scene_list_hash",
    "_seed_pack_hash",
    "_select_champions",
    "_split_scene_entries",
    "infer_scene_family",
]
