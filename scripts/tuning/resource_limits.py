"""Resource caps for process-based baseline tuning."""

from __future__ import annotations

import math
import os
from typing import Any


def parallel_worker_memory_gb(
    policy_name: str,
    *,
    tuning_profile: str,
    step_budget: str,
) -> float:
    base = {
        "baseline:astar_grid": 1.2,
        "baseline:mpc_lite": 1.1,
        "baseline:flow_field": 0.6,
        "baseline:potential_fields": 0.6,
    }.get(str(policy_name), 1.5)
    profile_mult = {
        "fast": 1.0,
        "balanced": 1.2,
        "deep": 1.45,
    }.get(str(tuning_profile), 1.0)
    budget_mult = {
        "default": 1.0,
        "static": 1.15,
        "dynamic": 1.25,
        "long": 1.75,
    }.get(str(step_budget), 1.0)
    return float(base * profile_mult * budget_mult)


def parallel_policy_worker_cap(policy_name: str) -> int:
    return {
        "baseline:astar_grid": 8,
        "baseline:mpc_lite": 8,
        "baseline:flow_field": 8,
        "baseline:potential_fields": 8,
    }.get(str(policy_name), 8)


def policy_cache_size_cap(policy_name: str) -> int:
    return {
        "baseline:astar_grid": 1,
        "baseline:mpc_lite": 2,
        "baseline:flow_field": 4,
        "baseline:potential_fields": 4,
    }.get(str(policy_name), 8)


def resolve_policy_cache_size(args: Any, policy_name: str) -> int:
    requested = max(1, int(getattr(args, "policy_cache_size", 8)))
    cap = max(1, int(policy_cache_size_cap(policy_name)))
    return min(requested, cap)


def resolve_process_worker_count(
    args: Any,
    policy_name: str,
    *,
    psutil_module: Any = None,
    cpu_count: int | None = None,
) -> tuple[int, dict[str, int | float | str]]:
    requested = max(1, int(getattr(args, "n_jobs", 1)))
    explicit_cap = int(getattr(args, "parallel_max_workers", 0) or 0)
    reserve_cpus = max(0, int(getattr(args, "parallel_reserve_cpus", 2) or 0))
    memory_fraction = float(getattr(args, "parallel_memory_fraction", 0.7) or 0.7)
    tuning_profile = str(getattr(args, "tuning_profile", "balanced") or "balanced")
    step_budget = str(getattr(args, "step_budget_name", "default") or "default")

    cpu_total = max(1, int(cpu_count if cpu_count is not None else (os.cpu_count() or 1)))
    cpu_cap = max(1, cpu_total - reserve_cpus)
    if explicit_cap > 0:
        cpu_cap = min(cpu_cap, explicit_cap)

    mem_cap = requested
    mem_available_gb = 0.0
    worker_gb = parallel_worker_memory_gb(
        policy_name,
        tuning_profile=tuning_profile,
        step_budget=step_budget,
    )
    policy_cap = max(1, int(parallel_policy_worker_cap(policy_name)))
    if psutil_module is not None:
        try:
            mem_available_gb = float(psutil_module.virtual_memory().available) / float(1024**3)
            budget_gb = max(0.0, mem_available_gb * memory_fraction)
            mem_cap = max(1, int(budget_gb // max(worker_gb, 0.25)))
        except Exception:
            mem_cap = requested

    effective = max(1, min(requested, cpu_cap, mem_cap, policy_cap))
    meta: dict[str, int | float | str] = {
        "requested": requested,
        "effective": effective,
        "cpu_total": cpu_total,
        "cpu_cap": cpu_cap,
        "reserve_cpus": reserve_cpus,
        "mem_cap": max(1, int(mem_cap)),
        "mem_available_gb": round(mem_available_gb, 2),
        "worker_gb": round(worker_gb, 2),
        "memory_fraction": memory_fraction,
        "policy_cap": policy_cap,
        "policy_name": str(policy_name),
        "tuning_profile": tuning_profile,
        "step_budget": step_budget,
    }
    return effective, meta


def parallel_policy_trial_cap(policy_name: str, *, tuning_profile: str, step_budget: str) -> int:
    base = {
        "baseline:astar_grid": 1,
        "baseline:mpc_lite": 2,
        "baseline:flow_field": 4,
        "baseline:potential_fields": 4,
    }.get(str(policy_name), 8)
    if str(step_budget) == "long":
        base = max(1, base - 1)
    if str(tuning_profile) == "deep":
        base = max(1, base - 1)
    return max(1, int(base))


def resolve_process_worker_trial_cap(args: Any, policy_name: str) -> tuple[int, dict[str, int | str]]:
    tuning_profile = str(getattr(args, "tuning_profile", "balanced") or "balanced")
    step_budget = str(getattr(args, "step_budget_name", "default") or "default")
    default_cap = parallel_policy_trial_cap(
        policy_name,
        tuning_profile=tuning_profile,
        step_budget=step_budget,
    )
    requested_cap = int(getattr(args, "parallel_max_trials_per_worker", 0) or 0)
    effective_cap = default_cap if requested_cap <= 0 else min(default_cap, max(1, requested_cap))
    meta: dict[str, int | str] = {
        "requested": requested_cap,
        "default_cap": default_cap,
        "effective": effective_cap,
        "policy_name": str(policy_name),
        "tuning_profile": tuning_profile,
        "step_budget": step_budget,
    }
    return effective_cap, meta


def build_process_worker_batches(
    *,
    total_trials: int,
    max_workers: int,
    trial_cap: int,
) -> list[list[tuple[int, int]]]:
    remaining = max(0, int(total_trials))
    worker_limit = max(1, int(max_workers))
    per_worker_cap = max(1, int(trial_cap))
    batches: list[list[tuple[int, int]]] = []
    launch_id = 0
    while remaining > 0:
        batch: list[tuple[int, int]] = []
        slots = min(worker_limit, math.ceil(remaining / per_worker_cap))
        for _ in range(slots):
            quota = min(per_worker_cap, remaining)
            if quota <= 0:
                break
            batch.append((launch_id, quota))
            launch_id += 1
            remaining -= quota
        if not batch:
            break
        batches.append(batch)
    return batches
