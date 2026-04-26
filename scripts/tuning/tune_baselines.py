"""Подбор гиперпараметров базовых политик по набору сценариев."""

import concurrent.futures as cf
import csv
import fcntl
import hashlib
import itertools
import json
import logging
import math
import multiprocessing as mp
import os
import random
import signal
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

try:
    import psutil
except Exception:  # pragma: no cover - опциональная зависимость
    psutil = None

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

from common.runtime.episode_runner import reset_policy_context
from env.config import EnvConfig
from env.pz_env import SwarmPZEnv
from scripts.common.artifacts import ensure_run_dir, write_config_audit, write_manifest
from scripts.common.config_guardrails import validate_config_guardrails
from common.runtime.env_overrides import split_env_runtime_fields
from scripts.common.experiment_result import ExperimentResult, write_experiment_result
from common.runtime.env_factory import apply_lite_metrics_cfg as _common_apply_lite_metrics_cfg
from common.runtime.env_factory import make_pz_env as _common_make_pz_env
from scripts.common.metrics_plots import write_summary_plots
from scripts.common.metrics_utils import (
    TUNING_METRIC_KEYS,
    ScoreWeights,
    aggregate_episode_metrics,
    aggregate_scene_metrics,
    metric_float,
    score_scalar,
    score_scalar_soft,
    score_tuple,
)
from scripts.common.numba_guard import log_numba_status
from scripts.common.path_utils import resolve_repo_path
from scripts.common.rng_manager import SeedManager
from scripts.common.scenario_eval import load_scenes, resolve_scene_paths as _shared_resolve_scene_paths, run_episode
from scripts.common.tune_types import AggregateMetrics, StageBRecord, TrialRecord
from scripts.tuning import eval_cache as eval_cache_helpers
from scripts.tuning import protocols as tuning_protocols
from scripts.tuning import reporting as tuning_reporting
from scripts.tuning.policy_factory import create_tuning_policy
from scripts.tuning import resource_limits
from scripts.tuning.scoring import make_weighted_scorers

logger = logging.getLogger(__name__)

DEFAULT_SPACES = {
    "baseline:astar_grid": {
        "alpha": [2.0, 3.0, 4.0],
        "max_cost": [0.30, 0.35, 0.40],
        "goal_radius": [5, 7, 9],
        "safe_max_cost": [0.03, 0.05, 0.07],
    },
    "baseline:mpc_lite": {
        "accel_levels": [
            [1.0],
            [0.5, 1.0],
            [0.35, 0.7, 1.0],
        ],
        "horizon": [1, 2, 3],
        "w_progress": [4.0, 5.0, 6.0],
        "w_risk": [4.0, 6.0, 8.0],
        "w_collision": [2.0, 4.0, 6.0],
        "safety_margin": [2.0, 3.0, 4.0],
        "fallback_score": [-0.5, -0.2, 0.0],
        "stuck_steps": [10, 20, 30],
    },
    "baseline:separation_steering": {
        "w_goal": [0.8, 1.0, 1.2],
        "w_avoid": [1.0, 1.5, 2.0],
        "w_sep": [1.0, 2.0, 3.0],
    },
    "baseline:greedy_safe": {
        "w_goal": [0.8, 1.0, 1.2],
        "w_avoid": [0.4, 0.7, 1.0],
        "w_wall": [0.2, 0.4, 0.6],
        "grid_radius": [1, 2, 3],
    },
    "baseline:potential_fields": {
        "k_att": [0.8, 1.0, 1.2],
        "k_rep": [20.0, 40.0, 60.0],
        "safety_margin": [2.0, 5.0, 8.0],
    },
    "baseline:flow_field": {
        "speed_gain": [0.6, 0.8, 1.0],
        "separation_gain": [0.0, 0.25, 0.5],
        "separation_radius": [2.0, 3.0, 4.0],
        "separation_power": [1.0, 1.5, 2.0],
    },
    "random": {},
    "zero": {},
    "greedy": {},
    "wall": {},
    "brake": {},
}


def _persistent_runtime_enabled(
    method: str,
    n_jobs: int,
    persistent_workers: bool,
    *,
    optuna_parallel_backend: str = "thread_debug",
    allow_unsafe_persistent_runtime: bool = False,
) -> bool:
    if not persistent_workers:
        return False
    if (
        str(method).strip().lower() == "optuna"
        and int(n_jobs) > 1
        and optuna_parallel_backend == "thread_debug"
        and not allow_unsafe_persistent_runtime
    ):
        return False
    return True


def _effective_optuna_n_jobs(
    method: str,
    n_jobs: int,
    *,
    allow_unsafe_parallel_optuna: bool = False,
) -> int:
    requested = max(1, int(n_jobs))
    if str(method).strip().lower() == "optuna" and requested > 1 and not allow_unsafe_parallel_optuna:
        return 1
    return requested


def _resolve_optuna_parallel_backend(
    method: str,
    n_jobs: int,
    *,
    allow_unsafe_parallel_optuna: bool = False,
) -> str:
    requested = max(1, int(n_jobs))
    if str(method).strip().lower() != "optuna" or requested <= 1:
        return "serial"
    if allow_unsafe_parallel_optuna:
        return "thread_debug"
    return "process"


def _parallel_worker_memory_gb(
    policy_name: str,
    *,
    tuning_profile: str,
    step_budget: str,
) -> float:
    return resource_limits.parallel_worker_memory_gb(
        policy_name,
        tuning_profile=tuning_profile,
        step_budget=step_budget,
    )


def _parallel_policy_worker_cap(policy_name: str) -> int:
    return resource_limits.parallel_policy_worker_cap(policy_name)


def _policy_cache_size_cap(policy_name: str) -> int:
    return resource_limits.policy_cache_size_cap(policy_name)


def _resolve_policy_cache_size(args, policy_name: str) -> int:
    return resource_limits.resolve_policy_cache_size(args, policy_name)


def _resolve_process_worker_count(args, policy_name: str) -> tuple[int, dict[str, int | float | str]]:
    return resource_limits.resolve_process_worker_count(
        args,
        policy_name,
        psutil_module=psutil,
        cpu_count=os.cpu_count(),
    )


def _parallel_policy_trial_cap(policy_name: str, *, tuning_profile: str, step_budget: str) -> int:
    return resource_limits.parallel_policy_trial_cap(
        policy_name,
        tuning_profile=tuning_profile,
        step_budget=step_budget,
    )


def _resolve_process_worker_trial_cap(args, policy_name: str) -> tuple[int, dict[str, int | str]]:
    return resource_limits.resolve_process_worker_trial_cap(args, policy_name)


def _build_process_worker_batches(
    *,
    total_trials: int,
    max_workers: int,
    trial_cap: int,
) -> list[list[tuple[int, int]]]:
    return resource_limits.build_process_worker_batches(
        total_trials=total_trials,
        max_workers=max_workers,
        trial_cap=trial_cap,
    )

DEFAULT_TUNING_PROFILES = {
    "fast": [
        {"label": "warmup", "scene_frac": 0.4, "seed_frac": 0.34, "max_steps_frac": 0.35},
        {"label": "focus", "scene_frac": 0.7, "seed_frac": 0.67, "max_steps_frac": 0.6},
        {"label": "full", "scene_frac": 1.0, "seed_frac": 1.0, "max_steps_frac": 1.0},
    ],
    "balanced": [
        {"label": "warmup", "scene_frac": 0.5, "seed_frac": 0.5, "max_steps_frac": 0.5},
        {"label": "full", "scene_frac": 1.0, "seed_frac": 1.0, "max_steps_frac": 1.0},
    ],
    "deep": [
        {"label": "warmup", "scene_frac": 0.5, "seed_frac": 0.5, "max_steps_frac": 0.5},
        {"label": "focus", "scene_frac": 0.8, "seed_frac": 0.8, "max_steps_frac": 0.8},
        {"label": "full", "scene_frac": 1.0, "seed_frac": 1.0, "max_steps_frac": 1.0},
    ],
}

POLICY_TUNING_PROFILE_OVERRIDES = {
    "baseline:astar_grid": {
        "fast": [
            {"label": "warmup", "scene_frac": 0.5, "seed_frac": 0.5, "max_steps_frac": 0.4},
            {"label": "full", "scene_frac": 1.0, "seed_frac": 1.0, "max_steps_frac": 1.0},
        ],
    },
    "baseline:mpc_lite": {
        "fast": [
            {"label": "warmup", "scene_frac": 0.5, "seed_frac": 0.5, "max_steps_frac": 0.4},
            {"label": "full", "scene_frac": 1.0, "seed_frac": 1.0, "max_steps_frac": 1.0},
        ],
    },
}

_PERSISTENT_ENV_CACHE: dict[str, SwarmPZEnv] = {}
_PERSISTENT_POLICY_CACHE: OrderedDict[str, object] = OrderedDict()
_PARALLEL_TRACE_LOCK = threading.Lock()
_RUNTIME_OWNER_LOCK = threading.Lock()
_RUNTIME_OWNERS: dict[int, dict[str, object]] = {}

CACHE_LINEAGE_PATHS = list(eval_cache_helpers.DEFAULT_CACHE_LINEAGE_PATHS)

FINISH_MIN_DEFAULT = 0.70
ALIVE_MIN_DEFAULT = 0.70
TRIAL_EXTRA_KEYS = ["score", "score_scalar", "value", "prune_value", "state", "reported_step"]

# Пороговые значения позволяют отсеивать решения, которые «выживают» ценой провала цели.


def _maybe_tqdm(iterable, total: int | None = None, desc: str | None = None):
    if tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc, leave=False)


def _parallel_debug_config(
    *,
    enabled: bool,
    trace_enabled: bool,
    trace_path: str,
    assert_ownership: bool,
    allow_unsafe_parallel_optuna: bool,
    allow_unsafe_persistent_runtime: bool,
) -> dict[str, object]:
    return {
        "enabled": bool(enabled),
        "trace_enabled": bool(trace_enabled),
        "trace_path": str(trace_path),
        "assert_ownership": bool(assert_ownership),
        "allow_unsafe_parallel_optuna": bool(allow_unsafe_parallel_optuna),
        "allow_unsafe_persistent_runtime": bool(allow_unsafe_persistent_runtime),
    }


def _make_parallel_debug(args, *, out_dir: Path) -> dict[str, object]:
    enabled = bool(getattr(args, "parallel_debug", False))
    trace_enabled = bool(getattr(args, "parallel_trace", False))
    assert_ownership = bool(getattr(args, "parallel_assert_ownership", True))
    allow_unsafe_parallel_optuna = bool(getattr(args, "allow_unsafe_parallel_optuna", False))
    allow_unsafe_persistent_runtime = bool(getattr(args, "allow_unsafe_persistent_runtime", False))
    trace_path = str(getattr(args, "parallel_trace_path", "") or "")
    if trace_enabled and not trace_path:
        trace_path = str(out_dir / "parallel_trace.jsonl")
    debug_cfg = _parallel_debug_config(
        enabled=enabled,
        trace_enabled=trace_enabled,
        trace_path=trace_path,
        assert_ownership=assert_ownership,
        allow_unsafe_parallel_optuna=allow_unsafe_parallel_optuna,
        allow_unsafe_persistent_runtime=allow_unsafe_persistent_runtime,
    )
    if trace_enabled and trace_path:
        Path(trace_path).parent.mkdir(parents=True, exist_ok=True)
        Path(trace_path).write_text("", encoding="utf-8")
    return debug_cfg


def _parallel_trace_write(debug_cfg: dict[str, object] | None, event: str, **payload) -> None:
    if not debug_cfg or not bool(debug_cfg.get("trace_enabled")):
        return
    trace_path = str(debug_cfg.get("trace_path") or "").strip()
    if not trace_path:
        return
    record = {
        "ts": time.time(),
        "pid": os.getpid(),
        "thread_id": threading.get_ident(),
        "thread_name": threading.current_thread().name,
        "event": event,
        **payload,
    }
    line = json.dumps(_json_normalize(record), ensure_ascii=False, sort_keys=True)
    with _PARALLEL_TRACE_LOCK:
        with Path(trace_path).open("a", encoding="utf-8") as handle:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            except Exception:
                pass
            handle.write(line + "\n")
            handle.flush()
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass


@contextmanager
def _runtime_claim(
    obj,
    *,
    kind: str,
    label: dict[str, object],
    debug_cfg: dict[str, object] | None,
):
    if obj is None:
        yield
        return
    enabled = bool(debug_cfg and debug_cfg.get("enabled") and debug_cfg.get("assert_ownership"))
    if not enabled:
        yield
        return
    obj_id = id(obj)
    owner_error: RuntimeError | None = None
    with _RUNTIME_OWNER_LOCK:
        state = _RUNTIME_OWNERS.get(obj_id)
        if state is None:
            _RUNTIME_OWNERS[obj_id] = {
                "kind": kind,
                "thread_id": threading.get_ident(),
                "label": dict(label),
                "depth": 1,
            }
            _parallel_trace_write(debug_cfg, "runtime_claim", kind=kind, obj_id=obj_id, label=label)
        elif int(state["thread_id"]) == threading.get_ident():
            state["depth"] = int(state.get("depth", 0)) + 1
            _parallel_trace_write(debug_cfg, "runtime_reenter", kind=kind, obj_id=obj_id, label=label)
        else:
            owner_error = RuntimeError(
                "Нарушение владения runtime-объектом: "
                f"{kind} obj_id={obj_id} уже занят thread={state['thread_id']} "
                f"label={state['label']}, новый label={label}"
            )
            _parallel_trace_write(
                debug_cfg,
                "runtime_ownership_violation",
                kind=kind,
                obj_id=obj_id,
                label=label,
                owner=state,
            )
    if owner_error is not None:
        raise owner_error
    try:
        yield
    finally:
        with _RUNTIME_OWNER_LOCK:
            state = _RUNTIME_OWNERS.get(obj_id)
            if state is not None:
                depth = int(state.get("depth", 1)) - 1
                if depth <= 0:
                    _RUNTIME_OWNERS.pop(obj_id, None)
                    _parallel_trace_write(debug_cfg, "runtime_release", kind=kind, obj_id=obj_id, label=label)
                else:
                    state["depth"] = depth


def _parallel_debug_label(
    *,
    policy_name: str | None,
    trial_number: int | None,
    stage_label: str | None,
    scene_id: str | None,
    scene_seed: int | None,
) -> dict[str, object]:
    return {
        "policy": policy_name,
        "trial": trial_number,
        "stage": stage_label,
        "scene": scene_id,
        "seed": scene_seed,
    }


WORKER_THREAD_ENV_VARS = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def _apply_worker_env_vars() -> None:
    for key, value in WORKER_THREAD_ENV_VARS.items():
        os.environ[key] = value


@contextmanager
def _worker_env_scope() -> None:
    previous = {key: os.environ.get(key) for key in WORKER_THREAD_ENV_VARS}
    try:
        _apply_worker_env_vars()
        yield
    finally:
        for key, old_value in previous.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def _warmup_worker_numba() -> None:
    try:
        from common.physics.model import apply_accel_dynamics_step

        apply_accel_dynamics_step(
            0.0,
            0.0,
            0.0,
            0.0,
            0.1,
            1.0,
            1.0,
            0.05,
            8.0,
        )
    except Exception:
        return


def _set_parent_death_signal(sig: int = signal.SIGTERM) -> bool:
    if os.name != "posix":
        return False
    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        prctl = libc.prctl
        prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
        prctl.restype = ctypes.c_int
        PR_SET_PDEATHSIG = 1
        rc = prctl(PR_SET_PDEATHSIG, int(sig), 0, 0, 0)
        if rc != 0:
            return False
        return True
    except Exception:
        return False


def _optuna_process_worker_initializer(expected_ppid: int) -> None:
    _apply_worker_env_vars()
    _set_parent_death_signal(signal.SIGTERM)
    # Если родитель умер до установки PDEATHSIG, worker не должен продолжать жить сиротой.
    if int(os.getppid()) != int(expected_ppid):
        raise SystemExit(0)
    _warmup_worker_numba()


def _terminate_spawn_children(*, sig: int = signal.SIGTERM, wait_s: float = 2.0) -> None:
    if psutil is None:
        return
    try:
        current = psutil.Process(os.getpid())
        children = current.children(recursive=True)
    except Exception:
        return
    tracked = []
    for child in children:
        try:
            cmdline = " ".join(child.cmdline())
        except Exception:
            cmdline = ""
        if "from multiprocessing.spawn import spawn_main" not in cmdline:
            continue
        tracked.append(child)
    if not tracked:
        return
    for child in tracked:
        try:
            child.send_signal(sig)
        except Exception:
            pass
    gone, alive = psutil.wait_procs(tracked, timeout=wait_s)
    del gone
    for child in alive:
        try:
            child.kill()
        except Exception:
            pass


def _stable_hash(text: str) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "little")


def resolve_scene_paths(scene_args: list[str] | None) -> list[Path]:
    return _shared_resolve_scene_paths(scene_args)


def _trial_to_row(policy_name: str, trial: TrialRecord, param_keys: list[str], extra_keys: list[str]) -> dict:
    row: dict[str, object] = {"policy": policy_name}
    params = trial.get("params", {}) or {}
    for k in param_keys:
        if k in params:
            row[k] = params.get(k)
    agg = trial.get("aggregate", {}) or {}
    for key in TUNING_METRIC_KEYS:
        row[key] = agg.get(key)
    for key in extra_keys:
        if key in trial:
            row[key] = trial.get(key)
    return row


def _trial_value(trial: TrialRecord) -> float:
    value = trial.get("value")
    if value is None:
        value = trial.get("score_scalar")
    if value is None:
        return -1.0e18
    try:
        return float(value)
    except Exception:
        return -1.0e18


def _load_policy_checkpoint(out_dir: Path, policy_name: str) -> dict:
    path = out_dir / f"tune_{policy_name.replace(':', '_')}.partial.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    return data


def _resolve_optuna_storage_path(
    out_dir: Path,
    policy_name: str,
    raw_path: str,
    *,
    multi_policy: bool,
    regime: str,
) -> Path:
    if not raw_path:
        return out_dir / f"optuna_{regime}_{policy_name.replace(':', '_')}.db"
    if "{policy}" in raw_path:
        formatted = raw_path.format(policy=policy_name.replace(":", "_"), regime=regime)
        return resolve_repo_path(formatted)
    path = resolve_repo_path(raw_path)
    if path.exists() and path.is_dir():
        return path / f"optuna_{regime}_{policy_name.replace(':', '_')}.db"
    if path.suffix.lower() == ".db":
        stem = path.stem
        if not stem.endswith(f"_{regime}"):
            stem = f"{stem}_{regime}"
        if multi_policy:
            return path.with_name(f"{stem}_{policy_name.replace(':', '_')}{path.suffix}")
        return path.with_name(f"{stem}{path.suffix}")
    return path / f"optuna_{regime}_{policy_name.replace(':', '_')}.db"


def _with_regime_suffix(name: str, regime: str) -> str:
    if name.endswith(f"_{regime}"):
        return name
    return f"{name}_{regime}"


def _resolve_study_name(args, policy_name: str, *, storage_url: str, resume_mode: bool, regime: str):
    if args.study_name:
        if "{policy}" in args.study_name:
            return args.study_name.format(policy=policy_name.replace(":", "_"), regime=regime)
        if args.policy and len(args.policy) > 1 and args.policy != ["all"]:
            return _with_regime_suffix(f"{args.study_name}_{policy_name.replace(':', '_')}", regime)
        return _with_regime_suffix(args.study_name, regime)
    if resume_mode:
        try:
            import optuna

            summaries = optuna.study.get_all_study_summaries(storage=storage_url)
        except Exception:
            summaries = []
        if len(summaries) == 1:
            return summaries[0].study_name
        if len(summaries) > 1:
            raise SystemExit("В storage найдено несколько studies. Укажи --study-name.")
        return _with_regime_suffix(policy_name.replace(":", "_"), regime)
    return _with_regime_suffix(f"{policy_name.replace(':', '_')}_{int(time.time())}", regime)


def _open_trial_csv(csv_path: Path, fieldnames: list[str], *, append: bool = False):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    mode = "a" if append else "w"
    f = csv_path.open(mode, encoding="utf-8", newline="")
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if not append or not file_exists or csv_path.stat().st_size == 0:
        writer.writeheader()
        f.flush()
    return f, writer


def _write_trial_row(
    writer: csv.DictWriter,
    file_handle,
    *,
    policy_name: str,
    trial: TrialRecord,
    param_keys: list[str],
    extra_keys: list[str],
) -> None:
    row = _trial_to_row(policy_name, trial, param_keys, extra_keys)
    writer.writerow(row)
    file_handle.flush()


def _write_policy_checkpoint(
    out_dir: Path,
    policy_name: str,
    args,
    trials: list[TrialRecord],
    best: TrialRecord | None,
    *,
    stageb_trials: list[StageBRecord] | None = None,
    best_stageb: StageBRecord | None = None,
    champions: dict[str, dict] | None = None,
    pareto_frontier: list[dict] | None = None,
    last_trial_number: int | None = None,
    seen_trial_numbers: list[int] | set[int] | None = None,
) -> None:
    payload = {
        "policy": policy_name,
        "config": vars(args),
        "best": best,
        "trials": trials,
    }
    if last_trial_number is not None:
        payload["last_trial_number"] = int(last_trial_number)
    if seen_trial_numbers is not None:
        payload["seen_trial_numbers"] = sorted(int(num) for num in seen_trial_numbers)
    if stageb_trials is not None:
        payload["stageB_trials"] = stageb_trials
    if best_stageb is not None:
        payload["best_stageB"] = best_stageb
    if champions is not None:
        payload["champions"] = champions
    if pareto_frontier is not None:
        payload["pareto_frontier"] = pareto_frontier
    out_path = out_dir / f"tune_{policy_name.replace(':', '_')}.partial.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _trial_from_optuna(
    trial,
    finish_min: float,
    alive_min: float,
    *,
    save_scenes: bool,
) -> TrialRecord:
    agg = trial.user_attrs.get("aggregate") or trial.user_attrs.get("aggregate_partial", {})
    score = trial.user_attrs.get("score") or trial.user_attrs.get("score_partial", None)
    score_scalar_val = trial.user_attrs.get("score_scalar")
    if score_scalar_val is None and agg:
        score_scalar_val = score_scalar(agg, finish_min, alive_min)
    payload: TrialRecord = {
        "params": dict(trial.params),
        "aggregate": agg,
        "score": score,
        "score_scalar": score_scalar_val,
        "value": trial.value,
        "prune_value": trial.user_attrs.get("prune_value"),
        "state": trial.state.name,
        "reported_step": trial.user_attrs.get("reported_step"),
        "number": trial.number,
        "search_stages": trial.user_attrs.get("search_stages"),
    }
    if save_scenes:
        payload["scenes"] = trial.user_attrs.get("scenes", None) or trial.user_attrs.get("scenes_partial", None)
    return payload


def _make_optuna_pruner(optuna, pruner_name: str, *, min_prune_scenes: int):
    if pruner_name == "none":
        return optuna.pruners.NopPruner()
    if pruner_name == "asha":
        return optuna.pruners.SuccessiveHalvingPruner(min_resource=max(1, int(min_prune_scenes)))
    return optuna.pruners.MedianPruner(n_warmup_steps=max(1, int(min_prune_scenes)))


def _make_optuna_storage(optuna, storage_url: str):
    return optuna.storages.RDBStorage(
        storage_url,
        engine_kwargs={
            "connect_args": {
                "timeout": 60,
            }
        },
    )


def _make_optuna_sampler(optuna, seed: int, *, parallel: bool):
    kwargs = {"seed": int(seed)}
    if parallel:
        kwargs["constant_liar"] = True
    return optuna.samplers.TPESampler(**kwargs)


def _append_study_trials(
    *,
    study,
    finish_min: float,
    alive_min: float,
    save_scenes: bool,
    trials: list[TrialRecord],
    best: TrialRecord | None,
    last_trial_number: int,
    csv_writer,
    csv_file,
    policy_name: str,
    param_keys: list[str],
    extra_keys: list[str],
    out_dir: Path,
    args,
    seen_trial_numbers: set[int] | None = None,
) -> tuple[TrialRecord | None, int, int]:
    appended = 0
    seen_numbers = seen_trial_numbers if seen_trial_numbers is not None else set()
    try:
        storage = getattr(study, "_storage", None)
        study_id = getattr(study, "_study_id", None)
        if storage is not None and study_id is not None and hasattr(storage, "read_trials_from_remote_storage"):
            storage.read_trials_from_remote_storage(study_id)
    except Exception:
        pass
    ordered = sorted(study.trials, key=lambda item: int(item.number))
    for trial in ordered:
        if int(trial.number) in seen_numbers:
            continue
        if trial.state.name in {"RUNNING", "WAITING"}:
            continue
        trial_row = _trial_from_optuna(
            trial,
            finish_min,
            alive_min,
            save_scenes=save_scenes,
        )
        trials.append(trial_row)
        if trial.state.name == "COMPLETE":
            if best is None or _trial_value(trial_row) > _trial_value(best):
                best = trial_row
        seen_numbers.add(int(trial.number))
        last_trial_number = max(int(last_trial_number), int(trial.number))
        _write_trial_row(
            csv_writer,
            csv_file,
            policy_name=policy_name,
            trial=trial_row,
            param_keys=param_keys,
            extra_keys=extra_keys,
        )
        appended += 1
    _write_policy_checkpoint(
        out_dir,
        policy_name,
        args,
        trials,
        best,
        last_trial_number=last_trial_number,
        seen_trial_numbers=seen_numbers,
    )
    return best, last_trial_number, appended


def _optuna_process_worker(payload: dict) -> dict:
    with _worker_env_scope():
        _warmup_worker_numba()
        import optuna

        storage = _make_optuna_storage(optuna, payload["storage_url"])
        sampler = _make_optuna_sampler(
            optuna,
            int(payload["seed"]) + int(payload["worker_id"]) * 100_003,
            parallel=True,
        )
        pruner = _make_optuna_pruner(
            optuna,
            payload["pruner"],
            min_prune_scenes=int(payload["min_prune_scenes"]),
        )
        study = optuna.load_study(
            study_name=payload["study_name"],
            storage=storage,
            sampler=sampler,
            pruner=pruner,
        )

        def objective(trial):
            params = _suggest_params(trial, payload["space"])
            scoring_weights = ScoreWeights(**payload["scoring"])

            def _prune_score_local(aggregate, finish_min, alive_min):
                del finish_min, alive_min
                return score_scalar_soft(aggregate, weights=scoring_weights)

            trial_record = _evaluate_candidate_progressive(
                policy_name=payload["policy_name"],
                params=params,
                stages=payload["search_stages"],
                success_threshold=float(payload["success_threshold"]),
                finish_min=float(payload["finish_min"]),
                alive_min=float(payload["alive_min"]),
                save_scenes=bool(payload["save_scenes"]),
                lite_metrics=bool(payload["lite_metrics"]),
                env_payload=dict(payload["env_payload"]),
                goal_radius=float(payload["goal_radius"]),
                aggregate_by_family=bool(payload["aggregate_by_family"]),
                base_seed=int(payload["seed"]),
                trial=trial,
                prune_score_fn=_prune_score_local,
                eval_cache=payload["eval_cache"],
                persistent_runtime=bool(payload["persistent_runtime"]),
                policy_cache_size=int(payload["policy_cache_size"]),
                information_regime=str(payload["information_regime"]),
                parallel_debug=payload["parallel_debug"],
            )
            trial.set_user_attr("aggregate", trial_record.get("aggregate"))
            trial.set_user_attr("score", list(trial_record.get("score", ())))
            trial.set_user_attr("score_scalar", trial_record.get("score_scalar"))
            if bool(payload["save_scenes"]):
                trial.set_user_attr("scenes", trial_record.get("scenes"))
            return float(trial_record.get("value", -1.0e18))

        study.optimize(objective, n_trials=int(payload["worker_trials"]), n_jobs=1)
        return {
            "worker_id": int(payload["worker_id"]),
            "worker_trials": int(payload["worker_trials"]),
            "eval_cache_stats": dict((payload.get("eval_cache") or {}).get("stats", {})),
        }


OPTUNA_SPACES = {
    "baseline:astar_grid": {
        "alpha": {"min": 1.5, "max": 5.0},
        "max_cost": {"min": 0.20, "max": 0.50},
        "goal_radius": {"min": 3, "max": 9, "type": "int"},
        "goal_radius_control": {"min": 2.0, "max": 6.0},
        "near_goal_speed_cap": {"min": 0.15, "max": 0.6},
        "near_goal_damping": {"min": 0.0, "max": 1.0},
        "near_goal_kp": {"min": 0.5, "max": 1.5},
        "safe_max_cost": {"min": 0.02, "max": 0.12},
        "safe_alpha_mult": {"min": 0.8, "max": 1.5},
        "near_goal_frac": {"min": 0.1, "max": 0.4},
        "near_goal_alpha": {"min": 0.6, "max": 1.0},
        "near_goal_max_cost": {"min": 0.25, "max": 0.9},
        "stuck_steps": {"min": 0, "max": 40, "type": "int"},
        "stuck_dist_eps": {"min": 0.2, "max": 1.5},
        "stuck_alpha": {"min": 0.3, "max": 1.0},
        "stuck_max_cost": {"min": 0.4, "max": 1.0},
        "escape_steps": {"min": 0, "max": 6, "type": "int"},
        "stop_risk_threshold": {"min": 0.0, "max": 0.1},
        "risk_speed_scale": {"min": 0.0, "max": 1.0},
        "risk_speed_floor": {"min": 0.0, "max": 0.4},
        "allow_diagonal": {"values": [False, True]},
        "safety_first": {"values": [True, False]},
    },
    "baseline:mpc_lite": {
        "accel_levels": {"values": [[1.0], [0.5, 1.0], [0.35, 0.7, 1.0]]},
        "horizon": {"min": 1, "max": 3, "type": "int"},
        "n_directions": {"min": 8, "max": 20, "type": "int"},
        "w_progress": {"min": 3.0, "max": 8.0},
        "w_risk": {"min": 3.0, "max": 10.0},
        "w_wall": {"min": 2.0, "max": 8.0},
        "w_collision": {"min": 1.0, "max": 6.0},
        "risk_hard_penalty": {"min": 2.0, "max": 10.0},
        "safety_first": {"values": [True, False]},
        "safety_margin": {"min": 1.0, "max": 6.0},
        "safety_min_score": {"min": -1.5, "max": 1.0},
        "idle_penalty": {"min": 0.0, "max": 0.6},
        "fallback_score": {"min": -1.0, "max": 0.5},
        "stuck_steps": {"min": 0, "max": 40, "type": "int"},
        "stuck_dist_eps": {"min": 0.2, "max": 1.5},
    },
    "baseline:separation_steering": {
        "w_goal": {"min": 0.5, "max": 2.0},
        "w_avoid": {"min": 0.5, "max": 3.0},
        "w_sep": {"min": 0.5, "max": 3.0},
    },
    "baseline:greedy_safe": {
        "w_goal": {"min": 0.5, "max": 2.0},
        "w_avoid": {"min": 0.0, "max": 1.5},
        "w_wall": {"min": 0.0, "max": 1.0},
        "grid_radius": {"min": 1, "max": 3, "type": "int"},
    },
    "baseline:potential_fields": {
        "k_att": {"min": 0.5, "max": 3.0},
        "k_rep": {"min": 10.0, "max": 80.0},
        "safety_margin": {"min": 1.0, "max": 10.0},
    },
    "baseline:flow_field": {
        "speed_gain": {"min": 0.5, "max": 1.0},
        "separation_gain": {"min": 0.0, "max": 0.8},
        "separation_radius": {"min": 1.5, "max": 5.0},
        "separation_power": {"min": 0.8, "max": 2.5},
    },
    "random": {},
    "zero": {},
    "greedy": {},
    "wall": {},
    "brake": {},
}

_aggregate_scenes = aggregate_scene_metrics
_score_tuple = score_tuple
_score_scalar = score_scalar
_score_scalar_soft = score_scalar_soft


def _resolve_env_overrides(args, cfg_raw):
    raw_env = None
    if cfg_raw is not None:
        try:
            raw_env = cfg_raw.get("env")
        except Exception:
            raw_env = None
    if raw_env is None:
        raw_env = getattr(args, "env", None)

    return split_env_runtime_fields(raw_env, default_max_steps=600, default_goal_radius=3.0)


def _make_env(env_payload: dict, *, max_steps: int, goal_radius: float, lite_metrics: bool) -> SwarmPZEnv:
    cfg = EnvConfig.from_dict(env_payload)
    return _common_make_pz_env(
        max_steps=max_steps,
        goal_radius=goal_radius,
        config=cfg,
        lite_metrics=bool(lite_metrics),
        reset=False,
    )


def _persistent_params_hash(params: dict) -> str:
    payload = json.dumps(_json_normalize(params), sort_keys=True)
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=8).hexdigest()


def _persistent_env_key(env_payload: dict, *, max_steps: int, goal_radius: float, lite_metrics: bool) -> str:
    payload = {
        "env": _json_normalize(env_payload),
        "max_steps": int(max_steps),
        "goal_radius": float(goal_radius),
        "lite_metrics": bool(lite_metrics),
    }
    return hashlib.blake2b(json.dumps(payload, sort_keys=True).encode("utf-8"), digest_size=12).hexdigest()


def _get_persistent_env(
    env_payload: dict,
    *,
    max_steps: int,
    goal_radius: float,
    lite_metrics: bool,
    persistent_runtime: bool,
):
    if not persistent_runtime:
        return _make_env(env_payload, max_steps=max_steps, goal_radius=goal_radius, lite_metrics=lite_metrics)
    key = _persistent_env_key(
        env_payload,
        max_steps=max_steps,
        goal_radius=goal_radius,
        lite_metrics=lite_metrics,
    )
    env = _PERSISTENT_ENV_CACHE.get(key)
    if env is None:
        env = _make_env(env_payload, max_steps=max_steps, goal_radius=goal_radius, lite_metrics=lite_metrics)
        _PERSISTENT_ENV_CACHE[key] = env
    return env


def _get_persistent_policy(
    policy_name: str,
    params: dict,
    *,
    env: SwarmPZEnv,
    seed: int,
    persistent_runtime: bool,
    policy_cache_size: int,
):
    if not persistent_runtime:
        return create_policy(policy_name, params, env, seed)
    env_key = _persistent_env_key(
        {
            "field_size": env.config.field_size,
            "n_agents": env.config.n_agents,
            "grid_width": env.config.grid_width,
            "grid_res": env.config.grid_res,
            "oracle_visibility": getattr(env.config, "oracle_visibility", None),
        },
        max_steps=int(env.max_steps),
        goal_radius=float(env.goal_radius),
        lite_metrics=bool(getattr(env.config, "debug_metrics_mode", "full") == "lite"),
    )
    key = f"{policy_name}:{_persistent_params_hash(params)}:{env_key}:{int(seed)}"
    policy = _PERSISTENT_POLICY_CACHE.get(key)
    if policy is not None:
        _PERSISTENT_POLICY_CACHE.move_to_end(key)
        return policy
    policy = create_policy(policy_name, params, env, seed)
    _PERSISTENT_POLICY_CACHE[key] = policy
    while len(_PERSISTENT_POLICY_CACHE) > max(1, int(policy_cache_size)):
        _PERSISTENT_POLICY_CACHE.popitem(last=False)
    return policy


def _validate_information_regime(args, env_cfg: EnvConfig) -> str:
    regime = str(getattr(args, "information_regime", "fair") or "fair").strip().lower()
    if regime not in {"fair", "privileged"}:
        raise SystemExit("information_regime must be one of: fair, privileged")
    if regime == "fair":
        violations = []
        visibility = str(getattr(env_cfg, "oracle_visibility", "") or "").strip().lower()
        if visibility not in {"", "none"}:
            violations.append(f"env.oracle_visibility={visibility}")
        if bool(getattr(env_cfg, "oracle_visible_to_baselines", False)):
            violations.append("env.oracle_visible_to_baselines=true")
        if bool(getattr(env_cfg, "oracle_visible_to_agents", False)):
            violations.append("env.oracle_visible_to_agents=true")
        if violations:
            joined = ", ".join(violations)
            raise SystemExit(f"Fair tuning forbids privileged oracle access: {joined}")
    return regime


def _validate_policy_params_for_regime(policy_name: str, params: dict, regime: str) -> None:
    if regime != "fair" or not params:
        return
    violations = []
    for key, value in params.items():
        key_norm = str(key).strip().lower()
        if not any(token in key_norm for token in ("oracle", "privileged", "full_map", "global_map", "map_aware")):
            continue
        is_enabled = False
        if isinstance(value, bool):
            is_enabled = bool(value)
        elif isinstance(value, (int, float)):
            is_enabled = float(value) > 0.0
        elif value is not None:
            text = str(value).strip().lower()
            is_enabled = text not in {"", "0", "false", "none", "off"}
        if is_enabled:
            violations.append(f"{policy_name}.{key}={value}")
    if violations:
        raise SystemExit(
            "Fair tuning forbids policy-level privileged params: " + ", ".join(sorted(violations))
        )


_scene_id = tuning_protocols._scene_id
infer_scene_family = tuning_protocols.infer_scene_family
_aggregate_by_family = tuning_protocols._aggregate_by_family
_scene_list_hash = tuning_protocols._scene_list_hash
_seed_pack_hash = tuning_protocols._seed_pack_hash
_resolve_seed_pack = tuning_protocols._resolve_seed_pack
_scene_family_map = tuning_protocols._scene_family_map
_family_breakdown = tuning_protocols._family_breakdown
_aggregate_with_protocol = tuning_protocols._aggregate_with_protocol
_risk_metric = tuning_protocols._risk_metric
_passes_gates = tuning_protocols._passes_gates
_champion_reason = tuning_protocols._champion_reason
_sample_from_spec = tuning_protocols._sample_from_spec


def _resolve_search_stages(
    args,
    *,
    search_scenes: list[dict],
    eval_seeds: list[int],
    max_steps: int,
    scene_families: dict[str, str],
    policy_name: str | None = None,
) -> list[dict]:
    return tuning_protocols._resolve_search_stages(
        args,
        search_scenes=search_scenes,
        eval_seeds=eval_seeds,
        max_steps=max_steps,
        scene_families=scene_families,
        default_tuning_profiles=DEFAULT_TUNING_PROFILES,
        policy_tuning_profile_overrides=POLICY_TUNING_PROFILE_OVERRIDES,
        policy_name=policy_name,
    )


def _pareto_frontier(rows: list[dict], *, args) -> list[dict]:
    return tuning_protocols._pareto_frontier(rows, args=args, trial_value_fn=_trial_value)


def _select_champions(rows: list[dict], *, args) -> dict[str, dict]:
    return tuning_protocols._select_champions(rows, args=args, trial_value_fn=_trial_value)


def _build_tuning_protocol(args, explicit_scene_paths: list[Path], explicit_scenes: list[dict]) -> dict:
    return tuning_protocols._build_tuning_protocol(
        args,
        explicit_scene_paths,
        explicit_scenes,
        resolve_scene_paths_fn=resolve_scene_paths,
        load_scenes_fn=load_scenes,
    )


def build_candidates(space: dict, method: str, n_samples: int, max_trials: int | None, seed: int):
    if not space:
        return [{}]

    if method == "grid":
        keys = []
        values = []
        for k, v in space.items():
            if isinstance(v, dict) and "values" in v:
                v = v["values"]
            if not isinstance(v, (list, tuple)):
                raise ValueError(f"Grid requires list values for {k}")
            keys.append(k)
            values.append(list(v))
        combos = [dict(zip(keys, vals)) for vals in itertools.product(*values)]
        if max_trials is not None and len(combos) > max_trials:
            rng = random.Random(seed)
            rng.shuffle(combos)
            combos = combos[:max_trials]
        return combos

    if method == "random":
        rng = random.Random(seed)
        out = []
        for _ in range(n_samples):
            params = {}
            for k, v in space.items():
                if isinstance(v, (list, tuple)):
                    params[k] = rng.choice(list(v))
                elif isinstance(v, dict):
                    params[k] = _sample_from_spec(v, rng)
                else:
                    raise ValueError(f"Bad space value for {k}: {v}")
            out.append(params)
        return out

    raise ValueError("method must be grid or random")


def _suggest_params(trial, space: dict) -> dict:
    params = {}
    for k, v in space.items():
        if isinstance(v, (list, tuple)):
            params[k] = trial.suggest_categorical(k, list(v))
            continue
        if isinstance(v, dict):
            if "values" in v:
                params[k] = trial.suggest_categorical(k, list(v["values"]))
                continue
            lo = v.get("min")
            hi = v.get("max")
            if lo is None or hi is None:
                raise ValueError(f"Bad range spec for {k}: {v}")
            typ = v.get("type", "float")
            log = bool(v.get("log", False))
            step = v.get("step", None)
            if typ == "int":
                if log:
                    params[k] = trial.suggest_int(k, int(lo), int(hi), log=True)
                elif step is not None:
                    params[k] = trial.suggest_int(k, int(lo), int(hi), step=int(step))
                else:
                    params[k] = trial.suggest_int(k, int(lo), int(hi))
            else:
                if step is not None:
                    params[k] = trial.suggest_float(k, float(lo), float(hi), step=float(step), log=log)
                else:
                    params[k] = trial.suggest_float(k, float(lo), float(hi), log=log)
            continue
        raise ValueError(f"Bad space value for {k}: {v}")
    return params


def create_policy(policy_name: str, params: dict, env: SwarmPZEnv, seed: int):
    return create_tuning_policy(policy_name, params, env, seed)


def evaluate_policy(
    env: SwarmPZEnv,
    policy,
    scenes: list,
    eval_seeds: list[int],
    success_threshold: float,
    finish_min: float,
    alive_min: float,
    trial=None,
    min_prune_scenes: int = 0,
    prune_score_fn=None,
    aggregate_fn=None,
    eval_cache: dict | None = None,
    cache_policy_name: str | None = None,
    cache_params: dict | None = None,
    parallel_debug: dict[str, object] | None = None,
    trial_number: int | None = None,
    stage_label: str | None = None,
):
    per_scene: dict[str, AggregateMetrics] = {}
    for idx, scene in enumerate(scenes):
        rows = []
        scene_id = scene.get("id", f"scene_{idx}")
        scene_offset = _stable_hash(str(scene_id)) % 100000
        for eval_seed in eval_seeds:
            scene_seed = int(eval_seed + scene_offset)
            cache_key = None
            cached_metrics = None
            if cache_policy_name is not None and cache_params is not None:
                cache_key = _episode_cache_key(
                    policy_name=cache_policy_name,
                    params=cache_params,
                    scene=scene,
                    scene_seed=scene_seed,
                    env_hash=(eval_cache or {}).get("env_hash", ""),
                    episode_max_steps=int(getattr(env, "max_steps", 0)),
                )
                cached_metrics = _eval_cache_read(eval_cache, key=cache_key)
            if cached_metrics is not None:
                _parallel_trace_write(
                    parallel_debug,
                    "eval_cache_hit",
                    label=_parallel_debug_label(
                        policy_name=cache_policy_name,
                        trial_number=trial_number,
                        stage_label=stage_label,
                        scene_id=scene_id,
                        scene_seed=scene_seed,
                    ),
                    cache_key=cache_key,
                )
                rows.append(cached_metrics)
                continue
            label = _parallel_debug_label(
                policy_name=cache_policy_name,
                trial_number=trial_number,
                stage_label=stage_label,
                scene_id=scene_id,
                scene_seed=scene_seed,
            )
            with _runtime_claim(env, kind="env", label=label, debug_cfg=parallel_debug):
                with _runtime_claim(policy, kind="policy", label=label, debug_cfg=parallel_debug):
                    _parallel_trace_write(parallel_debug, "policy_reset", label=label)
                    reset_policy_context(env, policy, scene_seed, policy_name=cache_policy_name)
                    _parallel_trace_write(parallel_debug, "policy_context", label=label)
                    _parallel_trace_write(parallel_debug, "run_episode_start", label=label)
                    metrics = run_episode(env, policy, scene, scene_seed, success_threshold)
                    _parallel_trace_write(parallel_debug, "run_episode_end", label=label)
            rows.append(metrics)
            if cache_key is not None:
                _eval_cache_write(eval_cache, key=cache_key, metrics=metrics)
        per_scene[scene_id] = aggregate_episode_metrics(rows)
        if trial is not None:
            agg = aggregate_fn(per_scene) if aggregate_fn is not None else _aggregate_scenes(per_scene)
            score = _score_tuple(agg, finish_min, alive_min)
            value = _score_scalar(agg, finish_min, alive_min)
            if prune_score_fn is None:
                prune_value = _score_scalar_soft(agg, finish_min, alive_min)
            else:
                prune_value = prune_score_fn(agg, finish_min, alive_min)
            trial.report(prune_value, step=idx + 1)
            trial.set_user_attr("aggregate_partial", agg)
            trial.set_user_attr("score_partial", list(score))
            trial.set_user_attr("value_partial", value)
            trial.set_user_attr("prune_value", prune_value)
            trial.set_user_attr("reported_step", idx + 1)
            trial.set_user_attr("scenes_partial", per_scene)
            if (idx + 1) >= int(min_prune_scenes) and trial.should_prune():
                import optuna

                raise optuna.TrialPruned
    return per_scene


def _evaluate_candidate_progressive(
    *,
    policy_name: str,
    params: dict,
    stages: list[dict],
    success_threshold: float,
    finish_min: float,
    alive_min: float,
    save_scenes: bool,
    lite_metrics: bool,
    env_payload: dict,
    goal_radius: float,
    aggregate_by_family: bool,
    base_seed: int,
    trial=None,
    prune_score_fn=None,
    eval_cache: dict | None = None,
    persistent_runtime: bool = False,
    policy_cache_size: int = 8,
    information_regime: str = "fair",
    parallel_debug: dict[str, object] | None = None,
) -> TrialRecord:
    final_result: TrialRecord | None = None
    stage_results: list[dict] = []
    _validate_policy_params_for_regime(policy_name, params, information_regime)
    for stage_idx, stage in enumerate(stages, start=1):
        stage_scene_families = stage["scene_families"]

        def aggregate_fn(data, *, _stage_scene_families=stage_scene_families):
            return _aggregate_with_protocol(
                data,
                scene_families=_stage_scene_families,
                aggregate_by_family=aggregate_by_family,
            )
        env = _get_persistent_env(
            env_payload,
            max_steps=int(stage["max_steps"]),
            goal_radius=goal_radius,
            lite_metrics=lite_metrics,
            persistent_runtime=persistent_runtime,
        )
        policy = _get_persistent_policy(
            policy_name,
            params,
            env=env,
            seed=base_seed,
            persistent_runtime=persistent_runtime,
            policy_cache_size=policy_cache_size,
        )
        per_scene = evaluate_policy(
            env,
            policy,
            stage["scenes"],
            stage["seeds"],
            success_threshold,
            finish_min,
            alive_min,
            aggregate_fn=aggregate_fn,
            eval_cache=eval_cache,
            cache_policy_name=policy_name,
            cache_params=params,
            parallel_debug=parallel_debug,
            trial_number=None if trial is None else int(getattr(trial, "number", -1)),
            stage_label=str(stage["label"]),
        )
        agg = aggregate_fn(per_scene)
        score = _score_tuple(agg, finish_min, alive_min)
        hard_value = _score_scalar(agg, finish_min, alive_min)
        soft_value = _score_scalar_soft(agg, finish_min, alive_min)
        stage_record = {
            "label": stage["label"],
            "scene_ids": list(stage["scene_ids"]),
            "seed_count": len(stage["seeds"]),
            "max_steps": int(stage["max_steps"]),
            "aggregate": agg,
            "score": score,
            "score_scalar": hard_value,
            "value": soft_value,
        }
        stage_results.append(stage_record)
        final_result = {
            "params": params,
            "aggregate": agg,
            "score": score,
            "score_scalar": hard_value,
            "value": soft_value,
            "search_stages": stage_results,
        }
        if save_scenes:
            final_result["scenes"] = per_scene
        if trial is not None:
            prune_value = prune_score_fn(agg, finish_min, alive_min) if prune_score_fn is not None else soft_value
            trial.report(prune_value, step=stage_idx)
            trial.set_user_attr("aggregate_partial", agg)
            trial.set_user_attr("score_partial", list(score))
            trial.set_user_attr("value_partial", hard_value)
            trial.set_user_attr("prune_value", prune_value)
            trial.set_user_attr("reported_step", stage_idx)
            trial.set_user_attr("search_stages", stage_results)
            if save_scenes:
                trial.set_user_attr("scenes_partial", per_scene)
            if trial.should_prune() and stage_idx < len(stages):
                import optuna

                raise optuna.TrialPruned
    if final_result is None:
        final_result = {
            "params": params,
            "aggregate": _aggregate_scenes({}),
            "score": _score_tuple({}, finish_min, alive_min),
            "score_scalar": -1.0e18,
            "value": -1.0e18,
            "search_stages": [],
        }
    return final_result


def _evaluate_candidate_worker(args):
    (
        policy_name,
        params,
        stages,
        success_threshold,
        finish_min,
        alive_min,
        save_scenes,
        lite_metrics,
        env_payload,
        goal_radius,
        aggregate_by_family,
        base_seed,
        eval_cache,
        persistent_runtime,
        policy_cache_size,
        information_regime,
        parallel_debug,
    ) = args
    return _evaluate_candidate_progressive(
        policy_name=policy_name,
        params=params,
        stages=stages,
        success_threshold=success_threshold,
        finish_min=finish_min,
        alive_min=alive_min,
        save_scenes=save_scenes,
        lite_metrics=lite_metrics,
        env_payload=env_payload,
        goal_radius=goal_radius,
        aggregate_by_family=aggregate_by_family,
        base_seed=base_seed,
        eval_cache=eval_cache,
        persistent_runtime=persistent_runtime,
        policy_cache_size=policy_cache_size,
        information_regime=information_regime,
        parallel_debug=parallel_debug,
    )


def _select_topk(trials: list[dict], k: int, finish_min: float, alive_min: float) -> list[dict]:
    if not trials or k <= 0:
        return []
    candidates = [t for t in trials if t.get("state", "COMPLETE") == "COMPLETE"]
    if not candidates:
        candidates = list(trials)

    def key_fn(t: dict) -> float:
        if t.get("value") is not None:
            return float(t["value"])
        if t.get("score_scalar") is not None:
            return float(t["score_scalar"])
        agg = t.get("aggregate") or {}
        return _score_scalar(agg, finish_min, alive_min)

    return sorted(candidates, key=key_fn, reverse=True)[:k]


def _merge_best_params(stageb_best: dict, out_path: Path, meta: dict | None = None) -> None:
    data = {"policies": {}, "meta": {}}
    if out_path.exists():
        try:
            data = json.loads(out_path.read_text(encoding="utf-8"))
        except Exception:
            data = {"policies": {}, "meta": {}}
    policies = data.get("policies") or {}
    for policy_name, row in stageb_best.items():
        if not policy_name:
            continue
        params = row.get("params", {})
        policies[policy_name] = params
    data["policies"] = policies
    if meta:
        data_meta = data.get("meta") or {}
        data_meta.update(meta)
        data["meta"] = data_meta
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _summarize_results(results: dict) -> dict:
    summary = {}
    for policy_name, payload in results.items():
        trials = payload.get("trials", []) or []
        states = {}
        for t in trials:
            st = t.get("state", "UNKNOWN")
            states[st] = states.get(st, 0) + 1
        best = payload.get("best") or {}
        best_stageb = payload.get("best_stageB") or payload.get("best_stageb") or {}
        summary[policy_name] = {
            "n_trials": len(trials),
            "states": states,
            "best_score_scalar": best.get("score_scalar"),
            "best_stageB_score_scalar": best_stageb.get("score_scalar"),
            "validation_status": payload.get("validation_status", "search_only"),
            "promotion_status": payload.get("promotion_status", "not_promoted"),
        }
    return summary


def _fmt_metric(value: float | None, digits: int = 3) -> str:
    return tuning_reporting.fmt_metric(value, digits)


def _aggregate_row(label: str, agg: dict | None, reason: str = "") -> str:
    return tuning_reporting.aggregate_row(label, agg, _risk_metric, reason)


def _write_tuning_report(out_dir: Path, protocol: dict, results: dict) -> Path:
    return tuning_reporting.write_tuning_report(
        out_dir,
        protocol,
        results,
        risk_metric=_risk_metric,
        champion_reason=_champion_reason,
    )


def _json_normalize(value):
    return eval_cache_helpers.json_normalize(value)


def _cache_lineage_files() -> list[Path]:
    return eval_cache_helpers.cache_lineage_files(CACHE_LINEAGE_PATHS)


def _compute_cache_code_hash(cache_version: str) -> tuple[str, list[str]]:
    return eval_cache_helpers.compute_cache_code_hash(cache_version, CACHE_LINEAGE_PATHS)


def _cleanup_eval_cache(base_root: Path, *, active_namespace: str, ttl_days: int) -> dict[str, int]:
    return eval_cache_helpers.cleanup_eval_cache(base_root, active_namespace=active_namespace, ttl_days=ttl_days)


def _episode_cache_env_hash(env_payload: dict, *, regime: str, goal_radius: float, success_threshold: float) -> str:
    return eval_cache_helpers.episode_cache_env_hash(
        env_payload,
        regime=regime,
        goal_radius=goal_radius,
        success_threshold=success_threshold,
    )


def _make_eval_cache(args, *, out_dir: Path, env_payload: dict, regime: str, goal_radius: float):
    return eval_cache_helpers.make_eval_cache(
        args,
        out_dir=out_dir,
        env_payload=env_payload,
        regime=regime,
        goal_radius=goal_radius,
    )


def _episode_cache_key(
    *,
    policy_name: str,
    params: dict,
    scene: dict,
    scene_seed: int,
    env_hash: str,
    episode_max_steps: int,
) -> str:
    return eval_cache_helpers.episode_cache_key(
        policy_name=policy_name,
        params=params,
        scene=scene,
        scene_seed=scene_seed,
        env_hash=env_hash,
        episode_max_steps=episode_max_steps,
    )


def _eval_cache_read(cache: dict | None, *, key: str):
    return eval_cache_helpers.eval_cache_read(cache, key=key)


def _eval_cache_write(cache: dict | None, *, key: str, metrics: dict) -> None:
    eval_cache_helpers.eval_cache_write(cache, key=key, metrics=metrics)


@dataclass
class TuneBaselinesConfig:
    policy: list[str] = field(default_factory=lambda: ["all"])
    tuning_profile: str = "balanced"
    step_budget_name: str = "default"
    search_fidelity: list[dict] = field(default_factory=list)
    policy_search_fidelity: dict = field(default_factory=dict)
    method: str = "grid"
    samples: int = 50
    max_trials: int = 200
    trials: int = 50
    pruner: str = "none"
    n_jobs: int = 1
    episodes: int = 5
    episodes_eval: int = 30
    stage_b: bool = False
    topk: int = 5
    min_prune_scenes: int = 2
    skip_empty: bool = False
    seed: int = 0
    success_threshold: float = 0.5
    finish_min: float = FINISH_MIN_DEFAULT
    alive_min: float = ALIVE_MIN_DEFAULT
    scoring: ScoreWeights = field(default_factory=ScoreWeights)
    aggregate_by_family: bool = True
    gate_finish_min: float = -1.0
    gate_alive_min: float = -1.0
    gate_max_risk: float = -1.0
    scenes: list[str] = field(default_factory=list)
    search_scenes: list[str] = field(default_factory=list)
    holdout_scenes: list[str] = field(default_factory=list)
    benchmark_scenes: list[str] = field(default_factory=list)
    ood_scenes: list[str] = field(default_factory=list)
    space: str = ""
    out_dir: str = ""
    out_root: str = "runs"
    run_id: str = ""
    out: str = ""
    save_scenes: bool = False
    optuna_db: str = ""
    study_name: str = ""
    resume: bool = False
    lite_metrics: bool = True
    eval_cache: bool = True
    cache_dir: str = ""
    cache_version: str = "v1"
    cache_cleanup: bool = True
    cache_ttl_days: int = 14
    persistent_workers: bool = True
    policy_cache_size: int = 8
    parallel_debug: bool = False
    parallel_trace: bool = False
    parallel_trace_path: str = ""
    parallel_assert_ownership: bool = True
    allow_unsafe_parallel_optuna: bool = False
    allow_unsafe_persistent_runtime: bool = False
    parallel_max_workers: int = 0
    parallel_reserve_cpus: int = 2
    parallel_memory_fraction: float = 0.7
    parallel_max_trials_per_worker: int = 0
    env: dict = field(default_factory=dict)
    search_seeds: list[int] = field(default_factory=list)
    holdout_seeds: list[int] = field(default_factory=list)
    report_seeds: list[int] = field(default_factory=list)
    information_regime: str = "fair"


def _apply_lite_metrics_cfg(cfg: EnvConfig, lite_metrics: bool) -> None:
    _common_apply_lite_metrics_cfg(cfg, lite_metrics)


def run(cfg: TuneBaselinesConfig, *, cfg_raw: object | None = None) -> None:
    args = cfg
    global _score_scalar
    global _score_scalar_soft
    validate_config_guardrails(cfg)
    SeedManager(int(args.seed)).seed_all()
    log_numba_status(logger)
    _score_scalar, _score_scalar_soft = make_weighted_scorers(args.scoring)
    policy_list = args.policy
    if len(policy_list) == 1 and policy_list[0] == "all":
        policy_list = list(DEFAULT_SPACES.keys())

    space_override = {}
    if args.space:
        data = yaml.safe_load(resolve_repo_path(args.space).read_text())
        if not isinstance(data, dict):
            raise SystemExit("space config must be a dict")
        space_override = data.get("policies", data)

    scene_paths = resolve_scene_paths(args.scenes)
    scenes = load_scenes(scene_paths)
    protocol = _build_tuning_protocol(args, scene_paths, scenes)
    search_scene_paths = protocol["search_paths"]
    holdout_scene_paths = protocol["holdout_paths"]
    benchmark_scene_paths = protocol["benchmark_paths"]
    ood_scene_paths = protocol["ood_paths"]
    search_scenes = protocol["search_scenes"]
    holdout_scenes = protocol["holdout_scenes"]
    benchmark_scenes = protocol["benchmark_scenes"]
    ood_scenes = protocol["ood_scenes"]
    search_seeds = protocol["search_seeds"]
    holdout_seeds = protocol["holdout_seeds"]
    report_seeds = protocol["report_seeds"]
    search_scene_families = _scene_family_map(search_scenes)
    holdout_scene_families = _scene_family_map(holdout_scenes)
    benchmark_scene_families = _scene_family_map(benchmark_scenes)
    ood_scene_families = _scene_family_map(ood_scenes)
    if not search_scenes:
        raise SystemExit("Tuning protocol produced an empty search split.")
    if args.stage_b and not holdout_scenes:
        raise SystemExit("Stage-B requires a non-empty holdout split. Set holdout_scenes explicitly or provide more scenes.")

    env_payload, max_steps, goal_radius = _resolve_env_overrides(args, cfg_raw)
    env_cfg_audit = EnvConfig.from_dict(env_payload)
    _apply_lite_metrics_cfg(env_cfg_audit, bool(args.lite_metrics))
    regime = _validate_information_regime(args, env_cfg_audit)
    out_dir_arg = args.out_dir or args.out
    out_dir = ensure_run_dir(
        category="tune",
        out_root=args.out_root,
        run_id=args.run_id or None,
        prefix=f"tune_{regime}",
        out_dir=out_dir_arg or None,
    )
    parallel_debug = _make_parallel_debug(args, out_dir=out_dir)
    optuna_parallel_backend = _resolve_optuna_parallel_backend(
        args.method,
        args.n_jobs,
        allow_unsafe_parallel_optuna=bool(
            parallel_debug.get("enabled") and parallel_debug.get("allow_unsafe_parallel_optuna")
        ),
    )
    persistent_runtime = _persistent_runtime_enabled(
        args.method,
        args.n_jobs,
        args.persistent_workers,
        optuna_parallel_backend=optuna_parallel_backend,
        allow_unsafe_persistent_runtime=bool(
            parallel_debug.get("enabled") and parallel_debug.get("allow_unsafe_persistent_runtime")
        ),
    )
    effective_optuna_n_jobs = _effective_optuna_n_jobs(
        args.method,
        args.n_jobs,
        allow_unsafe_parallel_optuna=bool(
            parallel_debug.get("enabled") and parallel_debug.get("allow_unsafe_parallel_optuna")
        ),
    )
    if bool(args.persistent_workers) and not persistent_runtime:
        logger.info(
            "persistent_workers отключён для method=%s n_jobs=%s: shared env/policy cache небезопасен в параллельном Optuna.",
            args.method,
            args.n_jobs,
        )
    if str(args.method).strip().lower() == "optuna" and int(args.n_jobs) > 1:
        if optuna_parallel_backend == "process":
            logger.info(
                "Optuna будет распараллелен через spawn-worker backend: n_jobs=%s, study.optimize внутри worker идёт с n_jobs=1.",
                args.n_jobs,
            )
        elif effective_optuna_n_jobs == 1:
            logger.warning(
                "Внутрипроцессный parallel Optuna отключён: n_jobs=%s -> 1. "
                "Для baseline-heavy search он даёт native-crash на связке Numba/NumPy. "
                "Параллельный тюнинг запускай отдельными процессами по политикам.",
                args.n_jobs,
            )
    if bool(parallel_debug.get("enabled")):
        logger.warning(
            "parallel_debug включён: trace=%s ownership_assert=%s unsafe_optuna=%s unsafe_persistent_runtime=%s backend=%s",
            parallel_debug.get("trace_enabled"),
            parallel_debug.get("assert_ownership"),
            parallel_debug.get("allow_unsafe_parallel_optuna"),
            parallel_debug.get("allow_unsafe_persistent_runtime"),
            optuna_parallel_backend,
        )
        _parallel_trace_write(
            parallel_debug,
            "parallel_debug_enabled",
            requested_n_jobs=int(args.n_jobs),
            effective_optuna_n_jobs=int(effective_optuna_n_jobs),
            persistent_runtime=bool(persistent_runtime),
            method=str(args.method),
            regime=regime,
            optuna_parallel_backend=optuna_parallel_backend,
        )
    protocol["summary"]["information_regime"] = regime
    protocol["summary"]["validation_status"] = "holdout_validated" if args.stage_b else "search_only"
    protocol["summary"]["promotion_status"] = "promotable" if args.stage_b else "not_promoted"
    write_manifest(out_dir, config=vars(args), command=["tune_baselines"])
    eval_cache = _make_eval_cache(
        args,
        out_dir=out_dir,
        env_payload=env_payload,
        regime=regime,
        goal_radius=goal_radius,
    )
    protocol["summary"]["eval_cache"] = {
        "enabled": bool(eval_cache.get("enabled")),
        "base_root": str(eval_cache.get("base_root")),
        "root": str(eval_cache.get("root")),
        "namespace": eval_cache.get("namespace"),
        "cache_version": eval_cache.get("cache_version"),
        "code_hash": eval_cache.get("code_hash"),
        "env_hash": eval_cache.get("env_hash"),
        "stats": dict(eval_cache.get("stats", {})),
        "cleanup": dict(eval_cache.get("cleanup", {})),
        "lineage_files": list(eval_cache.get("lineage_files", [])),
    }
    protocol["summary"]["parallel_debug"] = dict(parallel_debug)
    protocol["summary"]["optuna_parallel_backend"] = optuna_parallel_backend
    (out_dir / "eval_cache_manifest.json").write_text(
        json.dumps(protocol["summary"]["eval_cache"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    eval_seeds = search_seeds
    eval_seeds_eval = holdout_seeds
    default_search_stages = _resolve_search_stages(
        args,
        search_scenes=search_scenes,
        eval_seeds=eval_seeds,
        max_steps=max_steps,
        scene_families=search_scene_families,
    )
    protocol["summary"]["search_stages"] = [
        {
            "label": stage["label"],
            "scene_ids": stage["scene_ids"],
            "seed_count": len(stage["seeds"]),
            "max_steps": stage["max_steps"],
        }
        for stage in default_search_stages
    ]
    protocol["summary"]["search_stages_by_policy"] = {}
    audit_cfg = cfg_raw if cfg_raw is not None else vars(args)
    seed_union = sorted({*search_seeds, *holdout_seeds, *report_seeds})
    all_audit_paths = [*search_scene_paths, *holdout_scene_paths, *benchmark_scene_paths, *ood_scene_paths]
    write_config_audit(
        out_dir,
        cfg=audit_cfg,
        scenes=[str(p) for p in all_audit_paths],
        seeds=seed_union,
        env_cfg=env_cfg_audit,
    )

    def _write_protocol_payload() -> None:
        protocol["summary"]["eval_cache"]["stats"] = dict(eval_cache.get("stats", {}))
        (out_dir / "eval_cache_manifest.json").write_text(
            json.dumps(protocol["summary"]["eval_cache"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (out_dir / "tuning_protocol.json").write_text(
            json.dumps(
                {
                    "summary": protocol["summary"],
                    "search_paths": [str(path) for path in search_scene_paths],
                    "holdout_paths": [str(path) for path in holdout_scene_paths],
                    "benchmark_paths": [str(path) for path in benchmark_scene_paths],
                    "ood_paths": [str(path) for path in ood_scene_paths],
                    "search_seeds": search_seeds,
                    "holdout_seeds": holdout_seeds,
                    "report_seeds": report_seeds,
                    "information_regime": regime,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    _write_protocol_payload()

    results = {}
    stageb_rows = []
    stageb_best = {}
    stageb_writer = None
    stageb_file = None
    stageb_param_keys: list[str] = []
    if args.stage_b:
        for policy_name in policy_list:
            space = space_override.get(policy_name, OPTUNA_SPACES.get(policy_name, DEFAULT_SPACES.get(policy_name, {})))
            if space:
                stageb_param_keys.extend(list(space.keys()))
        stageb_param_keys = sorted(set(stageb_param_keys))
        stageb_csv = out_dir / "topk_stageB.csv"
        stageb_fieldnames = ["policy", "rank", *stageb_param_keys, *TUNING_METRIC_KEYS, "score_scalar", "stageA_value"]
        stageb_file, stageb_writer = _open_trial_csv(stageb_csv, stageb_fieldnames)

    def _search_aggregate(per_scene):
        return _aggregate_with_protocol(
            per_scene,
            scene_families=search_scene_families,
            aggregate_by_family=args.aggregate_by_family,
        )

    def _holdout_aggregate(per_scene):
        return _aggregate_with_protocol(
            per_scene,
            scene_families=holdout_scene_families,
            aggregate_by_family=args.aggregate_by_family,
        )

    def _benchmark_aggregate(per_scene):
        return _aggregate_with_protocol(
            per_scene,
            scene_families=benchmark_scene_families,
            aggregate_by_family=args.aggregate_by_family,
        )

    def _ood_aggregate(per_scene):
        return _aggregate_with_protocol(
            per_scene,
            scene_families=ood_scene_families,
            aggregate_by_family=args.aggregate_by_family,
        )

    def _validate_candidate(
        *,
        policy_name: str,
        champion_name: str,
        params: dict,
        split_name: str,
        split_scenes: list[dict],
        split_seeds: list[int],
        split_aggregate_fn,
        split_family_map: dict[str, str],
    ) -> dict | None:
        if not split_scenes or not split_seeds:
            return None
        _validate_policy_params_for_regime(policy_name, params, regime)
        env = _get_persistent_env(
            env_payload,
            max_steps=max_steps,
            goal_radius=goal_radius,
            lite_metrics=args.lite_metrics,
            persistent_runtime=persistent_runtime,
        )
        policy = _get_persistent_policy(
            policy_name,
            params,
            env=env,
            seed=args.seed,
            persistent_runtime=persistent_runtime,
            policy_cache_size=int(effective_policy_cache_size),
        )
        per_scene = evaluate_policy(
            env,
            policy,
            split_scenes,
            split_seeds,
            args.success_threshold,
            args.finish_min,
            args.alive_min,
            aggregate_fn=split_aggregate_fn,
            eval_cache=eval_cache,
            cache_policy_name=policy_name,
            cache_params=params,
            parallel_debug=parallel_debug,
            stage_label=split_name,
        )
        aggregate = split_aggregate_fn(per_scene)
        per_family = _family_breakdown(per_scene, split_family_map)
        plot_dir = out_dir / "tuning_plots" / policy_name.replace(":", "_") / champion_name / split_name
        plot_paths = write_summary_plots(plot_dir, per_scene, title=f"{policy_name} {champion_name} {split_name}")
        return {
            "aggregate": aggregate,
            "per_family": per_family,
            "plot_paths": {key: str(path) for key, path in plot_paths.items()},
        }
    for policy_name in policy_list:
        effective_policy_cache_size = _resolve_policy_cache_size(args, policy_name)
        if effective_policy_cache_size < int(args.policy_cache_size):
            logger.warning(
                "Снижаю policy_cache_size для %s: %s -> %s.",
                policy_name,
                args.policy_cache_size,
                effective_policy_cache_size,
            )
        search_stages = _resolve_search_stages(
            args,
            search_scenes=search_scenes,
            eval_seeds=eval_seeds,
            max_steps=max_steps,
            scene_families=search_scene_families,
            policy_name=policy_name,
        )
        protocol["summary"]["search_stages_by_policy"][policy_name] = [
            {
                "label": stage["label"],
                "scene_ids": stage["scene_ids"],
                "seed_count": len(stage["seeds"]),
                "max_steps": stage["max_steps"],
            }
            for stage in search_stages
        ]
        csv_path = out_dir / f"tune_{policy_name.replace(':', '_')}.csv"
        csv_file = None
        csv_writer = None
        param_keys: list[str] = []
        extra_keys = list(TRIAL_EXTRA_KEYS)
        last_trial_number = None
        seen_trial_numbers: set[int] = set()
        if args.method == "optuna":
            resume_mode = bool(args.resume or args.optuna_db)
            try:
                import optuna
            except Exception as exc:
                raise SystemExit(f"Optuna is required for --method optuna: {exc}") from exc

            space = space_override.get(policy_name, OPTUNA_SPACES.get(policy_name, DEFAULT_SPACES.get(policy_name, {})))
            if not space:
                space = {}
            param_keys = sorted(space.keys())
            fieldnames = ["policy", *param_keys, *TUNING_METRIC_KEYS, *extra_keys]
            csv_file, csv_writer = _open_trial_csv(
                csv_path,
                fieldnames,
                append=resume_mode and csv_path.exists(),
            )

            if not space:
                if args.skip_empty:
                    trials = []
                    best = None
                else:
                    prev_checkpoint = _load_policy_checkpoint(out_dir, policy_name) if resume_mode else {}
                    trials = prev_checkpoint.get("trials", [])
                    best = prev_checkpoint.get("best")
                    trial = _evaluate_candidate_progressive(
                        policy_name=policy_name,
                        params={},
                        stages=search_stages,
                        success_threshold=args.success_threshold,
                        finish_min=args.finish_min,
                        alive_min=args.alive_min,
                        save_scenes=args.save_scenes,
                        lite_metrics=args.lite_metrics,
                        env_payload=env_payload,
                        goal_radius=goal_radius,
                        aggregate_by_family=args.aggregate_by_family,
                        base_seed=args.seed,
                        eval_cache=eval_cache,
                        persistent_runtime=persistent_runtime,
                        policy_cache_size=int(effective_policy_cache_size),
                        information_regime=regime,
                        parallel_debug=parallel_debug,
                    )
                    trial["state"] = "COMPLETE"
                    trials.append(trial)
                    if best is None or _trial_value(trial) > _trial_value(best):
                        best = trial
                    _write_trial_row(
                        csv_writer,
                        csv_file,
                        policy_name=policy_name,
                        trial=trial,
                        param_keys=param_keys,
                        extra_keys=extra_keys,
                    )
                _write_policy_checkpoint(out_dir, policy_name, args, trials, best)
            else:
                storage_path = _resolve_optuna_storage_path(
                    out_dir,
                    policy_name,
                    args.optuna_db,
                    multi_policy=len(policy_list) > 1,
                    regime=regime,
                )
                storage_url = f"sqlite:///{storage_path}"
                study_name = _resolve_study_name(
                    args,
                    policy_name,
                    storage_url=storage_url,
                    resume_mode=resume_mode,
                    regime=regime,
                )
                pruner = _make_optuna_pruner(
                    optuna,
                    args.pruner,
                    min_prune_scenes=int(args.min_prune_scenes),
                )

                def objective(
                    trial,
                    _space=space,
                    _policy_name=policy_name,
                    _search_stages=search_stages,
                    _effective_policy_cache_size=effective_policy_cache_size,
                ):
                    params = _suggest_params(trial, _space)
                    scoring_weights = args.scoring

                    def _prune_score_local(aggregate, finish_min, alive_min):
                        del finish_min, alive_min
                        return score_scalar_soft(aggregate, weights=scoring_weights)
                    trial_record = _evaluate_candidate_progressive(
                        policy_name=_policy_name,
                        params=params,
                        stages=_search_stages,
                        success_threshold=args.success_threshold,
                        finish_min=args.finish_min,
                        alive_min=args.alive_min,
                        save_scenes=args.save_scenes,
                        lite_metrics=args.lite_metrics,
                        env_payload=env_payload,
                        goal_radius=goal_radius,
                        aggregate_by_family=args.aggregate_by_family,
                        base_seed=args.seed,
                        trial=trial,
                        prune_score_fn=_prune_score_local,
                        eval_cache=eval_cache,
                        persistent_runtime=persistent_runtime,
                        policy_cache_size=int(_effective_policy_cache_size),
                        information_regime=regime,
                        parallel_debug=parallel_debug,
                    )
                    trial.set_user_attr("aggregate", trial_record.get("aggregate"))
                    trial.set_user_attr("score", list(trial_record.get("score", ())))
                    trial.set_user_attr("score_scalar", trial_record.get("score_scalar"))
                    if args.save_scenes:
                        trial.set_user_attr("scenes", trial_record.get("scenes"))
                    return float(trial_record.get("value", -1.0e18))

                trials_total = int(args.trials)
                n_jobs = effective_optuna_n_jobs
                process_worker_jobs = int(args.n_jobs)
                process_worker_meta = None
                process_worker_trial_cap = 1
                process_worker_trial_meta = None
                if optuna_parallel_backend == "process" and int(args.n_jobs) > 1:
                    process_worker_jobs, process_worker_meta = _resolve_process_worker_count(args, policy_name)
                    if process_worker_jobs < int(args.n_jobs):
                        logger.warning(
                            "Снижаю число spawn-worker'ов для %s: %s -> %s "
                            "(cpu_cap=%s, mem_cap=%s, policy_cap=%s, avail_mem=%.2f GiB, est_worker=%.2f GiB).",
                            policy_name,
                            args.n_jobs,
                            process_worker_jobs,
                            process_worker_meta["cpu_cap"],
                            process_worker_meta["mem_cap"],
                            process_worker_meta["policy_cap"],
                            float(process_worker_meta["mem_available_gb"]),
                            float(process_worker_meta["worker_gb"]),
                        )
                    process_worker_trial_cap, process_worker_trial_meta = _resolve_process_worker_trial_cap(
                        args,
                        policy_name,
                    )
                    if process_worker_trial_meta["requested"] > 0 and process_worker_trial_cap < int(
                        process_worker_trial_meta["requested"]
                    ):
                        logger.warning(
                            "Снижаю max_trials_per_worker для %s: %s -> %s "
                            "(policy_default=%s, profile=%s, step_budget=%s).",
                            policy_name,
                            process_worker_trial_meta["requested"],
                            process_worker_trial_cap,
                            process_worker_trial_meta["default_cap"],
                            process_worker_trial_meta["tuning_profile"],
                            process_worker_trial_meta["step_budget"],
                        )
                    protocol["summary"].setdefault("process_workers", {})[policy_name] = dict(process_worker_meta)
                    protocol["summary"].setdefault("process_worker_trial_caps", {})[policy_name] = dict(
                        process_worker_trial_meta
                    )
                prev_checkpoint = _load_policy_checkpoint(out_dir, policy_name) if resume_mode else {}
                trials = prev_checkpoint.get("trials", [])
                best = prev_checkpoint.get("best")
                last_trial_number = int(prev_checkpoint.get("last_trial_number", -1))
                seen_trial_numbers = {
                    int(num)
                    for num in (
                        prev_checkpoint.get("seen_trial_numbers")
                        or [trial.get("number") for trial in trials if trial.get("number") is not None]
                    )
                }
                sampler = _make_optuna_sampler(
                    optuna,
                    args.seed,
                    parallel=optuna_parallel_backend == "process" and int(process_worker_jobs) > 1,
                )
                study = optuna.create_study(
                    study_name=study_name,
                    direction="maximize",
                    sampler=sampler,
                    pruner=pruner,
                    storage=_make_optuna_storage(optuna, storage_url),
                    load_if_exists=True,
                )
                pbar = tqdm(total=trials_total, desc=f"{policy_name} optuna", leave=False) if tqdm else None
                callback_lock = threading.Lock()

                def _callback(
                    study,
                    trial,
                    _pbar=pbar,
                    _callback_lock=callback_lock,
                    _policy_name=policy_name,
                    _param_keys=param_keys,
                    _extra_keys=extra_keys,
                    _csv_writer=csv_writer,
                    _csv_file=csv_file,
                    _trials=trials,
                    _out_dir=out_dir,
                    _args=args,
                    _seen_trial_numbers=seen_trial_numbers,
                ):
                    nonlocal best, last_trial_number
                    with _callback_lock:
                        if _pbar is not None:
                            _pbar.update(1)
                        trial_row = _trial_from_optuna(
                            trial,
                            args.finish_min,
                            args.alive_min,
                            save_scenes=args.save_scenes,
                        )
                        _trials.append(trial_row)
                        _seen_trial_numbers.add(int(trial.number))
                        if trial.state.name == "COMPLETE":
                            if best is None or _trial_value(trial_row) > _trial_value(best):
                                best = trial_row
                        if trial.number > last_trial_number:
                            last_trial_number = trial.number
                        _write_trial_row(
                            _csv_writer,
                            _csv_file,
                            policy_name=_policy_name,
                            trial=trial_row,
                            param_keys=_param_keys,
                            extra_keys=_extra_keys,
                        )
                        _write_policy_checkpoint(
                            _out_dir,
                            _policy_name,
                            _args,
                            _trials,
                            best,
                            last_trial_number=last_trial_number,
                            seen_trial_numbers=_seen_trial_numbers,
                        )

                callbacks = [_callback]
                try:
                    if optuna_parallel_backend == "process" and int(process_worker_jobs) > 1:
                        worker_batches = _build_process_worker_batches(
                            total_trials=trials_total,
                            max_workers=int(process_worker_jobs),
                            trial_cap=int(process_worker_trial_cap),
                        )
                        if len(worker_batches) > 1:
                            logger.info(
                                "Для %s включён recycle worker'ов: %s batch(ей), максимум %s trial(ов) на процесс.",
                                policy_name,
                                len(worker_batches),
                                process_worker_trial_cap,
                            )
                        worker_payload = {
                            "storage_url": storage_url,
                            "study_name": study_name,
                            "policy_name": policy_name,
                            "space": space,
                            "search_stages": search_stages,
                            "success_threshold": args.success_threshold,
                            "finish_min": args.finish_min,
                            "alive_min": args.alive_min,
                            "save_scenes": args.save_scenes,
                            "lite_metrics": args.lite_metrics,
                            "env_payload": env_payload,
                            "goal_radius": goal_radius,
                            "aggregate_by_family": args.aggregate_by_family,
                            "seed": args.seed,
                            "eval_cache": eval_cache,
                            "persistent_runtime": persistent_runtime,
                            "policy_cache_size": int(effective_policy_cache_size),
                            "information_regime": regime,
                            "parallel_debug": parallel_debug,
                            "pruner": args.pruner,
                            "min_prune_scenes": int(args.min_prune_scenes),
                            "scoring": (
                                asdict(args.scoring)
                                if hasattr(args.scoring, "__dataclass_fields__")
                                else _json_normalize(args.scoring)
                            ),
                        }
                        try:
                            ctx = mp.get_context("spawn")
                            with _worker_env_scope():
                                for batch_idx, batch in enumerate(worker_batches):
                                    if len(worker_batches) > 1:
                                        logger.debug(
                                            "Запускаю batch %s/%s для %s: %s",
                                            batch_idx + 1,
                                            len(worker_batches),
                                            policy_name,
                                            batch,
                                        )
                                    executor = cf.ProcessPoolExecutor(
                                        max_workers=len(batch),
                                        mp_context=ctx,
                                        initializer=_optuna_process_worker_initializer,
                                        initargs=(os.getpid(),),
                                    )
                                    try:
                                        futures = [
                                            executor.submit(
                                                _optuna_process_worker,
                                                {
                                                    **worker_payload,
                                                    "worker_id": worker_id,
                                                    "worker_trials": quota,
                                                },
                                            )
                                            for worker_id, quota in batch
                                        ]
                                        while True:
                                            best, last_trial_number, appended = _append_study_trials(
                                                study=study,
                                                finish_min=args.finish_min,
                                                alive_min=args.alive_min,
                                                save_scenes=args.save_scenes,
                                                trials=trials,
                                                best=best,
                                                last_trial_number=last_trial_number,
                                                csv_writer=csv_writer,
                                                csv_file=csv_file,
                                                policy_name=policy_name,
                                                param_keys=param_keys,
                                                extra_keys=extra_keys,
                                                out_dir=out_dir,
                                                args=args,
                                                seen_trial_numbers=seen_trial_numbers,
                                            )
                                            if pbar is not None and appended:
                                                pbar.update(appended)
                                            if all(future.done() for future in futures):
                                                break
                                            time.sleep(0.5)
                                        for future in futures:
                                            worker_result = future.result()
                                            worker_stats = dict(worker_result.get("eval_cache_stats", {}))
                                            for key in ("hits", "misses", "writes"):
                                                eval_cache["stats"][key] = int(eval_cache["stats"].get(key, 0)) + int(
                                                    worker_stats.get(key, 0)
                                                )
                                        best, last_trial_number, appended = _append_study_trials(
                                            study=study,
                                            finish_min=args.finish_min,
                                            alive_min=args.alive_min,
                                            save_scenes=args.save_scenes,
                                            trials=trials,
                                            best=best,
                                            last_trial_number=last_trial_number,
                                            csv_writer=csv_writer,
                                            csv_file=csv_file,
                                            policy_name=policy_name,
                                            param_keys=param_keys,
                                            extra_keys=extra_keys,
                                            out_dir=out_dir,
                                            args=args,
                                            seen_trial_numbers=seen_trial_numbers,
                                        )
                                        if pbar is not None and appended:
                                            pbar.update(appended)
                                    except BaseException:
                                        executor.shutdown(wait=False, cancel_futures=True)
                                        _terminate_spawn_children()
                                        raise
                                    else:
                                        executor.shutdown(wait=True, cancel_futures=False)
                        except (PermissionError, OSError) as exc:
                            logger.warning(
                                "spawn-worker backend недоступен (%s); fallback на serial study.optimize(n_jobs=1).",
                                exc,
                            )
                            study.optimize(objective, n_trials=trials_total, callbacks=callbacks, n_jobs=1)
                    else:
                        study.optimize(objective, n_trials=trials_total, callbacks=callbacks, n_jobs=n_jobs)
                finally:
                    if pbar is not None:
                        pbar.close()

                _write_policy_checkpoint(
                    out_dir,
                    policy_name,
                    args,
                    trials,
                    best,
                    last_trial_number=last_trial_number,
                    seen_trial_numbers=seen_trial_numbers,
                )
        else:
            space = space_override.get(policy_name, DEFAULT_SPACES.get(policy_name, {}))
            if not space and args.skip_empty:
                candidates = []
            else:
                candidates = build_candidates(space, args.method, args.samples, args.max_trials, args.seed)
                if not candidates:
                    candidates = [{}]

            param_keys = sorted({k for params in candidates for k in params.keys()})
            fieldnames = ["policy", *param_keys, *TUNING_METRIC_KEYS, *extra_keys]
            csv_file, csv_writer = _open_trial_csv(csv_path, fieldnames)

            best = None
            trials = []
            n_jobs = max(1, int(args.n_jobs))
            if n_jobs == 1:
                for params in _maybe_tqdm(candidates, total=len(candidates), desc=f"{policy_name} {args.method}"):
                    trial = _evaluate_candidate_progressive(
                        policy_name=policy_name,
                        params=params,
                        stages=search_stages,
                        success_threshold=args.success_threshold,
                        finish_min=args.finish_min,
                        alive_min=args.alive_min,
                        save_scenes=args.save_scenes,
                        lite_metrics=args.lite_metrics,
                        env_payload=env_payload,
                        goal_radius=goal_radius,
                        aggregate_by_family=args.aggregate_by_family,
                        base_seed=args.seed,
                        eval_cache=eval_cache,
                        persistent_runtime=persistent_runtime,
                        policy_cache_size=int(effective_policy_cache_size),
                        information_regime=regime,
                        parallel_debug=parallel_debug,
                    )
                    trial["state"] = "COMPLETE"
                    trials.append(trial)
                    if best is None or _trial_value(trial) > _trial_value(best):
                        best = trial
                    _write_trial_row(
                        csv_writer,
                        csv_file,
                        policy_name=policy_name,
                        trial=trial,
                        param_keys=param_keys,
                        extra_keys=extra_keys,
                    )
                    _write_policy_checkpoint(out_dir, policy_name, args, trials, best)
            else:
                payload = []
                for params in candidates:
                    payload.append(
                        (
                            policy_name,
                            params,
                            search_stages,
                            args.success_threshold,
                            args.finish_min,
                            args.alive_min,
                            args.save_scenes,
                            args.lite_metrics,
                            env_payload,
                            goal_radius,
                            args.aggregate_by_family,
                            args.seed,
                            eval_cache,
                            persistent_runtime,
                            int(effective_policy_cache_size),
                            regime,
                            parallel_debug,
                        )
                    )
                try:
                    ctx = mp.get_context("fork")
                except Exception:
                    ctx = mp.get_context()
                try:
                    with ctx.Pool(processes=n_jobs) as pool:
                        it = pool.imap_unordered(_evaluate_candidate_worker, payload)
                        for trial in _maybe_tqdm(it, total=len(payload), desc=f"{policy_name} {args.method}"):
                            trial["state"] = "COMPLETE"
                            trials.append(trial)
                            if best is None or _trial_value(trial) > _trial_value(best):
                                best = trial
                            _write_trial_row(
                                csv_writer,
                                csv_file,
                                policy_name=policy_name,
                                trial=trial,
                                param_keys=param_keys,
                                extra_keys=extra_keys,
                            )
                            _write_policy_checkpoint(out_dir, policy_name, args, trials, best)
                except Exception as exc:
                    print(f"[WARN] parallel pool failed ({exc}); falling back to n_jobs=1")
                    for params in _maybe_tqdm(candidates, total=len(candidates), desc=f"{policy_name} {args.method}"):
                        trial = _evaluate_candidate_progressive(
                            policy_name=policy_name,
                            params=params,
                            stages=search_stages,
                            success_threshold=args.success_threshold,
                            finish_min=args.finish_min,
                            alive_min=args.alive_min,
                            save_scenes=args.save_scenes,
                            lite_metrics=args.lite_metrics,
                            env_payload=env_payload,
                            goal_radius=goal_radius,
                            aggregate_by_family=args.aggregate_by_family,
                            base_seed=args.seed,
                            eval_cache=eval_cache,
                            persistent_runtime=persistent_runtime,
                            policy_cache_size=int(effective_policy_cache_size),
                            information_regime=regime,
                            parallel_debug=parallel_debug,
                        )
                        trial["state"] = "COMPLETE"
                        trials.append(trial)
                        if best is None or _trial_value(trial) > _trial_value(best):
                            best = trial
                        _write_trial_row(
                            csv_writer,
                            csv_file,
                            policy_name=policy_name,
                            trial=trial,
                            param_keys=param_keys,
                            extra_keys=extra_keys,
                        )
                        _write_policy_checkpoint(out_dir, policy_name, args, trials, best)

        best_stageA = best
        best_stageB = None
        stageB_trials = []
        champions = {}
        pareto_frontier = []
        validation = {}
        if args.stage_b and eval_seeds_eval and trials:
            topk = _select_topk(trials, int(args.topk), args.finish_min, args.alive_min)
            for rank, cand in enumerate(topk, start=1):
                params = cand.get("params", {})
                _validate_policy_params_for_regime(policy_name, params, regime)
                eval_env = _get_persistent_env(
                    env_payload,
                    max_steps=max_steps,
                    goal_radius=goal_radius,
                    lite_metrics=args.lite_metrics,
                    persistent_runtime=persistent_runtime,
                )
                policy = _get_persistent_policy(
                    policy_name,
                    params,
                    env=eval_env,
                    seed=args.seed,
                    persistent_runtime=persistent_runtime,
                    policy_cache_size=int(effective_policy_cache_size),
                )
                per_scene = evaluate_policy(
                    eval_env,
                    policy,
                    holdout_scenes,
                    eval_seeds_eval,
                    args.success_threshold,
                    args.finish_min,
                    args.alive_min,
                    aggregate_fn=_holdout_aggregate,
                    eval_cache=eval_cache,
                    cache_policy_name=policy_name,
                    cache_params=params,
                    parallel_debug=parallel_debug,
                    stage_label="holdout",
                )
                agg = _holdout_aggregate(per_scene)
                score = _score_tuple(agg, args.finish_min, args.alive_min)
                hard_value = _score_scalar(agg, args.finish_min, args.alive_min)
                value = _score_scalar_soft(agg, args.finish_min, args.alive_min)
                row = {
                    "policy": policy_name,
                    "rank": rank,
                    "params": params,
                    "aggregate": agg,
                    "score": score,
                    "score_scalar": hard_value,
                    "value": value,
                    "gate_pass": _passes_gates(agg, args),
                    "stageA_value": cand.get("value", cand.get("score_scalar")),
                }
                if args.save_scenes:
                    row["scenes"] = per_scene
                stageB_trials.append(row)
                stageb_rows.append(row)
                if best_stageB is None or _trial_value(row) > _trial_value(best_stageB):
                    best_stageB = row
                if stageb_writer is not None and stageb_file is not None:
                    stageb_row = {
                        "policy": policy_name,
                        "rank": rank,
                        **params,
                        **agg,
                        "score_scalar": hard_value,
                        "stageA_value": row.get("stageA_value"),
                    }
                    stageb_writer.writerow(stageb_row)
                    stageb_file.flush()
                _write_policy_checkpoint(
                    out_dir,
                    policy_name,
                    args,
                    trials,
                    best_stageA,
                    stageb_trials=stageB_trials,
                    best_stageb=best_stageB,
                    seen_trial_numbers=seen_trial_numbers,
                )
            if stageB_trials:
                champions = _select_champions(stageB_trials, args=args)
                pareto_frontier = _pareto_frontier(stageB_trials, args=args)
                if champions.get("balanced") is not None:
                    best_stageB = champions["balanced"]
            if best_stageB is not None:
                stageb_best[policy_name] = best_stageB

        if args.stage_b:
            candidate_set = champions if champions else ({"balanced": best_stageA} if best_stageA is not None else {})
            for champion_name, champion_row in candidate_set.items():
                params = champion_row.get("params", {}) if champion_row else {}
                validation[champion_name] = {
                    "reason": _champion_reason(champion_name, champion_row.get("aggregate") or {}),
                    "benchmark": _validate_candidate(
                        policy_name=policy_name,
                        champion_name=champion_name,
                        params=params,
                        split_name="benchmark",
                        split_scenes=benchmark_scenes,
                        split_seeds=report_seeds,
                        split_aggregate_fn=_benchmark_aggregate,
                        split_family_map=benchmark_scene_families,
                    ),
                    "ood": _validate_candidate(
                        policy_name=policy_name,
                        champion_name=champion_name,
                        params=params,
                        split_name="ood",
                        split_scenes=ood_scenes,
                        split_seeds=report_seeds,
                        split_aggregate_fn=_ood_aggregate,
                        split_family_map=ood_scene_families,
                    ),
                }

        if not trials:
            _write_policy_checkpoint(
                out_dir,
                policy_name,
                args,
                trials,
                best_stageA,
                last_trial_number=last_trial_number,
                seen_trial_numbers=seen_trial_numbers,
            )
        elif not args.stage_b:
            _write_policy_checkpoint(
                out_dir,
                policy_name,
                args,
                trials,
                best_stageA,
                last_trial_number=last_trial_number,
                seen_trial_numbers=seen_trial_numbers,
            )
        else:
            _write_policy_checkpoint(
                out_dir,
                policy_name,
                args,
                trials,
                best_stageA,
                stageb_trials=stageB_trials,
                best_stageb=best_stageB,
                champions=champions,
                pareto_frontier=pareto_frontier,
                last_trial_number=last_trial_number,
                seen_trial_numbers=seen_trial_numbers,
            )

        if best_stageB is not None:
            results[policy_name] = {
                "best": best_stageB,
                "best_stageA": best_stageA,
                "best_stageB": best_stageB,
                "champions": champions,
                "pareto_frontier": pareto_frontier,
                "validation": validation,
                "validation_status": "holdout_validated",
                "promotion_status": "promotable",
                "trials": trials,
                "stageB_trials": stageB_trials,
            }
        else:
            results[policy_name] = {
                "best": best_stageA,
                "validation": validation,
                "validation_status": "search_only",
                "promotion_status": "not_promoted",
                "trials": trials,
            }
        if csv_file is not None:
            csv_file.close()

        results_path = out_dir / "tune_results.json"
        results_path.write_text(
            json.dumps(
                {
                    "config": vars(args),
                    "results": results,
                    "summary": {
                        **_summarize_results(results),
                        "_eval_cache": dict(eval_cache.get("stats", {})),
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        _write_protocol_payload()

    if args.stage_b and stageb_rows:
        best_config_path = out_dir / f"best_config.{regime}.json"
        best_config_payload = {
            "policies": {
                k: {
                    "params": v.get("params", {}),
                    "aggregate": v.get("aggregate", {}),
                    "score": v.get("score"),
                    "score_scalar": v.get("score_scalar"),
                }
                for k, v in stageb_best.items()
            },
            "meta": {
                "episodes_eval": args.episodes_eval,
                "topk": args.topk,
                "seed": args.seed,
                "scenes": [Path(p).name for p in search_scene_paths],
                "information_regime": regime,
            },
        }
        best_config_path.write_text(json.dumps(best_config_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        if regime == "fair":
            best_params_path = resolve_repo_path("configs/best_policy_params.json")
            fair_params_path = resolve_repo_path("configs/best_policy_params.fair.json")
        else:
            best_params_path = resolve_repo_path("configs/best_policy_params.privileged.json")
            fair_params_path = None
        _merge_best_params(
            stageb_best,
            best_params_path,
            meta={
                "source": str(best_config_path),
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "information_regime": regime,
            },
        )
        if fair_params_path is not None:
            _merge_best_params(
                stageb_best,
                fair_params_path,
                meta={
                    "source": str(best_config_path),
                    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "information_regime": regime,
                },
            )
    if stageb_file is not None:
        stageb_file.close()

    json_path = out_dir / "tune_results.json"
    json_path.write_text(
        json.dumps(
            {
                "config": vars(args),
                "results": results,
                "summary": {
                    **_summarize_results(results),
                    "_eval_cache": dict(eval_cache.get("stats", {})),
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_protocol_payload()
    report_path = _write_tuning_report(out_dir, protocol, results)

    write_experiment_result(
        out_dir,
        ExperimentResult(
            name="tune_baselines",
            seed=int(args.seed),
            config=vars(args),
            metrics={"results": results},
            artifacts={"results_json": str(json_path), "tuning_report_md": str(report_path)},
        ),
    )

    print(f"[OK] {json_path}")


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig, OmegaConf

    from scripts.common.hydra_utils import apply_schema

    @hydra.main(version_base=None, config_path="../../configs/hydra", config_name="tuning/tune_baselines")
    def _run(cfg: DictConfig) -> None:
        cfg = apply_schema(cfg, TuneBaselinesConfig)
        data = OmegaConf.to_container(cfg, resolve=True)
        run(TuneBaselinesConfig(**data), cfg_raw=cfg)

    _run()
