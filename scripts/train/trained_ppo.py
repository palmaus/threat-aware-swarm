"""
Точка входа для обучения Threat-aware Swarm (PettingZoo + SuperSuit + SB3 PPO).

Запуск (Hydra):
  python -m scripts.train.trained_ppo run.total_timesteps=8000000 device=cuda
  ta-train run.total_timesteps=8000000 device=cuda

Примечания:
- Ожидает класс окружения в env/pz_env.py (SwarmPZEnv).
- Ожидает EnvConfig в env/config.py (EnvConfig).
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from gymnasium import spaces
from omegaconf import OmegaConf
from stable_baselines3 import PPO

try:  # Опционально: поддержка RNN/LSTM через sb3-contrib.
    from sb3_contrib import RecurrentPPO
except Exception:  # pragma: no cover - опциональная зависимость
    RecurrentPPO = None
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.vec_env import VecEnv

from scripts.common.config_guardrails import validate_config_guardrails
from common.runtime.env_factory import concat_vec_envs_safe as _concat_vec_envs_safe
from common.runtime.env_factory import make_pz_env as _common_make_pz_env
from common.runtime.env_factory import make_vec_env as _common_make_vec_env
from scripts.common.experiment_result import ExperimentResult, write_experiment_result
from scripts.common.logging_utils import (
    init_clearml,
    init_mlflow,
    log_artifacts_clearml,
    log_artifacts_mlflow,
    log_metrics,
    log_mlflow_pyfunc_model,
    log_params,
    resolve_tracking_cfg,
)
from scripts.common.model_registry import resolve_model_path
from scripts.common.numba_guard import log_numba_status
from scripts.common.path_utils import find_repo_root
from scripts.common.rng_manager import SeedManager
from scripts.train.callbacks import (
    SaveRunMetaCallback,
    SwarmCurriculumCallback,
    SwarmInfoMetricsCallback,
    TrackingMetricsCallback,
    _load_curriculum_config,
)
from scripts.train.curriculum_manager import CurriculumManager
from scripts.train.env_methods import call_env_method

logger = logging.getLogger(__name__)


def now_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def set_global_seed(seed: int) -> SeedManager:
    manager = SeedManager(int(seed))
    manager.seed_all()
    return manager


def _git_info(root: Path) -> tuple[str, bool]:
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(root),
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
        return rev, bool(dirty)
    except Exception:
        return "unknown", False


def safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _extract_episode_summary(env: Any) -> Any | None:
    """Достаёт последнюю сводку эпизода из VecEnv или базового окружения."""
    if env is None:
        return None
    # Стандартный путь для VecEnv.
    if hasattr(env, "env_method"):
        try:
            summaries = env.env_method("get_episode_summary")
            for summary in summaries:
                if summary is not None:
                    return summary
        except Exception:
            pass
    # Ручное развёртывание для нетипичных обёрток.
    cursor = env
    for _ in range(6):
        if hasattr(cursor, "get_episode_summary"):
            try:
                return cursor.get_episode_summary()
            except Exception:
                return None
        if hasattr(cursor, "venv"):
            cursor = cursor.venv
            continue
        if hasattr(cursor, "env"):
            cursor = cursor.env
            continue
        if hasattr(cursor, "envs"):
            try:
                envs = cursor.envs
                for item in envs:
                    if hasattr(item, "get_episode_summary"):
                        summary = item.get_episode_summary()
                        if summary is not None:
                            return summary
            except Exception:
                return None
        break
    return None


def _log_episode_summary_mlflow(mlflow_run: Any, env: Any, step: int) -> None:
    if mlflow_run is None:
        return
    summary = _extract_episode_summary(env)
    if summary is None:
        return
    metrics: dict[str, float] = {
        "episode/steps": safe_float(getattr(summary, "steps", float("nan"))),
        "episode/decision_steps": safe_float(getattr(summary, "decision_steps", float("nan"))),
        "episode/physics_steps": safe_float(getattr(summary, "physics_steps", float("nan"))),
        "episode/duration_sim_secs": safe_float(getattr(summary, "duration_sim_secs", float("nan"))),
    }
    cost_means = getattr(summary, "cost_means", None)
    if isinstance(cost_means, dict):
        for key, val in cost_means.items():
            metrics[f"episode/{key}"] = safe_float(val)
    log_metrics(mlflow_run, None, metrics, step=int(step))
    try:
        import mlflow

        tags: dict[str, str] = {}
        scene_id = getattr(summary, "scene_id", None)
        if scene_id is not None:
            tags["episode_scene"] = str(scene_id)
        seed = getattr(summary, "seed", None)
        if seed is not None:
            tags["episode_seed"] = str(seed)
        if tags:
            mlflow.set_tags(tags)
    except Exception:
        return


def _coerce_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def apply_resume_overrides(model: PPO, ppo_cfg: Any, warn_on_mismatch: bool = True) -> dict[str, Any]:
    """Обновляет безопасные гиперпараметры после PPO.load(). Возвращает применённые переопределения."""
    overrides: dict[str, Any] = {}

    def _maybe_update(name: str, new_value: Any, setter) -> None:
        new_val = _coerce_float(new_value)
        if new_val is None:
            return
        old_val = getattr(model, name, None)
        old_num = _coerce_float(old_val)
        if old_num is None or abs(old_num - new_val) > 1e-12:
            try:
                setter(new_val)
                overrides[name] = {"old": old_num, "new": new_val}
            except Exception as exc:
                logger.warning("Resume override '%s' failed: %s", name, exc)

    _maybe_update("ent_coef", getattr(ppo_cfg, "ent_coef", None), lambda v: setattr(model, "ent_coef", float(v)))

    def _set_learning_rate(v: float) -> None:
        model.learning_rate = float(v)
        model.lr_schedule = get_schedule_fn(float(v))
        try:
            model._setup_lr_schedule()
        except Exception as exc:
            logger.warning("Resume override learning_rate setup failed: %s", exc)

    _maybe_update("learning_rate", getattr(ppo_cfg, "learning_rate", None), _set_learning_rate)

    def _set_clip_range(v: float) -> None:
        model.clip_range = get_schedule_fn(float(v))

    _maybe_update("clip_range", getattr(ppo_cfg, "clip_range", None), _set_clip_range)
    _maybe_update("target_kl", getattr(ppo_cfg, "target_kl", None), lambda v: setattr(model, "target_kl", float(v)))
    _maybe_update("vf_coef", getattr(ppo_cfg, "vf_coef", None), lambda v: setattr(model, "vf_coef", float(v)))

    def _set_gamma(v: float) -> None:
        model.gamma = float(v)
        if hasattr(model, "rollout_buffer") and model.rollout_buffer is not None:
            try:
                model.rollout_buffer.gamma = float(v)
            except Exception as exc:
                logger.warning("Resume override gamma buffer update failed: %s", exc)

    _maybe_update("gamma", getattr(ppo_cfg, "gamma", None), _set_gamma)

    def _set_gae(v: float) -> None:
        model.gae_lambda = float(v)
        if hasattr(model, "rollout_buffer") and model.rollout_buffer is not None:
            try:
                model.rollout_buffer.gae_lambda = float(v)
            except Exception as exc:
                logger.warning("Resume override gae_lambda buffer update failed: %s", exc)

    _maybe_update("gae_lambda", getattr(ppo_cfg, "gae_lambda", None), _set_gae)

    if warn_on_mismatch:
        unsafe = {
            "n_steps": getattr(ppo_cfg, "n_steps", None),
            "batch_size": getattr(ppo_cfg, "batch_size", None),
            "n_epochs": getattr(ppo_cfg, "n_epochs", None),
        }
        for key, new_val in unsafe.items():
            old_val = getattr(model, key, None)
            if old_val is None:
                continue
            try:
                if int(old_val) != int(new_val):
                    print(
                        f"[WARN] Resume override ignored for {key}: "
                        f"loaded={old_val}, requested={new_val}. "
                        "Requires re-init of PPO."
                    )
            except Exception as exc:
                logger.warning("Resume override mismatch check failed for %s: %s", key, exc)

    if overrides:
        print(f"[Resume] Applied overrides: {overrides}")
    return overrides


def make_pz_env(
    max_steps: int,
    goal_radius: float,
    seed: int,
    initial_stage_params: dict[str, Any] | None = None,
    shared_curriculum: dict[str, Any] | None = None,
    env_overrides: dict[str, Any] | None = None,
) -> Any:
    return _common_make_pz_env(
        max_steps=max_steps,
        goal_radius=goal_radius,
        seed=seed,
        env_overrides=env_overrides,
        initial_stage_params=initial_stage_params,
        shared_curriculum=shared_curriculum,
    )


def concat_vec_envs_safe(
    env_fns, obs_space, act_space, num_vec_envs: int, num_cpus: int, base_class: str = "stable_baselines3"
):
    return _concat_vec_envs_safe(env_fns, obs_space, act_space, num_vec_envs, num_cpus, base_class=base_class)


def make_vec_env(
    max_steps: int,
    goal_radius: float,
    seed: int,
    num_vec_envs: int,
    num_cpus: int,
    initial_stage_params: dict[str, Any] | None = None,
    shared_curriculum: dict[str, Any] | None = None,
    env_overrides: dict[str, Any] | None = None,
) -> Any:
    return _common_make_vec_env(
        max_steps=max_steps,
        goal_radius=goal_radius,
        seed=seed,
        num_vec_envs=num_vec_envs,
        num_cpus=num_cpus,
        initial_stage_params=initial_stage_params,
        shared_curriculum=shared_curriculum,
        env_overrides=env_overrides,
        train_wrappers=True,
        vec_monitor=True,
        smoke_check=os.environ.get("TA_SMOKE") == "1",
    )


def _main_hydra() -> None:
    import hydra
    from omegaconf import DictConfig

    from scripts.common.hydra_schema import apply_schema

    @hydra.main(version_base=None, config_path="../../configs/hydra", config_name="train")
    def _run(cfg: DictConfig) -> None:
        cfg = apply_schema(cfg)
        train_with_args(cfg, hydra_cfg=cfg)

    _run()


def _ensure_repo_on_path() -> None:
    try:
        root = find_repo_root(Path.cwd())
    except Exception:
        logger.warning("Repo root detection failed; sys.path not updated")
        return
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def _is_external_ref(value: str | None) -> bool:
    if not value:
        return False
    token = str(value).strip()
    return token.startswith(("clearml:", "mlflow:"))


def _select_resume_model(run_dir: Path, resume_model: str) -> Path:
    model_ref = str(resume_model or "").strip()
    if _is_external_ref(model_ref):
        download_dir = run_dir / "models" / "external"
        return resolve_model_path(model_ref, download_dir=download_dir)
    if model_ref:
        path = Path(model_ref)
        if not path.is_absolute():
            path = run_dir / model_ref
        return path
    models_dir = run_dir / "models"
    if not models_dir.exists():
        raise FileNotFoundError(f"models dir not found: {models_dir}")
    candidates = sorted(models_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"no .zip models in {models_dir}")
    filtered = [p for p in candidates if p.name != "best_by_finished.zip"]
    return filtered[-1] if filtered else candidates[-1]


def _find_latest_run_dir(run_root: Path, run_name: str | None = None) -> Path | None:
    if not run_root.exists():
        return None
    candidates = sorted(run_root.glob("run_*"), key=lambda p: p.stat().st_mtime)
    if run_name:
        candidates = [p for p in candidates if run_name in p.name]
    return candidates[-1] if candidates else None


def _resolve_resume_run_dir(resume_cfg, run_cfg, root: Path) -> Path | None:
    token = str(getattr(resume_cfg, "run_dir", "") or "").strip()
    run_name = str(getattr(resume_cfg, "run_name", "") or "").strip()
    model_ref = str(getattr(resume_cfg, "model", "") or "").strip()
    if _is_external_ref(model_ref) and not token and not run_name:
        return None
    if token and ":" in token:
        prefix, suffix = token.split(":", 1)
        if prefix in {"latest", "last", "auto"}:
            token = prefix
            if suffix:
                run_name = suffix.strip()
    if not token:
        if not run_name:
            raise FileNotFoundError("resume.run_dir пустой. Укажите resume.run_dir или resume.run_name.")
        token = "latest"
    if token in {"latest", "last", "auto"}:
        run_root = Path(getattr(run_cfg, "out_dir", "runs"))
        if not run_root.is_absolute():
            run_root = root / run_root
        if not run_name and getattr(run_cfg, "run_name", ""):
            run_name = str(getattr(run_cfg, "run_name", "") or "").strip()
            if run_name:
                print(f"[Resume] resume.run_dir=latest: using run.run_name={run_name!r}")
        latest = _find_latest_run_dir(run_root, run_name or None)
        if latest is None:
            raise FileNotFoundError(f"resume latest run not found in {run_root} (run_name={run_name!r})")
        return latest
    path = Path(token)
    if not path.is_absolute():
        path = root / path
    return path


@dataclass
class RunContext:
    root: Path
    run_dir: Path
    run_id: str
    resume_run: Path | None
    resume_model_path: Path | None
    meta_dir: Path
    unused_hydra_run_dir: Path | None


@dataclass
class TrackingContext:
    tracking_cfg: Any
    mlflow_cfg: Any
    clearml_cfg: Any
    tracking_log_every: int
    system_metrics_enabled: bool
    clearml_task: Any
    mlflow_run: Any
    sb3_logger: Any


def _resolve_env_overrides(env_cfg) -> dict[str, Any]:
    try:
        env_overrides = OmegaConf.to_container(env_cfg, resolve=True)
    except Exception as exc:
        logger.warning("Env overrides parse failed: %s", exc)
        env_overrides = {}
    if not isinstance(env_overrides, dict):
        return {}
    return env_overrides


def _resolve_progress_bar(cfg) -> bool:
    if not cfg.progress_bar:
        return False
    try:
        import rich  # noqa: F401
        import tqdm  # noqa: F401

        return True
    except Exception as exc:
        import warnings

        warnings.warn(
            f"Прогресс‑бар отключён: нет rich/tqdm ({exc}). "
            "Установите `pip install rich tqdm` и повторите с --progress-bar.",
            stacklevel=2,
        )
    return False


def _setup_run_context(cfg, hydra_cfg) -> RunContext:
    run_cfg = cfg.run
    resume_cfg = cfg.resume
    root = find_repo_root(Path.cwd())
    resume_run: Path | None = None
    if resume_cfg.enabled:
        resume_run = _resolve_resume_run_dir(resume_cfg, run_cfg, root)
    unused_hydra_run_dir: Path | None = None
    if resume_run is not None:
        run_dir = resume_run
        run_id = run_dir.name
        if not run_dir.exists():
            raise FileNotFoundError(f"resume run_dir not found: {run_dir}")
        if hydra_cfg is not None:
            hydra_output = hydra_cfg.get("hydra", {}).get("runtime", {}).get("output_dir")
            if hydra_output:
                hydra_output_path = Path(hydra_output)
                if hydra_output_path != run_dir:
                    unused_hydra_run_dir = hydra_output_path
    else:
        run_id = f"run_{now_id()}" + (f"_{run_cfg.run_name}" if run_cfg.run_name else "")
        out_dir = Path(run_cfg.out_dir)
        if not out_dir.is_absolute():
            out_dir = root / out_dir
        run_dir = out_dir / run_id
        if hydra_cfg is not None:
            hydra_output = hydra_cfg.get("hydra", {}).get("runtime", {}).get("output_dir")
            if hydra_output:
                run_dir = Path(hydra_output)
                run_id = run_dir.name
    meta_dir = run_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "models").mkdir(parents=True, exist_ok=True)
    (run_dir / "tb").mkdir(parents=True, exist_ok=True)
    if hydra_cfg is not None:
        try:
            (meta_dir / "hydra_config.yaml").write_text(
                OmegaConf.to_yaml(hydra_cfg, resolve=True),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("Hydra config write failed: %s", exc)
    git_rev, git_dirty = _git_info(root)
    (meta_dir / "git.txt").write_text(
        f"commit={git_rev}\ndirty={int(git_dirty)}\n",
        encoding="utf-8",
    )
    resume_model_path: Path | None = None
    if resume_cfg.enabled:
        if resume_run is not None:
            resume_model_path = _select_resume_model(run_dir, resume_cfg.model)
        elif _is_external_ref(resume_cfg.model):
            resume_model_path = _select_resume_model(run_dir, resume_cfg.model)
        else:
            raise FileNotFoundError(
                "resume.enabled требует resume.run_dir или resume.model=mlflow:<run_id>[/artifact] "
                "или clearml:<task_id>[/artifact]."
            )
    return RunContext(
        root=root,
        run_dir=run_dir,
        run_id=run_id,
        resume_run=resume_run,
        resume_model_path=resume_model_path,
        meta_dir=meta_dir,
        unused_hydra_run_dir=unused_hydra_run_dir,
    )


def _init_tracking_context(hydra_cfg, run_id: str, run_dir: Path) -> TrackingContext:
    tracking_cfg = resolve_tracking_cfg(hydra_cfg)

    def _get_tracking(node, key, default=None):
        try:
            return node.get(key, default)
        except Exception:
            return default

    mlflow_cfg = _get_tracking(tracking_cfg, "mlflow", {})
    clearml_cfg = _get_tracking(tracking_cfg, "clearml", {})
    tracking_log_every = int(_get_tracking(tracking_cfg, "log_every_steps", 2048))
    system_metrics_enabled = bool(_get_tracking(_get_tracking(tracking_cfg, "system_metrics", {}), "enabled", False))
    clearml_task = init_clearml(clearml_cfg, run_name=run_id, hydra_cfg=hydra_cfg)
    mlflow_run = init_mlflow(mlflow_cfg, run_name=run_id)
    sb3_logger = configure(str(run_dir / "tb"), ["stdout", "tensorboard"])
    return TrackingContext(
        tracking_cfg=tracking_cfg,
        mlflow_cfg=mlflow_cfg,
        clearml_cfg=clearml_cfg,
        tracking_log_every=tracking_log_every,
        system_metrics_enabled=system_metrics_enabled,
        clearml_task=clearml_task,
        mlflow_run=mlflow_run,
        sb3_logger=sb3_logger,
    )


def _prepare_curriculum_context(
    cfg, root: Path
) -> tuple[
    list[dict[str, Any]],
    Path | None,
    dict[str, Any] | None,
    dict[str, Any] | None,
    mp.managers.SyncManager | None,
    float,
    dict[str, Any],
]:
    curriculum_cfg = cfg.curriculum
    vec_cfg = cfg.vec
    curriculum_stages: list[dict[str, Any]] = []
    curriculum_cfg_path: Path | None = None
    stage0_params: dict[str, Any] | None = None
    shared_curriculum: dict[str, Any] | None = None
    curriculum_manager: mp.managers.SyncManager | None = None
    curriculum_ema_alpha = float(curriculum_cfg.ema_alpha)
    curriculum_meta: dict[str, Any] = {}
    if curriculum_cfg.enabled:
        if curriculum_cfg.config:
            curriculum_cfg_path = Path(curriculum_cfg.config)
            if not curriculum_cfg_path.is_absolute():
                curriculum_cfg_path = root / curriculum_cfg_path
        else:
            candidate = root / "configs" / "curriculum" / "base.yaml"
            if candidate.exists():
                curriculum_cfg_path = candidate
        if curriculum_cfg_path and curriculum_cfg_path.exists():
            curriculum_stages, curriculum_meta = _load_curriculum_config(curriculum_cfg_path, curriculum_cfg.profile)
            if curriculum_meta.get("ema_alpha") is not None and curriculum_ema_alpha == 0.2:
                curriculum_ema_alpha = float(curriculum_meta["ema_alpha"])
        else:
            raise FileNotFoundError("Curriculum config not found. Use --curriculum-config.")
        if not curriculum_stages:
            logger.warning("Curriculum stages not found in %s", curriculum_cfg_path)
        else:
            stage0 = curriculum_stages[0] if curriculum_stages else {}
            if isinstance(stage0, dict) and isinstance(stage0.get("params"), dict):
                stage0_params = stage0["params"]
            else:
                stage0_params = stage0
            stage0_params = CurriculumManager.filter_stage_params(stage0_params) if stage0_params else None
        if vec_cfg.num_cpus > 1:
            curriculum_manager = mp.Manager()
            shared_curriculum = curriculum_manager.dict()
            shared_curriculum["params"] = dict(stage0_params or {})
            shared_curriculum["version"] = 0
    return (
        curriculum_stages,
        curriculum_cfg_path,
        stage0_params,
        shared_curriculum,
        curriculum_manager,
        curriculum_ema_alpha,
        curriculum_meta,
    )


def _build_env_with_curriculum(
    cfg,
    env_overrides: dict[str, Any],
    stage0_params: dict[str, Any] | None,
    shared_curriculum: dict[str, Any] | None,
    curriculum_stages: list[dict[str, Any]],
) -> VecEnv:
    run_cfg = cfg.run
    env_cfg = cfg.env
    vec_cfg = cfg.vec
    env = make_vec_env(
        max_steps=int(env_cfg.max_steps),
        goal_radius=float(env_cfg.goal_radius),
        seed=int(run_cfg.seed),
        num_vec_envs=int(vec_cfg.num_vec_envs),
        num_cpus=int(vec_cfg.num_cpus),
        initial_stage_params=stage0_params,
        shared_curriculum=shared_curriculum,
        env_overrides=env_overrides,
    )
    if curriculum_stages:
        params = stage0_params or {}
        if shared_curriculum is None:
            try:
                call_env_method(env, "apply_curriculum", params)
                env.reset()
            except Exception as exc:
                logger.warning("Initial curriculum apply failed: %s", exc)
    return env


def train_with_args(cfg, hydra_cfg=None) -> None:
    _ensure_repo_on_path()
    from env.pz_env import ENV_SCHEMA_VERSION

    run_cfg = cfg.run
    env_cfg = cfg.env
    vec_cfg = cfg.vec
    ppo_cfg = cfg.ppo
    use_rnn = bool(getattr(ppo_cfg, "use_rnn", False))
    curriculum_cfg = cfg.curriculum
    resume_cfg = cfg.resume
    if use_rnn and RecurrentPPO is None:
        raise RuntimeError("sb3-contrib is required for recurrent policies. Install `sb3-contrib`.")
    env_overrides = _resolve_env_overrides(env_cfg)
    progress_bar = _resolve_progress_bar(cfg)
    seed_manager = set_global_seed(int(run_cfg.seed))
    validate_config_guardrails(cfg)
    log_numba_status(logger)

    run_ctx = _setup_run_context(cfg, hydra_cfg)
    tracking_ctx = _init_tracking_context(hydra_cfg, run_ctx.run_id, run_ctx.run_dir)
    resume_model_path = run_ctx.resume_model_path
    resume_run = run_ctx.resume_run
    (
        curriculum_stages,
        curriculum_cfg_path,
        stage0_params,
        shared_curriculum,
        _curriculum_manager,
        curriculum_ema_alpha,
        curriculum_meta,
    ) = _prepare_curriculum_context(cfg, run_ctx.root)
    env = _build_env_with_curriculum(
        cfg,
        env_overrides=env_overrides,
        stage0_params=stage0_params,
        shared_curriculum=shared_curriculum,
        curriculum_stages=curriculum_stages,
    )
    run_dir = run_ctx.run_dir
    run_id = run_ctx.run_id
    meta_dir = run_ctx.meta_dir
    unused_hydra_run_dir = run_ctx.unused_hydra_run_dir
    _tracking_cfg = tracking_ctx.tracking_cfg
    mlflow_cfg = tracking_ctx.mlflow_cfg
    _clearml_cfg = tracking_ctx.clearml_cfg
    tracking_log_every = tracking_ctx.tracking_log_every
    system_metrics_enabled = tracking_ctx.system_metrics_enabled
    clearml_task = tracking_ctx.clearml_task
    mlflow_run = tracking_ctx.mlflow_run
    sb3_logger = tracking_ctx.sb3_logger

    def _get_tracking(node, key, default=None):
        try:
            return node.get(key, default)
        except Exception:
            return default

    obs_space = env.observation_space
    if hasattr(obs_space, "shape") and obs_space.shape is not None:
        obs_dim: object = int(np.prod(obs_space.shape))
    elif isinstance(obs_space, spaces.Dict):
        vec_space = obs_space.spaces.get("vector")
        grid_space = obs_space.spaces.get("grid")
        if vec_space is not None and grid_space is not None:
            obs_dim = {"vector": int(np.prod(vec_space.shape)), "grid": list(grid_space.shape)}
        else:
            obs_dim = "dict"
    else:
        obs_dim = "unknown"

    model_signature = {"obs_dim": obs_dim}
    if isinstance(obs_space, spaces.Dict):
        if vec_space is not None:
            model_signature["vector_dim"] = int(np.prod(vec_space.shape))
        if grid_space is not None:
            model_signature["grid_shape"] = list(grid_space.shape)
    (meta_dir / "model_signature.json").write_text(
        json.dumps(model_signature, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    def _parse_net_arch(val: Any) -> list[int]:
        if isinstance(val, (list, tuple)):
            return [int(v) for v in val]
        if isinstance(val, str):
            text = val.strip()
            if text.startswith("[") and text.endswith("]"):
                try:
                    import ast

                    parsed = ast.literal_eval(text)
                    if isinstance(parsed, (list, tuple)):
                        return [int(v) for v in parsed]
                except Exception as exc:
                    logger.warning("net_arch parse failed (%s): %s", text, exc)
            parts = [p.strip() for p in text.split(",") if p.strip()]
            return [int(p) for p in parts]
        return []

    # Парсим архитектуру сети, чтобы фиксировать ее в метаданных.
    net_arch = _parse_net_arch(ppo_cfg.net_arch)
    policy_kwargs = {"net_arch": net_arch, "log_std_init": ppo_cfg.log_std_init}
    if use_rnn:
        policy_kwargs["lstm_hidden_size"] = int(getattr(ppo_cfg, "lstm_hidden_size", 256))
        policy_kwargs["enable_critic_lstm"] = bool(getattr(ppo_cfg, "enable_critic_lstm", True))
    extractor_name = str(getattr(ppo_cfg, "cnn_extractor", "advanced")).lower().strip()
    try:
        from models.feature_extractors import AdvancedSwarmCNN, SmallSwarmCNN
    except Exception:  # pragma: no cover - запасной путь для разных cwd
        from threat_aware_swarm.models.feature_extractors import AdvancedSwarmCNN, SmallSwarmCNN
    extractor_cls = AdvancedSwarmCNN if extractor_name in {"advanced", "adv"} else SmallSwarmCNN
    policy_kwargs["normalize_images"] = False
    policy_kwargs["features_extractor_class"] = extractor_cls
    policy_kwargs["features_extractor_kwargs"] = {"features_dim": int(ppo_cfg.cnn_features_dim)}

    hparams = {
        "learning_rate": ppo_cfg.learning_rate,
        "n_steps": ppo_cfg.n_steps,
        "batch_size": ppo_cfg.batch_size,
        "n_epochs": ppo_cfg.n_epochs,
        "gamma": ppo_cfg.gamma,
        "gae_lambda": ppo_cfg.gae_lambda,
        "ent_coef": ppo_cfg.ent_coef,
        "vf_coef": ppo_cfg.vf_coef,
        "clip_range": ppo_cfg.clip_range,
        "target_kl": ppo_cfg.target_kl,
        "net_arch": net_arch,
        "log_std_init": ppo_cfg.log_std_init,
        "use_rnn": use_rnn,
        "lstm_hidden_size": int(getattr(ppo_cfg, "lstm_hidden_size", 256)),
        "enable_critic_lstm": bool(getattr(ppo_cfg, "enable_critic_lstm", True)),
    }
    if resume_model_path is not None:
        resume_meta = {"model_path": str(resume_model_path)}
        if resume_run is not None:
            resume_meta["run_dir"] = str(resume_run)
        if _is_external_ref(resume_cfg.model):
            resume_meta["model_ref"] = str(resume_cfg.model)
        hparams["resume"] = resume_meta
    if curriculum_stages:
        hparams["curriculum"] = {
            "config": str(curriculum_cfg_path) if curriculum_cfg_path else "",
            "profile": curriculum_cfg.profile,
            "metric_key": curriculum_cfg.metric,
            "threshold": curriculum_cfg.threshold,
            "min_steps_per_stage": curriculum_cfg.min_steps,
            "ema_alpha": curriculum_ema_alpha,
            "goal_radius_decay_steps": curriculum_cfg.goal_radius_decay_steps,
            "stages": curriculum_stages,
        }
    (meta_dir / "ppo_hparams.json").write_text(json.dumps(hparams, ensure_ascii=False, indent=2), encoding="utf-8")

    meta = {
        "run_id": run_id,
        "time_start": datetime.now().isoformat(timespec="seconds"),
        "algo": "PPO",
        "seed": run_cfg.seed,
        "obs_dim": obs_dim,
        "env_schema_version": ENV_SCHEMA_VERSION,
        "exp_id": cfg.exp_id,
        "env": {
            "max_steps": env_cfg.max_steps,
            "goal_radius": env_cfg.goal_radius,
            "num_vec_envs": vec_cfg.num_vec_envs,
            "config": dict(env_overrides or {}),
            "initial_stage_params": dict(stage0_params or {}),
        },
        "hparams": hparams,
        "model_signature": model_signature,
        "paths": {
            "run_dir": str(run_dir),
            "tb": str(run_dir / "tb"),
            "models": str(run_dir / "models"),
        },
    }
    if resume_model_path is not None:
        resume_meta = {"model_path": str(resume_model_path)}
        if resume_run is not None:
            resume_meta["run_dir"] = str(resume_run)
        if _is_external_ref(resume_cfg.model):
            resume_meta["model_ref"] = str(resume_cfg.model)
        meta["resume"] = resume_meta

    log_params(
        mlflow_run,
        clearml_task,
        {
            "run": {"id": run_id, "seed": run_cfg.seed, "device": cfg.device},
            "env": meta["env"],
            "hparams": hparams,
            "model_signature": model_signature,
            "resume": meta.get("resume", {}),
        },
    )
    if mlflow_run is not None:
        try:
            import mlflow

            mlflow.set_tags(
                {
                    "cnn_features_dim": str(ppo_cfg.cnn_features_dim),
                    "cnn_extractor": str(getattr(ppo_cfg, "cnn_extractor", "")),
                    "curriculum_enabled": str(bool(curriculum_stages)),
                    "rnn_enabled": str(bool(use_rnn)),
                    "lstm_hidden_size": str(getattr(ppo_cfg, "lstm_hidden_size", "")),
                    "enable_critic_lstm": str(getattr(ppo_cfg, "enable_critic_lstm", "")),
                }
            )
        except Exception as exc:
            logger.warning("MLflow set_tags failed: %s", exc)

    meta_cb = SaveRunMetaCallback(run_dir=run_dir, meta=meta)

    # Чекпоинты пишутся реже, чтобы не тормозить обучение.
    ckpt_cb = CheckpointCallback(
        save_freq=max(int(run_cfg.checkpoint_freq) // max(1, env.num_envs), 1),
        save_path=str(run_dir / "models"),
        name_prefix="swarm_ppo",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    metrics_cb = SwarmInfoMetricsCallback(log_every_steps=2048)

    callbacks_list = [meta_cb, metrics_cb, ckpt_cb]
    if mlflow_run is not None or clearml_task is not None:
        callbacks_list.append(
            TrackingMetricsCallback(
                mlflow_run=mlflow_run,
                clearml_task=clearml_task,
                log_every_steps=tracking_log_every,
                system_metrics=system_metrics_enabled,
            )
        )
        curriculum_state_path: Path | None = None
        curriculum_state_pickle_path: Path | None = None
    if curriculum_stages:

        def _merge_cfg(base: dict[str, Any] | None, override: dict[str, Any] | None) -> dict[str, Any]:
            merged: dict[str, Any] = dict(base or {})
            if override:
                for key, value in override.items():
                    if isinstance(value, dict) and isinstance(merged.get(key), dict):
                        nested = dict(merged[key])
                        nested.update(value)
                        merged[key] = nested
                    else:
                        merged[key] = value
            return merged

        def _reinit_env(stage_params: dict[str, Any]):
            params = dict(stage_params or {})
            max_steps = int(params.get("max_steps", env_cfg.max_steps))
            goal_radius = float(params.get("goal_radius", env_cfg.goal_radius))
            return make_vec_env(
                max_steps=max_steps,
                goal_radius=goal_radius,
                seed=int(run_cfg.seed),
                num_vec_envs=int(vec_cfg.num_vec_envs),
                num_cpus=int(vec_cfg.num_cpus),
                initial_stage_params=params,
                shared_curriculum=shared_curriculum,
                env_overrides=env_overrides,
            )

        curriculum_state_path = meta_dir / "curriculum_state.json"
        curriculum_state_pickle_path = meta_dir / "alp_state.pkl"
        adr_cfg = dict(vars(curriculum_cfg.adr)) if hasattr(curriculum_cfg, "adr") else {}
        alp_cfg = dict(vars(curriculum_cfg.alp)) if hasattr(curriculum_cfg, "alp") else {}
        if curriculum_meta.get("adr") is not None:
            adr_cfg = _merge_cfg(adr_cfg, curriculum_meta.get("adr") or {})
        if curriculum_meta.get("alp") is not None:
            alp_cfg = _merge_cfg(alp_cfg, curriculum_meta.get("alp") or {})
        if alp_cfg.get("seed") is None:
            alp_cfg["seed"] = seed_manager.child_seed("alp")
        curriculum_cb = SwarmCurriculumCallback(
            stages=curriculum_stages,
            metric_key=curriculum_cfg.metric,
            threshold=curriculum_cfg.threshold,
            min_steps_per_stage=curriculum_cfg.min_steps,
            ema_alpha=curriculum_ema_alpha,
            mark_steps=64,
            goal_radius_decay_steps=curriculum_cfg.goal_radius_decay_steps,
            adr_config=adr_cfg,
            alp_config=alp_cfg,
            shared_curriculum=shared_curriculum,
            reinit_env_fn=_reinit_env,
            reinit_on_stage_change=True,
            state_path=curriculum_state_path,
            state_pickle_path=curriculum_state_pickle_path,
        )
        callbacks_list.append(curriculum_cb)

    callbacks = CallbackList(callbacks_list)

    policy_name = "MultiInputLstmPolicy" if use_rnn else "MultiInputPolicy"
    resume_overrides: dict[str, Any] = {}
    if resume_model_path is not None:
        if use_rnn:
            model = RecurrentPPO.load(str(resume_model_path), env=env, device=cfg.device)
        else:
            model = PPO.load(str(resume_model_path), env=env, device=cfg.device)
        if bool(resume_cfg.apply_overrides):
            resume_overrides = apply_resume_overrides(
                model,
                ppo_cfg,
                warn_on_mismatch=bool(resume_cfg.warn_on_mismatch),
            )
    else:
        algo_cls = RecurrentPPO if use_rnn else PPO
        model = algo_cls(
            policy=policy_name,
            env=env,
            learning_rate=ppo_cfg.learning_rate,
            n_steps=ppo_cfg.n_steps,
            batch_size=ppo_cfg.batch_size,
            n_epochs=ppo_cfg.n_epochs,
            gamma=ppo_cfg.gamma,
            gae_lambda=ppo_cfg.gae_lambda,
            ent_coef=ppo_cfg.ent_coef,
            vf_coef=ppo_cfg.vf_coef,
            clip_range=ppo_cfg.clip_range,
            target_kl=ppo_cfg.target_kl,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=str(run_dir / "tb"),
            device=cfg.device,
        )
    model.set_logger(sb3_logger)

    if resume_overrides:
        try:
            hparams.setdefault("resume", {})["overrides"] = resume_overrides
            meta.setdefault("resume", {})["overrides"] = resume_overrides
            (meta_dir / "ppo_hparams.json").write_text(
                json.dumps(hparams, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            log_params(
                mlflow_run,
                clearml_task,
                {"resume_overrides": resume_overrides},
            )
        except Exception as exc:
            logger.warning("Resume override logging failed: %s", exc)

    def _write_model_meta(model_path: Path) -> None:
        meta_path = model_path.with_suffix(model_path.suffix + ".meta.json")
        meta = {
            "env_schema_version": ENV_SCHEMA_VERSION,
            "obs_dim": obs_dim,
            "run_id": run_id,
            "env_config": dict(env_overrides or {}),
            "initial_stage_params": dict(stage0_params or {}),
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    interrupted = False
    interrupt_path: Path | None = None
    try:
        model.learn(
            total_timesteps=int(run_cfg.total_timesteps),
            callback=callbacks,
            progress_bar=progress_bar,
            reset_num_timesteps=False if resume_model_path is not None else True,
        )
    except KeyboardInterrupt:
        interrupted = True
        print("[WARN] Обучение остановлено пользователем (Ctrl+C). Сохраняю прогресс.")
    finally:
        # Финальную модель сохраняем даже при исключениях, чтобы не терять прогресс.
        final_path = run_dir / "models" / "final.zip"
        model.save(str(final_path))
        _write_model_meta(final_path)
        if interrupted:
            interrupt_path = run_dir / "models" / "interrupt.zip"
            try:
                model.save(str(interrupt_path))
                _write_model_meta(interrupt_path)
            except Exception as exc:
                logger.warning("Interrupt checkpoint save failed: %s", exc)
                interrupt_path = None

        meta["time_end"] = datetime.now().isoformat(timespec="seconds")
        meta["final_model"] = str(final_path)
        meta["status"] = "interrupted" if interrupted else "completed"
        (run_dir / "meta" / "run.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        best_path = run_dir / "models" / "best_by_finished.zip"
        if best_path.exists():
            _write_model_meta(best_path)

        artifact_paths = [
            meta_dir / "run.json",
            meta_dir / "ppo_hparams.json",
            meta_dir / "model_signature.json",
        ]
        if curriculum_state_path is not None:
            artifact_paths.append(curriculum_state_path)
        log_artifacts_mlflow(mlflow_run, [meta_dir], artifact_path="meta")
        log_artifacts_clearml(clearml_task, [meta_dir], name_prefix="meta")
        model_artifacts = [final_path, best_path]
        if interrupt_path is not None:
            model_artifacts.append(interrupt_path)
        log_artifacts_mlflow(mlflow_run, model_artifacts, artifact_path="models")
        log_artifacts_clearml(clearml_task, model_artifacts, name_prefix="models")
        if mlflow_run is not None and bool(_get_tracking(mlflow_cfg, "log_tensorboard", True)):
            log_artifacts_mlflow(mlflow_run, [run_dir / "tb"], artifact_path="tensorboard")

        # Финальная сводка эпизода в MLflow (если доступно).
        _log_episode_summary_mlflow(mlflow_run, env, step=model.num_timesteps)

        try:
            write_experiment_result(
                run_dir,
                ExperimentResult(
                    name="train_ppo",
                    seed=int(run_cfg.seed),
                    config=meta,
                    metrics={"last_eval": meta.get("last_eval", {}), "status": meta.get("status", "")},
                    artifacts={
                        "run_dir": str(run_dir),
                        "models": str(run_dir / "models"),
                        "tb": str(run_dir / "tb"),
                        "meta": str(run_dir / "meta"),
                    },
                ),
            )
        except Exception as exc:
            logger.warning("ExperimentResult write failed: %s", exc)

        if mlflow_run is not None:
            try:
                import mlflow

                if curriculum_state_path and curriculum_state_path.exists():
                    state = json.loads(curriculum_state_path.read_text(encoding="utf-8"))
                    stage_name = state.get("stage_name")
                    stage_idx = state.get("stage_idx")
                    mlflow.set_tags(
                        {
                            "curriculum_stage": str(stage_name or stage_idx or ""),
                            "run_status": "interrupted" if interrupted else "completed",
                        }
                    )
            except Exception as exc:
                logger.warning("MLflow tag update failed: %s", exc)

        register_name = str(_get_tracking(mlflow_cfg, "register_model_name", "")).strip()
        if mlflow_run is not None:
            log_mlflow_pyfunc_model(final_path, register_name=register_name or None)

        env.close()
        if run_cfg.no_eval:
            pass

        if clearml_task is not None:
            try:
                clearml_task.close()
            except Exception as exc:
                logger.warning("ClearML task close failed: %s", exc)
        if mlflow_run is not None:
            try:
                import mlflow

                mlflow.end_run()
            except Exception as exc:
                logger.warning("MLflow end_run failed: %s", exc)

        cleanup_run_dir = bool(_get_tracking(mlflow_cfg, "cleanup_run_dir", False))
        if cleanup_run_dir:
            try:
                if run_dir.exists() and run_dir.name.startswith("run_"):
                    shutil.rmtree(run_dir, ignore_errors=True)
            except Exception as exc:
                logger.warning("Cleanup run_dir failed: %s", exc)
        if unused_hydra_run_dir is not None:
            try:
                if unused_hydra_run_dir.exists() and unused_hydra_run_dir.name.startswith("run_"):
                    shutil.rmtree(unused_hydra_run_dir, ignore_errors=True)
            except Exception as exc:
                logger.warning("Cleanup unused hydra dir failed: %s", exc)

    print(f"[ОК] Запуск сохранён в: {run_dir}")
    print(f"     Финальная модель: {final_path}")


def main() -> None:
    _main_hydra()


if __name__ == "__main__":
    main()
