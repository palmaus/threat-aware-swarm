"""Training callbacks for the PPO entrypoint."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf
from stable_baselines3.common.callbacks import BaseCallback

from scripts.common.logging_utils import collect_system_metrics, log_metrics
from scripts.train.curriculum_manager import CurriculumManager
from scripts.train.env_methods import call_env_method

logger = logging.getLogger(__name__)


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


class SwarmInfoMetricsCallback(BaseCallback):
    """
    Logs swarm metrics from infos into TensorBoard under the standard swarm/* keys.
    """

    def __init__(self, log_every_steps: int = 2048, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.log_every_steps = int(log_every_steps)
        self._last_log_step = 0
        self._collisions_since_log = 0
        self._steps_since_log = 0
        self._threat_collisions_since_log = 0
        self._threat_steps_since_log = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", None)
        if not infos:
            return True

        for inf in infos:
            if not isinstance(inf, dict) or not inf:
                continue
            collided = inf.get("collided", None)
            if collided is None:
                collided = _safe_float(inf.get("collision_speed", 0.0), 0.0) > 0.0
            if bool(collided):
                self._collisions_since_log += 1
            threat_hit = inf.get("threat_collided", None)
            if threat_hit is None:
                threat_hit = _safe_float(inf.get("inside_any", 0.0), 0.0) > 0.0
            if bool(threat_hit):
                self._threat_collisions_since_log += 1
        self._steps_since_log += 1
        self._threat_steps_since_log += 1

        if self.num_timesteps - self._last_log_step < self.log_every_steps:
            return True

        def _flag_or_nan(inf: dict[str, Any], key: str) -> float:
            if key not in inf:
                return float("nan")
            val = inf.get(key)
            if isinstance(val, bool):
                return 1.0 if val else 0.0
            return _safe_float(val, float("nan"))

        flag_keys = [
            "alive",
            "finished",
            "finished_alive",
            "newly_finished",
            "in_goal",
            "collided",
            "threat_collided",
        ]
        float_keys = [
            "in_goal_steps",
            "dist",
            "risk_p",
            "min_neighbor_dist",
            "min_dist_to_threat",
            "survival_time",
            "start_target_dist",
            "action_smoothness",
            "rew_total",
            "rew_progress",
            "rew_in_goal",
            "rew_center",
            "rew_finish_bonus",
            "rew_risk",
            "rew_wall",
            "rew_collision",
            "rew_speed",
            "rew_sep",
            "rew_death",
        ]
        values: dict[str, list[float]] = {key: [] for key in flag_keys + float_keys + ["path_ratio"]}
        for inf in infos:
            if not isinstance(inf, dict) or not inf:
                for key in values:
                    values[key].append(float("nan"))
                continue
            for key in flag_keys:
                values[key].append(_flag_or_nan(inf, key))
            for key in float_keys:
                values[key].append(_safe_float(inf.get(key, np.nan)))
            if "path_ratio" in inf:
                values["path_ratio"].append(_safe_float(inf.get("path_ratio", np.nan)))
            else:
                values["path_ratio"].append(_safe_float(inf.get("optimality_gap", np.nan)))

        def nanmean(xs: list[float]) -> float:
            arr = np.array(xs, dtype=np.float32)
            return float(np.nanmean(arr)) if arr.size else float("nan")

        alive_frac = nanmean(values["alive"])
        finished_frac = nanmean(values["finished"])
        finished_alive_frac = nanmean(values["finished_alive"])
        alive_arr = np.array(values["alive"], dtype=np.float32)
        finished_alive_arr = np.array(values["finished_alive"], dtype=np.float32)
        valid = np.isfinite(alive_arr) & np.isfinite(finished_alive_arr)
        if valid.any():
            alive_sum = float(np.nansum(alive_arr[valid]))
            finished_alive_sum = float(np.nansum(finished_alive_arr[valid]))
            finished_given_alive = finished_alive_sum / alive_sum if alive_sum > 0.0 else float("nan")
        else:
            finished_given_alive = float("nan")

        self.logger.record("swarm/alive_frac", alive_frac)
        self.logger.record("swarm/finished_frac", finished_frac)
        self.logger.record("swarm/finished_alive_frac", finished_alive_frac)
        self.logger.record("swarm/finished_given_alive", finished_given_alive)
        self.logger.record("swarm/newly_finished_frac", nanmean(values["newly_finished"]))
        self.logger.record("swarm/mean_in_goal_steps", nanmean(values["in_goal_steps"]))
        self.logger.record("swarm/in_goal_frac", nanmean(values["in_goal"]))
        self.logger.record("swarm/mean_dist", nanmean(values["dist"]))
        self.logger.record("swarm/mean_risk_p", nanmean(values["risk_p"]))
        self.logger.record("swarm/path_ratio", nanmean(values["path_ratio"]))
        self.logger.record("swarm/collision_frac", nanmean(values["collided"]))
        self.logger.record("swarm/collisions_total", float(self._collisions_since_log))
        if self._steps_since_log > 0:
            self.logger.record(
                "swarm/collisions_per_step", float(self._collisions_since_log) / float(self._steps_since_log)
            )
        self.logger.record("swarm/threat_collision_frac", nanmean(values["threat_collided"]))
        self.logger.record("swarm/threat_collisions_total", float(self._threat_collisions_since_log))
        if self._threat_steps_since_log > 0:
            self.logger.record(
                "swarm/threat_collisions_per_step",
                float(self._threat_collisions_since_log) / float(self._threat_steps_since_log),
            )

        dist_arr = np.array(values["min_dist_to_threat"], dtype=np.float32)
        dist_arr = dist_arr[np.isfinite(dist_arr)]
        self.logger.record(
            "swarm/min_dist_to_threat_mean", float(np.nanmean(dist_arr)) if dist_arr.size else float("nan")
        )

        st_arr = np.array(values["survival_time"], dtype=np.float32)
        st_arr = st_arr[np.isfinite(st_arr)]
        self.logger.record("swarm/survival_time_mean", float(np.nanmean(st_arr)) if st_arr.size else float("nan"))

        std_arr = np.array(values["start_target_dist"], dtype=np.float32)
        std_arr = std_arr[np.isfinite(std_arr)]
        self.logger.record("swarm/start_target_dist_mean", float(np.nanmean(std_arr)) if std_arr.size else float("nan"))

        mind_arr = np.array(values["min_neighbor_dist"], dtype=np.float32)
        mind_arr = mind_arr[np.isfinite(mind_arr) & (mind_arr > 1e-6)]
        self.logger.record(
            "swarm/min_neighbor_dist_mean", float(np.nanmean(mind_arr)) if mind_arr.size else float("nan")
        )
        self.logger.record("swarm/action_smoothness", nanmean(values["action_smoothness"]))

        self.logger.record("reward/total", nanmean(values["rew_total"]))
        self.logger.record("reward/progress", nanmean(values["rew_progress"]))
        self.logger.record("reward/in_goal", nanmean(values["rew_in_goal"]))
        self.logger.record("reward/center", nanmean(values["rew_center"]))
        self.logger.record("reward/finish_bonus", nanmean(values["rew_finish_bonus"]))
        self.logger.record("reward/risk", nanmean(values["rew_risk"]))
        self.logger.record("reward/wall", nanmean(values["rew_wall"]))
        self.logger.record("reward/collision", nanmean(values["rew_collision"]))
        self.logger.record("reward/speed", nanmean(values["rew_speed"]))
        self.logger.record("reward/sep", nanmean(values["rew_sep"]))
        self.logger.record("reward/death", nanmean(values["rew_death"]))

        self._last_log_step = self.num_timesteps
        self._collisions_since_log = 0
        self._steps_since_log = 0
        self._threat_collisions_since_log = 0
        self._threat_steps_since_log = 0
        return True


class SwarmCurriculumCallback(BaseCallback):
    """Thin SB3 callback delegating curriculum state to CurriculumManager."""

    def __init__(
        self,
        stages: list[dict[str, Any]],
        metric_key: str = "swarm/finished_frac",
        threshold: float = 0.7,
        min_steps_per_stage: int = 200_000,
        ema_alpha: float = 0.2,
        mark_steps: int = 64,
        goal_radius_decay_steps: int = 500_000,
        adr_config: dict[str, Any] | None = None,
        alp_config: dict[str, Any] | None = None,
        shared_curriculum: dict[str, Any] | None = None,
        reinit_env_fn=None,
        reinit_on_stage_change: bool = False,
        state_path: Path | None = None,
        state_pickle_path: Path | None = None,
        state_write_interval: int = 5000,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self._manager = CurriculumManager(
            stages=stages,
            metric_key=metric_key,
            threshold=threshold,
            min_steps_per_stage=min_steps_per_stage,
            ema_alpha=ema_alpha,
            mark_steps=mark_steps,
            goal_radius_decay_steps=goal_radius_decay_steps,
            adr_config=adr_config,
            alp_config=alp_config,
            shared_curriculum=shared_curriculum,
            reinit_on_stage_change=reinit_on_stage_change,
            state_path=state_path,
            state_pickle_path=state_pickle_path,
            state_write_interval=state_write_interval,
        )
        self._reinit_env_fn = reinit_env_fn
        self._pending_reinit = False
        self._pending_stage_params: dict[str, Any] | None = None

    def _apply_params(self, params: dict[str, Any], apply_env: bool) -> None:
        if not params:
            return
        if apply_env:
            try:
                call_env_method(self.training_env, "apply_curriculum", params)
            except Exception as exc:
                logger.warning("Curriculum apply failed: %s", exc)

    def _on_rollout_end(self) -> None:
        if not self._pending_reinit or not self._reinit_env_fn:
            return
        try:
            new_env = self._reinit_env_fn(self._pending_stage_params or {})
            try:
                self.training_env.close()
            except Exception as exc:
                logger.warning("Curriculum env close failed: %s", exc)
            self.model.set_env(new_env)
            self.training_env = new_env
            logger.info("[Curriculum] Env reinitialized for stage %s", self._manager.current_stage_idx)
        except Exception as exc:
            logger.warning("[Curriculum] Env reinit failed: %s", exc)
        finally:
            self._pending_reinit = False
            self._pending_stage_params = None

    def _on_training_start(self) -> None:
        if not self._manager.has_stages:
            return
        update = self._manager.initialize(self.num_timesteps)
        for key, value in update.log_values.items():
            self.logger.record(key, float(value))
        self._apply_params(update.apply_params, update.apply_env)

    def _on_step(self) -> bool:
        if not self._manager.has_stages:
            return True
        metric_key = self._manager.current_metric_key()
        metric_value = self.logger.name_to_value.get(metric_key)
        update = self._manager.step(self.num_timesteps, metric_value)
        for key, value in update.log_values.items():
            self.logger.record(key, float(value))
        self._apply_params(update.apply_params, update.apply_env)
        if update.reinit_params is not None and self._reinit_env_fn is not None:
            self._pending_reinit = True
            self._pending_stage_params = dict(update.reinit_params)
        return True


def _load_curriculum_config(path: Path, profile: str | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cfg = OmegaConf.load(path)
    data = OmegaConf.to_container(cfg, resolve=True)
    if data is None:
        return [], {}
    if isinstance(data, dict) and "curriculum" in data:
        data = data["curriculum"]

    if isinstance(data, dict):
        meta = {k: v for k, v in data.items() if k != "stages"}
        if "stages" in data:
            return list(data.get("stages") or []), meta
        if profile and profile in data and isinstance(data[profile], dict):
            prof = data[profile]
            if "stages" in prof:
                meta = {**meta, **{k: v for k, v in prof.items() if k != "stages"}}
                return list(prof.get("stages") or []), meta
    if isinstance(data, list):
        return list(data), {}
    return [], {}


class SaveRunMetaCallback(BaseCallback):
    """Writes meta/run.json once and keeps it updated with eval summaries."""

    def __init__(self, run_dir: Path, meta: dict[str, Any], verbose: int = 0):
        super().__init__(verbose=verbose)
        self.run_dir = run_dir
        self.meta_path = run_dir / "meta" / "run.json"
        self.meta = meta

    def _on_training_start(self) -> None:
        self.meta_path.parent.mkdir(parents=True, exist_ok=True)
        self.meta_path.write_text(json.dumps(self.meta, ensure_ascii=False, indent=2), encoding="utf-8")

    def update_eval(self, eval_dict: dict[str, Any]) -> None:
        self.meta.setdefault("eval_history", [])
        self.meta["eval_history"].append(eval_dict)
        self.meta["last_eval"] = eval_dict
        try:
            self.meta_path.write_text(json.dumps(self.meta, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.warning("Meta write failed: %s", exc)

    def _on_step(self) -> bool:
        return True


class TrackingMetricsCallback(BaseCallback):
    def __init__(self, mlflow_run, clearml_task, log_every_steps: int = 2048, system_metrics: bool = False):
        super().__init__()
        self.mlflow_run = mlflow_run
        self.clearml_task = clearml_task
        self.log_every_steps = int(log_every_steps)
        self.system_metrics = bool(system_metrics)
        self._last_log_step = 0

    def _on_step(self) -> bool:
        if self.log_every_steps <= 0:
            return True
        if (self.num_timesteps - self._last_log_step) < self.log_every_steps:
            return True
        self._last_log_step = self.num_timesteps
        metrics: dict[str, float] = {}
        for key, value in self.logger.name_to_value.items():
            if isinstance(value, (int, float, np.floating)):
                metrics[key] = float(value)
        if self.system_metrics:
            metrics.update(collect_system_metrics())
        if metrics:
            log_metrics(self.mlflow_run, self.clearml_task, metrics, step=self.num_timesteps)
        return True
