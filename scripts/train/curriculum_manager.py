"""Менеджер curriculum, отделённый от SB3‑callback'ов."""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_ADR_PARAM_MAP = {
    "drag_coeff": ("dr_drag_min", "dr_drag_max"),
    "max_speed": ("dr_max_speed_min", "dr_max_speed_max"),
    "max_accel": ("dr_max_accel_min", "dr_max_accel_max"),
    "mass": ("dr_mass_min", "dr_mass_max"),
}


@dataclass
class CurriculumUpdate:
    log_values: dict[str, float]
    apply_params: dict[str, Any]
    apply_env: bool
    reinit_params: dict[str, Any] | None


class CurriculumManager:
    def __init__(
        self,
        stages: list[dict[str, Any]],
        *,
        metric_key: str = "swarm/finished_frac",
        threshold: float = 0.7,
        min_steps_per_stage: int = 200_000,
        ema_alpha: float = 0.2,
        mark_steps: int = 64,
        goal_radius_decay_steps: int = 500_000,
        shared_curriculum: dict[str, Any] | None = None,
        reinit_on_stage_change: bool = False,
        state_path: Path | None = None,
        state_pickle_path: Path | None = None,
        state_write_interval: int = 5000,
        adr_config: dict[str, Any] | None = None,
        alp_config: dict[str, Any] | None = None,
    ) -> None:
        self.stages = stages or []
        self.metric_key = metric_key
        self.threshold = float(threshold)
        self.min_steps_per_stage = int(min_steps_per_stage)
        self.ema_alpha = float(np.clip(ema_alpha, 0.0, 1.0))
        self.mark_steps = int(max(1, mark_steps))
        self.goal_radius_decay_steps = int(max(1, goal_radius_decay_steps))
        self.current_stage_idx = 0
        self.last_switch_step = 0
        self._ema: float | None = None
        self._mark_left = 0
        self._stage_goal_start: float | None = None
        self._stage_goal_base: float | None = None
        self._stage_goal_decay: int | None = None
        self._last_goal_update = -1
        self._last_goal_value: float | None = None
        self._stage_motion: dict[str, Any] | None = None
        self._motion_speed_start: float | None = None
        self._motion_speed_end: float | None = None
        self._motion_speed_steps: int | None = None
        self._last_motion_update = -1
        self._shared_curriculum = shared_curriculum
        self._reinit_on_stage_change = bool(reinit_on_stage_change)
        self._state_path = state_path
        self._state_pickle_path = state_pickle_path
        self._state_write_interval = int(max(1, state_write_interval))
        self._last_state_write = -1
        self._resume_state: dict[str, Any] | None = None
        self._resume_aux_state: dict[str, Any] | None = None

        self._adr_cfg: dict[str, Any] = {}
        self._adr_enabled = False
        self._adr_metric = self.metric_key
        self._adr_threshold = 0.8
        self._adr_interval = 50_000
        self._adr_step = 0.1
        self._adr_params: dict[str, Any] = {}
        self._adr_scale = 0.0
        self._adr_last_update = -1

        self._alp_cfg: dict[str, Any] = {}
        self._alp_enabled = False
        self._alp_metric = self.metric_key
        self._alp_interval = 50_000
        self._alp_epsilon = 0.1
        self._alp_buckets: list[dict[str, Any]] = []
        self._alp_last_update = -1
        self._alp_current_idx = 0
        self._alp_last_metric: list[float | None] = []
        self._alp_lp: list[float] = []
        self._alp_rng = np.random.default_rng()

        self._set_adr_config(adr_config or {})
        self._set_alp_config(alp_config or {})

    @property
    def has_stages(self) -> bool:
        return bool(self.stages)

    @staticmethod
    def filter_stage_params(stage: dict[str, Any]) -> dict[str, Any]:
        params: dict[str, Any] = {}
        for key, value in stage.items():
            if key in {
                "name",
                "threshold",
                "min_steps",
                "metric_key",
                "goal_radius_start",
                "goal_radius_base",
                "goal_radius_decay_steps",
                "target_motion_speed_start",
                "target_motion_speed_end",
                "target_motion_speed_steps",
                "adr",
                "alp",
                "params",
            }:
                continue
            params[key] = value
        return params

    def _update_shared_curriculum(self, params: dict[str, Any]) -> None:
        if not self._shared_curriculum:
            return
        try:
            payload = dict(self._shared_curriculum.get("params") or {})
            payload.update(params)
            self._shared_curriculum["params"] = payload
            version = int(self._shared_curriculum.get("version", 0))
            self._shared_curriculum["version"] = version + 1
        except Exception as exc:
            logger.warning("Shared curriculum update failed: %s", exc)

    def _load_state(self) -> dict[str, Any] | None:
        if self._state_path is None:
            return None
        try:
            if not self._state_path.exists():
                return None
            raw = self._state_path.read_text(encoding="utf-8")
            data = json.loads(raw)
            if not isinstance(data, dict):
                return None
            return data
        except Exception as exc:
            logger.warning("Curriculum state load failed: %s", exc)
            return None

    def _load_aux_state(self) -> dict[str, Any] | None:
        if self._state_pickle_path is None:
            return None
        try:
            if not self._state_pickle_path.exists():
                return None
            with self._state_pickle_path.open("rb") as f:
                data = pickle.load(f)
            if not isinstance(data, dict):
                return None
            return data
        except Exception as exc:
            logger.warning("Curriculum aux state load failed: %s", exc)
            return None

    def _set_adr_config(self, cfg: dict[str, Any]) -> None:
        self._adr_cfg = dict(cfg or {})
        self._adr_enabled = bool(self._adr_cfg.get("enabled", False))
        self._adr_metric = str(self._adr_cfg.get("metric", self.metric_key))
        self._adr_threshold = float(self._adr_cfg.get("threshold", 0.8))
        self._adr_interval = int(max(1, self._adr_cfg.get("interval_steps", 50_000)))
        self._adr_step = float(self._adr_cfg.get("step", 0.1))
        self._adr_params = dict(self._adr_cfg.get("params") or {})
        self._adr_scale = 0.0
        self._adr_last_update = -1

    def _set_alp_config(self, cfg: dict[str, Any]) -> None:
        self._alp_cfg = dict(cfg or {})
        self._alp_enabled = bool(self._alp_cfg.get("enabled", False))
        self._alp_metric = str(self._alp_cfg.get("metric", self.metric_key))
        self._alp_interval = int(max(1, self._alp_cfg.get("interval_steps", 50_000)))
        self._alp_epsilon = float(self._alp_cfg.get("epsilon", 0.1))
        self._alp_buckets = list(self._alp_cfg.get("buckets") or [])
        self._alp_last_update = -1
        self._alp_current_idx = 0
        self._alp_last_metric = [None for _ in self._alp_buckets]
        self._alp_lp = [0.0 for _ in self._alp_buckets]
        seed = self._alp_cfg.get("seed", None)
        self._alp_rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

    def _save_state(self, num_timesteps: int, force: bool = False) -> None:
        if self._state_path is None:
            return
        if not force and self._last_state_write >= 0:
            if num_timesteps - self._last_state_write < self._state_write_interval:
                return
        state = {
            "stage_idx": int(self.current_stage_idx),
            "stage_name": self.stages[self.current_stage_idx].get("name")
            if self.current_stage_idx < len(self.stages)
            else None,
            "last_switch_step": int(self.last_switch_step),
            "ema": None if self._ema is None else float(self._ema),
            "metric_key": str(self.metric_key),
            "threshold": float(self.threshold),
            "min_steps_per_stage": int(self.min_steps_per_stage),
            "num_timesteps": int(num_timesteps),
            "adr": {
                "enabled": bool(self._adr_enabled),
                "scale": float(self._adr_scale),
                "last_update": int(self._adr_last_update),
                "params": dict(self._adr_params),
            },
            "alp": {
                "enabled": bool(self._alp_enabled),
                "current_idx": int(self._alp_current_idx),
                "last_update": int(self._alp_last_update),
                "last_metric": [None if v is None else float(v) for v in self._alp_last_metric],
                "lp": [float(v) for v in self._alp_lp],
                "bucket_count": len(self._alp_buckets),
            },
        }
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            self._state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
            self._last_state_write = int(num_timesteps)
        except Exception as exc:
            logger.warning("Curriculum state write failed: %s", exc)
        self._save_aux_state()

    def _save_aux_state(self) -> None:
        if self._state_pickle_path is None:
            return
        payload = {
            "adr": {
                "scale": float(self._adr_scale),
                "last_update": int(self._adr_last_update),
                "params": dict(self._adr_params),
            },
            "alp": {
                "current_idx": int(self._alp_current_idx),
                "last_update": int(self._alp_last_update),
                "last_metric": [None if v is None else float(v) for v in self._alp_last_metric],
                "lp": [float(v) for v in self._alp_lp],
                "bucket_count": len(self._alp_buckets),
                "rng_state": getattr(self._alp_rng, "bit_generator", None).state if self._alp_rng is not None else None,
            },
        }
        try:
            self._state_pickle_path.parent.mkdir(parents=True, exist_ok=True)
            with self._state_pickle_path.open("wb") as f:
                pickle.dump(payload, f)
        except Exception as exc:
            logger.warning("Curriculum aux state write failed: %s", exc)

    def _restore_dynamic_state(self, resume_state: dict[str, Any] | None) -> None:
        if not resume_state:
            return
        adr = resume_state.get("adr") if isinstance(resume_state, dict) else None
        if isinstance(adr, dict):
            try:
                self._adr_scale = float(adr.get("scale", self._adr_scale))
                self._adr_last_update = int(adr.get("last_update", self._adr_last_update))
            except Exception:
                pass
        alp = resume_state.get("alp") if isinstance(resume_state, dict) else None
        if isinstance(alp, dict):
            try:
                bucket_count = int(alp.get("bucket_count", len(self._alp_buckets)))
                if bucket_count == len(self._alp_buckets):
                    self._alp_current_idx = int(alp.get("current_idx", self._alp_current_idx))
                    self._alp_last_update = int(alp.get("last_update", self._alp_last_update))
                    last_metric = alp.get("last_metric", None)
                    if isinstance(last_metric, list) and len(last_metric) == len(self._alp_buckets):
                        self._alp_last_metric = [None if v is None else float(v) for v in last_metric]
                    lp = alp.get("lp", None)
                    if isinstance(lp, list) and len(lp) == len(self._alp_buckets):
                        self._alp_lp = [float(v) for v in lp]
            except Exception:
                pass
        aux = self._resume_aux_state
        if isinstance(aux, dict):
            alp_aux = aux.get("alp")
            if isinstance(alp_aux, dict):
                try:
                    rng_state = alp_aux.get("rng_state")
                    if rng_state:
                        self._alp_rng = np.random.default_rng()
                        self._alp_rng.bit_generator.state = rng_state
                except Exception:
                    pass

    def _stage_params(self, idx: int) -> dict[str, Any]:
        stage = self.stages[idx] if idx < len(self.stages) else {}
        params = stage.get("params") if isinstance(stage, dict) else None
        if isinstance(params, dict):
            return self.filter_stage_params(params)
        return self.filter_stage_params(stage)

    def _stage_threshold(self, idx: int) -> float:
        stage = self.stages[idx] if idx < len(self.stages) else {}
        if isinstance(stage, dict) and stage.get("threshold") is not None:
            return float(stage["threshold"])
        return self.threshold

    def _stage_min_steps(self, idx: int) -> int:
        stage = self.stages[idx] if idx < len(self.stages) else {}
        if isinstance(stage, dict) and stage.get("min_steps") is not None:
            return int(stage["min_steps"])
        return self.min_steps_per_stage

    def _stage_metric_key(self, idx: int) -> str:
        stage = self.stages[idx] if idx < len(self.stages) else {}
        if isinstance(stage, dict) and stage.get("metric_key"):
            return str(stage["metric_key"])
        return self.metric_key

    def current_metric_key(self) -> str:
        return self._stage_metric_key(self.current_stage_idx)

    def _apply_stage(self, idx: int, num_timesteps: int, *, initial: bool = False) -> dict[str, Any]:
        params = self._stage_params(idx)
        stage = self.stages[idx] if idx < len(self.stages) else {}
        stage_params = stage.get("params") if isinstance(stage, dict) else None
        stage_view = stage_params if isinstance(stage_params, dict) else stage
        stage_meta = stage if isinstance(stage, dict) else {}

        self._stage_goal_start = stage_view.get("goal_radius_start")
        self._stage_goal_base = stage_view.get("goal_radius_base", stage_view.get("goal_radius"))
        self._stage_goal_decay = stage_view.get("goal_radius_decay_steps", self.goal_radius_decay_steps)
        self._stage_motion = stage_view.get("target_motion")
        self._motion_speed_start = stage_view.get("target_motion_speed_start")
        self._motion_speed_end = stage_view.get("target_motion_speed_end")
        self._motion_speed_steps = stage_view.get("target_motion_speed_steps")
        adr_cfg = None
        alp_cfg = None
        if "adr" in stage_meta:
            adr_cfg = stage_meta.get("adr")
        elif isinstance(stage_view, dict) and "adr" in stage_view:
            adr_cfg = stage_view.get("adr")
        if "alp" in stage_meta:
            alp_cfg = stage_meta.get("alp")
        elif isinstance(stage_view, dict) and "alp" in stage_view:
            alp_cfg = stage_view.get("alp")
        if adr_cfg is not None:
            self._set_adr_config(adr_cfg or {})
        if alp_cfg is not None:
            self._set_alp_config(alp_cfg or {})
        self._last_goal_update = -1
        self._last_goal_value = None
        self._last_motion_update = -1
        self._update_shared_curriculum(params)
        self.current_stage_idx = idx
        if not initial:
            self.last_switch_step = num_timesteps
            self._mark_left = self.mark_steps
        name = self.stages[idx].get("name") if idx < len(self.stages) else None
        name = name or f"stage_{idx}"
        logger.info("[Curriculum] STEP %s: stage -> %s", num_timesteps, name)
        return params

    def _update_adr(self, num_timesteps: int, metric_value: float | None) -> tuple[dict[str, Any], dict[str, float]]:
        if not self._adr_enabled:
            return {}, {}
        if metric_value is None or not np.isfinite(metric_value):
            return {}, {}
        if num_timesteps - self._adr_last_update < self._adr_interval:
            return {}, {}
        if float(metric_value) < float(self._adr_threshold):
            return {}, {}

        self._adr_scale = float(min(1.0, self._adr_scale + self._adr_step))
        self._adr_last_update = int(num_timesteps)
        apply_params: dict[str, Any] = {"domain_randomization": True}
        log_values: dict[str, float] = {"curriculum/adr_scale": float(self._adr_scale)}

        for name, bounds in self._adr_params.items():
            min_key = None
            max_key = None
            if name in _ADR_PARAM_MAP:
                min_key, max_key = _ADR_PARAM_MAP[name]
            try:
                base = float(bounds.get("base")) if bounds.get("base") is not None else None
                min_val = float(bounds.get("min")) if bounds.get("min") is not None else None
                max_val = float(bounds.get("max")) if bounds.get("max") is not None else None
            except Exception:
                continue
            if base is None:
                if min_val is None or max_val is None:
                    continue
                base = 0.5 * (min_val + max_val)
            if min_val is None:
                min_val = base
            if max_val is None:
                max_val = base
            cur_min = base - (base - min_val) * self._adr_scale
            cur_max = base + (max_val - base) * self._adr_scale
            if min_key and max_key:
                apply_params[min_key] = float(cur_min)
                apply_params[max_key] = float(cur_max)
                log_values[f"curriculum/adr_{name}_min"] = float(cur_min)
                log_values[f"curriculum/adr_{name}_max"] = float(cur_max)
            else:
                target = max_val if max_val != base else min_val
                if target is None:
                    continue
                cur_val = float(base + (target - base) * self._adr_scale)
                apply_params[name] = cur_val
                log_values[f"curriculum/adr_{name}"] = cur_val

        if len(apply_params) > 1:
            self._update_shared_curriculum(apply_params)
        return apply_params, log_values

    def _update_alp(
        self, num_timesteps: int, metric_value: float | None, *, force: bool = False
    ) -> tuple[dict[str, Any], dict[str, float]]:
        if not self._alp_enabled or not self._alp_buckets:
            return {}, {}
        if not force and (metric_value is None or not np.isfinite(metric_value)):
            return {}, {}
        if not force and num_timesteps - self._alp_last_update < self._alp_interval:
            return {}, {}

        if metric_value is not None and np.isfinite(metric_value):
            last = self._alp_last_metric[self._alp_current_idx]
            if last is None:
                lp = 0.0
            else:
                lp = abs(float(metric_value) - float(last))
            self._alp_lp[self._alp_current_idx] = float(lp)
            self._alp_last_metric[self._alp_current_idx] = float(metric_value)

        weights = np.array(self._alp_lp, dtype=np.float32) + float(self._alp_epsilon)
        if not np.any(weights > 0):
            weights = np.ones_like(weights, dtype=np.float32)
        probs = weights / float(np.sum(weights))
        self._alp_current_idx = int(self._alp_rng.choice(len(self._alp_buckets), p=probs))
        self._alp_last_update = int(num_timesteps)
        bucket = self._alp_buckets[self._alp_current_idx]
        params = dict(bucket.get("params") or bucket)
        log_values = {
            "curriculum/alp_bucket": float(self._alp_current_idx),
            "curriculum/alp_lp": float(self._alp_lp[self._alp_current_idx]),
        }
        if params:
            self._update_shared_curriculum(params)
        return params, log_values

    def initialize(self, num_timesteps: int) -> CurriculumUpdate:
        if not self.stages:
            return CurriculumUpdate({}, {}, False, None)
        self._resume_state = self._load_state()
        self._resume_aux_state = self._load_aux_state()
        if self._resume_state:
            idx = int(self._resume_state.get("stage_idx", 0))
            if idx < 0 or idx >= len(self.stages):
                idx = 0
            params = self._apply_stage(idx, num_timesteps, initial=True)
            try:
                self.last_switch_step = int(self._resume_state.get("last_switch_step", 0))
            except Exception as exc:
                logger.warning("Curriculum resume last_switch_step failed: %s", exc)
                self.last_switch_step = 0
            try:
                ema_val = self._resume_state.get("ema", None)
                self._ema = float(ema_val) if ema_val is not None else None
            except Exception as exc:
                logger.warning("Curriculum resume ema failed: %s", exc)
                self._ema = None
            self._restore_dynamic_state(self._resume_state)
            logger.info("[Curriculum] RESUME: stage=%s step=%s", idx, num_timesteps)
        else:
            params = self._apply_stage(0, num_timesteps, initial=True)
        log_values: dict[str, float] = {"curriculum/stage_idx": float(self.current_stage_idx)}
        apply_params = dict(params or {})
        apply_env = self._shared_curriculum is None

        adr_params, adr_logs = self._update_adr(num_timesteps, None)
        if adr_params:
            apply_params.update(adr_params)
            apply_env = apply_env or self._shared_curriculum is None
        log_values.update(adr_logs)

        alp_params, alp_logs = self._update_alp(num_timesteps, None, force=True)
        if alp_params:
            apply_params.update(alp_params)
            apply_env = apply_env or self._shared_curriculum is None
        log_values.update(alp_logs)

        self._save_state(num_timesteps, force=True)
        return CurriculumUpdate(log_values, apply_params, apply_env=apply_env, reinit_params=None)

    def step(self, num_timesteps: int, metric_value: float | None) -> CurriculumUpdate:
        if not self.stages:
            return CurriculumUpdate({}, {}, False, None)
        log_values: dict[str, float] = {"curriculum/stage_idx": float(self.current_stage_idx)}
        if self._mark_left > 0:
            log_values["curriculum/switch_flag"] = 1.0
            log_values["curriculum/switch_to"] = float(self.current_stage_idx)
            self._mark_left -= 1

        steps_in_stage = num_timesteps - self.last_switch_step
        apply_params: dict[str, Any] = {}
        apply_env = False
        reinit_params: dict[str, Any] | None = None

        if self._stage_goal_start is not None and self._stage_goal_base is not None:
            decay_steps = int(self._stage_goal_decay or self.goal_radius_decay_steps)
            decay_steps = max(1, decay_steps)
            decay_factor = max(0.0, 1.0 - steps_in_stage / float(decay_steps))
            start_r = float(self._stage_goal_start)
            base_r = float(self._stage_goal_base)
            current_r = base_r + (start_r - base_r) * decay_factor
            if self._last_goal_update < 0 or (num_timesteps - self._last_goal_update) >= 64:
                if self._last_goal_value is None or abs(current_r - self._last_goal_value) > 1e-4:
                    apply_params["goal_radius"] = current_r
                    self._update_shared_curriculum({"goal_radius": current_r})
                    apply_env = True
                    self._last_goal_value = current_r
                    self._last_goal_update = num_timesteps
            log_values["curriculum/goal_radius"] = float(current_r)

        if self._stage_motion and self._motion_speed_start is not None and self._motion_speed_end is not None:
            speed_steps = int(self._motion_speed_steps or self.goal_radius_decay_steps)
            speed_steps = max(1, speed_steps)
            speed_factor = min(1.0, max(0.0, steps_in_stage / float(speed_steps)))
            speed = (
                float(self._motion_speed_start)
                + (float(self._motion_speed_end) - float(self._motion_speed_start)) * speed_factor
            )
            if self._last_motion_update < 0 or (num_timesteps - self._last_motion_update) >= 64:
                motion = dict(self._stage_motion)
                motion["speed"] = speed
                apply_params["target_motion"] = motion
                self._update_shared_curriculum({"target_motion": motion})
                apply_env = True
                self._last_motion_update = num_timesteps
            log_values["curriculum/target_speed"] = float(speed)

        if self.current_stage_idx < len(self.stages) - 1:
            current_val = metric_value
            if current_val is not None:
                try:
                    current_val = float(current_val)
                except Exception as exc:
                    logger.warning("Curriculum metric cast failed: %s", exc)
                    current_val = None
            if current_val is not None and np.isfinite(current_val):
                if self._ema is None:
                    self._ema = float(current_val)
                else:
                    self._ema = self.ema_alpha * float(current_val) + (1.0 - self.ema_alpha) * float(self._ema)
                log_values["curriculum/metric_raw"] = float(current_val)
                log_values["curriculum/metric_ema"] = float(self._ema)
                min_steps = self._stage_min_steps(self.current_stage_idx)
                if num_timesteps - self.last_switch_step >= min_steps:
                    threshold = self._stage_threshold(self.current_stage_idx)
                    if self._ema >= threshold:
                        params = self._apply_stage(self.current_stage_idx + 1, num_timesteps, initial=False)
                        apply_params.update(params)
                        if self._shared_curriculum is None:
                            apply_env = True
                        if self._reinit_on_stage_change:
                            reinit_params = params
                        self._save_state(num_timesteps, force=True)

        adr_metric_val = metric_value if self._adr_metric == self.current_metric_key() else None
        adr_params, adr_logs = self._update_adr(num_timesteps, adr_metric_val)
        if adr_params:
            apply_params.update(adr_params)
            if self._shared_curriculum is None:
                apply_env = True
        log_values.update(adr_logs)

        alp_metric_val = metric_value if self._alp_metric == self.current_metric_key() else None
        alp_params, alp_logs = self._update_alp(num_timesteps, alp_metric_val)
        if alp_params:
            apply_params.update(alp_params)
            if self._shared_curriculum is None:
                apply_env = True
        log_values.update(alp_logs)

        self._save_state(num_timesteps, force=False)
        return CurriculumUpdate(log_values, apply_params, apply_env, reinit_params)
