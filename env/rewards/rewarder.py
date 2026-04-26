from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from env.rewards.cost_schema import COST_KEYS, costs_from_parts, ensure_costs
from env.rewards.dto import RewardBreakdown, RewardOutput, RewardRuntimeView
from env.rewards.metrics import MetricsFn
from env.rewards.pipeline import RewardFn
from env.runtime.results import StepResult, ThreatMeta
from env.state import SimState

COMPACT_INFO_KEYS = {
    "dist",
    "alive",
    "in_goal",
    "finished",
    "finished_alive",
    "in_goal_steps",
    "newly_finished",
    "risk_p",
    "min_neighbor_dist",
    "energy",
    "energy_level",
    "path_len",
    "optimal_len",
    "optimality_gap",
    "path_ratio",
    "threat_collided",
    "threat_collisions",
    "min_dist_to_threat",
    "survival_time",
    "start_target_dist",
    "action_smoothness",
    "rew_total",
}

FAST_INFO_KEYS = {
    "dist",
    "alive",
    "in_goal",
    "finished",
    "finished_alive",
    "newly_finished",
    "risk_p",
    "energy_level",
    "rew_total",
}


@dataclass
class RewardConfig:
    """Настройки формирования наград.
    Значения по умолчанию стабильны для PPO.
    """

    # Параметры прогресса к цели.
    w_progress: float = 5.0  # Вес прогресса (уменьшение расстояния до цели).
    w_finish_bonus: float = 50.0  # Бонус за завершение задачи.
    w_in_goal_step: float = 1.0  # Награда за нахождение в зоне цели.
    w_center: float = 1.0  # Дополнительная награда за приближение к центру цели.

    # Параметры безопасности.
    death_penalty: float = 200.0  # Штраф за смерть агента.
    w_risk: float = 2.0  # Штраф за риск столкновения с угрозой.
    w_wall: float = 20.0  # Штраф за приближение к стенам.
    wall_collision_penalty: float = 0.05  # Штраф за удар о стену, масштабируется скоростью.

    # Регуляризация движения.
    w_speed: float = 0.05  # Штраф за скорость, чтобы уменьшить шум.
    brake_dist: float = 10.0  # Дистанция начала торможения при сближении с целью.
    w_action_change: float = 0.05  # Штраф за резкое изменение действия (рывок/jerk).
    w_time: float = 0.0  # Штраф за время (каждый шаг), 0 = отключено.

    # Энергетика.
    w_energy: float = 0.0  # Штраф за расход энергии (нормализованный).
    energy_drain_hover: float = 0.0  # Подхватывается из EnvConfig.battery.drain_hover.
    energy_drain_thrust: float = 0.0  # Подхватывается из EnvConfig.battery.drain_thrust.

    # Параметры дистанции между агентами.
    sep_radius: float = 1.5  # Минимальная дистанция до соседа.
    w_sep: float = 0.0  # Штраф за близость к соседу (временно отключен).
    sep_disable_in_goal: bool = True  # Не штрафовать за близость в зоне цели.

    # Список включённых компонент награды (None = все).
    components: tuple[str, ...] | None = None


class SwarmRewarder:
    def __init__(self, reward_cfg: RewardConfig, *, field_size: float, goal_radius: float) -> None:
        self.reward_cfg = reward_cfg
        self.reward_fn = RewardFn(reward_cfg, field_size, goal_radius)
        self.metrics_fn = MetricsFn()

    def _filter_info(self, info: dict, *, tier: str) -> dict:
        if tier == "full":
            return info
        if tier == "fast":
            filtered = {k: info[k] for k in FAST_INFO_KEYS if k in info}
            return ensure_costs(filtered)
        filtered = {k: info[k] for k in COMPACT_INFO_KEYS if k in info}
        for key in COST_KEYS:
            if key in info:
                filtered[key] = info[key]
        return filtered

    def _info_tier(self, runtime: RewardRuntimeView, *, debug_metrics: bool) -> tuple[str, bool, bool]:
        runtime_mode = str(getattr(runtime.config, "runtime_mode", "full")).strip().lower()
        runtime_fast = runtime_mode in {"train_fast", "fast"}
        debug_mode = str(getattr(runtime.config, "debug_metrics_mode", "full")).lower()
        infos_mode = str(getattr(runtime.config, "infos_mode", "full")).lower()
        full_debug = bool(debug_metrics) and debug_mode != "lite" and not runtime_fast
        if runtime_fast:
            return "fast", full_debug, runtime_fast
        if infos_mode == "compact":
            return "compact", full_debug, runtime_fast
        return "full", full_debug, runtime_fast

    def sync_from_engine(
        self,
        *,
        field_size: float,
        goal_radius: float,
        battery_drain_hover: float | None = None,
        battery_drain_thrust: float | None = None,
    ) -> None:
        self.reward_fn.field_size = float(field_size)
        self.reward_fn.goal_radius = float(goal_radius)
        if battery_drain_hover is not None:
            self.reward_cfg.energy_drain_hover = float(battery_drain_hover)
        if battery_drain_thrust is not None:
            self.reward_cfg.energy_drain_thrust = float(battery_drain_thrust)
        if hasattr(self.reward_fn, "sync"):
            self.reward_fn.sync(field_size=float(field_size), goal_radius=float(goal_radius))

    def build_reset_infos(self, state: SimState, runtime: RewardRuntimeView, agents: list[str]) -> dict[str, dict]:
        infos: dict[str, dict] = {}
        info_tier, _full_debug, runtime_fast = self._info_tier(runtime, debug_metrics=True)
        for i, agent_id in enumerate(agents):
            info = self.metrics_fn(state, i, include_vectors=not runtime_fast)
            if not runtime_fast:
                opt_gap = float(runtime.optimality_gap[i])
                info.update(
                    {
                        "optimal_len": float(runtime.optimal_len[i]),
                        "path_len": float(runtime.path_len[i]),
                        "optimality_gap": opt_gap,
                        "path_ratio": opt_gap,
                        "threat_collisions": int(runtime.threat_collisions[i]),
                        "threat_collided": 0.0,
                        "min_dist_to_threat": float(runtime.min_threat_dist[i]),
                        "survival_time": float("nan"),
                        "start_target_dist": (
                            float(runtime.start_dists[i]) if runtime.start_dists is not None else float("nan")
                        ),
                    }
                )
            infos[agent_id] = self._filter_info(info, tier=info_tier)
        return infos

    def build_step_result(
        self,
        step: StepResult,
        runtime: RewardRuntimeView,
        agents: list[str],
        *,
        debug_metrics: bool = True,
    ) -> RewardOutput:
        prev_state = step.prev_state
        state = step.state
        threat_meta = step.threat_meta
        prev_threat_meta = step.prev_threat_meta
        info_tier, full_debug, runtime_fast = self._info_tier(runtime, debug_metrics=debug_metrics)

        rewards: dict[str, float] = {}
        infos: dict[str, dict] = {}
        terminations: dict[str, bool] = {}
        truncations: dict[str, bool] = {}

        smooth_vals = None
        action_change_sq = None
        if hasattr(state, "last_action") and hasattr(prev_state, "last_action"):
            try:
                delta_all = state.last_action - prev_state.last_action
                action_change_sq = np.sum(delta_all * delta_all, axis=1).astype(np.float32)
                smooth_vals = np.sqrt(action_change_sq).astype(np.float32)
            except Exception:
                smooth_vals = None
                action_change_sq = None

        min_wall = None
        if full_debug or float(getattr(self.reward_cfg, "w_wall", 0.0)) > 0.0:
            try:
                min_wall = np.min(state.walls, axis=1)
            except Exception:
                min_wall = None

        speed_norm = None
        if full_debug or float(getattr(self.reward_cfg, "w_speed", 0.0)) > 0.0:
            try:
                speed_norm = np.linalg.norm(state.vel, axis=1).astype(np.float32)
            except Exception:
                speed_norm = None

        if hasattr(self.reward_fn, "prepare_step"):
            self.reward_fn.prepare_step(
                min_wall=min_wall,
                speed_norm=speed_norm,
                action_change_sq=action_change_sq,
            )

        batch_rewards = None
        batch_parts = None
        if hasattr(self.reward_fn, "compute_all"):
            try:
                batch_rewards, batch_parts = self.reward_fn.compute_all(prev_state, state)
            except Exception:
                batch_rewards, batch_parts = None, None

        for i, agent_id in enumerate(agents):
            info = self.metrics_fn(state, i, include_vectors=not runtime_fast)
            if not runtime_fast:
                info["path_len"] = float(runtime.path_len[i])
                info["optimal_len"] = float(runtime.optimal_len[i])
                opt_gap = float(runtime.optimality_gap[i])
                info["optimality_gap"] = opt_gap
                info["path_ratio"] = opt_gap
                info["threat_collided"] = float(threat_meta.inside_any[i])
                info["threat_collisions"] = int(runtime.threat_collisions[i])
                info["min_dist_to_threat"] = float(runtime.min_threat_dist[i])
                info["survival_time"] = int(runtime.death_step[i]) if runtime.death_step[i] >= 0 else float("nan")
                info["start_target_dist"] = (
                    float(runtime.start_dists[i]) if runtime.start_dists is not None else float("nan")
                )
            if smooth_vals is not None and i < smooth_vals.shape[0]:
                info["action_smoothness"] = float(smooth_vals[i])
            else:
                info["action_smoothness"] = float("nan")

            if full_debug:
                info["died_this_step"] = float(step.died_this_step[i])
                info["dist_to_nearest_threat"] = float(threat_meta.dist[i])
                info["nearest_threat_margin"] = float(threat_meta.margin[i])
                info["nearest_threat_radius"] = float(threat_meta.radius[i])
                info["nearest_threat_intensity"] = float(threat_meta.intensity[i])
                info["nearest_threat_id"] = int(threat_meta.idx[i])
                info["inside_nearest"] = float(threat_meta.margin[i] > 0.0)
                info["inside_any"] = float(threat_meta.inside_any[i])
                info["any_inside_intensity_sum"] = float(threat_meta.inside_any_intensity_sum[i])
                if prev_threat_meta is None:
                    prev_threat_meta = ThreatMeta(
                        dist=np.full_like(threat_meta.dist, np.nan),
                        margin=np.full_like(threat_meta.margin, np.nan),
                        radius=np.full_like(threat_meta.radius, np.nan),
                        intensity=np.full_like(threat_meta.intensity, np.nan),
                        idx=np.full_like(threat_meta.idx, -1),
                        inside_any=np.zeros_like(threat_meta.inside_any, dtype=bool),
                        inside_any_intensity_sum=np.zeros_like(threat_meta.inside_any_intensity_sum, dtype=np.float32),
                    )
                info["dist_to_nearest_threat_before"] = float(prev_threat_meta.dist[i])
                info["nearest_threat_margin_before"] = float(prev_threat_meta.margin[i])
                info["nearest_threat_radius_before"] = float(prev_threat_meta.radius[i])
                info["nearest_threat_intensity_before"] = float(prev_threat_meta.intensity[i])
                info["nearest_threat_id_before"] = int(prev_threat_meta.idx[i])
                info["inside_nearest_before"] = float(prev_threat_meta.margin[i] > 0.0)
                info["inside_any_before"] = float(prev_threat_meta.inside_any[i])
                info["any_inside_intensity_sum_before"] = float(prev_threat_meta.inside_any_intensity_sum[i])
                if min_wall is not None and i < min_wall.shape[0]:
                    info["min_wall_dist"] = float(min_wall[i])
                else:
                    info["min_wall_dist"] = float(np.min(state.walls[i]))
                info["alive_before"] = float(step.alive_before[i])
                info["alive_after"] = float(state.alive[i])
                info["pos_before"] = prev_state.pos[i].astype(np.float32).tolist()
                info["pos_after"] = state.pos[i].astype(np.float32).tolist()
                info["risk_p_before"] = float(prev_state.risk_p[i])
                info["risk_p_after"] = float(state.risk_p[i])
                info["collision_speed"] = float(state.collision_speed[i])
                info["collided"] = float(runtime.last_collision[i])

            if batch_rewards is not None and batch_parts is not None:
                reward_val = float(batch_rewards[i])
                parts = {k: float(v[i]) for k, v in batch_parts.items()}
                rewards[agent_id] = reward_val
                info["rew_total"] = reward_val
                for key, val in parts.items():
                    info[key] = float(val)
                info.update(costs_from_parts(parts, include_time=True))
            else:
                breakdown = RewardBreakdown.from_output(self.reward_fn(prev_state, state, i))
                rewards[agent_id] = float(breakdown.total)
                info["rew_total"] = float(breakdown.total)
                if breakdown.parts:
                    for key, val in breakdown.parts.items():
                        info[key] = float(val)

                # Унифицированные cost‑метрики: отрицание reward‑частей.
                info.update(costs_from_parts(breakdown.parts, include_time=True))

            terminations[agent_id] = bool(step.done and (not step.is_timeout))
            truncations[agent_id] = bool(step.done and step.is_timeout)
            infos[agent_id] = self._filter_info(ensure_costs(info), tier=info_tier)

        return RewardOutput(
            rewards=rewards,
            infos=infos,
            terminations=terminations,
            truncations=truncations,
        )

    def build_step_outputs(
        self,
        step: StepResult,
        runtime: RewardRuntimeView,
        agents: list[str],
        *,
        debug_metrics: bool = True,
    ) -> tuple[dict[str, float], dict[str, dict], dict[str, bool], dict[str, bool]]:
        out = self.build_step_result(step, runtime, agents, debug_metrics=debug_metrics)
        return out.as_tuple()
