"""Shared episode metric accumulation for eval/bench/debug rollouts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from env.rewards.cost_schema import COST_KEYS


def aggregate_infos(infos: dict[str, dict[str, Any]]) -> dict[str, float]:
    alive = []
    finished = []
    finished_alive = []
    in_goal = []
    dist = []
    risk = []

    for inf in infos.values():
        if not isinstance(inf, dict) or not inf:
            continue
        alive.append(1.0 if inf.get("alive", False) else 0.0)
        finished.append(1.0 if inf.get("finished", False) else 0.0)
        finished_alive.append(1.0 if inf.get("finished_alive", False) else 0.0)
        in_goal.append(1.0 if inf.get("in_goal", False) else 0.0)
        dist.append(float(inf.get("dist", float("nan"))))
        risk.append(float(inf.get("risk_p", float("nan"))))

    def nanmean(xs: Sequence[float]) -> float:
        arr = np.array(xs, dtype=np.float32)
        return float(np.nanmean(arr)) if arr.size else float("nan")

    alive_sum = float(np.nansum(alive)) if alive else 0.0
    finished_alive_sum = float(np.nansum(finished_alive)) if finished_alive else 0.0
    finished_given_alive = finished_alive_sum / alive_sum if alive_sum > 0.0 else float("nan")

    return {
        "alive_frac": nanmean(alive),
        "finished_frac": nanmean(finished),
        "finished_given_alive": finished_given_alive,
        "in_goal_frac": nanmean(in_goal),
        "mean_dist": nanmean(dist),
        "mean_risk_p": nanmean(risk),
    }


def aggregate_vector_infos(infos: Any) -> dict[str, float]:
    """Aggregate VectorEnv infos with the same core semantics as PettingZoo infos."""

    if isinstance(infos, dict):
        infos_list = [infos]
    else:
        infos_list = list(infos) if infos is not None else []
    payload = {str(idx): inf for idx, inf in enumerate(infos_list) if isinstance(inf, dict)}
    summary = aggregate_infos(payload)
    mind = [float(inf.get("min_neighbor_dist", float("nan"))) for inf in payload.values()]
    mind_arr = np.array(mind, dtype=np.float32)
    mind_arr = mind_arr[np.isfinite(mind_arr) & (mind_arr > 1e-6)]
    summary["min_neighbor_dist_mean"] = float(np.nanmean(mind_arr)) if mind_arr.size else float("nan")
    return summary


@dataclass
class EpisodeMetricsAccumulator:
    n_agents: int
    success_threshold: float = 0.5
    cost_keys: Sequence[str] = tuple(COST_KEYS)
    start_energy: np.ndarray | None = None
    finish_step: list[int | None] = field(default_factory=list)
    first_finish_step: int | None = None
    risk_sum: float = 0.0
    risk_sum_alive: float = 0.0
    smooth_sum: float = 0.0
    smooth_count: int = 0
    cost_sum: dict[str, float] = field(default_factory=dict)
    last_infos: dict[str, dict[str, Any]] | None = None

    @classmethod
    def from_env(cls, env: Any, *, success_threshold: float = 0.5) -> "EpisodeMetricsAccumulator":
        start_energy = None
        try:
            start_energy = np.asarray(env.get_state().energy, dtype=np.float32).copy()
        except Exception:
            start_energy = None
        n_agents = int(getattr(env, "n_agents", len(getattr(env, "possible_agents", []) or [])))
        return cls(
            n_agents=n_agents,
            success_threshold=float(success_threshold),
            start_energy=start_energy,
            finish_step=[None for _ in range(max(n_agents, 0))],
            cost_sum={key: 0.0 for key in COST_KEYS},
        )

    def update(self, env: Any, infos: dict[str, dict[str, Any]], *, decision_step: int) -> None:
        self.last_infos = infos
        agents = list(getattr(env, "possible_agents", []) or [])
        risks = [infos.get(a, {}).get("risk_p", float("nan")) for a in agents]
        if risks:
            self.risk_sum += float(np.nanmean(risks))
        alive_mask = [bool(infos.get(a, {}).get("alive", 0.0)) for a in agents]
        if any(alive_mask):
            self.risk_sum_alive += float(np.nanmean([r for r, alive in zip(risks, alive_mask) if alive]))

        for key in self.cost_keys:
            vals = [infos.get(a, {}).get(key, float("nan")) for a in agents]
            vals = [v for v in vals if v == v]
            if vals:
                self.cost_sum[key] = self.cost_sum.get(key, 0.0) + float(np.mean(vals))

        smooth_vals = [infos.get(a, {}).get("action_smoothness", float("nan")) for a in agents]
        smooth_vals = [v for v in smooth_vals if v == v]
        if smooth_vals:
            self.smooth_sum += float(np.mean(smooth_vals))
            self.smooth_count += 1

        if self.first_finish_step is None:
            for inf in infos.values():
                if isinstance(inf, dict) and inf.get("newly_finished", 0.0):
                    self.first_finish_step = int(decision_step)
                    break
        for idx, agent_id in enumerate(agents):
            if idx >= len(self.finish_step):
                break
            inf = infos.get(agent_id, {})
            if self.finish_step[idx] is None and isinstance(inf, dict) and inf.get("newly_finished", 0.0):
                self.finish_step[idx] = int(decision_step)

    def summary(self, env: Any, *, steps: int) -> dict[str, float]:
        infos = self.last_infos or {}
        agents = list(getattr(env, "possible_agents", []) or [])
        n_agents = max(int(getattr(env, "n_agents", self.n_agents)), 1)
        finished_end = sum(1 for a in agents if infos.get(a, {}).get("finished", 0.0))
        alive_end = sum(1 for a in agents if infos.get(a, {}).get("alive", 0.0))
        finished_frac_end = finished_end / n_agents
        alive_frac_end = alive_end / n_agents
        finished_given_alive_end = finished_end / max(alive_end, 1) if alive_end > 0 else float("nan")

        times = [step for step in self.finish_step if step is not None]
        time_to_goal_mean = float(np.mean(times)) if times else float("nan")

        min_neighbor = [infos.get(a, {}).get("min_neighbor_dist", float("nan")) for a in agents]
        sep_radius = float(getattr(getattr(env, "reward", None), "sep_radius", float("nan")))
        collision_like = (
            float(np.mean([1.0 if m < sep_radius else 0.0 for m in min_neighbor if np.isfinite(m)]))
            if min_neighbor
            else float("nan")
        )

        path_len = [infos.get(a, {}).get("path_len", float("nan")) for a in agents]
        path_len_mean = float(np.nanmean(path_len)) if path_len else float("nan")
        path_ratio_vals = [infos.get(a, {}).get("path_ratio", float("nan")) for a in agents]
        path_ratio_mean = float(np.nanmean(path_ratio_vals)) if path_ratio_vals else float("nan")

        energy_efficiency = float("nan")
        try:
            end_energy = np.asarray(env.get_state().energy, dtype=np.float32)
            if self.start_energy is not None and end_energy.size:
                used = np.maximum(0.0, self.start_energy - end_energy)
                used_mean = float(np.mean(used)) if used.size else float("nan")
                if used_mean > 1e-6 and path_len_mean == path_len_mean:
                    energy_efficiency = path_len_mean / used_mean
        except Exception:
            energy_efficiency = float("nan")

        risk_avg = self.risk_sum_alive / max(steps, 1)
        if risk_avg != risk_avg:
            risk_avg = self.risk_sum / max(steps, 1)
        safety_score = float("nan")
        if risk_avg == risk_avg:
            safety_score = float(1.0 - float(np.clip(risk_avg, 0.0, 1.0)))

        cost_means = {
            key: float(self.cost_sum.get(key, 0.0) / max(steps, 1)) if steps > 0 else float("nan")
            for key in self.cost_keys
        }
        return {
            "finished_count_end": float(finished_end),
            "alive_count_end": float(alive_end),
            "first_finish_step": float(self.first_finish_step if self.first_finish_step is not None else -1),
            "finished_given_alive_end": finished_given_alive_end,
            "success_all_finished": float(finished_end == n_agents),
            "success_any_finished": float(finished_end > 0),
            "success_frac_ge_0_5": float(finished_frac_end >= 0.5),
            "success": float(finished_frac_end >= float(self.success_threshold)),
            "finished_frac_end": float(finished_frac_end),
            "alive_frac_end": float(alive_frac_end),
            "deaths": float(n_agents - alive_end),
            "time_to_goal_mean": time_to_goal_mean,
            "risk_integral_all": self.risk_sum / max(steps, 1),
            "risk_integral_alive": self.risk_sum_alive / max(steps, 1),
            "collision_like": collision_like,
            "episode_len": float(steps),
            "path_ratio": path_ratio_mean,
            "action_smoothness": float(self.smooth_sum / self.smooth_count)
            if self.smooth_count > 0
            else float("nan"),
            "energy_efficiency": energy_efficiency,
            "safety_score": safety_score,
            **cost_means,
        }
