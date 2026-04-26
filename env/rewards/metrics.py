"""Сбор компактных метрик для info, пригодных для логирования и UI."""

from __future__ import annotations

from env.state import SimState


class MetricsFn:
    """Собирает минимальный набор метрик для info без тяжёлых вычислений."""

    def __call__(self, state: SimState, idx: int, *, include_vectors: bool = True) -> dict:
        info = {
            "dist": float(state.dists[idx]),
            "alive": float(state.alive[idx]),
            "in_goal": float(state.in_goal[idx]),
            "finished": float(state.finished[idx]),
            "finished_alive": float(bool(state.finished[idx]) and bool(state.alive[idx])),
            "in_goal_steps": int(state.in_goal_steps[idx]),
            "newly_finished": float(state.newly_finished[idx]),
            "risk_p": float(state.risk_p[idx]),
            "min_neighbor_dist": float(state.min_neighbor_dist[idx]),
            "energy": float(state.energy[idx]) if state.energy is not None else float("nan"),
            "energy_level": float(state.energy_level[idx]) if state.energy_level is not None else float("nan"),
        }
        if include_vectors:
            info["target_pos"] = state.target_pos
            info["target_vel"] = state.target_vel
            info["pos"] = state.pos[idx]
        return info
