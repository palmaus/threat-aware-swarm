from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from env.state import SimState


@dataclass
class RewardContext:
    reward: object
    field_size: float
    goal_radius: float
    min_wall: np.ndarray | None = None
    speed_norm: np.ndarray | None = None
    action_change_sq: np.ndarray | None = None


class RewardComponent:
    """Базовый компонент награды."""

    name = ""
    key = ""

    def compute(self, prev: SimState, cur: SimState, idx: int, ctx: RewardContext) -> float:
        raise NotImplementedError


class DeathPenalty(RewardComponent):
    name = "death"
    key = "rew_death"

    def compute(self, prev: SimState, cur: SimState, idx: int, ctx: RewardContext) -> float:
        died = bool(prev.alive[idx]) and (not bool(cur.alive[idx]))
        if not died:
            return 0.0
        return -float(ctx.reward.death_penalty)


class ProgressReward(RewardComponent):
    name = "progress"
    key = "rew_progress"

    def compute(self, prev: SimState, cur: SimState, idx: int, ctx: RewardContext) -> float:
        if not bool(cur.alive[idx]) or bool(cur.finished[idx]):
            return 0.0
        return float(ctx.reward.w_progress) * float(prev.dists[idx] - cur.dists[idx])


class InGoalReward(RewardComponent):
    name = "in_goal"
    key = "rew_in_goal"

    def compute(self, _prev: SimState, cur: SimState, idx: int, ctx: RewardContext) -> float:
        if not bool(cur.alive[idx]) or bool(cur.finished[idx]):
            return 0.0
        if not bool(cur.in_goal[idx]):
            return 0.0
        return float(ctx.reward.w_in_goal_step)


class CenterReward(RewardComponent):
    name = "center"
    key = "rew_center"

    def compute(self, _prev: SimState, cur: SimState, idx: int, ctx: RewardContext) -> float:
        if not bool(cur.alive[idx]) or bool(cur.finished[idx]):
            return 0.0
        if not bool(cur.in_goal[idx]):
            return 0.0
        center_bonus = max(0.0, 1.0 - float(cur.dists[idx] / max(ctx.goal_radius, 1e-6)))
        return float(ctx.reward.w_center) * center_bonus


class FinishBonusReward(RewardComponent):
    name = "finish_bonus"
    key = "rew_finish_bonus"

    def compute(self, _prev: SimState, cur: SimState, idx: int, ctx: RewardContext) -> float:
        if not bool(cur.alive[idx]):
            return 0.0
        if not bool(cur.newly_finished[idx]):
            return 0.0
        return float(ctx.reward.w_finish_bonus)


class RiskPenalty(RewardComponent):
    name = "risk"
    key = "rew_risk"

    def compute(self, _prev: SimState, cur: SimState, idx: int, ctx: RewardContext) -> float:
        if not bool(cur.alive[idx]) or bool(cur.finished[idx]):
            return 0.0
        return -float(ctx.reward.w_risk) * float(cur.risk_p[idx])


class WallPenalty(RewardComponent):
    name = "wall"
    key = "rew_wall"

    def compute(self, _prev: SimState, cur: SimState, idx: int, ctx: RewardContext) -> float:
        if not bool(cur.alive[idx]) or bool(cur.finished[idx]):
            return 0.0
        if ctx.min_wall is not None and idx < ctx.min_wall.shape[0]:
            min_wall = float(ctx.min_wall[idx])
        else:
            walls = cur.walls[idx]
            min_wall = float(np.min(walls))
        if min_wall >= 0.05:
            return 0.0
        return -float(ctx.reward.w_wall) * (0.05 - min_wall)


class CollisionPenalty(RewardComponent):
    name = "collision"
    key = "rew_collision"

    def compute(self, _prev: SimState, cur: SimState, idx: int, ctx: RewardContext) -> float:
        if not bool(cur.alive[idx]) or bool(cur.finished[idx]):
            return 0.0
        impact = float(cur.collision_speed[idx])
        if impact <= 0.0:
            return 0.0
        return -float(ctx.reward.wall_collision_penalty) * impact


class SpeedPenalty(RewardComponent):
    name = "speed"
    key = "rew_speed"

    def compute(self, _prev: SimState, cur: SimState, idx: int, ctx: RewardContext) -> float:
        if not bool(cur.alive[idx]) or bool(cur.finished[idx]):
            return 0.0
        if float(cur.dists[idx]) >= float(ctx.reward.brake_dist):
            return 0.0
        if ctx.speed_norm is not None and idx < ctx.speed_norm.shape[0]:
            speed = float(ctx.speed_norm[idx])
        else:
            speed = float(np.linalg.norm(cur.vel[idx]))
        gate = max(
            0.0, (float(ctx.reward.brake_dist) - float(cur.dists[idx])) / max(float(ctx.reward.brake_dist), 1e-6)
        )
        return -float(ctx.reward.w_speed) * speed * gate


class SeparationPenalty(RewardComponent):
    name = "separation"
    key = "rew_sep"

    def compute(self, _prev: SimState, cur: SimState, idx: int, ctx: RewardContext) -> float:
        if not bool(cur.alive[idx]) or bool(cur.finished[idx]):
            return 0.0
        if float(ctx.reward.w_sep) <= 0.0:
            return 0.0
        if ctx.reward.sep_disable_in_goal and bool(cur.in_goal[idx]):
            return 0.0
        md = float(cur.min_neighbor_dist[idx])
        if not np.isfinite(md) or md >= float(ctx.reward.sep_radius):
            return 0.0
        return -float(ctx.reward.w_sep) * (float(ctx.reward.sep_radius) - md) / max(float(ctx.reward.sep_radius), 1e-6)


class ActionChangePenalty(RewardComponent):
    name = "action_change"
    key = "rew_action_change"

    def compute(self, prev: SimState, cur: SimState, idx: int, ctx: RewardContext) -> float:
        if not bool(cur.alive[idx]) or bool(cur.finished[idx]):
            return 0.0
        if float(getattr(ctx.reward, "w_action_change", 0.0)) <= 0.0:
            return 0.0
        if ctx.action_change_sq is not None and idx < ctx.action_change_sq.shape[0]:
            delta_sq = float(ctx.action_change_sq[idx])
        else:
            delta = cur.last_action[idx] - prev.last_action[idx]
            delta_sq = float(np.dot(delta, delta))
        return -float(ctx.reward.w_action_change) * delta_sq


class TimePenalty(RewardComponent):
    name = "time"
    key = "rew_time"

    def compute(self, _prev: SimState, cur: SimState, idx: int, ctx: RewardContext) -> float:
        if not bool(cur.alive[idx]) or bool(cur.finished[idx]):
            return 0.0
        if float(getattr(ctx.reward, "w_time", 0.0)) <= 0.0:
            return 0.0
        return -float(ctx.reward.w_time)


class EnergyPenalty(RewardComponent):
    name = "energy"
    key = "rew_energy"

    def compute(self, prev: SimState, cur: SimState, idx: int, ctx: RewardContext) -> float:
        if not bool(cur.alive[idx]) or bool(cur.finished[idx]):
            return 0.0
        if float(getattr(ctx.reward, "w_energy", 0.0)) <= 0.0:
            return 0.0
        if not (hasattr(cur, "energy") and hasattr(prev, "energy")):
            return 0.0
        drain = float(max(0.0, float(prev.energy[idx]) - float(cur.energy[idx])))
        max_drain = (
            float(getattr(ctx.reward, "energy_drain_hover", 0.0))
            + float(getattr(cur, "max_thrust", 0.0)) * float(getattr(ctx.reward, "energy_drain_thrust", 0.0))
        ) * float(getattr(cur, "dt", 0.0))
        if max_drain <= 0.0:
            return 0.0
        return -float(ctx.reward.w_energy) * (drain / max_drain)


class RewardPipeline:
    """Компонентная награда с возможностью включения/выключения через конфиг."""

    def __init__(self, reward_cfg, field_size: float, goal_radius: float) -> None:
        self.ctx = RewardContext(reward=reward_cfg, field_size=float(field_size), goal_radius=float(goal_radius))
        self.registry = {
            "death": DeathPenalty(),
            "progress": ProgressReward(),
            "in_goal": InGoalReward(),
            "center": CenterReward(),
            "finish_bonus": FinishBonusReward(),
            "risk": RiskPenalty(),
            "wall": WallPenalty(),
            "collision": CollisionPenalty(),
            "speed": SpeedPenalty(),
            "separation": SeparationPenalty(),
            "action_change": ActionChangePenalty(),
            "time": TimePenalty(),
            "energy": EnergyPenalty(),
        }
        self._all_components = list(self.registry.values())
        configured = getattr(reward_cfg, "components", None)
        if configured:
            enabled = [str(name) for name in configured]
            self.components = [self.registry[name] for name in enabled if name in self.registry]
        else:
            self.components = list(self._all_components)
        self._enabled_names = {comp.name for comp in self.components}
        self._all_keys = [comp.key for comp in self._all_components]

    def set_step_cache(
        self,
        *,
        min_wall: np.ndarray | None = None,
        speed_norm: np.ndarray | None = None,
        action_change_sq: np.ndarray | None = None,
    ) -> None:
        self.ctx.min_wall = min_wall
        self.ctx.speed_norm = speed_norm
        self.ctx.action_change_sq = action_change_sq

    def __call__(self, prev: SimState, cur: SimState, idx: int) -> tuple[float, dict]:
        rew = 0.0
        parts = dict.fromkeys(self._all_keys, 0.0)
        for comp in self.components:
            val = float(comp.compute(prev, cur, idx, self.ctx))
            if val == 0.0:
                continue
            rew += val
            parts[comp.key] += val
        return float(rew), parts

    def compute_all(self, prev: SimState, cur: SimState) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        n = int(np.asarray(cur.alive).shape[0])
        parts: dict[str, np.ndarray] = {k: np.zeros(n, dtype=np.float32) for k in self._all_keys}
        enabled = self._enabled_names
        active = np.asarray(cur.alive, dtype=bool) & (~np.asarray(cur.finished, dtype=bool))

        if "death" in enabled:
            died = np.asarray(prev.alive, dtype=bool) & (~np.asarray(cur.alive, dtype=bool))
            parts["rew_death"] = (-float(self.ctx.reward.death_penalty) * died.astype(np.float32)).astype(np.float32)

        if "progress" in enabled:
            delta = (np.asarray(prev.dists, dtype=np.float32) - np.asarray(cur.dists, dtype=np.float32))
            parts["rew_progress"] = float(self.ctx.reward.w_progress) * delta * active.astype(np.float32)

        if "in_goal" in enabled:
            mask = active & np.asarray(cur.in_goal, dtype=bool)
            parts["rew_in_goal"] = float(self.ctx.reward.w_in_goal_step) * mask.astype(np.float32)

        if "center" in enabled:
            mask = active & np.asarray(cur.in_goal, dtype=bool)
            center_bonus = 1.0 - (np.asarray(cur.dists, dtype=np.float32) / max(float(self.ctx.goal_radius), 1e-6))
            center_bonus = np.clip(center_bonus, 0.0, 1.0)
            parts["rew_center"] = float(self.ctx.reward.w_center) * center_bonus * mask.astype(np.float32)

        if "finish_bonus" in enabled:
            mask = np.asarray(cur.alive, dtype=bool) & np.asarray(cur.newly_finished, dtype=bool)
            parts["rew_finish_bonus"] = float(self.ctx.reward.w_finish_bonus) * mask.astype(np.float32)

        if "risk" in enabled:
            parts["rew_risk"] = -float(self.ctx.reward.w_risk) * np.asarray(cur.risk_p, dtype=np.float32) * active.astype(
                np.float32
            )

        if "wall" in enabled:
            if self.ctx.min_wall is not None:
                min_wall = np.asarray(self.ctx.min_wall, dtype=np.float32)
            else:
                min_wall = np.min(np.asarray(cur.walls, dtype=np.float32), axis=1)
            mask = active & (min_wall < 0.05)
            parts["rew_wall"] = (
                -float(self.ctx.reward.w_wall) * (0.05 - min_wall) * mask.astype(np.float32)
            )

        if "collision" in enabled:
            impact = np.asarray(cur.collision_speed, dtype=np.float32)
            mask = active & (impact > 0.0)
            parts["rew_collision"] = -float(self.ctx.reward.wall_collision_penalty) * impact * mask.astype(np.float32)

        if "speed" in enabled:
            if float(self.ctx.reward.w_speed) > 0.0:
                if self.ctx.speed_norm is not None:
                    speed = np.asarray(self.ctx.speed_norm, dtype=np.float32)
                else:
                    speed = np.linalg.norm(np.asarray(cur.vel, dtype=np.float32), axis=1)
                dists = np.asarray(cur.dists, dtype=np.float32)
                gate = (float(self.ctx.reward.brake_dist) - dists) / max(float(self.ctx.reward.brake_dist), 1e-6)
                gate = np.clip(gate, 0.0, 1.0)
                mask = active & (dists < float(self.ctx.reward.brake_dist))
                parts["rew_speed"] = -float(self.ctx.reward.w_speed) * speed * gate * mask.astype(np.float32)

        if "separation" in enabled:
            if float(self.ctx.reward.w_sep) > 0.0:
                md = np.asarray(cur.min_neighbor_dist, dtype=np.float32)
                mask = active.copy()
                if bool(getattr(self.ctx.reward, "sep_disable_in_goal", True)):
                    mask &= ~np.asarray(cur.in_goal, dtype=bool)
                mask &= np.isfinite(md) & (md < float(self.ctx.reward.sep_radius))
                parts["rew_sep"] = (
                    -float(self.ctx.reward.w_sep)
                    * (float(self.ctx.reward.sep_radius) - md)
                    / max(float(self.ctx.reward.sep_radius), 1e-6)
                    * mask.astype(np.float32)
                )

        if "action_change" in enabled:
            if float(getattr(self.ctx.reward, "w_action_change", 0.0)) > 0.0:
                if self.ctx.action_change_sq is not None:
                    delta_sq = np.asarray(self.ctx.action_change_sq, dtype=np.float32)
                else:
                    delta = np.asarray(cur.last_action, dtype=np.float32) - np.asarray(prev.last_action, dtype=np.float32)
                    delta_sq = np.sum(delta * delta, axis=1).astype(np.float32)
                parts["rew_action_change"] = -float(self.ctx.reward.w_action_change) * delta_sq * active.astype(
                    np.float32
                )

        if "time" in enabled:
            if float(getattr(self.ctx.reward, "w_time", 0.0)) > 0.0:
                parts["rew_time"] = -float(self.ctx.reward.w_time) * active.astype(np.float32)

        if "energy" in enabled:
            if float(getattr(self.ctx.reward, "w_energy", 0.0)) > 0.0:
                if hasattr(cur, "energy") and hasattr(prev, "energy"):
                    prev_energy = np.asarray(prev.energy, dtype=np.float32)
                    cur_energy = np.asarray(cur.energy, dtype=np.float32)
                    drain = np.maximum(0.0, prev_energy - cur_energy)
                    max_drain = (
                        float(getattr(self.ctx.reward, "energy_drain_hover", 0.0))
                        + float(getattr(cur, "max_thrust", 0.0)) * float(getattr(self.ctx.reward, "energy_drain_thrust", 0.0))
                    ) * float(getattr(cur, "dt", 0.0))
                    if max_drain > 0.0:
                        parts["rew_energy"] = -float(self.ctx.reward.w_energy) * (drain / max_drain) * active.astype(
                            np.float32
                        )

        total = np.zeros(n, dtype=np.float32)
        for arr in parts.values():
            total += arr
        return total.astype(np.float32), parts


class RewardFn:
    """Суммарная награда как сочетание прогресса, безопасности и регуляризаций."""

    def __init__(self, reward_cfg, field_size: float, goal_radius: float):
        self.reward = reward_cfg
        self.field_size = float(field_size)
        self.goal_radius = float(goal_radius)
        self._pipeline = RewardPipeline(reward_cfg, field_size, goal_radius)

    def sync(self, field_size: float, goal_radius: float) -> None:
        self.field_size = float(field_size)
        self.goal_radius = float(goal_radius)
        self._pipeline = RewardPipeline(self.reward, field_size, goal_radius)

    def prepare_step(
        self,
        *,
        min_wall: np.ndarray | None = None,
        speed_norm: np.ndarray | None = None,
        action_change_sq: np.ndarray | None = None,
    ) -> None:
        if hasattr(self, "_pipeline"):
            self._pipeline.set_step_cache(
                min_wall=min_wall,
                speed_norm=speed_norm,
                action_change_sq=action_change_sq,
            )

    def __call__(self, prev: SimState, cur: SimState, idx: int) -> tuple[float, dict]:
        return self._pipeline(prev, cur, idx)

    def compute_all(self, prev: SimState, cur: SimState) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        return self._pipeline.compute_all(prev, cur)
