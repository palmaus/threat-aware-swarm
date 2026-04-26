from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from baselines.controllers import WaypointController
from common.physics.model import apply_accel_dynamics
from scripts.common.artifacts import ensure_run_dir


@dataclass
class CalibrationConfig:
    steps: int = 400
    dt: float = 0.1
    field_size: float = 100.0
    max_speed: float = 4.0
    max_accel: float = 2.0
    drag: float = 0.05
    goal_radius: float = 1.5
    path: str = "square"
    side: float = 20.0
    output_dir: str = ""


def _parse_args() -> CalibrationConfig:
    parser = argparse.ArgumentParser(description="Калибровка WaypointController")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--field-size", type=float, default=100.0)
    parser.add_argument("--max-speed", type=float, default=4.0)
    parser.add_argument("--max-accel", type=float, default=2.0)
    parser.add_argument("--drag", type=float, default=0.05)
    parser.add_argument("--goal-radius", type=float, default=1.5)
    parser.add_argument("--path", choices=("square", "sine"), default="square")
    parser.add_argument("--side", type=float, default=20.0)
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()
    return CalibrationConfig(
        steps=args.steps,
        dt=args.dt,
        field_size=args.field_size,
        max_speed=args.max_speed,
        max_accel=args.max_accel,
        drag=args.drag,
        goal_radius=args.goal_radius,
        path=args.path,
        side=args.side,
        output_dir=args.output_dir,
    )


def _square_waypoints(center: np.ndarray, side: float) -> list[np.ndarray]:
    half = 0.5 * side
    return [
        center + np.array([half, half], dtype=np.float32),
        center + np.array([-half, half], dtype=np.float32),
        center + np.array([-half, -half], dtype=np.float32),
        center + np.array([half, -half], dtype=np.float32),
    ]


def _sine_target(t: float, center: np.ndarray, side: float) -> np.ndarray:
    return center + np.array([0.5 * side * np.sin(t), 0.5 * side * np.cos(t)], dtype=np.float32)


def _build_obs(to_target: np.ndarray, vel: np.ndarray, field_size: float, max_speed: float) -> dict[str, np.ndarray]:
    vec = np.zeros((13,), dtype=np.float32)
    vec[0:2] = to_target / max(field_size, 1e-6)
    vec[2:4] = vel / max(max_speed, 1e-6)
    return {"vector": vec, "grid": None}


def _write_csv(path: Path, rows: Iterable[dict[str, float]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    lines = [",".join(keys)]
    for row in rows:
        lines.append(",".join(str(row.get(k, "")) for k in keys))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _summarize(dist_series: list[float], goal_radius: float) -> dict[str, float]:
    if not dist_series:
        return {"settling_time": float("nan"), "overshoot": float("nan")}
    settling_time = float("nan")
    overshoot = 0.0
    reached = False
    first_idx = None
    for i, d in enumerate(dist_series):
        if d <= goal_radius and first_idx is None:
            first_idx = i
        if first_idx is not None and not reached:
            if d <= goal_radius:
                reached = True
        if reached and d > goal_radius:
            overshoot = max(overshoot, d - goal_radius)
    if first_idx is not None:
        settling_time = float(first_idx)
    return {"settling_time": settling_time, "overshoot": float(overshoot)}


def run(cfg: CalibrationConfig) -> Path:
    run_dir = ensure_run_dir(
        category="controller_calibration",
        out_root="runs",
        run_id=None,
        prefix=cfg.path,
        out_dir=cfg.output_dir or None,
    )

    controller = WaypointController(
        goal_radius_control=cfg.goal_radius,
        near_goal_speed_cap=0.3,
        near_goal_damping=0.2,
        near_goal_kp=1.0,
        risk_speed_scale=0.0,
        risk_speed_floor=1.0,
    )
    info = {
        "max_speed": cfg.max_speed,
        "max_accel": cfg.max_accel,
        "dt": cfg.dt,
        "drag": cfg.drag,
        "accel_tau": 0.35,
    }

    pos = np.zeros((1, 2), dtype=np.float32)
    vel = np.zeros((1, 2), dtype=np.float32)
    center = np.array([cfg.field_size * 0.5, cfg.field_size * 0.5], dtype=np.float32)

    waypoints = _square_waypoints(center, cfg.side)
    wp_idx = 0

    rows: list[dict[str, float]] = []
    dist_series: list[float] = []
    for t in range(cfg.steps):
        if cfg.path == "sine":
            target = _sine_target(t * cfg.dt, center, cfg.side)
        else:
            target = waypoints[wp_idx]
        desired_vec = target - pos[0]
        dist = float(np.linalg.norm(desired_vec))
        dist_series.append(dist)
        if dist <= cfg.goal_radius and cfg.path == "square":
            wp_idx = (wp_idx + 1) % len(waypoints)
            target = waypoints[wp_idx]
            desired_vec = target - pos[0]

        obs = _build_obs(desired_vec, vel[0], cfg.field_size, cfg.max_speed)
        action = controller.compute_action(
            desired_vec,
            obs,
            dist_m=dist,
            dist_normed=False,
            field_size=cfg.field_size,
            in_goal=dist <= cfg.goal_radius,
            risk_p=0.0,
            stop_risk_threshold=0.0,
            info=info,
            to_target=desired_vec,
        )
        accel = np.asarray(action, dtype=np.float32) * float(cfg.max_accel)
        pos, vel = apply_accel_dynamics(pos, vel, accel[None, :], cfg.dt, drag=cfg.drag, max_speed=cfg.max_speed)

        rows.append(
            {
                "t": float(t * cfg.dt),
                "pos_x": float(pos[0, 0]),
                "pos_y": float(pos[0, 1]),
                "vel_x": float(vel[0, 0]),
                "vel_y": float(vel[0, 1]),
                "target_x": float(target[0]),
                "target_y": float(target[1]),
                "dist": dist,
                "action_x": float(action[0]),
                "action_y": float(action[1]),
            }
        )

    summary = _summarize(dist_series, cfg.goal_radius)
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(run_dir / "trace.csv", rows)
    return run_dir


def main() -> None:
    cfg = _parse_args()
    out = run(cfg)
    print(f"[OK] {out}")


if __name__ == "__main__":
    main()
