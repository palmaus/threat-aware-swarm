"""Диагностика расчёта метрик и риска в среде на случайных действиях."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from env.config import EnvConfig
from scripts.common.artifacts import ensure_run_dir, write_manifest
from common.runtime.env_factory import make_pz_env as _common_make_pz_env
from common.runtime.env_factory import pettingzoo_to_vec_env


def _risk_stats(values):
    arr = np.array(values, dtype=np.float32)
    if arr.size == 0:
        return {
            "min": float("nan"),
            "mean": float("nan"),
            "max": float("nan"),
            "frac_positive": float("nan"),
        }
    return {
        "min": float(np.nanmin(arr)),
        "mean": float(np.nanmean(arr)),
        "max": float(np.nanmax(arr)),
        "frac_positive": float(np.mean(arr > 0.0)),
    }


def run_direct(max_steps: int, seed: int, episodes: int):
    cfg = EnvConfig()
    env = _common_make_pz_env(max_steps=max_steps, goal_radius=3.0, config=cfg, reset=False)

    risk_vals = []
    min_dist_any = []
    min_margin = []
    hit_steps = 0
    near_5_steps = 0
    near_10_steps = 0
    speeds = []
    threats_count = []
    threat_radii = []
    infos_empty = 0
    risk_missing = 0
    steps_total = 0
    inside_nearest = 0
    inside_nearest_intensity_zero = 0
    inside_any = 0
    inside_any_intensity_zero = 0

    episode_geometry = []

    # Прогон случайных действий позволяет проверить корректность метрик без политики.
    for ep in range(episodes):
        _obs, _infos = env.reset(seed=seed + ep)
        state = env.get_state()
        threats_count.append(len(state.threats))
        threat_radii.extend([float(t.radius) for t in state.threats])
        episode_geometry.append(_episode_geometry(env, max_steps))
        for _ in range(max_steps):
            steps_total += 1
            actions = {a: env.action_space(a).sample() for a in env.possible_agents}
            _obs, _rewards, terms, truncs, infos = env.step(actions)
            state = env.get_state()
            if not infos:
                infos_empty += 1
            for inf in infos.values():
                if not isinstance(inf, dict):
                    infos_empty += 1
                    continue
                if "risk_p" not in inf:
                    risk_missing += 1
                else:
                    risk_vals.append(float(inf.get("risk_p", 0.0)))
                if float(inf.get("inside_nearest_before", 0.0)) > 0.0:
                    inside_nearest += 1
                    if float(inf.get("nearest_threat_intensity_before", 0.0)) == 0.0:
                        inside_nearest_intensity_zero += 1
                if float(inf.get("inside_any_before", 0.0)) > 0.0:
                    inside_any += 1
                    if float(inf.get("any_inside_intensity_sum_before", 0.0)) == 0.0:
                        inside_any_intensity_zero += 1
            step_min = _min_dist_to_threats(state.pos, state.threats)
            if step_min is not None:
                min_dist_any.append(step_min["min_dist"])
                min_margin.append(step_min["min_margin"])
                if step_min["hit"]:
                    hit_steps += 1
                if step_min["near_5"]:
                    near_5_steps += 1
                if step_min["near_10"]:
                    near_10_steps += 1
            speeds.append(float(np.mean(np.linalg.norm(state.vel, axis=1))))
            if all(terms.values()) or all(truncs.values()):
                break

    geometry = {
        "threats_count_mean": float(np.mean(threats_count)) if threats_count else float("nan"),
        "threat_radius_mean": float(np.mean(threat_radii)) if threat_radii else float("nan"),
        "min_dist_any_p10": _pctl(min_dist_any, 10),
        "min_dist_any_p50": _pctl(min_dist_any, 50),
        "min_dist_any_p90": _pctl(min_dist_any, 90),
        "min_margin_any_p10": _pctl(min_margin, 10),
        "min_margin_any_p50": _pctl(min_margin, 50),
        "min_margin_any_p90": _pctl(min_margin, 90),
        "hit_rate": float(hit_steps / max(len(min_dist_any), 1)),
        "near_rate_5": float(near_5_steps / max(len(min_dist_any), 1)),
        "near_rate_10": float(near_10_steps / max(len(min_dist_any), 1)),
        "mean_speed": float(np.mean(speeds)) if speeds else float("nan"),
    }

    return {
        "risk_stats": _risk_stats(risk_vals),
        "risk_samples": risk_vals[:20],
        "geometry": geometry,
        "episode_geometry": episode_geometry,
        "infos_loss": {
            "infos_empty_steps": int(infos_empty),
            "risk_missing": int(risk_missing),
            "steps_total": int(steps_total),
        },
        "risk_zero_inside_breakdown": {
            "inside_nearest_count": int(inside_nearest),
            "inside_nearest_intensity_zero": int(inside_nearest_intensity_zero),
            "inside_any_count": int(inside_any),
            "inside_any_intensity_zero": int(inside_any_intensity_zero),
        },
    }


def run_wrapped(max_steps: int, seed: int, episodes: int):
    cfg = EnvConfig()
    env = _common_make_pz_env(max_steps=max_steps, goal_radius=3.0, config=cfg, reset=False)
    vec_env = pettingzoo_to_vec_env(env)

    risk_vals = []
    infos_seen = []
    infos_empty = 0
    risk_missing = 0
    steps_total = 0
    for ep in range(episodes):
        try:
            _obs = vec_env.reset(seed=seed + ep)
        except TypeError:
            _obs = vec_env.reset()
        for _ in range(max_steps):
            steps_total += 1
            try:
                actions = np.array([vec_env.action_space.sample() for _ in range(vec_env.num_envs)])
            except Exception:
                actions = vec_env.action_space.sample()
            _obs, _rewards, terms, truncs, infos = vec_env.step(actions)
            infos_seen.append(infos)
            if not infos:
                infos_empty += 1
            for inf in infos:
                if not isinstance(inf, dict):
                    infos_empty += 1
                    continue
                if "risk_p" not in inf:
                    risk_missing += 1
                else:
                    risk_vals.append(float(inf.get("risk_p", 0.0)))
            if np.all(terms) or np.all(truncs):
                break

    return {
        "infos_types": [str(type(i)) for i in infos_seen[:3]],
        "risk_stats": _risk_stats(risk_vals),
        "risk_samples": risk_vals[:20],
        "infos_loss": {
            "infos_empty_steps": int(infos_empty),
            "risk_missing": int(risk_missing),
            "steps_total": int(steps_total),
        },
    }


def run_sanity_scenario(seed: int = 0):
    cfg = EnvConfig(n_agents=2)
    env = _common_make_pz_env(max_steps=5, goal_radius=3.0, config=cfg, reset=False)
    _obs, _infos = env.reset(seed=seed)
    env.clear_threats()
    env.add_static_threat(np.array([50.0, 50.0], dtype=np.float32), radius=10.0, intensity=0.2)
    env.set_agent_positions(np.array([[50.0, 50.0], [60.0, 50.0]], dtype=np.float32))

    actions = {a: np.zeros(2, dtype=np.float32) for a in env.possible_agents}
    _obs, _rewards, _terms, _truncs, infos = env.step(actions)
    risk0 = infos[env.possible_agents[0]].get("risk_p", 0.0)
    risk1 = infos[env.possible_agents[1]].get("risk_p", 0.0)
    state = env.get_state()
    threat = state.threats[0]

    dist0 = float(np.linalg.norm(state.pos[0] - threat.position))
    dist1 = float(np.linalg.norm(state.pos[1] - threat.position))

    ok = float(risk0) > 0.0
    report = {
        "risk_p_agent0": float(risk0),
        "risk_p_agent1": float(risk1),
        "dist0": dist0,
        "dist1": dist1,
        "radius": float(threat.radius),
        "intensity": float(threat.intensity),
        "ok": ok,
    }
    return report


@dataclass
class DebugEnvMetricsConfig:
    max_steps: int = 200
    seed: int = 0
    episodes: int = 50
    out: str = ""
    out_dir: str = ""
    out_root: str = "runs"
    run_id: str = ""
    no_sanity: bool = False


def run(cfg: DebugEnvMetricsConfig) -> None:
    args = cfg

    direct = run_direct(args.max_steps, args.seed, args.episodes)
    wrapped = run_wrapped(args.max_steps, args.seed, args.episodes)
    sanity = None if args.no_sanity else run_sanity_scenario(args.seed)

    report = {
        "direct": direct,
        "wrapped": wrapped,
        "sanity": sanity,
        "verdict": _verdict(direct, wrapped),
        "likely_infos_loss": _likely_infos_loss(direct, wrapped),
    }

    out_dir = ensure_run_dir(
        category="diagnostics",
        out_root=args.out_root,
        run_id=args.run_id or None,
        prefix="env_metrics",
        out_dir=args.out_dir or None,
    )
    out_path = out_dir / "env_metrics.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ОК] {out_path}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    diag_dir = out_dir
    geo_csv = diag_dir / f"geometry_stats_{ts}.csv"
    geo_json = diag_dir / f"geometry_stats_{ts}.json"
    with geo_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "threats_count_mean",
                "threat_radius_mean",
                "min_dist_any_p10",
                "min_dist_any_p50",
                "min_dist_any_p90",
                "min_margin_any_p10",
                "min_margin_any_p50",
                "min_margin_any_p90",
                "hit_rate",
                "near_rate_5",
                "near_rate_10",
                "mean_speed",
            ],
        )
        writer.writeheader()
        writer.writerow(direct.get("geometry", {}))
    geom = direct.get("geometry", {})
    bug = _diagnostics_bug(geom)
    geom["diagnostics_bug"] = bug["flag"]
    geom["diagnostics_bug_reason"] = bug["reason"]
    geo_json.write_text(json.dumps(geom, ensure_ascii=False, indent=2), encoding="utf-8")
    write_manifest(out_dir, config=vars(args), command=["debug_env_metrics"])
    print(f"[ОК] {geo_csv}")
    print(f"[ОК] {geo_json}")

    ep_csv = diag_dir / f"episode_geometry_{ts}.csv"
    ep_json = diag_dir / f"episode_geometry_{ts}.json"
    _write_episode_geometry(ep_csv, ep_json, direct.get("episode_geometry", []))
    print(f"[ОК] {ep_csv}")
    print(f"[ОК] {ep_json}")

    if sanity is not None and not sanity.get("ok", False):
        raise SystemExit(
            f"Sanity scenario failed: risk_p_agent0={sanity.get('risk_p_agent0')} "
            f"dist0={sanity.get('dist0')} radius={sanity.get('radius')}"
        )


def main() -> None:
    import hydra
    from omegaconf import DictConfig, OmegaConf

    from scripts.common.hydra_utils import apply_schema

    @hydra.main(version_base=None, config_path="../../configs/hydra", config_name="debug/debug_env_metrics")
    def _run(cfg: DictConfig) -> None:
        cfg = apply_schema(cfg, DebugEnvMetricsConfig)
        data = OmegaConf.to_container(cfg, resolve=True)
        run(DebugEnvMetricsConfig(**data))

    _run()


def _verdict(direct, wrapped):
    d_mean = direct["risk_stats"]["mean"]
    w_mean = wrapped["risk_stats"]["mean"]
    if d_mean == 0.0 and w_mean == 0.0:
        return "risk_p остаётся нулевым уже в env"
    if d_mean > 0.0 and (w_mean == 0.0 or w_mean != w_mean):
        return "risk_p пропадает после обёрток"
    return "risk_p присутствует и в env, и после обёрток"


def _likely_infos_loss(direct, wrapped):
    d = direct.get("infos_loss", {})
    w = wrapped.get("infos_loss", {})
    if not d or not w:
        return False
    d_rate = d.get("risk_missing", 0) / max(d.get("steps_total", 1), 1)
    w_rate = w.get("risk_missing", 0) / max(w.get("steps_total", 1), 1)
    return bool(w_rate > d_rate + 0.1)


def _pctl(values, q):
    arr = np.array(values, dtype=np.float32)
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, q))


def _min_dist_to_threats(positions: np.ndarray, threats: list):
    if not threats:
        return None
    min_dist = float("inf")
    min_margin = float("inf")
    nearest_radius = float("nan")
    nearest_margin = float("nan")
    hit = False
    near_5 = False
    near_10 = False
    for pos in positions:
        for t in threats:
            d = float(np.linalg.norm(pos - t.position))
            margin = float(t.radius) - d
            if d < min_dist:
                min_dist = d
                nearest_radius = float(t.radius)
                nearest_margin = float(t.radius) - d
            if margin < min_margin:
                min_margin = margin
            if d <= float(t.radius):
                hit = True
            if d <= float(t.radius) + 5.0:
                near_5 = True
            if d <= float(t.radius) + 10.0:
                near_10 = True
    return {
        "min_dist": float(min_dist),
        "min_margin": float(min_margin),
        "nearest_radius": float(nearest_radius),
        "nearest_margin": float(nearest_margin),
        "hit": bool(hit),
        "near_5": bool(near_5),
        "near_10": bool(near_10),
    }


def _episode_geometry(env: SwarmPZEnv, max_steps: int):
    state = env.get_state()
    start_center = np.mean(state.pos, axis=0).astype(np.float32)
    target_pos = state.target_pos.copy()
    threats = [
        {"pos": t.position.tolist(), "radius": float(t.radius), "intensity": float(t.intensity)}
        for t in state.threats
    ]
    start_to_target = float(np.linalg.norm(start_center - target_pos))

    min_dist_any = float("inf")
    min_margin_any = float("inf")
    min_threat_radius = float("nan")
    min_threat_dist = float("nan")
    min_threat_margin = float("nan")
    hit_steps = 0
    near_5_steps = 0
    near_10_steps = 0
    steps = 0

    per_agent_min = np.full(env.n_agents, np.inf, dtype=np.float32)
    per_agent_hit = np.zeros(env.n_agents, dtype=np.int32)
    per_agent_near5 = np.zeros(env.n_agents, dtype=np.int32)
    per_agent_near10 = np.zeros(env.n_agents, dtype=np.int32)

    for _ in range(max_steps):
        steps += 1
        actions = {a: env.action_space(a).sample() for a in env.possible_agents}
        _obs, _rewards, terms, truncs, _infos = env.step(actions)
        state = env.get_state()

        step_min = _min_dist_to_threats(state.pos, state.threats)
        if step_min is not None:
            if step_min["min_dist"] <= min_dist_any:
                min_dist_any = step_min["min_dist"]
                min_threat_radius = step_min["nearest_radius"]
                min_threat_dist = step_min["min_dist"]
                min_threat_margin = step_min["nearest_margin"]
            min_margin_any = min(min_margin_any, step_min["min_margin"])
            if step_min["hit"]:
                hit_steps += 1
            if step_min["near_5"]:
                near_5_steps += 1
            if step_min["near_10"]:
                near_10_steps += 1

        if state.threats:
            first_radius = float(state.threats[0].radius)
            for i in range(env.n_agents):
                d_i = min(float(np.linalg.norm(state.pos[i] - t.position)) for t in state.threats)
                per_agent_min[i] = min(per_agent_min[i], d_i)
                if d_i <= first_radius:
                    per_agent_hit[i] += 1
                if d_i <= first_radius + 5.0:
                    per_agent_near5[i] += 1
                if d_i <= first_radius + 10.0:
                    per_agent_near10[i] += 1

        if all(terms.values()) or all(truncs.values()):
            break

    corridor_5, corridor_10, min_threat_seg = _threats_to_path(start_center, target_pos, state.threats)

    return {
        "start_center": start_center.tolist(),
        "target_pos": target_pos.tolist(),
        "threats": threats,
        "dist_start_target": start_to_target,
        "min_threat_dist_to_path": min_threat_seg,
        "threats_in_corridor_5": corridor_5,
        "threats_in_corridor_10": corridor_10,
        "min_dist_any": float(min_dist_any) if min_dist_any != float("inf") else float("nan"),
        "min_margin_any": float(min_margin_any) if min_margin_any != float("inf") else float("nan"),
        "sanity_nearest_radius": float(min_threat_radius),
        "sanity_nearest_dist": float(min_threat_dist),
        "sanity_nearest_margin": float(min_threat_margin),
        "hit_rate_episode": float(hit_steps / max(steps, 1)),
        "near_rate_5_episode": float(near_5_steps / max(steps, 1)),
        "near_rate_10_episode": float(near_10_steps / max(steps, 1)),
        "min_dist_agents_p10": _pctl(per_agent_min, 10),
        "min_dist_agents_p50": _pctl(per_agent_min, 50),
        "min_dist_agents_p90": _pctl(per_agent_min, 90),
        "hit_rate_agents_mean": float(np.mean(per_agent_hit / max(steps, 1))) if steps else float("nan"),
        "near_rate_5_agents_mean": float(np.mean(per_agent_near5 / max(steps, 1))) if steps else float("nan"),
        "near_rate_10_agents_mean": float(np.mean(per_agent_near10 / max(steps, 1))) if steps else float("nan"),
    }


def _threats_to_path(start: np.ndarray, target: np.ndarray, threats: list):
    if not threats:
        return 0.0, 0.0, float("nan")
    dists = []
    in_5 = 0
    in_10 = 0
    for t in threats:
        d = _point_to_segment_dist(t.position, start, target)
        dists.append(d)
        if d <= 5.0:
            in_5 += 1
        if d <= 10.0:
            in_10 += 1
    return in_5 / len(threats), in_10 / len(threats), float(min(dists))


def _point_to_segment_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray):
    ap = p - a
    ab = b - a
    ab_len = float(np.dot(ab, ab))
    if ab_len <= 1e-8:
        return float(np.linalg.norm(ap))
    t = float(np.dot(ap, ab) / ab_len)
    t = max(0.0, min(1.0, t))
    proj = a + t * ab
    return float(np.linalg.norm(p - proj))


def _write_episode_geometry(csv_path: Path, json_path: Path, rows: list):
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    if not rows:
        csv_path.write_text("", encoding="utf-8")
        return
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _diagnostics_bug(geom: dict):
    radius = geom.get("sanity_nearest_radius")
    dist = geom.get("sanity_nearest_dist")
    margin = geom.get("sanity_nearest_margin")
    if radius is None or dist is None or margin is None:
        return {"flag": False, "reason": ""}
    if radius != radius or dist != dist or margin != margin:
        return {"flag": False, "reason": ""}
    expected = float(radius) - float(dist)
    if abs(expected - float(margin)) > 1e-3:
        return {"flag": True, "reason": f"margin!=radius-dist ({margin} vs {expected})"}
    return {"flag": False, "reason": ""}


if __name__ == "__main__":
    main()
