"""Диагностика метрик финиша и времени в цели по эпизодам."""

import csv
import json
from dataclasses import dataclass

import numpy as np

from baselines.factory import create_baseline_policy
from env.config import EnvConfig
from scripts.common.artifacts import ensure_run_dir, write_manifest
from common.runtime.env_factory import make_pz_env as _common_make_pz_env
from scripts.common.episode_runner import policy_actions


def run_episode(env, policy, max_steps, seed, trace=False):
    obs, infos = env.reset(seed=seed)
    if hasattr(policy, "reset"):
        policy.reset(seed)
    n_agents = len(env.possible_agents)
    min_dist = np.full(n_agents, np.inf, dtype=np.float32)
    time_in_goal = np.zeros(n_agents, dtype=np.int32)
    max_streak = np.zeros(n_agents, dtype=np.int32)
    cur_streak = np.zeros(n_agents, dtype=np.int32)
    first_in_goal = np.full(n_agents, -1, dtype=np.int32)
    first_finish = np.full(n_agents, -1, dtype=np.int32)
    trace_rows = []

    for t in range(max_steps):
        decision_step = t + 1
        actions = policy_actions(env, policy, obs, infos)

        obs, _rewards, terms, truncs, infos = env.step(actions)

        if trace:
            # Трассировка пишет агрегаты по шагам для последующей визуализации.
            in_goal_flags = [bool(infos[a].get("in_goal", False)) for a in env.possible_agents]
            finished_flags = [bool(infos[a].get("finished", False)) for a in env.possible_agents]
            dists = [float(infos[a].get("dist", float("nan"))) for a in env.possible_agents]
            trace_rows.append(
                {
                    "step_idx": t,
                    "min_dist_to_target": float(np.nanmin(dists)) if dists else float("nan"),
                    "in_goal_frac": float(np.mean(in_goal_flags)) if in_goal_flags else float("nan"),
                    "finished_frac": float(np.mean(finished_flags)) if finished_flags else float("nan"),
                    "any_in_goal": float(any(in_goal_flags)),
                    "any_finished": float(any(finished_flags)),
                }
            )

        for i, agent_id in enumerate(env.possible_agents):
            inf = infos.get(agent_id, {})
            d = float(inf.get("dist", float("nan")))
            if d == d:
                min_dist[i] = min(min_dist[i], d)
            in_goal = bool(inf.get("in_goal", False))
            if in_goal:
                time_in_goal[i] += 1
                cur_streak[i] += 1
                if first_in_goal[i] < 0:
                    first_in_goal[i] = decision_step
            else:
                cur_streak[i] = 0
            max_streak[i] = max(max_streak[i], cur_streak[i])
            if inf.get("newly_finished", 0.0) and first_finish[i] < 0:
                first_finish[i] = decision_step

        if all(terms.values()) or all(truncs.values()):
            break

    summary = {
        "min_dist_to_target": float(np.mean(min_dist)),
        "time_in_goal_total": float(np.mean(time_in_goal)),
        "max_in_goal_streak": float(np.mean(max_streak)),
        "first_in_goal_step": int(np.min(first_in_goal[first_in_goal >= 0])) if np.any(first_in_goal >= 0) else -1,
        "first_finish_step": int(np.min(first_finish[first_finish >= 0])) if np.any(first_finish >= 0) else -1,
    }
    return summary, trace_rows


@dataclass
class FinishDebugConfig:
    policy: str = "baseline:potential_fields"
    episodes: int = 20
    max_steps: int = 600
    seed: int = 0
    trace_episode: int = -1
    out_dir: str = ""
    out_root: str = "runs"
    run_id: str = ""


def run(cfg: FinishDebugConfig) -> None:
    args = cfg

    cfg = EnvConfig()
    env = _common_make_pz_env(max_steps=args.max_steps, goal_radius=3.0, config=cfg, reset=False)

    policy = create_baseline_policy(args.policy, env=env, seed=args.seed)

    rows = []
    trace_rows = []
    for ep in range(args.episodes):
        summary, trace = run_episode(
            env,
            policy,
            args.max_steps,
            args.seed + ep,
            trace=(ep == args.trace_episode),
        )
        rows.append(summary)
        if trace:
            trace_rows = trace

    out_dir = ensure_run_dir(
        category="diagnostics",
        out_root=args.out_root,
        run_id=args.run_id or None,
        prefix="finish_debug",
        out_dir=args.out_dir or None,
    )
    json_path = out_dir / "finish_debug.json"
    csv_path = out_dir / "finish_debug.csv"

    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    if trace_rows:
        t_json = out_dir / "finish_trace.json"
        t_csv = out_dir / "finish_trace.csv"
        t_json.write_text(json.dumps(trace_rows, ensure_ascii=False, indent=2), encoding="utf-8")
        with t_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(trace_rows[0].keys()))
            writer.writeheader()
            for r in trace_rows:
                writer.writerow(r)
        print(f"[ОК] {t_json}")
        print(f"[ОК] {t_csv}")

    write_manifest(out_dir, config=vars(args), command=["finish_debug"])
    print(f"[ОК] {json_path}")
    print(f"[ОК] {csv_path}")


def main() -> None:
    import hydra
    from omegaconf import DictConfig, OmegaConf

    from scripts.common.hydra_utils import apply_schema

    @hydra.main(version_base=None, config_path="../../configs/hydra", config_name="debug/finish_debug")
    def _run(cfg: DictConfig) -> None:
        cfg = apply_schema(cfg, FinishDebugConfig)
        data = OmegaConf.to_container(cfg, resolve=True)
        run(FinishDebugConfig(**data))

    _run()


if __name__ == "__main__":
    main()
