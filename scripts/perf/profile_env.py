from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from env.config import EnvConfig
from scripts.common.artifacts import ensure_run_dir
from common.runtime.env_factory import make_pz_env as _common_make_pz_env


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Профилирование шага среды (FPS).")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--agents", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--goal-radius", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="")
    return parser.parse_args()


def run() -> Path:
    args = _parse_args()
    cfg = EnvConfig()
    cfg.n_agents = int(args.agents)
    env = _common_make_pz_env(
        max_steps=int(args.max_steps),
        goal_radius=float(args.goal_radius),
        config=cfg,
        reset=False,
    )
    env.reset(seed=int(args.seed))

    actions = {agent: np.zeros((2,), dtype=np.float32) for agent in env.possible_agents}
    start = time.perf_counter()
    for _ in range(int(args.steps)):
        env.step(actions)
    elapsed = time.perf_counter() - start
    fps = float(args.steps) / max(elapsed, 1e-6)

    run_dir = ensure_run_dir(
        category="perf",
        out_root="runs",
        run_id=None,
        prefix=f"{args.agents}a",
        out_dir=args.out_dir or None,
    )
    payload = {
        "steps": int(args.steps),
        "agents": int(args.agents),
        "elapsed_sec": float(elapsed),
        "fps": fps,
    }
    (run_dir / "perf.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))
    return run_dir


if __name__ == "__main__":
    run()
