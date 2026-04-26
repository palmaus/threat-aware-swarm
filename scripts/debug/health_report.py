from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import hydra
import numpy as np
from omegaconf import OmegaConf

from env.config import EnvConfig
from scripts.common.artifacts import ensure_run_dir, write_manifest
from common.runtime.env_factory import make_pz_env as _common_make_pz_env
from scripts.common.hydra_utils import apply_schema
from scripts.common.logging_utils import collect_system_metrics
from scripts.common.numba_guard import log_numba_status

logger = logging.getLogger(__name__)


def _numba_status() -> dict[str, object]:
    status: dict[str, object] = {}
    try:
        from common import physics_model

        status["physics_model.apply_accel_dynamics_step"] = bool(
            getattr(physics_model.apply_accel_dynamics_step, "signatures", None)
        )
    except Exception as exc:
        status["physics_model.apply_accel_dynamics_step"] = None
        status["physics_model.error"] = str(exc)

    try:
        from env import oracle as oracle_mod

        status["oracle.numba_available"] = bool(getattr(oracle_mod, "_NUMBA_AVAILABLE", False))
    except Exception as exc:
        status["oracle.numba_available"] = None
        status["oracle.error"] = str(exc)

    try:
        from baselines import astar_grid as astar_mod

        status["astar_grid.numba_available"] = bool(getattr(astar_mod, "_NUMBA_AVAILABLE", False))
    except Exception as exc:
        status["astar_grid.numba_available"] = None
        status["astar_grid.error"] = str(exc)

    return status


@dataclass
class HealthReportConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    steps: int = 500
    agents: int = 1
    max_steps: int = 200
    goal_radius: float = 3.0
    seed: int = 0
    out_dir: str = ""
    out_root: str = "runs"


def run(cfg: HealthReportConfig) -> Path:
    args = cfg
    env_cfg = args.env
    env_cfg.n_agents = int(args.agents)

    env = _common_make_pz_env(
        max_steps=int(args.max_steps),
        goal_radius=float(args.goal_radius),
        config=env_cfg,
        reset=False,
    )
    env.reset(seed=int(args.seed))

    actions = {agent: np.zeros((2,), dtype=np.float32) for agent in env.possible_agents}

    start = time.perf_counter()
    for _ in range(int(args.steps)):
        env.step(actions)
    elapsed = time.perf_counter() - start

    steps = int(args.steps)
    fps = float(steps) / max(elapsed, 1e-6)
    step_ms = (elapsed * 1000.0) / max(steps, 1)

    run_dir = ensure_run_dir(
        category="health",
        out_root=args.out_root,
        run_id=None,
        prefix=f"{args.agents}a",
        out_dir=args.out_dir or None,
    )

    payload = {
        "steps": steps,
        "agents": int(args.agents),
        "elapsed_sec": float(elapsed),
        "fps": float(fps),
        "step_ms": float(step_ms),
        "system": collect_system_metrics(),
        "numba": _numba_status(),
    }

    out_path = run_dir / "health_report.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_manifest(run_dir, config=OmegaConf.to_container(OmegaConf.structured(args), resolve=True), command=["health_report"])
    log_numba_status(logger)

    print(json.dumps(payload, ensure_ascii=False))
    return out_path


@hydra.main(config_path="../../configs/hydra", config_name="debug/health_report", version_base=None)
def main(cfg: HealthReportConfig) -> None:
    cfg = apply_schema(cfg, HealthReportConfig)
    data = OmegaConf.to_container(cfg, resolve=True)
    cfg = HealthReportConfig(**(data or {}))
    run(cfg)


if __name__ == "__main__":
    main()
