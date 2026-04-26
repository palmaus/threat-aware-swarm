"""Быстрый перф‑бенчмарк схем наблюдения."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from env.config import EnvConfig
from common.runtime.env_factory import make_pz_env as _common_make_pz_env


def _build_env(schema: str, n_agents: int) -> SwarmPZEnv:
    cfg = EnvConfig()
    cfg.n_agents = int(n_agents)
    cfg.obs_schema_version = schema
    if schema == "obs@1694:v5":
        cfg.grid_width = 41
        cfg.grid_res = 1.0
    else:
        raise ValueError(f"Неизвестная схема: {schema}")
    return _common_make_pz_env(max_steps=200, goal_radius=3.0, config=cfg, reset=False)


def run_bench(
    schema: str = "obs@1694:v5",
    steps: int = 2000,
    n_agents: int = 20,
    seed: int = 0,
) -> dict[str, float | int | str]:
    env = _build_env(schema, n_agents)
    rng = np.random.default_rng(seed)
    env.reset(seed=seed)

    start = time.perf_counter()
    for _ in range(int(steps)):
        actions = {agent: rng.uniform(-1.0, 1.0, size=(2,)).astype(np.float32) for agent in env.possible_agents}
        _obs, _rewards, terminations, truncations, _infos = env.step(actions)
        if all(terminations.values()) or all(truncations.values()):
            env.reset(seed=seed)
    elapsed = time.perf_counter() - start
    steps_per_sec = float("inf") if elapsed <= 0 else float(steps) / elapsed
    return {
        "schema": schema,
        "steps": int(steps),
        "elapsed_sec": elapsed,
        "steps_per_sec": steps_per_sec,
        "obs_dim": int(env.obs_dim),
        "grid_width": int(env.grid_width),
        "grid_res": float(env.obs_builder.grid_res),
    }


def _print_result(result: dict[str, float | int | str]) -> None:
    print(
        "schema={schema} obs_dim={obs_dim} grid={grid_width}x{grid_width} res={grid_res:.1f} "
        "steps={steps} elapsed={elapsed_sec:.3f}s steps_per_sec={steps_per_sec:.1f}".format(**result)
    )


@dataclass
class BenchObsPerfConfig:
    steps: int = 2000
    n_agents: int = 20
    seed: int = 0
    schema: str = "obs@1694:v5"


def run(cfg: BenchObsPerfConfig) -> int:
    result = run_bench(cfg.schema, cfg.steps, cfg.n_agents, cfg.seed)
    _print_result(result)
    return 0


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig, OmegaConf

    from scripts.common.hydra_utils import apply_schema

    @hydra.main(version_base=None, config_path="../../configs/hydra", config_name="bench/obs_perf")
    def _run(cfg: DictConfig) -> None:
        cfg = apply_schema(cfg, BenchObsPerfConfig)
        data = OmegaConf.to_container(cfg, resolve=True)
        raise SystemExit(run(BenchObsPerfConfig(**data)))

    _run()
