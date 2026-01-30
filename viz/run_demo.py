"""Demo runner for environment (random policy).

Запуск:
  python -m viz.run_demo
"""

from __future__ import annotations

import numpy as np

from env.pz_env import SwarmPZEnv
from env.visualizer import SwarmVisualizer


def main():
    env = SwarmPZEnv(render_mode=None)
    obs, _ = env.reset()

    viz = SwarmVisualizer(config=env.config)

    for _ in range(600):
        actions = {a: env.action_space(a).sample() for a in env.agents}
        obs, rewards, terms, truncs, infos = env.step(actions)
        # визуализация
        viz.render(env.sim)
        if any(terms.values()) or any(truncs.values()):
            obs, _ = env.reset()

    viz.close()


if __name__ == "__main__":
    main()
