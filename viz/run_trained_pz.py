import os
import argparse

import numpy as np
import pygame
import supersuit as ss
from stable_baselines3 import PPO

from env.config import EnvConfig
from env.pz_env import SwarmPZEnv
from env.visualizer import SwarmVisualizer


def find_model_file(filename):
    start_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(start_dir) == "viz":
        start_dir = os.path.dirname(start_dir)
    for root, _, files in os.walk(start_dir):
        if filename in files:
            return os.path.join(root, filename)
    return None


def create_eval_env(max_steps=600):
    config = EnvConfig()
    pz_env = SwarmPZEnv(config=config, max_steps=max_steps)
    if hasattr(ss, "black_death_v3"):
        pz_env = ss.black_death_v3(pz_env)
    elif hasattr(ss, "black_death_v2"):
        pz_env = ss.black_death_v2(pz_env)
    pz_env = ss.clip_actions_v0(pz_env)
    pz_env = ss.dtype_v0(pz_env, np.float32)
    return pz_env


def _resolve_model_path(model_arg: str) -> str:
    if model_arg and os.path.isfile(model_arg):
        return model_arg
    if model_arg and (not model_arg.endswith(".zip")) and os.path.isfile(model_arg + ".zip"):
        return model_arg + ".zip"

    name = os.path.basename(model_arg) if model_arg else "best_by_finished.zip"
    if not name.endswith(".zip"):
        name = name + ".zip"

    path = find_model_file(name)
    if path:
        return path

    if model_arg:
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_arg)
        if os.path.isfile(p):
            return p
        if (not p.endswith(".zip")) and os.path.isfile(p + ".zip"):
            return p + ".zip"

    raise FileNotFoundError(f"Модель не найдена: {model_arg}")


def _adapt_obs(obs: np.ndarray, expected_dim: int) -> np.ndarray:
    obs = np.asarray(obs, dtype=np.float32)
    if obs.ndim != 1:
        return obs
    if obs.shape[0] == expected_dim:
        return obs
    if obs.shape[0] > expected_dim:
        return obs[:expected_dim]
    out = np.zeros((expected_dim,), dtype=np.float32)
    out[: obs.shape[0]] = obs
    return out


def run_demo(model_arg: str, deterministic: bool, max_steps: int):
    model_path = _resolve_model_path(model_arg)

    print(f"Загрузка модели из {model_path}...")
    custom_objects = {
        "learning_rate": 0.0,
        "lr_schedule": lambda _: 0.0,
        "clip_range": lambda _: 0.0,
    }
    try:
        model = PPO.load(model_path, custom_objects=custom_objects)
    except Exception:
        model = PPO.load(model_path)

    expected_dim = None
    try:
        space = model.observation_space
        if hasattr(space, "shape") and space.shape is not None and len(space.shape) == 1:
            expected_dim = int(space.shape[0])
    except Exception:
        expected_dim = None

    env = create_eval_env(max_steps=max_steps)
    base = getattr(env, "unwrapped", env)

    viz = SwarmVisualizer(base.config)

    out = env.reset()
    if isinstance(out, tuple) and len(out) == 2:
        obs, _ = out
    else:
        obs = out

    done = False
    print("Запуск визуализации... (Нажми ESC для выхода)")

    while not done:
        actions = {}
        for agent_id, o in obs.items():
            if o is None:
                continue
            oo = o
            if expected_dim is not None:
                oo = _adapt_obs(o, expected_dim)
            act, _ = model.predict(oo, deterministic=deterministic)
            actions[agent_id] = act

        step_out = env.step(actions)
        if isinstance(step_out, tuple) and len(step_out) == 5:
            obs, _, terminations, truncations, _ = step_out
            done = bool(any(terminations.values()) or any(truncations.values()))
        elif isinstance(step_out, tuple) and len(step_out) == 4:
            obs, _, dones, _ = step_out
            done = bool(any(dones.values())) if isinstance(dones, dict) else bool(dones)
        else:
            raise RuntimeError("Неожиданный формат результата step()")

        sim = getattr(base, "sim", None)
        target_pos = getattr(base, "target_pos", None)
        goal_radius = getattr(base, "goal_radius", 3.0)

        in_goal_mask = None
        finished_mask = None
        if sim is not None and target_pos is not None:
            try:
                pos = sim.agents_pos
                alive = sim.agents_active
                d = np.linalg.norm(pos - target_pos, axis=1)
                in_goal_mask = (d <= float(goal_radius)) & alive
            except Exception:
                in_goal_mask = None

        finished_mask = getattr(base, "finished", None)
        if finished_mask is not None:
            finished_mask = np.asarray(finished_mask, dtype=bool)

        if sim is not None and target_pos is not None:
            if not viz.render(sim, target_pos, float(goal_radius), in_goal_mask=in_goal_mask, finished_mask=finished_mask):
                break

        keys = pygame.key.get_pressed()
        if keys[pygame.K_ESCAPE]:
            break

    pygame.quit()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="runs/run_20260118_203327/models/best_by_finished.zip",
    )
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--max-steps", type=int, default=600)
    args = parser.parse_args()

    run_demo(args.model, deterministic=(not args.stochastic), max_steps=args.max_steps)


if __name__ == "__main__":
    main()
