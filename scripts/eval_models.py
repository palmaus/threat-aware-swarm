"""
Запуск:
  python scripts/eval_models.py --registry model_registry.csv --mode both \
      --n-episodes 20 --max-steps 600 --deterministic --seed 0
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import supersuit as ss
from stable_baselines3 import PPO

VERSION = "v6"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _as_obs(x):
    # Преобразование кортежа в наблюдение, если это необходимо
    if isinstance(x, tuple) and len(x) == 2:
        return x[0]
    return x


def make_vec_env(max_steps: int, goal_radius: float, seed: int):
    from env.config import EnvConfig
    from env.pz_env import SwarmPZEnv

    cfg = EnvConfig()
    pz_env = SwarmPZEnv(cfg, max_steps=max_steps, goal_radius=goal_radius)

    # Возвращает MarkovVectorEnv из SuperSuit (API Gymnasium vector, 5-элементный кортеж)
    vec_env = ss.pettingzoo_env_to_vec_env_v1(pz_env)

    # Установка seed через reset (у MarkovVectorEnv нет метода .seed())
    try:
        vec_env.reset(seed=seed)
    except TypeError:
        vec_env.reset()
    return vec_env


def adapt_obs_to_dim(obs: np.ndarray, expected_dim: int) -> np.ndarray:
    # Адаптация наблюдений к ожидаемым размерам
    obs = np.asarray(obs)
    if obs.ndim == 1:
        cur = obs.shape[0]
        if cur == expected_dim:
            return obs
        if cur > expected_dim:
            return obs[:expected_dim]
        pad = np.zeros((expected_dim - cur,), dtype=obs.dtype)
        return np.concatenate([obs, pad], axis=0)

    if obs.ndim == 2:
        cur = obs.shape[1]
        if cur == expected_dim:
            return obs
        if cur > expected_dim:
            return obs[:, :expected_dim]
        pad = np.zeros((obs.shape[0], expected_dim - cur), dtype=obs.dtype)
        return np.concatenate([obs, pad], axis=1)

    return obs


def safe_float(x: Any, default: float = float("nan")) -> float:
    # Безопасное преобразование в float с обработкой исключений
    try:
        return float(x)
    except Exception:
        return default


def aggregate_infos(infos: Any) -> Dict[str, float]:
    """
    Ожидается:
      - Gymnasium VectorEnv: infos — это list[dict] длины n_env
      - Старый gym: info может быть dict; нормализуем до list
    """
    if isinstance(infos, dict):
        infos_list = [infos]
    else:
        infos_list = list(infos) if infos is not None else []

    alive, finished, in_goal, dist, risk, mind = [], [], [], [], [], []
    for inf in infos_list:
        if not isinstance(inf, dict):
            continue
        a = inf.get("alive", 1.0)
        alive.append(1.0 if bool(a) else 0.0 if isinstance(a, bool) else safe_float(a, 1.0))
        finished.append(1.0 if inf.get("finished", False) else 0.0)
        in_goal.append(1.0 if inf.get("in_goal", False) else 0.0)
        dist.append(safe_float(inf.get("dist", np.nan)))
        risk.append(safe_float(inf.get("risk_p", 0.0), 0.0))
        mind.append(safe_float(inf.get("min_neighbor_dist", np.nan)))

    def nanmean(xs):
        arr = np.array(xs, dtype=np.float32)
        return float(np.nanmean(arr)) if arr.size else float("nan")

    mind_arr = np.array(mind, dtype=np.float32)
    mind_arr = mind_arr[np.isfinite(mind_arr) & (mind_arr > 1e-6)]
    min_neighbor = float(np.nanmean(mind_arr)) if mind_arr.size else float("nan")

    return {
        "alive_frac": nanmean(alive),
        "finished_frac": nanmean(finished),
        "in_goal_frac": nanmean(in_goal),
        "mean_dist": nanmean(dist),
        "mean_risk_p": nanmean(risk),
        "min_neighbor_dist_mean": min_neighbor,
    }


def load_model(path: Path) -> PPO:
    # Загрузка модели PPO с пользовательскими параметрами
    custom_objects = {
        "learning_rate": 0.0,
        "lr_schedule": (lambda _: 0.0),
        "clip_range": 0.2,
    }
    return PPO.load(str(path), device="cpu", custom_objects=custom_objects)


def step_env(env, actions):
    """
    Возвращает: obs, rewards, dones, infos
    Работает с:
      - 5-элементным кортежем: obs, rewards, terminated, truncated, infos
      - 4-элементным кортежем: obs, rewards, dones, infos
    """
    out = env.step(actions)
    if isinstance(out, tuple) and len(out) == 5:
        obs, rewards, terminated, truncated, infos = out
        dones = np.logical_or(terminated, truncated)
        return obs, rewards, dones, infos
    if isinstance(out, tuple) and len(out) == 4:
        obs, rewards, dones, infos = out
        return obs, rewards, dones, infos
    raise RuntimeError(f"Неподдерживаемый формат возвращаемого значения env.step(): {type(out)} len={len(out) if hasattr(out,'__len__') else 'NA'}")


def eval_one_model(
    model_path: Path,
    mode: str,
    n_episodes: int,
    max_steps: int,
    deterministic: bool,
    seed: int,
    goal_radius: float,
) -> Dict[str, Any]:
    # Оценка одной модели
    assert mode in ("fixed", "random")

    model = load_model(model_path)
    expected_dim = int(model.observation_space.shape[0])

    env = make_vec_env(max_steps=max_steps, goal_radius=goal_radius, seed=seed)

    ep_metrics: List[Dict[str, float]] = []
    ep_steps: List[int] = []

    for ep in range(n_episodes):
        ep_seed = seed if mode == "fixed" else (seed + 1000 + ep)

        try:
            obs = env.reset(seed=ep_seed)
        except TypeError:
            obs = env.reset()
        obs = adapt_obs_to_dim(_as_obs(obs), expected_dim)

        last_infos = None
        for t in range(max_steps):
            obs_for_model = adapt_obs_to_dim(obs, expected_dim)
            actions, _ = model.predict(obs_for_model, deterministic=deterministic)

            obs, rewards, dones, infos = step_env(env, actions)
            obs = adapt_obs_to_dim(_as_obs(obs), expected_dim)
            last_infos = infos

            if np.all(dones):
                ep_steps.append(t + 1)
                break
        else:
            ep_steps.append(max_steps)

        if last_infos is not None:
            ep_metrics.append(aggregate_infos(last_infos))

    def mean_key(k: str) -> float:
        vals = [m.get(k, np.nan) for m in ep_metrics]
        arr = np.array(vals, dtype=np.float32)
        return float(np.nanmean(arr)) if arr.size else float("nan")

    out = {
        "eval_status": "ok",
        "eval_error": "",
        "steps": float(np.mean(ep_steps)) if ep_steps else float("nan"),
        "alive_frac": mean_key("alive_frac"),
        "finished_frac": mean_key("finished_frac"),
        "in_goal_frac": mean_key("in_goal_frac"),
        "mean_dist": mean_key("mean_dist"),
        "mean_risk_p": mean_key("mean_risk_p"),
        "min_neighbor_dist_mean": mean_key("min_neighbor_dist_mean"),
        "expected_obs_dim": expected_dim,
        "last_eval_time": time.time(),
        "eval_models_version": VERSION,
    }
    try:
        env.close()
    except Exception:
        pass
    return out


def read_registry(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    # Чтение реестра моделей из CSV файла
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        for r in reader:
            rows.append(r)
    return rows, fieldnames


def write_registry(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    # Запись обновлённого реестра моделей в CSV файл
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def ensure_fields(fieldnames: List[str], prefix: str) -> List[str]:
    # Убедиться, что все необходимые поля присутствуют в реестре
    required = [
        f"{prefix}alive_frac",
        f"{prefix}finished_frac",
        f"{prefix}in_goal_frac",
        f"{prefix}mean_dist",
        f"{prefix}mean_risk_p",
        f"{prefix}min_neighbor_dist_mean",
        f"{prefix}steps",
        f"{prefix}expected_obs_dim",
        f"{prefix}eval_status",
        f"{prefix}eval_error",
        f"{prefix}last_eval_time",
        f"{prefix}eval_models_version",
    ]
    for c in required:
        if c not in fieldnames:
            fieldnames.append(c)
    return fieldnames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", required=True)
    ap.add_argument("--mode", default="fixed", choices=["fixed", "random", "both"])
    ap.add_argument("--n-episodes", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--goal-radius", type=float, default=3.0)
    args = ap.parse_args()

    reg_path = Path(args.registry)
    rows, fieldnames = read_registry(reg_path)

    modes = ["fixed", "random"] if args.mode == "both" else [args.mode]
    for m in modes:
        fieldnames = ensure_fields(fieldnames, prefix=f"{m}_")
    write_registry(reg_path, rows, fieldnames)

    print(f"[INFO] eval_models.py {VERSION} -> {Path(__file__).resolve()}")

    for row in rows:
        model_path = Path(row.get("path", ""))
        if not model_path.exists():
            for m in modes:
                row[f"{m}_eval_status"] = "missing"
                row[f"{m}_eval_error"] = "model file not found"
                row[f"{m}_last_eval_time"] = time.time()
                row[f"{m}_eval_models_version"] = VERSION
            write_registry(reg_path, rows, fieldnames)
            print(f"[ERR] Missing: {model_path}")
            continue

        for m in modes:
            try:
                res = eval_one_model(
                    model_path=model_path,
                    mode=m,
                    n_episodes=args.n_episodes,
                    max_steps=args.max_steps,
                    deterministic=args.deterministic,
                    seed=args.seed,
                    goal_radius=args.goal_radius,
                )
                for k, v in res.items():
                    row[f"{m}_{k}"] = v
                print(f"[OK] {m:6s} {model_path} finished_frac={row.get(f'{m}_finished_frac')}")
            except Exception as e:
                row[f"{m}_eval_status"] = "error"
                row[f"{m}_eval_error"] = str(e)
                row[f"{m}_last_eval_time"] = time.time()
                row[f"{m}_eval_models_version"] = VERSION
                print(f"[ERR] {m:6s} {model_path}: {e}")
            write_registry(reg_path, rows, fieldnames)

    print(f"[OK] Registry updated -> {reg_path}")


if __name__ == "__main__":
    main()
