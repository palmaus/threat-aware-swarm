"""
trained_ppo.py — точка входа для обучения Threat-aware Swarm (PettingZoo + SuperSuit + SB3 PPO)

Запуск:
  python trained_ppo.py --total-timesteps 8_000_000 --device cuda

Примечания:
- Ожидает класс окружения в: env/pz_env.py (SwarmPZEnv)
- Ожидает EnvConfig в: env/config.py (EnvConfig)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecMonitor

def now_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def repo_root() -> Path:
    return Path(__file__).resolve().parent

def safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return default

def make_pz_env(max_steps: int, goal_radius: float, seed: int) -> Any:
    # Локальные импорты, чтобы скрипт мог запускаться при установке в editable или в Colab.
    from env.config import EnvConfig
    from env.pz_env import SwarmPZEnv

    cfg = EnvConfig()
    env = SwarmPZEnv(cfg, max_steps=max_steps, goal_radius=goal_radius)

    # Сеансирование (seeding) PettingZoo не всегда полностью детерминировано, но это помогает.
    # Также устанавливаем numpy/random глобально в set_global_seed.
    try:
        env.reset(seed=seed)
    except Exception:
        pass

    return env

def make_vec_env(
    max_steps: int,
    goal_radius: float,
    seed: int,
    num_vec_envs: int,
    num_cpus: int,
) -> Any:
    pz_env = make_pz_env(max_steps=max_steps, goal_radius=goal_radius, seed=seed)

    # PettingZoo ParallelEnv -> VecEnv (по одному "окружению" на агента)
    vec_env = ss.pettingzoo_env_to_vec_env_v1(pz_env)

    # Опционально конкатенируем несколько копий (рандомизация через глобальный сид).
    # Это увеличивает размер батча и стабилизирует PPO.
    if num_vec_envs > 1:
        vec_env = ss.concat_vec_envs_v1(
            vec_env,
            num_vec_envs=num_vec_envs,
            num_cpus=num_cpus,
            base_class="stable_baselines3",
        )

    # Мониторы для статистики по эпизодам
    vec_env = VecMonitor(vec_env)
    return vec_env


class SwarmInfoMetricsCallback(BaseCallback):
    """
    Логирует метрики, специфичные для роя, взятые из infos (ожидается, что окружение кладёт ключи в dict info).
    Мы агрегируем по векторизованным окружениям и логируем:
      swarm/alive_frac
      swarm/finished_frac
      swarm/in_goal_frac
      swarm/mean_dist
      swarm/mean_risk_p
      swarm/min_neighbor_dist_mean
    """
    def __init__(self, log_every_steps: int = 2048, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.log_every_steps = int(log_every_steps)
        self._last_log_step = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_log_step < self.log_every_steps:
            return True

        infos = self.locals.get("infos", None)
        if not infos:
            return True

        # infos — список словарей длины n_envs
        alive = []
        finished = []
        in_goal = []
        dist = []
        risk = []
        mind = []

        for inf in infos:
            if not isinstance(inf, dict):
                continue
            alive.append(safe_float(inf.get("alive", 1.0), 1.0))
            finished.append(1.0 if inf.get("finished", False) else 0.0)
            in_goal.append(1.0 if inf.get("in_goal", False) else 0.0)
            dist.append(safe_float(inf.get("dist", np.nan)))
            risk.append(safe_float(inf.get("risk_p", 0.0), 0.0))
            mind.append(safe_float(inf.get("min_neighbor_dist", np.nan)))

        def nanmean(xs: List[float]) -> float:
            arr = np.array(xs, dtype=np.float32)
            return float(np.nanmean(arr)) if arr.size else float("nan")

        self.logger.record("swarm/alive_frac", nanmean(alive))
        self.logger.record("swarm/finished_frac", nanmean(finished))
        self.logger.record("swarm/in_goal_frac", nanmean(in_goal))
        self.logger.record("swarm/mean_dist", nanmean(dist))
        self.logger.record("swarm/mean_risk_p", nanmean(risk))

        # Игнорируем нули/NaN для минимальной дистанции до соседей (0 часто означает "не вычислено")
        mind_arr = np.array(mind, dtype=np.float32)
        mind_arr = mind_arr[np.isfinite(mind_arr) & (mind_arr > 1e-6)]
        self.logger.record("swarm/min_neighbor_dist_mean", float(np.nanmean(mind_arr)) if mind_arr.size else float("nan"))

        self._last_log_step = self.num_timesteps
        return True


class SaveRunMetaCallback(BaseCallback):
    """
    Записывает meta/run.json один раз (при старте обучения) и обновляет его последней информацией об eval.
    """
    def __init__(self, run_dir: Path, meta: Dict[str, Any], verbose: int = 0):
        super().__init__(verbose=verbose)
        self.run_dir = run_dir
        self.meta_path = run_dir / "meta" / "run.json"
        self.meta = meta

    def _on_training_start(self) -> None:
        self.meta_path.parent.mkdir(parents=True, exist_ok=True)
        self.meta_path.write_text(json.dumps(self.meta, ensure_ascii=False, indent=2), encoding="utf-8")

    def update_eval(self, eval_dict: Dict[str, Any]) -> None:
        self.meta.setdefault("eval_history", [])
        self.meta["eval_history"].append(eval_dict)
        # держим последний
        self.meta["last_eval"] = eval_dict
        try:
            self.meta_path.write_text(json.dumps(self.meta, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _on_step(self) -> bool:
        return True


class EvalToMetaCallback(BaseCallback):
    """
    Обёртка для EvalCallback, которая также отправляет результаты в run.json через SaveRunMetaCallback.
    """
    def __init__(self, eval_cb: EvalCallback, meta_cb: SaveRunMetaCallback):
        super().__init__(verbose=0)
        self.eval_cb = eval_cb
        self.meta_cb = meta_cb

    def _on_training_start(self) -> None:
        self.eval_cb.init_callback(self.model)

    def _on_step(self) -> bool:
        # Даем EvalCallback выполнить свою работу
        cont = self.eval_cb.on_step()
        # Если произошла оценка, EvalCallback хранит last_mean_reward и n_eval_episodes и т.д.
        # К сожалению, он не предоставляет чистого хука; мы детектируем по счётчикам timesteps.
        # Логируем минимальную запись периодически.
        if self.eval_cb.n_calls > 0 and self.eval_cb.eval_freq > 0:
            if (self.eval_cb.n_calls % self.eval_cb.eval_freq) == 0:
                rec = {
                    "timestep": int(self.num_timesteps),
                    "mean_reward": float(getattr(self.eval_cb, "last_mean_reward", np.nan)),
                    "best_mean_reward": float(getattr(self.eval_cb, "best_mean_reward", np.nan)),
                    "time": datetime.now().isoformat(timespec="seconds"),
                }
                self.meta_cb.update_eval(rec)
        return cont


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="", help="Optional run name suffix")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--total-timesteps", type=int, default=8_000_000)
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--goal-radius", type=float, default=3.0)

    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--num-vec-envs", type=int, default=1, help="How many copies of the VecEnv to concat (batch size multiplier)")
    ap.add_argument("--num-cpus", type=int, default=1)

    # PPO гиперпараметры (разумные значения по умолчанию)
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--n-steps", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--n-epochs", type=int, default=10)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--ent-coef", type=float, default=0.0)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--clip-range", type=float, default=0.2)
    ap.add_argument("--target-kl", type=float, default=0.01)

    ap.add_argument("--net-arch", default="256,256", help="Comma-separated hidden layer sizes, e.g. 256,256")
    ap.add_argument("--log-std-init", type=float, default=-1.0)

    # Сохранение / оценка
    ap.add_argument("--eval-freq", type=int, default=200_000)
    ap.add_argument("--n-eval-episodes", type=int, default=20)
    ap.add_argument("--checkpoint-freq", type=int, default=1_000_000)
    ap.add_argument("--deterministic-eval", action="store_true")

    ap.add_argument("--out-dir", default="train/models", help="Where to create run_*/")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    run_id = f"run_{now_id()}" + (f"_{args.run_name}" if args.run_name else "")
    run_dir = repo_root() / args.out_dir / run_id
    (run_dir / "meta").mkdir(parents=True, exist_ok=True)
    (run_dir / "models").mkdir(parents=True, exist_ok=True)
    (run_dir / "tb").mkdir(parents=True, exist_ok=True)

    # Логгер (SB3)
    sb3_logger = configure(str(run_dir / "tb"), ["stdout", "tensorboard"])

    # Создаём окружения
    env = make_vec_env(
        max_steps=args.max_steps,
        goal_radius=args.goal_radius,
        seed=args.seed,
        num_vec_envs=args.num_vec_envs,
        num_cpus=args.num_cpus,
    )

    eval_env = make_vec_env(
        max_steps=args.max_steps,
        goal_radius=args.goal_radius,
        seed=args.seed + 10_000,
        num_vec_envs=1,
        num_cpus=1,
    )

    # Парсим архитектуру сети
    net_arch = [int(x.strip()) for x in args.net_arch.split(",") if x.strip()]
    policy_kwargs = dict(net_arch=net_arch, log_std_init=args.log_std_init)

    meta = {
        "run_id": run_id,
        "time_start": datetime.now().isoformat(timespec="seconds"),
        "algo": "PPO",
        "seed": args.seed,
        "env": {
            "max_steps": args.max_steps,
            "goal_radius": args.goal_radius,
            "num_vec_envs": args.num_vec_envs,
        },
        "hparams": {
            "learning_rate": args.learning_rate,
            "n_steps": args.n_steps,
            "batch_size": args.batch_size,
            "n_epochs": args.n_epochs,
            "gamma": args.gamma,
            "gae_lambda": args.gae_lambda,
            "ent_coef": args.ent_coef,
            "vf_coef": args.vf_coef,
            "clip_range": args.clip_range,
            "target_kl": args.target_kl,
            "net_arch": net_arch,
            "log_std_init": args.log_std_init,
        },
        "paths": {
            "run_dir": str(run_dir),
            "tb": str(run_dir / "tb"),
            "models": str(run_dir / "models"),
        },
    }

    meta_cb = SaveRunMetaCallback(run_dir=run_dir, meta=meta)

    # Периодические контрольные точки
    ckpt_cb = CheckpointCallback(
        save_freq=max(args.checkpoint_freq // max(1, env.num_envs), 1),
        save_path=str(run_dir / "models"),
        name_prefix="swarm_ppo",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    # Eval callback (лучшая модель по среднему вознаграждению на eval окружении)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(run_dir / "models"),
        log_path=str(run_dir / "meta"),
        eval_freq=max(args.eval_freq // max(1, env.num_envs), 1),
        n_eval_episodes=args.n_eval_episodes,
        deterministic=args.deterministic_eval,
        render=False,
        verbose=0,
    )

    eval_to_meta = EvalToMetaCallback(eval_cb=eval_cb, meta_cb=meta_cb)

    metrics_cb = SwarmInfoMetricsCallback(log_every_steps=2048)

    callbacks = CallbackList([meta_cb, metrics_cb, ckpt_cb, eval_to_meta])

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        clip_range=args.clip_range,
        target_kl=args.target_kl,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=str(run_dir / "tb"),
        device=args.device,
    )
    model.set_logger(sb3_logger)

    try:
        model.learn(total_timesteps=int(args.total_timesteps), callback=callbacks, progress_bar=True)
    finally:
        # Всегда сохраняем финальную модель
        final_path = run_dir / "models" / "final.zip"
        model.save(str(final_path))

        meta["time_end"] = datetime.now().isoformat(timespec="seconds")
        meta["final_model"] = str(final_path)
        (run_dir / "meta" / "run.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        env.close()
        eval_env.close()

    print(f"[ОК] Запуск сохранён в: {run_dir}")
    print(f"     Финальная модель: {final_path}")


if __name__ == "__main__":
    main()
