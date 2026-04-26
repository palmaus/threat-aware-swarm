"""
Запуск:
  python -m scripts.eval.eval_models --registry model_registry.csv --mode both \
      --n-episodes 20 --max-steps 600 --deterministic --seed 0
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from gymnasium import spaces
from stable_baselines3 import PPO

try:
    from sb3_contrib import RecurrentPPO
except Exception:  # pragma: no cover - опциональная зависимость
    RecurrentPPO = None

from common.runtime.env_factory import make_vec_env as _common_make_vec_env
from scripts.common.episode_metrics import aggregate_vector_infos
from scripts.common.rl_episode_runner import VectorModelState, predict_vector_model_actions
from scripts.common.path_utils import resolve_repo_path
from scripts.common.rng_manager import SeedManager

VERSION = "v7"


def _as_obs(x):
    # Нормализуем reset() в наблюдение, так как Gymnasium может вернуть (obs, info).
    if isinstance(x, tuple) and len(x) == 2:
        return x[0]
    return x


def _read_json(path: Path) -> dict[str, Any]:
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}
    return {}


def _find_run_dir(model_path: Path) -> Path | None:
    parent = model_path.parent
    if parent.name == "models":
        return parent.parent
    return None


def _load_model_meta(model_path: Path) -> dict[str, Any]:
    meta = _read_json(Path(str(model_path) + ".meta.json"))
    run_dir = _find_run_dir(model_path)
    if run_dir is not None:
        run_meta = _read_json(run_dir / "meta" / "run.json")
        if run_meta:
            meta.setdefault("run_meta", run_meta)
    return meta


def _env_overrides_from_meta(model_meta: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key in ("env_config", "env_overrides"):
        value = model_meta.get(key)
        if isinstance(value, dict):
            payload.update(value)
    initial_stage = model_meta.get("initial_stage_params")
    if isinstance(initial_stage, dict):
        payload.update(initial_stage)
    run_meta = model_meta.get("run_meta")
    if isinstance(run_meta, dict):
        env_payload = run_meta.get("env")
        if isinstance(env_payload, dict):
            config_payload = env_payload.get("config")
            if isinstance(config_payload, dict):
                payload.update(config_payload)
            initial_stage = env_payload.get("initial_stage_params")
            if isinstance(initial_stage, dict):
                payload.update(initial_stage)
            if not payload:
                payload.update({k: v for k, v in env_payload.items() if k not in {"num_vec_envs"}})
    # Eval suites own episode length and success radius; keep those standardized.
    payload.pop("max_steps", None)
    payload.pop("goal_radius", None)
    return payload


def make_vec_env(max_steps: int, goal_radius: float, seed: int, env_overrides: dict[str, Any] | None = None):
    return _common_make_vec_env(
        max_steps=max_steps,
        goal_radius=goal_radius,
        seed=seed,
        env_overrides=env_overrides,
        train_wrappers=False,
        vec_monitor=False,
    )


def safe_float(x: Any, default: float = float("nan")) -> float:
    # Безопасное приведение к float для логирования и метрик.
    try:
        return float(x)
    except Exception:
        return default


def aggregate_infos(infos: Any) -> dict[str, float]:
    return aggregate_vector_infos(infos)


def load_model(path: Path) -> PPO:
    # Загружаем PPO в режиме eval, чтобы избежать влияния планировщика на метрики.
    custom_objects = {
        "learning_rate": 0.0,
        "lr_schedule": (lambda _: 0.0),
        "clip_range": 0.2,
    }
    try:
        return PPO.load(str(path), device="cpu", custom_objects=custom_objects)
    except Exception:
        if RecurrentPPO is None:
            raise
        return RecurrentPPO.load(str(path), device="cpu", custom_objects=custom_objects)


def step_env(env, actions):
    """
    Возвращает: obs, rewards, dones, infos.
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
    raise RuntimeError(
        f"Неподдерживаемый формат возвращаемого значения env.step(): {type(out)} len={len(out) if hasattr(out, '__len__') else 'NA'}"
    )


def eval_one_model(
    model_path: Path,
    mode: str,
    n_episodes: int,
    max_steps: int,
    deterministic: bool,
    seed: int,
    goal_radius: float,
) -> dict[str, Any]:
    # Оценка одной модели в фиксированном или случайном режиме.
    assert mode in ("fixed", "random")

    model = load_model(model_path)
    is_recurrent = bool(getattr(getattr(model, "policy", None), "lstm_actor", None))
    space = getattr(model, "observation_space", None)
    if not isinstance(space, spaces.Dict):
        return {
            "eval_status": "несовместимо",
            "eval_error": "unsupported_obs",
            "eval_error_message": "Поддерживаются только Dict‑наблюдения (obs@1694:v5).",
            "env_schema_version": "unknown",
        }

    model_meta = _load_model_meta(model_path)
    env_schema_version = model_meta.get("env_schema_version") or "unknown"
    env_overrides = _env_overrides_from_meta(model_meta)

    env = make_vec_env(max_steps=max_steps, goal_radius=goal_radius, seed=seed, env_overrides=env_overrides)

    ep_metrics: list[dict[str, float]] = []
    ep_steps: list[int] = []
    ep_returns: list[float] = []
    rng = np.random.RandomState(seed + 1000) if mode == "random" else None

    for ep in range(n_episodes):
        ep_seed = (seed + ep) if mode == "fixed" else int(rng.randint(0, 2**31 - 1))

        try:
            obs = env.reset(seed=ep_seed)
        except TypeError:
            obs = env.reset()
        obs = _as_obs(obs)

        last_infos = None
        ep_return = 0.0
        model_state = VectorModelState()
        for t in range(max_steps):
            actions = predict_vector_model_actions(model, obs, model_state, deterministic=deterministic)

            obs, rewards, dones, infos = step_env(env, actions)
            obs = _as_obs(obs)
            last_infos = infos
            try:
                ep_return += float(np.mean(rewards))
            except Exception:
                ep_return += safe_float(rewards, 0.0)

            if is_recurrent:
                model_state.mark_dones(dones)
            if np.all(dones):
                ep_steps.append(t + 1)
                break
        else:
            ep_steps.append(max_steps)
        ep_returns.append(ep_return)

        if last_infos is not None:
            ep_metrics.append(aggregate_infos(last_infos))

    def mean_key(k: str) -> float:
        vals = [m.get(k, np.nan) for m in ep_metrics]
        arr = np.array(vals, dtype=np.float32)
        return float(np.nanmean(arr)) if arr.size else float("nan")

    def mean_std_key(k: str) -> tuple[float, float]:
        vals = [m.get(k, np.nan) for m in ep_metrics]
        arr = np.array(vals, dtype=np.float32)
        if not arr.size:
            return float("nan"), float("nan")
        return float(np.nanmean(arr)), float(np.nanstd(arr))

    fga_mean, fga_std = mean_std_key("finished_given_alive")
    alive_mean, alive_std = mean_std_key("alive_frac")
    dist_mean, dist_std = mean_std_key("mean_dist")

    out = {
        "eval_status": "ок",
        "eval_error": "",
        "eval_error_message": "",
        "env_schema_version": env_schema_version,
        "steps": float(np.mean(ep_steps)) if ep_steps else float("nan"),
        "eval_mean": float(np.mean(ep_returns)) if ep_returns else float("nan"),
        "eval_std": float(np.std(ep_returns)) if ep_returns else float("nan"),
        "eval_dt": datetime.now().isoformat(timespec="seconds"),
        "alive_frac": mean_key("alive_frac"),
        "finished_frac": mean_key("finished_frac"),
        "finished_given_alive": mean_key("finished_given_alive"),
        "in_goal_frac": mean_key("in_goal_frac"),
        "mean_dist": mean_key("mean_dist"),
        "mean_risk_p": mean_key("mean_risk_p"),
        "min_neighbor_dist_mean": mean_key("min_neighbor_dist_mean"),
        "finished_given_alive_mean": fga_mean,
        "finished_given_alive_std": fga_std,
        "alive_frac_mean": alive_mean,
        "alive_frac_std": alive_std,
        "mean_dist_mean": dist_mean,
        "mean_dist_std": dist_std,
        "last_eval_time": time.time(),
        "eval_models_version": VERSION,
    }
    try:
        env.close()
    except Exception:
        pass
    return out


def read_registry(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        for r in reader:
            rows.append(r)
    return rows, fieldnames


def write_registry(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def ensure_fields(fieldnames: list[str], prefix: str) -> list[str]:
    required = [
        f"{prefix}alive_frac",
        f"{prefix}finished_frac",
        f"{prefix}in_goal_frac",
        f"{prefix}mean_dist",
        f"{prefix}mean_risk_p",
        f"{prefix}min_neighbor_dist_mean",
        f"{prefix}steps",
        f"{prefix}eval_status",
        f"{prefix}eval_error",
        f"{prefix}eval_error_message",
        f"{prefix}last_eval_time",
        f"{prefix}eval_models_version",
    ]
    for c in required:
        if c not in fieldnames:
            fieldnames.append(c)
    return fieldnames


def ensure_eval_fields(fieldnames: list[str]) -> list[str]:
    required = [
        "eval_fixed_mean",
        "eval_fixed_std",
        "eval_random_mean",
        "eval_random_std",
        "eval_dt",
    ]
    for c in required:
        if c not in fieldnames:
            fieldnames.append(c)
    return fieldnames


def ensure_protocol_fields(fieldnames: list[str]) -> list[str]:
    required = [
        "eval_fixed/finished_given_alive_mean",
        "eval_fixed/finished_given_alive_std",
        "eval_fixed/alive_frac_mean",
        "eval_fixed/alive_frac_std",
        "eval_fixed/mean_dist_mean",
        "eval_fixed/mean_dist_std",
        "eval_fixed/error_message",
        "eval_random/finished_given_alive_mean",
        "eval_random/finished_given_alive_std",
        "eval_random/alive_frac_mean",
        "eval_random/alive_frac_std",
        "eval_random/mean_dist_mean",
        "eval_random/mean_dist_std",
        "eval_random/error_message",
        "eval_dt",
    ]
    for c in required:
        if c not in fieldnames:
            fieldnames.append(c)
    return fieldnames


def eval_run_dir(
    run_dir: Path,
    modes: list[str],
    n_episodes: int,
    max_steps: int,
    deterministic: bool,
    seed: int,
    goal_radius: float,
    suite_name: str | None = None,
) -> list[dict[str, Any]]:
    eval_dir = run_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    model_entries = []
    final_path = run_dir / "models" / "final.zip"
    if final_path.exists():
        model_entries.append(("final", final_path))
    best_path = run_dir / "models" / "best_by_finished.zip"
    if best_path.exists():
        model_entries.append(("best_by_finished", best_path))

    results: list[dict[str, Any]] = []
    for name, path in model_entries:
        row: dict[str, Any] = {
            "model_name": name,
            "path": str(path),
        }
        for m in modes:
            try:
                res = eval_one_model(
                    model_path=path,
                    mode=m,
                    n_episodes=n_episodes,
                    max_steps=max_steps,
                    deterministic=deterministic,
                    seed=seed,
                    goal_radius=goal_radius,
                )
                row.update({f"{m}_{k}": v for k, v in res.items()})
                row[f"eval_{m}_mean"] = res.get("eval_mean", float("nan"))
                row[f"eval_{m}_std"] = res.get("eval_std", float("nan"))
                row[f"eval_{m}/finished_given_alive_mean"] = res.get("finished_given_alive_mean", float("nan"))
                row[f"eval_{m}/finished_given_alive_std"] = res.get("finished_given_alive_std", float("nan"))
                row[f"eval_{m}/alive_frac_mean"] = res.get("alive_frac_mean", float("nan"))
                row[f"eval_{m}/alive_frac_std"] = res.get("alive_frac_std", float("nan"))
                row[f"eval_{m}/mean_dist_mean"] = res.get("mean_dist_mean", float("nan"))
                row[f"eval_{m}/mean_dist_std"] = res.get("mean_dist_std", float("nan"))
                row[f"eval_{m}/error_message"] = res.get("eval_error_message", "")
                row["eval_dt"] = res.get("eval_dt", datetime.now().isoformat(timespec="seconds"))
            except Exception as e:
                row[f"{m}_eval_status"] = "ошибка"
                row[f"{m}_eval_error"] = "исключение"
                row[f"{m}_eval_error_message"] = f"Ошибка: {e}"
                row[f"{m}_last_eval_time"] = time.time()
                row[f"{m}_eval_models_version"] = VERSION
                row[f"eval_{m}_mean"] = float("nan")
                row[f"eval_{m}_std"] = float("nan")
                row[f"eval_{m}/finished_given_alive_mean"] = float("nan")
                row[f"eval_{m}/finished_given_alive_std"] = float("nan")
                row[f"eval_{m}/alive_frac_mean"] = float("nan")
                row[f"eval_{m}/alive_frac_std"] = float("nan")
                row[f"eval_{m}/mean_dist_mean"] = float("nan")
                row[f"eval_{m}/mean_dist_std"] = float("nan")
                row[f"eval_{m}/error_message"] = f"Ошибка: {e}"
                row["eval_dt"] = datetime.now().isoformat(timespec="seconds")
        results.append(row)

    suffix = f"_{suite_name}" if suite_name else ""
    json_path = eval_dir / f"eval_results{suffix}.json"
    csv_path = eval_dir / f"eval_results{suffix}.csv"
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    fieldnames: list[str] = []
    for row in results:
        for k in row.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    return results


@dataclass
class EvalModelsConfig:
    registry: str = ""
    run_dir: str = ""
    mode: str = "both"
    suite: str = ""
    n_episodes: int = 20
    max_steps: int = 600
    seed: int = 0
    deterministic: bool = False
    goal_radius: float = 3.0


def run(cfg: EvalModelsConfig) -> None:
    args = cfg

    suite_cfg = None
    suite_name = args.suite.strip()
    if suite_name:
        cfg_path = Path(__file__).resolve().parents[2] / "configs" / "hydra" / "eval" / "eval_suite.yaml"
        if not cfg_path.exists():
            print(f"[ПРЕД] Файл suite не найден: {cfg_path}. Использую пустой набор.")
            suites = {}
        else:
            suites = yaml.safe_load(cfg_path.read_text()) or {}
        suite_cfg = (suites.get("suites", {}) or {}).get(suite_name)
        if not suite_cfg:
            raise SystemExit(f"Неизвестный suite: {suite_name}")
        args.mode = suite_cfg.get("mode", args.mode)
        args.n_episodes = int(suite_cfg.get("n_episodes", args.n_episodes))
        args.max_steps = int(suite_cfg.get("max_steps", args.max_steps))
        args.goal_radius = float(suite_cfg.get("goal_radius", args.goal_radius))
        args.seed = int(suite_cfg.get("seed", args.seed))

    if not args.registry and not args.run_dir:
        raise SystemExit("Нужно указать --registry или --run-dir")

    modes = ["fixed", "random"] if args.mode == "both" else [args.mode]

    SeedManager(int(args.seed)).seed_all()

    print(f"[ИНФО] eval_models.py {VERSION} -> {Path(__file__).resolve()}")

    if args.run_dir:
        run_dir = resolve_repo_path(args.run_dir)
        _results = eval_run_dir(
            run_dir=run_dir,
            modes=modes,
            n_episodes=args.n_episodes,
            max_steps=args.max_steps,
            deterministic=args.deterministic,
            seed=args.seed,
            goal_radius=args.goal_radius,
            suite_name=suite_name or None,
        )
        print(f"[ОК] Артефакты eval -> {run_dir / 'eval'}")
        if not args.registry:
            return

    reg_path = resolve_repo_path(args.registry)
    rows, fieldnames = read_registry(reg_path)
    for m in modes:
        fieldnames = ensure_fields(fieldnames, prefix=f"{m}_")
    fieldnames = ensure_eval_fields(fieldnames)
    fieldnames = ensure_protocol_fields(fieldnames)
    write_registry(reg_path, rows, fieldnames)

    def proto_key(mode: str, name: str) -> str:
        return f"eval_{mode}/{name}"

    for row in rows:
        model_path = Path(row.get("path", ""))
        if not model_path.exists():
            for m in modes:
                row[f"{m}_eval_status"] = "нет_файла"
                row[f"{m}_eval_error"] = "файл модели не найден"
                row[f"{m}_eval_error_message"] = "файл модели не найден"
                row[f"{m}_last_eval_time"] = time.time()
                row[f"{m}_eval_models_version"] = VERSION
                row[f"eval_{m}_mean"] = float("nan")
                row[f"eval_{m}_std"] = float("nan")
                row[proto_key(m, "finished_given_alive_mean")] = float("nan")
                row[proto_key(m, "finished_given_alive_std")] = float("nan")
                row[proto_key(m, "alive_frac_mean")] = float("nan")
                row[proto_key(m, "alive_frac_std")] = float("nan")
                row[proto_key(m, "mean_dist_mean")] = float("nan")
                row[proto_key(m, "mean_dist_std")] = float("nan")
                row[proto_key(m, "error_message")] = "файл модели не найден"
                row["eval_dt"] = datetime.now().isoformat(timespec="seconds")
            print(f"[ОШИБКА] Нет файла: {model_path}")
            write_registry(reg_path, rows, fieldnames)
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
                    if k in {"eval_mean", "eval_std", "eval_dt"}:
                        continue
                    row[f"{m}_{k}"] = v
                row[f"{m}_eval_error_message"] = res.get("eval_error_message", "")
                row[f"eval_{m}_mean"] = res.get("eval_mean", float("nan"))
                row[f"eval_{m}_std"] = res.get("eval_std", float("nan"))
                row[proto_key(m, "finished_given_alive_mean")] = res.get("finished_given_alive_mean", float("nan"))
                row[proto_key(m, "finished_given_alive_std")] = res.get("finished_given_alive_std", float("nan"))
                row[proto_key(m, "alive_frac_mean")] = res.get("alive_frac_mean", float("nan"))
                row[proto_key(m, "alive_frac_std")] = res.get("alive_frac_std", float("nan"))
                row[proto_key(m, "mean_dist_mean")] = res.get("mean_dist_mean", float("nan"))
                row[proto_key(m, "mean_dist_std")] = res.get("mean_dist_std", float("nan"))
                row[proto_key(m, "error_message")] = res.get("eval_error_message", "")
                row["eval_dt"] = res.get("eval_dt", datetime.now().isoformat(timespec="seconds"))
                print(f"[ОК] {m:6s} {model_path} finished_frac={row.get(f'{m}_finished_frac')}")
            except Exception as e:
                row[f"{m}_eval_status"] = "ошибка"
                row[f"{m}_eval_error"] = "исключение"
                row[f"{m}_eval_error_message"] = f"Ошибка: {e}"
                row[f"{m}_last_eval_time"] = time.time()
                row[f"{m}_eval_models_version"] = VERSION
                row[f"eval_{m}_mean"] = float("nan")
                row[f"eval_{m}_std"] = float("nan")
                row[proto_key(m, "finished_given_alive_mean")] = float("nan")
                row[proto_key(m, "finished_given_alive_std")] = float("nan")
                row[proto_key(m, "alive_frac_mean")] = float("nan")
                row[proto_key(m, "alive_frac_std")] = float("nan")
                row[proto_key(m, "mean_dist_mean")] = float("nan")
                row[proto_key(m, "mean_dist_std")] = float("nan")
                row[proto_key(m, "error_message")] = f"Ошибка: {e}"
                row["eval_dt"] = datetime.now().isoformat(timespec="seconds")
                print(f"[ОШИБКА] {m:6s} {model_path}: {e}")
        write_registry(reg_path, rows, fieldnames)

    print(f"[ОК] Реестр обновлён -> {reg_path}")


def main() -> None:
    import hydra
    from omegaconf import DictConfig, OmegaConf

    from scripts.common.hydra_utils import apply_schema

    @hydra.main(version_base=None, config_path="../../configs/hydra", config_name="eval/models")
    def _run(cfg: DictConfig) -> None:
        cfg = apply_schema(cfg, EvalModelsConfig)
        data = OmegaConf.to_container(cfg, resolve=True)
        run(EvalModelsConfig(**data))

    _run()


if __name__ == "__main__":
    main()
