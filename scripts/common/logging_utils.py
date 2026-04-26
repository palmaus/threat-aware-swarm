from __future__ import annotations

import os
import shutil
import socket
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


def flatten_dict(d: dict[str, Any], prefix: str = "", sep: str = ".") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{sep}{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten_dict(v, key, sep=sep))
        else:
            out[key] = v
    return out


def _normalize_tags(tags: Any) -> dict[str, str]:
    if tags is None:
        return {}
    if isinstance(tags, dict):
        return {str(k): str(v) for k, v in tags.items()}
    if isinstance(tags, (list, tuple, set)):
        return {str(t): "true" for t in tags if str(t).strip()}
    if isinstance(tags, str):
        return {t.strip(): "true" for t in tags.split(",") if t.strip()}
    return {}


def resolve_tracking_cfg(hydra_cfg: Any | None) -> dict[str, Any]:
    if hydra_cfg is not None and getattr(hydra_cfg, "get", None):
        tracking = hydra_cfg.get("tracking", None)
        if tracking:
            return tracking
        logging_cfg = hydra_cfg.get("logging", None)
        if logging_cfg:
            try:
                tracking = logging_cfg.get("tracking", None)
            except Exception:
                tracking = None
            if tracking:
                return tracking
    return {}


def init_mlflow(cfg: dict[str, Any], run_name: str):
    if not cfg or not bool(cfg.get("enabled", False)):
        return None
    try:
        import mlflow
    except Exception:
        return None
    try:
        tracking_uri = str(cfg.get("tracking_uri", "")).strip()
        if tracking_uri:
            parsed = urlparse(tracking_uri)
            host = parsed.hostname
            if host:
                try:
                    socket.getaddrinfo(host, parsed.port or 80, proto=socket.IPPROTO_TCP)
                except OSError as exc:
                    print(f"[WARN] MLflow disabled (unreachable {host}): {exc}")
                    return None
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        experiment = str(cfg.get("experiment", "")).strip()
        if experiment:
            mlflow.set_experiment(experiment)
        run = mlflow.start_run(run_name=run_name)
        tags = _normalize_tags(cfg.get("tags"))
        if tags:
            mlflow.set_tags(tags)
        return run
    except Exception as exc:
        print(f"[WARN] MLflow disabled: {exc}")
        return None


def init_clearml(cfg: dict[str, Any], run_name: str, hydra_cfg: Any | None = None):
    if not cfg or not bool(cfg.get("enabled", False)):
        return None
    has_creds = bool(
        os.getenv("CLEARML_API_ACCESS_KEY")
        or os.getenv("CLEARML_API_SECRET_KEY")
        or Path("~/.clearml.conf").expanduser().exists()
    )
    if not has_creds:
        print("[WARN] ClearML disabled: credentials not configured.")
        return None
    try:
        from clearml import Task
    except Exception:
        return None
    project = str(cfg.get("project", "Threat-Aware-Swarm"))
    base_name = str(cfg.get("task_name", "train_ppo"))
    task_name = f"{base_name}_{run_name}" if run_name else base_name
    tags = cfg.get("tags", [])
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",") if t.strip()]
    output_uri = str(cfg.get("output_uri", "")).strip()
    task = Task.init(
        project_name=project,
        task_name=task_name,
        tags=list(tags) if tags else None,
        output_uri=output_uri or None,
        auto_connect_frameworks={"tensorboard": True, "pytorch": True, "matplotlib": True},
    )
    try:
        from omegaconf import OmegaConf

        if hydra_cfg is not None:
            task.connect(OmegaConf.to_container(hydra_cfg, resolve=True))
    except Exception:
        pass
    return task


def log_metrics(mlflow_run, clearml_task, metrics: dict[str, float], step: int):
    if not metrics:
        return
    if mlflow_run is not None:
        try:
            import mlflow

            mlflow.log_metrics(metrics, step=step)
        except Exception:
            pass
    if clearml_task is not None:
        try:
            logger = clearml_task.get_logger()
            for key, value in metrics.items():
                logger.report_scalar(title=key, series="value", value=value, iteration=step)
        except Exception:
            pass


def collect_system_metrics() -> dict[str, float]:
    metrics: dict[str, float] = {}
    try:
        import os

        import psutil

        proc = psutil.Process(os.getpid())
        vm = psutil.virtual_memory()
        metrics["system/cpu_percent"] = float(psutil.cpu_percent(interval=None))
        metrics["system/ram_percent"] = float(vm.percent)
        metrics["system/ram_used_gb"] = float(vm.used / (1024**3))
        metrics["system/proc_rss_mb"] = float(proc.memory_info().rss / (1024**2))
    except Exception:
        pass

    try:
        import torch

        if torch.cuda.is_available():
            metrics["system/gpu_mem_alloc_mb"] = float(torch.cuda.memory_allocated() / (1024**2))
            metrics["system/gpu_mem_reserved_mb"] = float(torch.cuda.memory_reserved() / (1024**2))
            total = float(torch.cuda.get_device_properties(0).total_memory / (1024**2))
            metrics["system/gpu_mem_total_mb"] = total
    except Exception:
        pass

    return metrics


def log_params(mlflow_run, clearml_task, params: dict[str, Any]):
    if mlflow_run is not None:
        try:
            import mlflow

            flat = flatten_dict(params)
            mlflow.log_params({k: v for k, v in flat.items() if v is not None})
        except Exception:
            pass
    if clearml_task is not None:
        try:
            clearml_task.connect(params)
        except Exception:
            pass


def log_artifacts_mlflow(mlflow_run, paths: list[Path], artifact_path: str | None = None):
    if mlflow_run is None:
        return
    try:
        import mlflow
    except Exception:
        return
    for p in paths:
        if p is None or not p.exists():
            continue
        try:
            if p.is_dir():
                mlflow.log_artifacts(str(p), artifact_path=artifact_path)
            else:
                mlflow.log_artifact(str(p), artifact_path=artifact_path)
        except Exception:
            pass


def log_artifacts_clearml(clearml_task, paths: list[Path], name_prefix: str = ""):
    if clearml_task is None:
        return
    prefix = f"{name_prefix}_" if name_prefix else ""
    for p in paths:
        if p is None or not p.exists():
            continue
        try:
            clearml_task.upload_artifact(name=f"{prefix}{p.name}", artifact_object=str(p))
        except Exception:
            pass


def download_clearml_artifact(ref: str, download_dir: Path) -> Path | None:
    """
    Скачать артефакт из ClearML по ссылке вида:
      clearml:<task_id>
      clearml:<task_id>/<artifact_name>
      clearml:<task_id>:<artifact_name>
    Возвращает путь к локальной копии артефакта (если найден).
    """
    if not ref or not ref.startswith("clearml:"):
        return None
    token = ref.split("clearml:", 1)[1].strip()
    if not token:
        return None

    task_id = token
    artifact_name = ""
    if "/" in token:
        task_id, artifact_name = token.split("/", 1)
    elif ":" in token:
        task_id, artifact_name = token.split(":", 1)
    task_id = task_id.strip()
    artifact_name = artifact_name.strip()
    if not task_id:
        return None

    try:
        from clearml import Task
    except Exception as exc:
        raise RuntimeError("ClearML is required for resume.model=clearml:...") from exc

    task = Task.get_task(task_id=task_id)
    if task is None:
        raise FileNotFoundError(f"ClearML task not found: {task_id}")

    artifacts = getattr(task, "artifacts", {}) or {}
    if not isinstance(artifacts, dict):
        artifacts = {}

    candidates = []
    if artifact_name:
        candidates.append(artifact_name)
    candidates.extend(
        [
            "models_best_by_finished.zip",
            "models_final.zip",
            "models_interrupt.zip",
            "final.zip",
        ]
    )

    found = None
    for name in candidates:
        if name in artifacts:
            found = artifacts[name]
            artifact_name = name
            break
    if found is None:
        raise FileNotFoundError(f"ClearML artifact not found for task={task_id}. Available: {list(artifacts.keys())}")

    local_copy = found.get_local_copy()
    if not local_copy:
        raise FileNotFoundError(f"ClearML artifact download failed: {artifact_name}")
    src = Path(local_copy)
    download_dir.mkdir(parents=True, exist_ok=True)
    dst = download_dir / artifact_name
    try:
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
        return dst
    except Exception:
        return src


def log_mlflow_pyfunc_model(model_path: Path, register_name: str | None = None):
    try:
        import mlflow
        import mlflow.pyfunc
        import stable_baselines3
    except Exception:
        return

    class Sb3PpoModel(mlflow.pyfunc.PythonModel):
        def load_context(self, context):
            self.model = stable_baselines3.PPO.load(context.artifacts["sb3_model"])

        def predict(self, context, model_input):
            import numpy as np

            obs = np.asarray(model_input)
            actions, _ = self.model.predict(obs, deterministic=True)
            return actions

    try:
        mlflow.pyfunc.log_model(
            artifact_path="models/sb3",
            python_model=Sb3PpoModel(),
            artifacts={"sb3_model": str(model_path)},
        )
        if register_name:
            run_id = mlflow.active_run().info.run_id
            model_uri = f"runs:/{run_id}/models/sb3"
            mlflow.register_model(model_uri, register_name)
    except Exception:
        return
