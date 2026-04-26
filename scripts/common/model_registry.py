from __future__ import annotations

from pathlib import Path

from scripts.common.logging_utils import download_clearml_artifact
from scripts.common.mlflow_utils import download_mlflow_artifact
from scripts.common.path_utils import resolve_repo_path


def resolve_model_path(
    model_ref: str,
    *,
    artifact_path: str = "models/final.zip",
    download_dir: Path | None = None,
    tracking_uri: str | None = None,
) -> Path:
    """
    Resolve model reference from:
      - local paths
      - mlflow:<run_id>[/artifact] or mlflow:<run_id>:<artifact>
      - clearml:<task_id>[/artifact] or clearml:<task_id>:<artifact>
    """
    token = str(model_ref or "").strip()
    if not token:
        raise ValueError("model_ref is empty")

    def _normalize_mlflow_artifact(name: str) -> str:
        if not name:
            return artifact_path
        alias = name.strip().lower()
        if alias in {"final", "final.zip"}:
            return "models/final.zip"
        if alias in {"best", "best_by_finished", "best_by_finished.zip"}:
            return "models/best_by_finished.zip"
        if alias in {"interrupt", "interrupt.zip"}:
            return "models/interrupt.zip"
        return name

    def _normalize_clearml_artifact(name: str) -> str:
        if not name:
            return ""
        alias = name.strip().lower()
        if alias in {"final", "final.zip"}:
            return "models_final.zip"
        if alias in {"best", "best_by_finished", "best_by_finished.zip"}:
            return "models_best_by_finished.zip"
        if alias in {"interrupt", "interrupt.zip"}:
            return "models_interrupt.zip"
        return name

    if token.startswith("mlflow:"):
        payload = token.split("mlflow:", 1)[1].strip()
        if not payload:
            raise ValueError("mlflow run_id is empty")
        if ":" in payload:
            run_id, art = payload.split(":", 1)
        elif "/" in payload:
            run_id, art = payload.split("/", 1)
        else:
            run_id, art = payload, artifact_path
        dst = download_dir or (Path("out") / "mlflow_models" / run_id)
        return download_mlflow_artifact(
            run_id,
            _normalize_mlflow_artifact(art),
            dst,
            tracking_uri=tracking_uri,
        )

    if token.startswith("clearml:"):
        payload = token.split("clearml:", 1)[1].strip()
        if not payload:
            raise ValueError("clearml task_id is empty")
        if ":" in payload:
            task_id, art = payload.split(":", 1)
        elif "/" in payload:
            task_id, art = payload.split("/", 1)
        else:
            task_id, art = payload, ""
        artifact_name = _normalize_clearml_artifact(art)
        ref = f"clearml:{task_id}:{artifact_name}" if artifact_name else f"clearml:{task_id}"
        dst = download_dir or (Path("out") / "clearml_models")
        path = download_clearml_artifact(ref, dst)
        if path is None:
            raise FileNotFoundError(f"ClearML artifact not found for {token!r}")
        return path

    path = resolve_repo_path(token)
    if not path.exists():
        raise FileNotFoundError(f"model path not found: {path}")
    return path
