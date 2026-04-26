from __future__ import annotations

from pathlib import Path


def download_mlflow_artifact(
    run_id: str,
    artifact_path: str,
    dst_dir: Path,
    tracking_uri: str | None = None,
) -> Path:
    if not run_id:
        raise ValueError("run_id is required")
    dst_dir.mkdir(parents=True, exist_ok=True)
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except Exception as exc:
        raise RuntimeError("mlflow is not available") from exc
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    local_path = client.download_artifacts(run_id, artifact_path, str(dst_dir))
    return Path(local_path)
