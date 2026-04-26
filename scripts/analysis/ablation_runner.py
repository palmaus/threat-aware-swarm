from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from scripts.common.artifacts import ensure_run_dir
from scripts.common.path_utils import resolve_repo_path


@dataclass
class AblationRun:
    name: str
    command: list[str]
    run_name: str | None
    out_dir: str | None
    env: dict[str, str]


@dataclass
class AblationSpec:
    name: str
    runs: list[AblationRun]
    metrics: list[dict[str, str]]
    use_ray: bool = False
    parallel: int = 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ablation Runner")
    parser.add_argument("--config", required=True, help="Путь к YAML‑спеке абляции.")
    parser.add_argument("--dry-run", action="store_true", help="Печатает команды без запуска.")
    parser.add_argument("--parallel", type=int, default=0, help="Число параллельных запусков.")
    parser.add_argument("--python-bin", default=sys.executable, help="Интерпретатор Python.")
    return parser.parse_args()


def _load_spec(path: Path) -> AblationSpec:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    runs = []
    for raw in data.get("runs", []):
        cmd = [str(x) for x in raw.get("command", [])]
        runs.append(
            AblationRun(
                name=str(raw.get("name")),
                command=cmd,
                run_name=raw.get("run_name"),
                out_dir=raw.get("out_dir"),
                env={str(k): str(v) for k, v in (raw.get("env") or {}).items()},
            )
        )
    return AblationSpec(
        name=str(data.get("name", path.stem)),
        runs=runs,
        metrics=list(data.get("metrics") or []),
        use_ray=bool(data.get("use_ray", False)),
        parallel=int(data.get("parallel", 1)),
    )


def _run_cmd(cmd: list[str], env: dict[str, str], cwd: Path) -> int:
    proc = subprocess.run(cmd, env=env, cwd=str(cwd), check=False)
    return int(proc.returncode)


def _find_latest_run(out_dir: Path, run_name: str) -> Path | None:
    if not out_dir.exists():
        return None
    candidates = [p for p in out_dir.glob("run_*") if run_name in p.name]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _load_experiment_result(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "meta" / "experiment_result.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _extract_metric(payload: dict[str, Any], path: str) -> Any:
    cur: Any = payload
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def _render_table(rows: list[dict[str, Any]], metric_keys: list[str]) -> str:
    headers = ["name", *metric_keys]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        line = [str(row.get("name", ""))]
        for key in metric_keys:
            line.append(str(row.get(key, "")))
        lines.append("| " + " | ".join(line) + " |")
    return "\n".join(lines) + "\n"


def run(spec: AblationSpec, *, dry_run: bool, python_bin: str) -> Path:
    out_dir = ensure_run_dir(category="ablation", out_root="runs", run_id=None, prefix=spec.name)
    results: list[dict[str, Any]] = []

    def _exec_run(run: AblationRun) -> tuple[str, int, dict[str, Any] | None]:
        cmd = [python_bin if part == "{python}" else part for part in run.command]
        if not cmd:
            return run.name, 1, None
        if dry_run:
            print(f"[ablation:{spec.name}] {run.name}: {' '.join(cmd)}")
            return run.name, 0, None
        env = os.environ.copy()
        env.update(run.env)
        rc = _run_cmd(cmd, env, resolve_repo_path("."))
        result = None
        if run.run_name and run.out_dir:
            latest = _find_latest_run(resolve_repo_path(run.out_dir), run.run_name)
            if latest is not None:
                result = _load_experiment_result(latest)
        return run.name, rc, result

    if spec.use_ray:
        try:
            import ray
            from ray import remote

            ray.init(ignore_reinit_error=True)

            @remote
            def _ray_exec(run: AblationRun) -> tuple[str, int, dict[str, Any] | None]:
                return _exec_run(run)

            futures = [_ray_exec.remote(run) for run in spec.runs]
            for name, rc, payload in ray.get(futures):
                if rc != 0:
                    print(f"[WARN] run failed: {name} rc={rc}")
                results.append(_format_result(name, payload, spec.metrics))
        except Exception as exc:
            print(f"[WARN] Ray unavailable ({exc}); fallback to sequential.")
            for run in spec.runs:
                name, rc, payload = _exec_run(run)
                if rc != 0:
                    print(f"[WARN] run failed: {name} rc={rc}")
                results.append(_format_result(name, payload, spec.metrics))
    elif spec.parallel > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=spec.parallel) as ex:
            futs = {ex.submit(_exec_run, run): run.name for run in spec.runs}
            for fut in as_completed(futs):
                name, rc, payload = fut.result()
                if rc != 0:
                    print(f"[WARN] run failed: {name} rc={rc}")
                results.append(_format_result(name, payload, spec.metrics))
    else:
        for run in spec.runs:
            name, rc, payload = _exec_run(run)
            if rc != 0:
                print(f"[WARN] run failed: {name} rc={rc}")
            results.append(_format_result(name, payload, spec.metrics))

    metric_keys = [m.get("label", m.get("key", "")) for m in spec.metrics]
    report_md = out_dir / "report.md"
    report_csv = out_dir / "report.json"
    report_md.write_text(_render_table(results, metric_keys), encoding="utf-8")
    report_csv.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "name": spec.name,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "metrics": spec.metrics,
                "runs": [r.__dict__ for r in spec.runs],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return out_dir


def _format_result(name: str, payload: dict[str, Any] | None, metrics: list[dict[str, str]]) -> dict[str, Any]:
    row: dict[str, Any] = {"name": name}
    if payload is None:
        row["status"] = "no_result"
        return row
    row["status"] = payload.get("metrics", {}).get("status", payload.get("status", "ok"))
    for metric in metrics:
        key = metric.get("key", "")
        label = metric.get("label", key)
        if not key:
            continue
        row[label] = _extract_metric(payload, key)
    return row


def main() -> None:
    args = _parse_args()
    spec_path = resolve_repo_path(args.config)
    spec = _load_spec(spec_path)
    if args.parallel > 0:
        spec.parallel = int(args.parallel)
    out = run(spec, dry_run=bool(args.dry_run), python_bin=str(args.python_bin))
    print(f"[OK] {out}")


if __name__ == "__main__":
    main()
