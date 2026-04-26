"""
Сводит run‑директорию в runs/index/summary.csv и runs/index/summary.json.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from scripts.common.path_utils import resolve_repo_path


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _find_event_files(tb_dir: Path) -> list[Path]:
    if not tb_dir.exists():
        return []
    return sorted(tb_dir.glob("**/events.out.tfevents*"))


def _best_scalar_from_tb(tb_dir: Path, tag: str) -> float:
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except Exception:
        return float("nan")

    best = float("nan")
    for f in _find_event_files(tb_dir):
        try:
            ea = event_accumulator.EventAccumulator(str(f))
            ea.Reload()
            if tag not in ea.Tags().get("scalars", []):
                continue
            vals = [s.value for s in ea.Scalars(tag)]
            if not vals:
                continue
            cur = max(vals)
            if best != best or cur > best:
                best = cur
        except Exception:
            continue
    return best


def _select_eval_row(eval_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not eval_rows:
        return None
    for name in ("best_by_finished", "final"):
        for row in eval_rows:
            if row.get("model_name") == name:
                return row
    return eval_rows[0]


def summarize_run(run_dir: Path) -> dict[str, Any]:
    meta = _read_json(run_dir / "meta" / "run.json") or {}
    eval_rows = _read_json(run_dir / "eval" / "eval_results_fixed_20.json")
    if eval_rows is None:
        eval_rows = _read_json(run_dir / "eval" / "eval_results.json")
    eval_rows = eval_rows or []
    if isinstance(eval_rows, dict):
        eval_rows = [eval_rows]

    chosen = _select_eval_row(eval_rows) or {}
    robust_rows = _read_json(run_dir / "eval" / "eval_results_random_100.json") or []
    robust = _select_eval_row(robust_rows) or {}

    tb_dir = run_dir / "tb"
    best_eval_reward = _best_scalar_from_tb(tb_dir, "eval/mean_reward")
    best_rollout_rew = _best_scalar_from_tb(tb_dir, "rollout/ep_rew_mean")

    summary = {
        "run_id": meta.get("run_id", run_dir.name),
        "run_dir": str(run_dir),
        "time_start": meta.get("time_start", ""),
        "time_end": meta.get("time_end", ""),
        "model_selected": chosen.get("model_name", ""),
        "env_schema_version": meta.get("env_schema_version", "unknown"),
        "eval_fixed/finished_given_alive_mean": chosen.get("eval_fixed/finished_given_alive_mean", float("nan")),
        "eval_fixed/alive_frac_mean": chosen.get("eval_fixed/alive_frac_mean", float("nan")),
        "eval_fixed/mean_dist_mean": chosen.get("eval_fixed/mean_dist_mean", float("nan")),
        "eval_random/finished_given_alive_mean": robust.get("eval_random/finished_given_alive_mean", float("nan")),
        "eval_random/alive_frac_mean": robust.get("eval_random/alive_frac_mean", float("nan")),
        "eval_random/mean_dist_mean": robust.get("eval_random/mean_dist_mean", float("nan")),
        "tb_eval/mean_reward_max": best_eval_reward,
        "tb_rollout/ep_rew_mean_max": best_rollout_rew,
        "summary_dt": datetime.now().isoformat(timespec="seconds"),
    }
    return summary


def _read_summary_json(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _find_latest_bench_eval(out_root: Path) -> list[dict[str, Any]]:
    bench_dir = out_root / "bench"
    if not bench_dir.exists():
        return []
    candidates = sorted(bench_dir.glob("**/eval_results.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in candidates:
        data = _read_json(path)
        if isinstance(data, list):
            return data
    return []


def _write_summary(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "summary.json"
    csv_path = out_dir / "summary.csv"

    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    fieldnames: list[str] = []
    for row in rows:
        for k in row.keys():
            if k not in fieldnames:
                fieldnames.append(k)

    import csv

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


@dataclass
class SummarizeRunConfig:
    run_dir: str = ""
    out_root: str = "runs"
    out: str = ""


def run(cfg: SummarizeRunConfig) -> None:
    args = cfg

    run_dir = resolve_repo_path(args.run_dir)
    out_root = resolve_repo_path(args.out) if args.out else resolve_repo_path(args.out_root or "runs")
    out_dir = out_root / "index"

    summary = summarize_run(run_dir)

    rows = _read_summary_json(out_dir / "summary.json")
    rows = [r for r in rows if r.get("run_id") != summary.get("run_id")]
    rows.append(summary)

    _write_summary(out_dir, rows)

    baseline_rows = _find_latest_bench_eval(out_root)
    if baseline_rows:
        rows = _read_summary_json(out_dir / "summary.json")
        for b in baseline_rows:
            run_id = f"baseline:{b.get('model_name', 'unknown')}"
            entry = {
                "run_id": run_id,
                "run_dir": "",
                "model_selected": b.get("model_name", ""),
                "eval_fixed/finished_given_alive_mean": b.get("eval_fixed/finished_given_alive_mean", float("nan")),
                "eval_fixed/alive_frac_mean": b.get("eval_fixed/alive_frac_mean", float("nan")),
                "eval_fixed/mean_dist_mean": b.get("eval_fixed/mean_dist_mean", float("nan")),
                "summary_dt": datetime.now().isoformat(timespec="seconds"),
            }
            rows = [r for r in rows if r.get("run_id") != run_id]
            rows.append(entry)
        _write_summary(out_dir, rows)

    print(f"[ОК] Сводка -> {out_dir / 'summary.csv'}")


def main() -> None:
    import hydra
    from omegaconf import DictConfig, OmegaConf

    from scripts.common.hydra_utils import apply_schema

    @hydra.main(version_base=None, config_path="../../configs/hydra", config_name="eval/summarize")
    def _run(cfg: DictConfig) -> None:
        cfg = apply_schema(cfg, SummarizeRunConfig)
        data = OmegaConf.to_container(cfg, resolve=True)
        run(SummarizeRunConfig(**data))

    _run()


if __name__ == "__main__":
    main()
