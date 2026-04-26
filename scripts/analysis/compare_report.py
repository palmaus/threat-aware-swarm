from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from scripts.analysis.demo_pack_report import METRIC_HIGHER_BETTER, METRICS, _improvement_percent
from scripts.common.path_utils import get_runs_dir, resolve_repo_path


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_latest_eval(root: Path, group: str) -> Path | None:
    if not root.exists():
        return None
    candidates = sorted(root.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in candidates:
        try:
            payload = _load_json(path)
        except Exception:
            continue
        policy = str(payload.get("policy") or payload.get("config", {}).get("policy") or "").lower()
        if group == "ppo" and policy.startswith("ppo"):
            return path
        if group == "baseline" and not policy.startswith("ppo"):
            return path
    return None


def _format_val(val: float | None) -> str:
    if val is None or val != val:
        return "-"
    return f"{val:.4f}"


def _build_report(baseline: dict, ppo: dict) -> list[str]:
    lines = ["# Comparative report", ""]
    lines.append(f"_Updated: {datetime.now().isoformat(timespec='seconds')}_")
    lines.append("")
    lines.append("| Metric | Baseline mean | PPO mean | % Improvement |")
    lines.append("| --- | --- | --- | --- |")
    for key in METRICS:
        base_mean = baseline.get(key, {}).get("mean")
        ppo_mean = ppo.get(key, {}).get("mean")
        diff = _improvement_percent(base_mean, ppo_mean, higher_better=METRIC_HIGHER_BETTER.get(key, True))
        diff_str = "-" if diff is None else f"{diff:+.2f}%"
        lines.append(
            f"| {key} | {_format_val(base_mean)} | {_format_val(ppo_mean)} | {diff_str} |"
        )
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description="Сравнение baseline vs PPO по eval_protocol.")
    parser.add_argument("--baseline", default="", help="Путь к baseline eval_protocol JSON.")
    parser.add_argument("--ppo", default="", help="Путь к PPO eval_protocol JSON.")
    parser.add_argument("--out", default="", help="Путь к Markdown отчёту.")
    args = parser.parse_args()

    eval_root = get_runs_dir() / "eval_protocol"
    baseline_path = resolve_repo_path(args.baseline) if args.baseline else _find_latest_eval(eval_root, "baseline")
    ppo_path = resolve_repo_path(args.ppo) if args.ppo else _find_latest_eval(eval_root, "ppo")

    if baseline_path is None or not baseline_path.exists():
        raise SystemExit("Не найден baseline eval_protocol JSON.")
    if ppo_path is None or not ppo_path.exists():
        raise SystemExit("Не найден PPO eval_protocol JSON.")

    baseline_payload = _load_json(baseline_path)
    ppo_payload = _load_json(ppo_path)
    baseline = baseline_payload.get("aggregate") or {}
    ppo = ppo_payload.get("aggregate") or {}

    lines = _build_report(baseline, ppo)
    if args.out:
        out_path = resolve_repo_path(args.out)
    else:
        out_path = get_runs_dir() / "compare_report" / "compare_report.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[compare_report] out={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
