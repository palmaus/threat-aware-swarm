from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

from scripts.common.metrics_utils import aggregate_scene_metrics
from scripts.common.path_utils import get_runs_dir, resolve_repo_path

SUMMARY_KEYS = [
    "success_rate",
    "alive_frac_end",
    "risk_integral_alive",
    "path_ratio",
    "collision_like",
    "time_to_goal_mean",
]


def _find_latest_results(root: Path) -> Path | None:
    if not root.exists():
        return None
    candidates = sorted(root.glob("**/results.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _format_val(val: float | None, digits: int = 4) -> str:
    if val is None or val != val:
        return "-"
    return f"{val:.{digits}f}"


def _copy_plots(src_dir: Path, dst_dir: Path) -> dict[str, Path]:
    copied: dict[str, Path] = {}
    if not src_dir.exists():
        return copied
    dst_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(src_dir.glob("*.*")):
        if path.suffix.lower() not in {".png", ".svg"}:
            continue
        target = dst_dir / path.name
        shutil.copy2(path, target)
        copied[path.stem] = target
    return copied


def _build_report(
    *,
    title: str,
    source: Path,
    aggregate: dict[str, float],
    plot_dir: Path,
    copied: dict[str, Path],
) -> str:
    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"_Updated: {datetime.now().isoformat(timespec='seconds')}_")
    lines.append("")
    lines.append(f"Source: `{source}`")
    lines.append("")

    def _img(name: str) -> None:
        svg = plot_dir / f"{name}.svg"
        png = plot_dir / f"{name}.png"
        if svg.exists():
            lines.append(f"![{name}]({svg.as_posix()})")
            lines.append("")
            return
        if png.exists():
            lines.append(f"![{name}]({png.as_posix()})")
            lines.append("")

    _img("success_rate_by_scene")
    _img("path_ratio_distribution")
    _img("collisions_by_scene")

    lines.append("## Aggregate metrics")
    lines.append("| Metric | Value |")
    lines.append("| --- | --- |")
    for key in SUMMARY_KEYS:
        lines.append(f"| {key} | {_format_val(aggregate.get(key))} |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build quick sanity report from latest eval_scenarios output.")
    parser.add_argument("--src", default="", help="Path to eval_scenarios results.json.")
    parser.add_argument("--out", default="", help="Output directory for sanity report.")
    parser.add_argument("--title", default="Sanity Report", help="Report title.")
    args = parser.parse_args()

    src_path = resolve_repo_path(args.src) if args.src else _find_latest_results(get_runs_dir() / "scenarios")
    if src_path is None or not src_path.exists():
        raise SystemExit("eval_scenarios results.json not found.")

    payload = json.loads(src_path.read_text(encoding="utf-8"))
    per_scene = payload.get("scenes") or {}
    if not isinstance(per_scene, dict):
        per_scene = {}
    aggregate = aggregate_scene_metrics(per_scene)

    out_dir = resolve_repo_path(args.out) if args.out else (get_runs_dir() / "sanity_report")
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_src = src_path.parent / "plots"
    plot_dst = out_dir / "plots"
    copied = _copy_plots(plot_src, plot_dst)

    report_text = _build_report(
        title=str(args.title),
        source=src_path,
        aggregate={k: float(aggregate.get(k, float("nan"))) for k in SUMMARY_KEYS},
        plot_dir=plot_dst,
        copied=copied,
    )
    report_path = out_dir / "sanity_report.md"
    report_path.write_text(report_text, encoding="utf-8")
    print(f"[sanity_report] out={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
