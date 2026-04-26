from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from scripts.common.artifacts import ensure_run_dir
from scripts.common.path_utils import resolve_repo_path

CORE_COLUMNS = [
    "policy",
    "success_rate",
    "finished_frac_end",
    "alive_frac_end",
    "risk_integral_all",
    "path_ratio",
    "time_to_goal_mean",
    "safety_score",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a release-friendly benchmark bundle from bench runs.")
    parser.add_argument("--inputs", nargs="+", required=True, help="One or more benchmark run directories.")
    parser.add_argument("--labels", nargs="*", default=None, help="Optional labels matching --inputs.")
    parser.add_argument("--out-dir", default="")
    return parser.parse_args()


def load_results(run_dir: Path, *, label: str) -> list[dict]:
    csv_path = run_dir / "results.csv"
    manifest_path = run_dir / "manifest.json"
    rows: list[dict] = []
    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with csv_path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            row["run_label"] = label
            row["run_dir"] = str(run_dir)
            row["config"] = manifest.get("config", {})
            rows.append(row)
    return rows


def _float(row: dict, key: str) -> float | None:
    raw = row.get(key)
    if raw in (None, "", "nan", "NaN"):
        return None
    try:
        return float(raw)
    except Exception:
        return None


def summarize_rows(rows: list[dict]) -> dict:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(str(row["run_label"]), []).append(row)
    summary: dict[str, dict] = {}
    for label, group_rows in groups.items():
        best_success = max(group_rows, key=lambda r: (_float(r, "success_rate") or float("-inf")))
        best_safety = max(group_rows, key=lambda r: (_float(r, "safety_score") or float("-inf")))
        best_path = min(
            group_rows,
            key=lambda r: (_float(r, "path_ratio") if _float(r, "path_ratio") is not None else float("inf")),
        )
        summary[label] = {
            "rows": len(group_rows),
            "best_success_policy": best_success["policy"],
            "best_success_rate": _float(best_success, "success_rate"),
            "best_safety_policy": best_safety["policy"],
            "best_safety_score": _float(best_safety, "safety_score"),
            "best_path_policy": best_path["policy"],
            "best_path_ratio": _float(best_path, "path_ratio"),
        }
    return summary


def write_release_markdown(out_path: Path, rows: list[dict], summary: dict) -> None:
    lines = [
        "# Benchmark Release Bundle",
        "",
        "This bundle is meant for release/readme-facing result packaging. It does not replace raw run artifacts.",
        "",
    ]
    for label, info in summary.items():
        lines.extend(
            [
                f"## {label}",
                "",
                f"- rows: `{info['rows']}`",
                f"- best success: `{info['best_success_policy']}` (`{info['best_success_rate']}`)",
                f"- best safety: `{info['best_safety_policy']}` (`{info['best_safety_score']}`)",
                f"- best path ratio: `{info['best_path_policy']}` (`{info['best_path_ratio']}`)",
                "",
                "| policy | success_rate | finished_frac_end | alive_frac_end | risk_integral_all | path_ratio | time_to_goal_mean | safety_score |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in [r for r in rows if r["run_label"] == label]:
            values = [row.get(col, "-") or "-" for col in CORE_COLUMNS]
            lines.append("| " + " | ".join(values) + " |")
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = _parse_args()
    labels = list(args.labels or [])
    if labels and len(labels) != len(args.inputs):
        raise SystemExit("--labels must match --inputs length")
    if not labels:
        labels = [Path(item).name for item in args.inputs]
    all_rows: list[dict] = []
    for raw, label in zip(args.inputs, labels):
        run_dir = resolve_repo_path(raw)
        all_rows.extend(load_results(run_dir, label=label))
    summary = summarize_rows(all_rows)
    run_dir = ensure_run_dir(category="release", out_root="runs", out_dir=args.out_dir or None, prefix="benchmark")
    (run_dir / "benchmark_release.json").write_text(
        json.dumps({"summary": summary, "rows": all_rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (run_dir / "leaderboard.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["run_label", "run_dir", *CORE_COLUMNS])
        writer.writeheader()
        for row in all_rows:
            writer.writerow({key: row.get(key, "") for key in ["run_label", "run_dir", *CORE_COLUMNS]})
    write_release_markdown(run_dir / "benchmark_release.md", all_rows, summary)
    print(f"[build_benchmark_release] out={run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
