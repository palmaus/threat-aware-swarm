"""Tuning report rendering helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable


def fmt_metric(value: float | None, digits: int = 3) -> str:
    try:
        val = float(value)
    except Exception:
        return "--"
    if val != val:
        return "--"
    return f"{val:.{digits}f}"


def aggregate_row(label: str, agg: dict | None, risk_metric: Callable[[dict], float], reason: str = "") -> str:
    agg = agg or {}
    return (
        f"| {label} | {fmt_metric(agg.get('finished_frac_end'))} | {fmt_metric(agg.get('alive_frac_end'))} | "
        f"{fmt_metric(risk_metric(agg), 4)} | {fmt_metric(agg.get('time_to_goal_mean'), 2)} | "
        f"{fmt_metric(agg.get('path_ratio'), 3)} | {reason} |"
    )


def write_tuning_report(
    out_dir: Path,
    protocol: dict,
    results: dict,
    *,
    risk_metric: Callable[[dict], float],
    champion_reason: Callable[[str, dict], str],
) -> Path:
    report_payload = {"protocol": protocol["summary"], "policies": {}}
    lines = [
        "# Tuning Report",
        "",
        "## Protocol",
        "",
        f"- Information regime: {protocol['summary'].get('information_regime', '--')}",
        f"- Search scenes: {', '.join(protocol['summary'].get('search_scene_ids', [])) or '--'}",
        f"- Holdout scenes: {', '.join(protocol['summary'].get('holdout_scene_ids', [])) or '--'}",
        f"- Benchmark scenes: {', '.join(protocol['summary'].get('benchmark_scene_ids', [])) or '--'}",
        f"- OOD scenes: {', '.join(protocol['summary'].get('ood_scene_ids', [])) or '--'}",
        f"- Validation status: {protocol['summary'].get('validation_status', '--')}",
        f"- Promotion status: {protocol['summary'].get('promotion_status', '--')}",
        "",
    ]
    if protocol["summary"].get("validation_status") == "search_only":
        lines.extend(
            [
                "> This run is search-only. Do not use these params as benchmark-grade tuned results until Stage-B/validation is enabled.",
                "",
            ]
        )
    for policy_name, payload in results.items():
        policy_report = {
            "champions": payload.get("champions", {}),
            "validation": payload.get("validation", {}),
            "validation_status": payload.get("validation_status", "search_only"),
            "promotion_status": payload.get("promotion_status", "not_promoted"),
        }
        report_payload["policies"][policy_name] = policy_report
        lines.extend(
            [
                f"## {policy_name}",
                "",
                f"- Validation status: {payload.get('validation_status', 'search_only')}",
                f"- Promotion status: {payload.get('promotion_status', 'not_promoted')}",
                "",
                "| champion/split | finish | alive | risk | time | path_ratio | note |",
                "|---|---:|---:|---:|---:|---:|---|",
            ]
        )
        champions = payload.get("champions") or {}
        if not champions and payload.get("best"):
            champions = {"balanced": payload["best"]}
        for champion_name, champion_row in champions.items():
            lines.append(
                aggregate_row(
                    f"{champion_name}:holdout",
                    champion_row.get("aggregate"),
                    risk_metric,
                    champion_reason(champion_name, champion_row.get("aggregate") or {}),
                )
            )
            validation = (payload.get("validation") or {}).get(champion_name, {})
            lines.append(
                aggregate_row(
                    f"{champion_name}:benchmark",
                    (validation.get("benchmark") or {}).get("aggregate"),
                    risk_metric,
                    validation.get("reason", ""),
                )
            )
            lines.append(
                aggregate_row(
                    f"{champion_name}:ood",
                    (validation.get("ood") or {}).get("aggregate"),
                    risk_metric,
                    "",
                )
            )
        balanced_validation = (payload.get("validation") or {}).get("balanced", {})
        benchmark_family = (balanced_validation.get("benchmark") or {}).get("per_family") or {}
        if benchmark_family:
            lines.extend(
                [
                    "",
                    "Per-family benchmark breakdown:",
                    "",
                    "| family | finish | alive | risk | time | path_ratio |",
                    "|---|---:|---:|---:|---:|---:|",
                ]
            )
            for family, agg in sorted(benchmark_family.items()):
                lines.append(
                    f"| {family} | {fmt_metric(agg.get('finished_frac_end'))} | {fmt_metric(agg.get('alive_frac_end'))} | "
                    f"{fmt_metric(risk_metric(agg), 4)} | {fmt_metric(agg.get('time_to_goal_mean'), 2)} | "
                    f"{fmt_metric(agg.get('path_ratio'), 3)} |"
                )
        lines.append("")
    json_path = out_dir / "tuning_report.json"
    md_path = out_dir / "tuning_report.md"
    json_path.write_text(json.dumps(report_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return md_path
