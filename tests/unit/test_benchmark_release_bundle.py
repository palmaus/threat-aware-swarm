from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.analysis.build_benchmark_release import load_results, summarize_rows, write_release_markdown


def _write_run(root: Path, name: str, rows: list[dict]) -> Path:
    run_dir = root / name
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "manifest.json").open("w", encoding="utf-8") as fh:
        json.dump({"config": {"policy": "all"}}, fh)
    with (run_dir / "results.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "policy",
                "success_rate",
                "finished_frac_end",
                "alive_frac_end",
                "risk_integral_all",
                "path_ratio",
                "time_to_goal_mean",
                "safety_score",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return run_dir


def test_summarize_rows_selects_expected_champions(tmp_path: Path):
    run_dir = _write_run(
        tmp_path,
        "fair_static",
        [
            {
                "policy": "astar",
                "success_rate": "0.8",
                "finished_frac_end": "0.8",
                "alive_frac_end": "1.0",
                "risk_integral_all": "0.01",
                "path_ratio": "0.9",
                "time_to_goal_mean": "100",
                "safety_score": "0.95",
            },
            {
                "policy": "mpc",
                "success_rate": "0.7",
                "finished_frac_end": "0.7",
                "alive_frac_end": "0.9",
                "risk_integral_all": "0.02",
                "path_ratio": "0.8",
                "time_to_goal_mean": "90",
                "safety_score": "0.90",
            },
        ],
    )
    rows = load_results(run_dir, label="fair-static")
    summary = summarize_rows(rows)
    assert summary["fair-static"]["best_success_policy"] == "astar"
    assert summary["fair-static"]["best_path_policy"] == "mpc"


def test_write_release_markdown_contains_group_table(tmp_path: Path):
    rows = [
        {
            "run_label": "fair-static",
            "policy": "astar",
            "success_rate": "0.8",
            "finished_frac_end": "0.8",
            "alive_frac_end": "1.0",
            "risk_integral_all": "0.01",
            "path_ratio": "0.9",
            "time_to_goal_mean": "100",
            "safety_score": "0.95",
        }
    ]
    summary = {
        "fair-static": {
            "rows": 1,
            "best_success_policy": "astar",
            "best_success_rate": 0.8,
            "best_safety_policy": "astar",
            "best_safety_score": 0.95,
            "best_path_policy": "astar",
            "best_path_ratio": 0.9,
        }
    }
    out = tmp_path / "bundle.md"
    write_release_markdown(out, rows, summary)
    text = out.read_text(encoding="utf-8")
    assert "fair-static" in text
    assert "| astar | 0.8 |" in text
