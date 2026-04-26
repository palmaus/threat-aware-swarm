from __future__ import annotations

from pathlib import Path

from scripts.perf.profile_baselines import (
    compare_profiles,
    gate_regressions,
    main,
    metric_from_row,
    normalize_policy_name,
    write_summary_markdown,
)


def test_normalize_policy_name_adds_prefix():
    assert normalize_policy_name("astar_grid") == "baseline:astar_grid"
    assert normalize_policy_name("baseline:mpc_lite") == "baseline:mpc_lite"


def test_metric_from_row_prefers_wall_then_step_only():
    row = {"step_only_s_per_step": 0.2, "wall_time_s_per_step": 0.1}
    assert metric_from_row(row, preferred="wall") == 0.1
    assert metric_from_row(row, preferred="profile") == 0.2


def test_compare_profiles_uses_preferred_metric():
    current = {"baseline:astar_grid": {"wall_time_s_per_step": 0.12, "step_only_s_per_step": 0.5}}
    reference = {"baseline:astar_grid": {"wall_time_s_per_step": 0.1, "step_only_s_per_step": 0.4}}
    out = compare_profiles(current, reference, preferred="wall")
    assert round(out["baseline:astar_grid"]["delta_pct"], 1) == 20.0


def test_gate_regressions_flags_only_over_threshold():
    comparison = {
        "baseline:astar_grid": {"delta_pct": 4.9},
        "baseline:mpc_lite": {"delta_pct": 5.1},
    }
    assert gate_regressions(comparison, max_regression_pct=5.0) == ["baseline:mpc_lite"]


def test_write_summary_markdown_includes_comparison(tmp_path: Path):
    out = tmp_path / "summary.md"
    results = {"baseline:astar_grid": {"wall_time_s_per_step": 0.1, "step_only_s_per_step": 0.2}}
    comparison = {
        "baseline:astar_grid": {
            "reference": 0.09,
            "current": 0.1,
            "delta_pct": 11.1,
            "preferred_metric": "wall",
        }
    }
    write_summary_markdown(out, scenario="scenarios/S7_dynamic_chaser.yaml", steps=600, results=results, comparison=comparison)
    text = out.read_text(encoding="utf-8")
    assert "baseline:astar_grid" in text
    assert "Сравнение" in text
    assert "+11.1%" in text


def test_main_writes_profile_and_comparison(tmp_path: Path, monkeypatch):
    reference_path = tmp_path / "reference.json"
    reference_path.write_text(
        '{"baseline:astar_grid": {"wall_time_s_per_step": 0.2, "step_only_s_per_step": 0.3}}',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "scripts.perf.profile_baselines._parse_args",
        lambda: type(
            "Args",
            (),
            {
                "scenario": "scenarios/S7_dynamic_chaser.yaml",
                "policies": ["baseline:astar_grid"],
                "steps": 10,
                "seed": 0,
                "goal_radius": 3.0,
                "lite_metrics": True,
                "out_dir": str(tmp_path),
                "compare": str(reference_path),
                "metric": "wall",
                "max_regression_pct": 1000.0,
            },
        )()
    )
    monkeypatch.setattr(
        "scripts.perf.profile_baselines.profile_policy",
        lambda *args, **kwargs: {
            "wall_time_total_sec": 1.0,
            "wall_time_s_per_step": 0.1,
            "step_only_total_time_sec": 2.0,
            "step_only_s_per_step": 0.2,
            "top": [],
        },
    )
    assert main() == 0
    assert (tmp_path / "profile.json").exists()
    assert (tmp_path / "comparison.json").exists()
    assert (tmp_path / "summary.md").exists()
