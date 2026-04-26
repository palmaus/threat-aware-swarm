"""Интеграционный смоук‑тест тюнинга бейзлайнов на малом наборе сцен."""

import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.integration
def test_tune_baselines_smoke(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[2]
    out_dir = tmp_path / "tune"
    scene = repo_root / "scenarios" / "S0_sanity_no_threats.yaml"

    cmd = [
        sys.executable,
        "-m",
        "scripts.tuning.tune_baselines",
        "policy=['baseline:astar_grid']",
        "tuning/profile=fast",
        "method=random",
        "samples=1",
        "episodes=1",
        "seed=0",
        f"scenes=[{scene}]",
        f"out_dir={out_dir}",
        "pruner=none",
    ]

    proc = subprocess.run(cmd, cwd=str(repo_root), text=True, capture_output=True)
    assert proc.returncode == 0, f"tune_baselines failed\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"

    results = json.loads((out_dir / "tune_results.json").read_text(encoding="utf-8"))
    policy_summary = results["summary"]["baseline:astar_grid"]
    assert policy_summary["states"]["COMPLETE"] == 1
    assert policy_summary["validation_status"] == "search_only"
    assert (out_dir / "tune_results.json").exists()
    assert (out_dir / "tuning_protocol.json").exists()
    assert (out_dir / "tuning_report.json").exists()
    assert (out_dir / "tuning_report.md").exists()
    assert (out_dir / "eval_cache_manifest.json").exists()
    csv_path = out_dir / "tune_baseline_astar_grid.csv"
    assert csv_path.exists()
    assert csv_path.stat().st_size > 0


@pytest.mark.integration
def test_tune_baselines_parallel_trace_smoke(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[2]
    out_dir = tmp_path / "tune_trace"
    scene = repo_root / "scenarios" / "S0_sanity_no_threats.yaml"

    cmd = [
        sys.executable,
        "-m",
        "scripts.tuning.tune_baselines",
        "policy=['baseline:astar_grid']",
        "tuning/profile=fast",
        "method=random",
        "samples=1",
        "episodes=1",
        "seed=0",
        "eval_cache=false",
        "parallel_debug=true",
        "parallel_trace=true",
        f"scenes=[{scene}]",
        f"out_dir={out_dir}",
        "pruner=none",
    ]

    proc = subprocess.run(cmd, cwd=str(repo_root), text=True, capture_output=True)
    assert proc.returncode == 0, f"tune_baselines trace failed\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    trace_path = out_dir / "parallel_trace.jsonl"
    assert trace_path.exists()
    assert "run_episode_start" in trace_path.read_text(encoding="utf-8")


@pytest.mark.integration
def test_tune_baselines_optuna_smoke(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[2]
    out_dir = tmp_path / "tune_optuna"
    scene = repo_root / "scenarios" / "S0_sanity_no_threats.yaml"

    cmd = [
        sys.executable,
        "-m",
        "scripts.tuning.tune_baselines",
        "policy=['baseline:astar_grid']",
        "tuning/profile=balanced",
        "method=optuna",
        "trials=1",
        "n_jobs=1",
        "pruner=none",
        "episodes=1",
        f"scenes=[{scene}]",
        f"out_dir={out_dir}",
    ]

    proc = subprocess.run(cmd, cwd=str(repo_root), text=True, capture_output=True)
    assert proc.returncode == 0, f"tune_baselines optuna failed\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"

    results = json.loads((out_dir / "tune_results.json").read_text(encoding="utf-8"))
    policy_summary = results["summary"]["baseline:astar_grid"]
    assert policy_summary["n_trials"] >= 1
    assert policy_summary["validation_status"] == "search_only"
    assert (out_dir / "tune_baseline_astar_grid.csv").exists()


@pytest.mark.integration
def test_tune_baselines_optuna_parallel_smoke(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[2]
    out_dir = tmp_path / "tune_optuna_parallel"
    scene = repo_root / "scenarios" / "S0_sanity_no_threats.yaml"

    cmd = [
        sys.executable,
        "-m",
        "scripts.tuning.tune_baselines",
        "policy=['baseline:astar_grid']",
        "tuning/profile=balanced",
        "method=optuna",
        "trials=3",
        "n_jobs=2",
        "pruner=none",
        "episodes=1",
        "eval_cache=false",
        "parallel_debug=true",
        "parallel_trace=true",
        f"scenes=[{scene}]",
        f"out_dir={out_dir}",
    ]

    proc = subprocess.run(cmd, cwd=str(repo_root), text=True, capture_output=True)
    assert proc.returncode == 0, f"tune_baselines optuna parallel failed\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"

    results = json.loads((out_dir / "tune_results.json").read_text(encoding="utf-8"))
    policy_summary = results["summary"]["baseline:astar_grid"]
    assert policy_summary["n_trials"] == 3
    assert policy_summary["validation_status"] == "search_only"
    assert (out_dir / "tune_baseline_astar_grid.csv").exists()
    trace_lines = (out_dir / "parallel_trace.jsonl").read_text(encoding="utf-8").strip().splitlines()
    trace_records = [json.loads(line) for line in trace_lines]
    worker_pids = {record["pid"] for record in trace_records if record.get("event") == "run_episode_start"}
    if len(worker_pids) < 2:
        combined = proc.stdout + "\n" + proc.stderr
        assert (
            "spawn-worker backend недоступен" in combined
            or "Снижаю число spawn-worker'ов" in combined
        )


@pytest.mark.integration
@pytest.mark.tuning_stageb
def test_tune_baselines_optuna_stageb_smoke(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[2]
    out_dir = tmp_path / "tune_optuna_stageb"
    search_scene = repo_root / "scenarios" / "S0_sanity_no_threats.yaml"
    holdout_scene = repo_root / "scenarios" / "S18_hard_maze.yaml"
    benchmark_scene = repo_root / "scenarios" / "S1_single_threat_blocking.yaml"
    ood_scene = repo_root / "scenarios" / "S11_dynamic_gate.yaml"

    cmd = [
        sys.executable,
        "-m",
        "scripts.tuning.tune_baselines",
        "policy=['baseline:astar_grid']",
        "tuning/profile=balanced",
        "method=optuna",
        "trials=1",
        "n_jobs=1",
        "pruner=none",
        "episodes=1",
        "episodes_eval=1",
        "stage_b=true",
        "topk=1",
        f"search_scenes=[{search_scene}]",
        f"holdout_scenes=[{holdout_scene}]",
        f"benchmark_scenes=[{benchmark_scene}]",
        f"ood_scenes=[{ood_scene}]",
        f"out_dir={out_dir}",
    ]

    proc = subprocess.run(cmd, cwd=str(repo_root), text=True, capture_output=True)
    assert proc.returncode == 0, f"tune_baselines optuna stage_b failed\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"

    results = json.loads((out_dir / "tune_results.json").read_text(encoding="utf-8"))
    policy_summary = results["summary"]["baseline:astar_grid"]
    assert policy_summary["validation_status"] == "holdout_validated"
    assert policy_summary["promotion_status"] == "promotable"
    assert (out_dir / "topk_stageB.csv").exists()


@pytest.mark.integration
def test_tune_baselines_parallel_random_smoke(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[2]
    out_dir = tmp_path / "tune_parallel"
    scene = repo_root / "scenarios" / "S0_sanity_no_threats.yaml"

    cmd = [
        sys.executable,
        "-m",
        "scripts.tuning.tune_baselines",
        "policy=['baseline:potential_fields']",
        "tuning/profile=balanced",
        "method=random",
        "samples=2",
        "n_jobs=2",
        "episodes=1",
        "stage_b=false",
        f"scenes=[{scene}]",
        f"out_dir={out_dir}",
    ]

    proc = subprocess.run(cmd, cwd=str(repo_root), text=True, capture_output=True)
    assert proc.returncode == 0, f"tune_baselines parallel failed\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"

    results = json.loads((out_dir / "tune_results.json").read_text(encoding="utf-8"))
    policy_summary = results["summary"]["baseline:potential_fields"]
    assert policy_summary["n_trials"] == 2
    assert policy_summary["states"]["COMPLETE"] == 2
    assert policy_summary["validation_status"] == "search_only"
    assert (out_dir / "tune_baseline_potential_fields.csv").exists()
