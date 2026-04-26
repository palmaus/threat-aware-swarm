"""Интеграционный смоук‑тест полного цикла обучения и оценки."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.integration
def test_one_cycle(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[2]
    runs_dir = tmp_path / "runs"
    out_dir = tmp_path / "out"

    env = os.environ.copy()
    env["TA_SKIP_AUTO_EVAL"] = "1"
    env["TA_SKIP_VECS_MONITOR"] = "1"

    cmd = [
        sys.executable,
        "-m",
        "scripts.train.trained_ppo",
        "run.total_timesteps=128",
        "env.max_steps=50",
        "vec.num_vec_envs=1",
        "vec.num_cpus=1",
        "ppo.n_steps=32",
        "ppo.batch_size=32",
        "ppo.n_epochs=1",
        "ppo.use_rnn=false",
        "run.no_eval=true",
        "run.eval_freq=100000000",
        "run.checkpoint_freq=100000000",
        f"run.out_dir={runs_dir}",
        "run.run_name=smoke",
        "device=cpu",
        "logging.tracking.mlflow.enabled=false",
        "logging.tracking.clearml.enabled=false",
        "logging.tracking.mlflow.cleanup_run_dir=false",
    ]

    proc = subprocess.run(cmd, cwd=str(repo_root), env=env, text=True, capture_output=True)
    assert proc.returncode == 0, f"train failed\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"

    run_dirs = sorted(runs_dir.glob("run_*"))
    assert run_dirs, "expected run_* directory"
    run_dir = run_dirs[-1]
    assert (run_dir / "meta" / "run.json").exists()
    assert (run_dir / "models" / "final.zip").exists()

    cmd = [
        sys.executable,
        "-m",
        "scripts.eval.eval_models",
        f"run_dir={run_dir}",
        "mode=fixed",
        "n_episodes=2",
        "max_steps=50",
        "deterministic=true",
        "seed=0",
    ]
    proc = subprocess.run(cmd, cwd=str(repo_root), text=True, capture_output=True)
    assert proc.returncode == 0, f"eval failed\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"

    eval_path = run_dir / "eval" / "eval_results.json"
    assert eval_path.exists()
    json.loads(eval_path.read_text())

    cmd = [
        sys.executable,
        "-m",
        "scripts.eval.summarize_run",
        f"run_dir={run_dir}",
        f"out={out_dir}",
    ]
    proc = subprocess.run(cmd, cwd=str(repo_root), text=True, capture_output=True)
    assert proc.returncode == 0, f"summary failed\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"

    summary_path = out_dir / "index" / "summary.json"
    assert summary_path.exists()
    data = json.loads(summary_path.read_text())
    assert any(row.get("run_id") == run_dir.name for row in data)
