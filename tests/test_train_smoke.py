"""Смоук‑тест запуска из командной строки обучения на минимальных параметрах."""

import os
import subprocess
import sys
from pathlib import Path


def test_train_smoke(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = tmp_path / "runs"

    cmd = [
        sys.executable,
        "-m",
        "scripts.train.trained_ppo",
        "run.total_timesteps=256",
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
        f"run.out_dir={out_dir}",
        "logging.tracking.mlflow.enabled=false",
        "logging.tracking.clearml.enabled=false",
        "logging.tracking.mlflow.cleanup_run_dir=false",
    ]

    env = os.environ.copy()
    env["TA_SKIP_VECS_MONITOR"] = "1"
    env["TA_SKIP_AUTO_EVAL"] = "1"
    proc = subprocess.run(
        cmd,
        cwd=str(repo_root),
        env=env,
        text=True,
        capture_output=True,
    )

    assert proc.returncode == 0, f"train failed\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"

    run_dirs = sorted(out_dir.glob("run_*"))
    assert run_dirs, "expected run_* directory"
    meta_dir = run_dirs[-1] / "meta"
    assert (meta_dir / "run.json").exists()
