import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

if importlib.util.find_spec("env") is None or importlib.util.find_spec("scripts") is None:
    raise pytest.UsageError("Package not installed. Run: pip install -e .")

if importlib.util.find_spec("ui") is None:
    sys.path.insert(0, str(ROOT))


def pytest_addoption(parser):
    parser.addoption(
        "--run-tuning-stageb",
        action="store_true",
        default=False,
        help="Run heavy tuning stage-B integration smoke tests.",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-tuning-stageb"):
        return

    skip_stageb = pytest.mark.skip(reason="need --run-tuning-stageb to run")
    for item in items:
        if "tuning_stageb" in item.keywords:
            item.add_marker(skip_stageb)


@pytest.fixture()
def env():
    try:
        import gymnasium  # noqa: F401
    except Exception:
        pytest.skip("gymnasium not installed")
    from env.config import EnvConfig
    from env.pz_env import SwarmPZEnv

    cfg = EnvConfig()
    return SwarmPZEnv(cfg, max_steps=50, goal_radius=3.0)
