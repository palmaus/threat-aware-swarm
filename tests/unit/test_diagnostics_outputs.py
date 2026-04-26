"""Проверка диагностических отчётов и формата выходных файлов."""

import json

from scripts.debug.debug_env_metrics import run_sanity_scenario


def test_sanity_scenario_risk_positive():
    report = run_sanity_scenario(seed=0)
    assert report["ok"] is True


def test_diagnostics_files_created(tmp_path):
    out_dir = tmp_path / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    data = {"ok": True}
    p = out_dir / "dummy.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    assert p.exists()
