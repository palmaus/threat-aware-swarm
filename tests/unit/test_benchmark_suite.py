"""Проверка генерации отчёта benchmark suite."""

from scripts.analysis.benchmark_suite import _write_markdown


def test_benchmark_suite_markdown(tmp_path):
    out_path = tmp_path / "bench.md"
    results = {
        "baseline:random": {
            "overall": {
                "success_rate": 0.5,
                "alive_frac_end": 1.0,
                "risk_integral_alive": 0.1,
                "energy_efficiency": 2.0,
                "safety_score": 0.9,
            }
        }
    }
    _write_markdown(out_path, results)
    text = out_path.read_text(encoding="utf-8")
    assert "baseline:random" in text
    assert "Benchmark suite" in text
