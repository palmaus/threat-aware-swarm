from types import SimpleNamespace

import scripts.debug.monitor_tuning_memory as mtm


def test_summarize_returns_peaks():
    samples = [
        mtm.MemorySample(0.0, 1, 100.0, 150.0, 1, 1, 1000.0, 500.0, 200.0, 0.0),
        mtm.MemorySample(1.0, 1, 120.0, 310.0, 3, 2, 1400.0, 300.0, 260.0, 128.0),
    ]
    summary = mtm._summarize(samples)
    assert summary["samples"] == 2
    assert summary["peak_tree_rss_mb"] == 310.0
    assert summary["peak_spawn_workers"] == 2
    assert summary["peak_swap_used_mb"] == 128.0
    assert summary["final_tree_rss_mb"] == 310.0


def test_collect_sample_counts_spawn_workers(monkeypatch):
    class DummyProc:
        def __init__(self, pid, rss, cmd):
            self.pid = pid
            self._rss = rss
            self._cmd = cmd

        def memory_info(self):
            return SimpleNamespace(rss=self._rss)

        def children(self, recursive=True):
            assert recursive is True
            return [
                DummyProc(11, 50 * 1024 * 1024, "python3 -c from multiprocessing.spawn import spawn_main"),
                DummyProc(12, 25 * 1024 * 1024, "python3 worker.py"),
            ]

        def cmdline(self):
            return self._cmd.split()

    monkeypatch.setattr(
        mtm.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(used=2 * 1024**3, available=6 * 1024**3, cached=512 * 1024**2),
    )
    monkeypatch.setattr(
        mtm.psutil,
        "swap_memory",
        lambda: SimpleNamespace(used=256 * 1024**2),
    )

    sample = mtm._collect_sample(DummyProc(10, 100 * 1024 * 1024, "python3 main.py"), elapsed_s=1.5)
    assert sample.main_rss_mb == 100.0
    assert sample.tree_rss_mb == 175.0
    assert sample.child_count == 2
    assert sample.spawn_worker_count == 1
