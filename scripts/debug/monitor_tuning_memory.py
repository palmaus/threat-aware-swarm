"""Мониторинг памяти для долгих tuning/debug запусков.

Скрипт запускает указанную команду, раз в N секунд снимает:
- RSS основного процесса;
- суммарный RSS дерева процессов;
- число `spawn_main` worker'ов;
- системную память/кэш/подкачку.

Вывод нужен, чтобы отличить:
1. рост heap/RSS самого тюнера;
2. сиротские `spawn`-worker'ы;
3. рост системного page cache / swap.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import psutil


@dataclass
class MemorySample:
    elapsed_s: float
    pid: int
    main_rss_mb: float
    tree_rss_mb: float
    child_count: int
    spawn_worker_count: int
    system_used_mb: float
    system_available_mb: float
    system_cached_mb: float
    swap_used_mb: float


def _cmdline(proc: psutil.Process) -> str:
    try:
        return " ".join(proc.cmdline())
    except Exception:
        return ""


def _collect_sample(proc: psutil.Process, *, elapsed_s: float) -> MemorySample:
    processes = [proc]
    try:
        processes.extend(proc.children(recursive=True))
    except Exception:
        pass

    tree_rss = 0
    spawn_workers = 0
    for item in processes:
        try:
            tree_rss += int(item.memory_info().rss)
        except Exception:
            continue
        if "from multiprocessing.spawn import spawn_main" in _cmdline(item):
            spawn_workers += 1

    main_rss = 0
    try:
        main_rss = int(proc.memory_info().rss)
    except Exception:
        pass

    vm = psutil.virtual_memory()
    swap = psutil.swap_memory()
    cached = getattr(vm, "cached", 0) or 0
    return MemorySample(
        elapsed_s=round(elapsed_s, 2),
        pid=int(proc.pid),
        main_rss_mb=round(main_rss / 1024 / 1024, 1),
        tree_rss_mb=round(tree_rss / 1024 / 1024, 1),
        child_count=max(0, len(processes) - 1),
        spawn_worker_count=spawn_workers,
        system_used_mb=round(vm.used / 1024 / 1024, 1),
        system_available_mb=round(vm.available / 1024 / 1024, 1),
        system_cached_mb=round(cached / 1024 / 1024, 1),
        swap_used_mb=round(swap.used / 1024 / 1024, 1),
    )


def _summarize(samples: list[MemorySample]) -> dict[str, float | int]:
    if not samples:
        return {
            "samples": 0,
            "peak_tree_rss_mb": 0.0,
            "peak_main_rss_mb": 0.0,
            "peak_spawn_workers": 0,
            "peak_system_used_mb": 0.0,
            "peak_system_cached_mb": 0.0,
            "peak_swap_used_mb": 0.0,
        }
    return {
        "samples": len(samples),
        "peak_tree_rss_mb": max(item.tree_rss_mb for item in samples),
        "peak_main_rss_mb": max(item.main_rss_mb for item in samples),
        "peak_spawn_workers": max(item.spawn_worker_count for item in samples),
        "peak_system_used_mb": max(item.system_used_mb for item in samples),
        "peak_system_cached_mb": max(item.system_cached_mb for item in samples),
        "peak_swap_used_mb": max(item.swap_used_mb for item in samples),
        "final_tree_rss_mb": samples[-1].tree_rss_mb,
        "final_system_used_mb": samples[-1].system_used_mb,
        "final_system_cached_mb": samples[-1].system_cached_mb,
        "final_swap_used_mb": samples[-1].swap_used_mb,
    }


def _write_csv(path: Path, samples: list[MemorySample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(samples[0]).keys()) if samples else list(asdict(MemorySample(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)).keys()))
        writer.writeheader()
        for sample in samples:
            writer.writerow(asdict(sample))


def main() -> int:
    parser = argparse.ArgumentParser(description="Мониторинг памяти дерева процессов для тюнинга.")
    parser.add_argument("--interval", type=float, default=1.0, help="Интервал сэмплирования, секунды.")
    parser.add_argument("--timeout", type=float, default=0.0, help="Жёсткий timeout; 0 = без лимита.")
    parser.add_argument("--csv", type=Path, required=True, help="Куда писать CSV со сэмплами.")
    parser.add_argument("--summary-json", type=Path, required=True, help="Куда писать summary JSON.")
    parser.add_argument("--cwd", type=Path, default=Path.cwd(), help="Рабочая директория команды.")
    parser.add_argument("cmd", nargs=argparse.REMAINDER, help="Команда после `--`.")
    args = parser.parse_args()

    command = list(args.cmd)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit("Нужно передать команду после `--`.")

    start = time.time()
    proc = subprocess.Popen(command, cwd=args.cwd, preexec_fn=os.setsid)
    process = psutil.Process(proc.pid)
    samples: list[MemorySample] = []
    timed_out = False
    exit_code = None

    try:
        while True:
            elapsed = time.time() - start
            samples.append(_collect_sample(process, elapsed_s=elapsed))
            exit_code = proc.poll()
            if exit_code is not None:
                break
            if args.timeout > 0 and elapsed >= args.timeout:
                timed_out = True
                os.killpg(proc.pid, signal.SIGTERM)
                exit_code = proc.wait(timeout=10)
                break
            time.sleep(max(0.1, float(args.interval)))
    finally:
        if exit_code is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except Exception:
                pass
            try:
                exit_code = proc.wait(timeout=10)
            except Exception:
                exit_code = -1

    summary = _summarize(samples)
    summary.update(
        {
            "command": command,
            "cwd": str(args.cwd),
            "returncode": int(exit_code),
            "timed_out": bool(timed_out),
            "elapsed_s": round(time.time() - start, 2),
        }
    )

    _write_csv(args.csv, samples)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return int(exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
