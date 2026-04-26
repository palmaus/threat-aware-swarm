"""
Анализирует скаляры TensorBoard и ищет точки деградации.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from scripts.common.path_utils import resolve_repo_path


@dataclass
class ChangePoint:
    tag: str
    step: int
    delta: float
    direction: str
    value_before: float
    value_after: float


def _find_event_files(tb_dir: Path) -> list[Path]:
    if not tb_dir.exists():
        return []
    return sorted(tb_dir.glob("**/events.out.tfevents*"))


def _load_scalars(tb_dir: Path, tag: str) -> list[tuple[int, float]]:
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except Exception:
        return []

    series: list[tuple[int, float]] = []
    for f in _find_event_files(tb_dir):
        try:
            ea = event_accumulator.EventAccumulator(str(f))
            ea.Reload()
            if tag not in ea.Tags().get("scalars", []):
                continue
            for s in ea.Scalars(tag):
                series.append((int(s.step), float(s.value)))
        except Exception:
            continue

    if not series:
        return []
    series.sort(key=lambda x: x[0])
    return series


def _change_point(series: list[tuple[int, float]], window: int = 20) -> ChangePoint | None:
    if len(series) < max(10, window * 2):
        return None

    steps = np.array([s for s, _ in series], dtype=np.int64)
    values = np.array([v for _, v in series], dtype=np.float32)

    # Сглаживаем шум скаляров, чтобы выделить устойчивые изменения.
    kernel = np.ones(window, dtype=np.float32) / float(window)
    smooth = np.convolve(values, kernel, mode="valid")
    smooth_steps = steps[window - 1 :]

    diffs = np.diff(smooth)
    if diffs.size == 0:
        return None

    idx = int(np.argmax(np.abs(diffs)))
    delta = float(diffs[idx])
    step = int(smooth_steps[idx + 1])
    direction = "up" if delta > 0 else "down"
    value_before = float(smooth[idx])
    value_after = float(smooth[idx + 1])
    return ChangePoint("", step, delta, direction, value_before, value_after)


def _summarize_tag(tb_dir: Path, tag: str) -> tuple[ChangePoint | None, float | None]:
    series = _load_scalars(tb_dir, tag)
    if not series:
        return None, None

    cp = _change_point(series)
    last_val = series[-1][1]
    if cp is not None:
        cp.tag = tag
    return cp, float(last_val)


def _hypotheses(changes: dict[str, ChangePoint]) -> list[str]:
    notes: list[str] = []

    entropy = changes.get("train/entropy_loss")
    kl = changes.get("train/approx_kl")
    clip = changes.get("train/clip_fraction")
    ev = changes.get("train/explained_variance")
    finished = changes.get("swarm/finished_given_alive")
    mean_dist = changes.get("swarm/mean_dist")

    if entropy and entropy.direction == "down" and (kl or clip):
        if (kl and kl.direction == "up") or (clip and clip.direction == "up"):
            notes.append(
                "Возможны слишком большие обновления политики: энтропия падает, а approx_kl/clip_fraction растут."
            )

    if ev and ev.direction == "down" and mean_dist and mean_dist.direction == "up":
        notes.append("Возможен дрейф value‑функции: explained_variance падает, mean_dist ухудшается.")

    if finished and finished.direction == "down" and entropy and entropy.direction == "down":
        notes.append("Возможен коллапс политики: finished_given_alive падает вместе с энтропией.")

    return notes[:3]


@dataclass
class AnalyzeTrainingConfig:
    run_dir: str = ""
    out_root: str = "runs"
    out_dir: str = ""


def run(cfg: AnalyzeTrainingConfig) -> None:
    args = cfg

    run_dir = resolve_repo_path(args.run_dir)
    tb_dir = run_dir / "tb"
    if args.out_dir:
        out_dir = resolve_repo_path(args.out_dir)
    else:
        out_dir = resolve_repo_path(args.out_root) / "analysis" / run_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    tags = [
        "swarm/finished_given_alive",
        "swarm/mean_dist",
        "train/entropy_loss",
        "train/approx_kl",
        "train/clip_fraction",
        "train/explained_variance",
    ]

    changes: dict[str, ChangePoint] = {}
    last_vals: dict[str, float] = {}
    for tag in tags:
        cp, last = _summarize_tag(tb_dir, tag)
        if cp is not None:
            changes[tag] = cp
        if last is not None:
            last_vals[tag] = last

    # Первая точка нужна, чтобы интерпретировать первичный сбой.
    first = None
    if changes:
        first = sorted(changes.values(), key=lambda c: c.step)[0]

    # Последующие точки показывают цепочку деградаций после первого сдвига.
    follow = []
    if first is not None:
        follow = [c for c in changes.values() if c.step > first.step]
        follow.sort(key=lambda c: c.step)

    hypotheses = _hypotheses(changes)

    run_id = run_dir.name
    out_path = out_dir / "analysis.md"

    lines = []
    lines.append(f"# Анализ обучения: {run_id}")
    lines.append("")
    lines.append(f"Сгенерировано: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")

    if first is None:
        lines.append("Точки смены режима не найдены (мало данных или нет TensorBoard).")
    else:
        lines.append("## Что сломалось первым")
        lines.append(
            f"- {first.tag} @ step {first.step}: {first.direction} (Δ={first.delta:.4f}, {first.value_before:.4f} → {first.value_after:.4f})"
        )

        if follow:
            lines.append("")
            lines.append("## Что изменилось дальше")
            for c in follow[:5]:
                lines.append(
                    f"- {c.tag} @ step {c.step}: {c.direction} (Δ={c.delta:.4f}, {c.value_before:.4f} → {c.value_after:.4f})"
                )

    if last_vals:
        lines.append("")
        lines.append("## Последние значения")
        for tag in tags:
            if tag in last_vals:
                lines.append(f"- {tag}: {last_vals[tag]:.4f}")

    lines.append("")
    lines.append("## Возможные причины")
    if hypotheses:
        for h in hypotheses:
            lines.append(f"- {h}")
    else:
        lines.append("- Недостаточно данных, чтобы предложить причины.")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[ОК] Анализ -> {out_path}")


def main() -> None:
    import hydra
    from omegaconf import DictConfig, OmegaConf

    from scripts.common.hydra_utils import apply_schema

    @hydra.main(version_base=None, config_path="../../configs/hydra", config_name="analysis/analyze_training")
    def _run(cfg: DictConfig) -> None:
        cfg = apply_schema(cfg, AnalyzeTrainingConfig)
        data = OmegaConf.to_container(cfg, resolve=True)
        run(AnalyzeTrainingConfig(**data))

    _run()


if __name__ == "__main__":
    main()
