from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from scripts.common.path_utils import get_project_root, get_runs_dir, resolve_repo_path

METRICS = [
    "success_rate",
    "alive_frac_end",
    "risk_integral_alive",
    "time_to_goal_mean",
    "energy_efficiency",
    "safety_score",
]

METRIC_HIGHER_BETTER = {
    "success_rate": True,
    "alive_frac_end": True,
    "risk_integral_alive": False,
    "time_to_goal_mean": False,
    "energy_efficiency": True,
    "safety_score": True,
}


def _policy_group(payload: dict) -> str:
    policy = payload.get("policy") or payload.get("config", {}).get("policy") or ""
    policy = str(policy).strip().lower()
    if policy.startswith("ppo"):
        return "ppo"
    return "baseline"


def _find_latest_by_group(root: Path) -> dict[str, tuple[Path, dict]]:
    if not root.exists():
        return {}
    results: dict[str, tuple[Path, dict]] = {}
    candidates = sorted(root.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        group = _policy_group(payload)
        if group not in results:
            results[group] = (path, payload)
        if "baseline" in results and "ppo" in results:
            break
    return results


def _format_stat(stat: dict) -> str:
    mean = stat.get("mean")
    std = stat.get("std")
    if mean is None or mean != mean:
        return "-"
    if std is None or std != std:
        return f"{mean:.4f}"
    return f"{mean:.4f} ± {std:.4f}"


def _improvement_percent(base: float | None, ppo: float | None, *, higher_better: bool) -> float | None:
    if base is None or ppo is None:
        return None
    if base != base or ppo != ppo:
        return None
    denom = abs(base)
    if denom < 1e-8:
        denom = 1.0
    if higher_better:
        return 100.0 * (ppo - base) / denom
    return 100.0 * (base - ppo) / denom


def _build_table(aggregate: dict[str, dict]) -> list[str]:
    lines = ["| Metric | Mean ± Std | 95% CI | N |", "| --- | --- | --- | --- |"]
    for key in METRICS:
        stat = aggregate.get(key, {})
        ci = stat.get("ci95")
        n = stat.get("n")
        ci_str = "-" if ci is None or ci != ci else f"{ci:.4f}"
        n_str = "-" if n is None else str(int(n))
        lines.append(f"| {key} | {_format_stat(stat)} | {ci_str} | {n_str} |")
    return lines


def _render_section(title: str, aggregate: dict[str, dict]) -> list[str]:
    lines = [f"#### {title}", ""]
    lines.extend(_build_table(aggregate))
    return lines


def _render_block(
    baseline: dict[str, dict] | None,
    ppo: dict[str, dict] | None,
    image_path: str | None,
    compare_path: str | None,
) -> str:
    lines = [f"_Updated: {datetime.now().isoformat(timespec='seconds')}_", ""]
    if image_path:
        lines.append(f"![Demo Metrics]({image_path})")
        lines.append("")
    if compare_path:
        lines.append(f"![Demo Comparison]({compare_path})")
        lines.append("")
    if baseline and ppo:
        diffs = []
        for key in METRICS:
            base_mean = baseline.get(key, {}).get("mean")
            ppo_mean = ppo.get(key, {}).get("mean")
            diff = _improvement_percent(base_mean, ppo_mean, higher_better=METRIC_HIGHER_BETTER.get(key, True))
            if diff is not None:
                diffs.append(diff)
        if diffs:
            avg = sum(diffs) / float(len(diffs))
            lines.append(f"**Baseline vs PPO % improvement (mean across metrics): {avg:+.2f}%**")
            lines.append("")
    if baseline:
        lines.extend(_render_section("Baseline", baseline))
    else:
        lines.append("#### Baseline")
        lines.append("")
        lines.append("_Нет данных. Запусти demo_pack для baseline._")
    lines.append("")
    if ppo:
        lines.extend(_render_section("PPO", ppo))
    else:
        lines.append("#### PPO")
        lines.append("")
        lines.append("_Нет данных. Установи DEMO_PPO_MODEL и запусти demo_pack._")
    return "\n".join(lines)


def _replace_block(text: str, start: str, end: str, body: str) -> str:
    if start not in text or end not in text:
        raise ValueError("README markers not found")
    head, rest = text.split(start, 1)
    _, tail = rest.split(end, 1)
    return f"{head}{start}\n{body}\n{end}{tail}"


def _write_image(path: Path, baseline: dict[str, dict] | None, ppo: dict[str, dict] | None) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False
    rows = 1 + (1 if ppo else 0)
    fig, axes = plt.subplots(rows, 1, figsize=(8.2, 2.4 * rows))
    if rows == 1:
        axes = [axes]

    def add_table(ax, title: str, aggregate: dict[str, dict]) -> None:
        # Простая визуализация таблицы для README/GIF.
        ax.axis("off")
        ax.set_title(title, fontsize=10, pad=6)
        data = []
        for key in METRICS:
            stat = aggregate.get(key, {})
            ci = stat.get("ci95")
            n = stat.get("n")
            ci_str = "-" if ci is None or ci != ci else f"{ci:.4f}"
            n_str = "-" if n is None else str(int(n))
            data.append([key, _format_stat(stat), ci_str, n_str])
        table = ax.table(
            cellText=data,
            colLabels=["Metric", "Mean ± Std", "95% CI", "N"],
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.2)

    if baseline:
        add_table(axes[0], "Baseline", baseline)
    if ppo and rows > 1:
        add_table(axes[1], "PPO", ppo)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return True


def _write_compare_image(path: Path, baseline: dict[str, dict], ppo: dict[str, dict], svg_path: Path | None = None) -> bool:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return False

    labels = list(METRICS)
    diffs = []
    for key in labels:
        b = baseline.get(key, {}).get("mean", float("nan"))
        p = ppo.get(key, {}).get("mean", float("nan"))
        diff = _improvement_percent(
            b if b == b else None,
            p if p == p else None,
            higher_better=METRIC_HIGHER_BETTER.get(key, True),
        )
        diffs.append(0.0 if diff is None else float(diff))

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ["#22C55E" if v >= 0 else "#EF4444" for v in diffs]
    ax.bar(x, diffs, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Improvement, %")
    ax.set_title("Demo Pack: PPO vs Baseline (% improvement)")
    ax.axhline(0.0, color="#94A3B8", linewidth=1)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    if svg_path is not None:
        svg_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(svg_path)
    plt.close(fig)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Генерация demo-таблицы для README.")
    parser.add_argument("--input-baseline", default="", help="JSON eval_protocol для baseline.")
    parser.add_argument("--input-ppo", default="", help="JSON eval_protocol для PPO.")
    parser.add_argument("--out", default="", help="Выходной Markdown-файл.")
    parser.add_argument("--update-readme", action="store_true", help="Обновить README между маркерами.")
    parser.add_argument("--readme", default="README.md", help="Путь к README.")
    parser.add_argument("--image-out", default="docs/images/demo_metrics.png", help="Путь к PNG-таблице.")
    parser.add_argument(
        "--compare-image-out",
        default="docs/images/demo_metrics_compare.png",
        help="Путь к PNG-графику сравнения baseline vs PPO.",
    )
    parser.add_argument(
        "--compare-image-svg",
        default="docs/images/demo_metrics_compare.svg",
        help="Путь к SVG-графику сравнения baseline vs PPO.",
    )
    args = parser.parse_args()

    eval_root = get_runs_dir() / "eval_protocol"
    baseline_payload = None
    ppo_payload = None
    if args.input_baseline:
        baseline_payload = json.loads(resolve_repo_path(args.input_baseline).read_text(encoding="utf-8"))
    if args.input_ppo:
        ppo_payload = json.loads(resolve_repo_path(args.input_ppo).read_text(encoding="utf-8"))
    if baseline_payload is None or (ppo_payload is None):
        latest = _find_latest_by_group(eval_root)
        if baseline_payload is None and "baseline" in latest:
            baseline_payload = latest["baseline"][1]
        if ppo_payload is None and "ppo" in latest:
            ppo_payload = latest["ppo"][1]
    if baseline_payload is None:
        raise SystemExit("Не найден baseline eval_protocol JSON. Запусти demo_pack или укажи --input-baseline.")

    baseline = baseline_payload.get("aggregate") or {}
    ppo = ppo_payload.get("aggregate") if ppo_payload else None

    image_path = resolve_repo_path(args.image_out)
    image_written = _write_image(image_path, baseline, ppo)
    if image_written:
        try:
            image_ref = str(image_path.relative_to(get_project_root())).replace("\\", "/")
        except Exception:
            image_ref = str(image_path).replace("\\", "/")
    else:
        image_ref = None

    compare_ref = None
    if ppo:
        compare_path = resolve_repo_path(args.compare_image_out)
        compare_svg_path = resolve_repo_path(args.compare_image_svg)
        compare_written = _write_compare_image(compare_path, baseline, ppo, svg_path=compare_svg_path)
        if compare_written:
            try:
                compare_ref = str(compare_svg_path.relative_to(get_project_root())).replace("\\", "/")
            except Exception:
                compare_ref = str(compare_svg_path).replace("\\", "/")

    block = _render_block(baseline, ppo, image_ref, compare_ref)

    if args.out:
        out_path = resolve_repo_path(args.out)
    else:
        out_path = get_runs_dir() / "demo_pack" / "README_metrics.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(block, encoding="utf-8")

    if args.update_readme:
        readme_path = resolve_repo_path(args.readme)
        text = readme_path.read_text(encoding="utf-8")
        updated = _replace_block(
            text,
            "<!-- DEMO_METRICS_START -->",
            "<!-- DEMO_METRICS_END -->",
            block,
        )
        readme_path.write_text(updated, encoding="utf-8")

    print("[demo_report] baseline=", "ok" if baseline_payload else "missing")
    print("[demo_report] ppo=", "ok" if ppo_payload else "missing")
    print(f"[demo_report] out={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
