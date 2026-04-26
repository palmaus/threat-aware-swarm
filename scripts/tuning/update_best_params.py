"""Сбор лучших параметров из результатов тюнинга в единый файл."""

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from scripts.common.metrics_utils import METRIC_KEYS as COMMON_METRIC_KEYS, ScoreWeights, score_scalar
from scripts.common.path_utils import resolve_repo_path

METRIC_KEYS = set(COMMON_METRIC_KEYS)


def _parse_value(raw: str):
    if raw is None:
        return None
    s = str(raw).strip()
    if s == "":
        return None
    lower = s.lower()
    if lower in {"nan", "none", "null"}:
        return float("nan")
    if lower in {"true", "false"}:
        return lower == "true"
    try:
        if "." not in s and "e" not in s and "E" not in s:
            return int(s)
        return float(s)
    except Exception:
        return s


def _score_from_metrics(m: dict, finish_min: float, alive_min: float, weights: ScoreWeights) -> float:
    return score_scalar(m, finish_min=finish_min, alive_min=alive_min, weights=weights)


def _best_row(rows: list[dict]) -> dict | None:
    if not rows:
        return None
    if any("value" in r and r["value"] is not None for r in rows):
        candidates = [r for r in rows if r.get("state", "COMPLETE") == "COMPLETE" and r.get("value") is not None]
        if not candidates:
            candidates = [r for r in rows if r.get("value") is not None]
        return max(candidates, key=lambda r: r["value"]) if candidates else None
    return max(rows, key=lambda r: r["score"])


@dataclass
class UpdateBestParamsConfig:
    tune_dir: str = "runs/tune"
    out: str = "configs/best_policy_params.json"
    finish_min: float = 0.95
    alive_min: float = 0.95
    scoring: ScoreWeights = field(default_factory=ScoreWeights)
    backup: bool = True
    merge_existing: bool = True


def run(cfg: UpdateBestParamsConfig) -> None:
    args = cfg

    tune_dir = resolve_repo_path(args.tune_dir)
    if tune_dir.is_file() and tune_dir.name.startswith("tune_") and tune_dir.suffix == ".csv":
        csv_paths = [tune_dir]
    else:
        csv_paths = sorted(tune_dir.rglob("tune_*.csv"))
    if not csv_paths:
        raise SystemExit(f"No tune_*.csv found in {tune_dir}")

    policies: dict[str, dict] = {}
    meta = {
        "source_dir": str(tune_dir),
        "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "files": [],
    }

    for path in csv_paths:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows = []
            for row in reader:
                policy = row.get("policy") or ""
                if not policy:
                    continue
                metrics = {}
                params = {}
                extra = {}
                for k, v in row.items():
                    if k == "policy":
                        continue
                    val = _parse_value(v)
                    if k in METRIC_KEYS:
                        metrics[k] = val
                    elif k in {"value", "state"}:
                        extra[k] = val
                    else:
                        params[k] = val
                score = _score_from_metrics(metrics, args.finish_min, args.alive_min, args.scoring)
                rows.append(
                    {
                        "policy": policy,
                        "params": params,
                        "metrics": metrics,
                        "score": score,
                        **extra,
                    }
                )

        if not rows:
            continue

        policy_name = rows[0]["policy"]
        best = _best_row(rows)
        if best is None:
            continue

        policies[policy_name] = best["params"]
        meta["files"].append(
            {
                "policy": policy_name,
                "path": str(path),
                "best_score": best.get("score"),
                "best_value": best.get("value"),
            }
        )

    out_path = resolve_repo_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict = {}
    if args.merge_existing and out_path.exists():
        try:
            existing = json.loads(out_path.read_text()) or {}
        except Exception:
            existing = {}

    merged = dict(existing) if isinstance(existing, dict) else {}
    base_policies = merged.get("policies")
    if not isinstance(base_policies, dict):
        base_policies = {}
    base_policies.update(policies)
    merged["policies"] = base_policies
    merged["meta"] = meta

    if args.backup and out_path.exists():
        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        backup_path = Path(str(out_path) + f".bak_{stamp}")
        backup_path.write_text(out_path.read_text(), encoding="utf-8")
        meta["backup"] = str(backup_path)

    out_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] {out_path}")


def main() -> None:
    import hydra
    from omegaconf import DictConfig, OmegaConf

    from scripts.common.hydra_utils import apply_schema

    @hydra.main(version_base=None, config_path="../../configs/hydra", config_name="tuning/update_best_params")
    def _run(cfg: DictConfig) -> None:
        cfg = apply_schema(cfg, UpdateBestParamsConfig)
        data = OmegaConf.to_container(cfg, resolve=True)
        run(UpdateBestParamsConfig(**data))

    _run()


if __name__ == "__main__":
    main()
